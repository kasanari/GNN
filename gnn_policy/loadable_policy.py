import gymnasium as gym
import torch as th
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from gnn_policy.functional import (
    sample_action_and_node,
    sample_action_then_node,
    sample_node_then_action,
)
from gnn_policy.gnn_extractor import GNNExtractor
from gnn_policy.gnns import MultiMessagePassing
from gnn_policy.gnns.graph_net_local import LocalMultiMessagePassing
from gnn_policy.node_extractor import NodeExtractor


def create_masks(obs: dict[str, Tensor]):
    action_mask = obs["mask_0"].bool()
    node_mask = obs["mask_1"].bool().flatten()
    return action_mask, node_mask


def collate(obs: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """
    Preprocess the observation if needed and extract features.

    :param obs: Observation
    :param features_extractor: The features extractor to use.
    :return: The extracted features
    """

    datalist: list[Data] = [
        Data(
            x=obs["nodes"][i].float(),
            edge_index=obs["edge_index"][i].T.long(),
            edge_attr=th.zeros(0),
        )
        for i in range(obs["nodes"].shape[0])
    ]

    batch = Batch.from_data_list(datalist)

    return batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.num_graphs


def gnn_str_to_class(gnn_class: str) -> type[nn.Module]:
    if gnn_class == "LocalMultiMessagePassing":
        return LocalMultiMessagePassing
    elif gnn_class == "MultiMessagePassing":
        return MultiMessagePassing
    else:
        raise ValueError(f"Unknown GNN class: {gnn_class}")


class LoadablePolicy(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_mode: str,
        separate_actor_critic: bool,
        gnn_steps: int,
        gnn_class: str,
        node_feature_dim: int,
        num_actions: int,
        embedding_dim: int,
        device: th.device | str = "auto",
    ):
        super().__init__()
        activation_fn = nn.Mish

        self.device = device
        edge_dim = 2

        gnn_class = gnn_str_to_class(gnn_class)

        self.features_extractor = NodeExtractor(
            observation_space,
            node_feature_dim,
            output_dim=embedding_dim,
            activation_fn=activation_fn,
        )

        self.edge_dim = 1

        self.create_masks = create_masks
        self.collate = collate

        self.sde_features_extractor = None
        self.action_order: str = action_mode

        self.gnn_extractor = GNNExtractor(
            gnn_class,
            embedding_dim,
            edge_dim=edge_dim,
            activation_fn=activation_fn,
            device=device,
            steps=gnn_steps,
        )

        if separate_actor_critic:
            self.vf_gnn_extractor = GNNExtractor(
                gnn_class,
                embedding_dim,
                edge_dim=edge_dim,
                activation_fn=activation_fn,
                device=device,
                steps=gnn_steps,
            )

        self.activation_fn = activation_fn
        self.gnn_steps: int = gnn_steps
        self.emb_size: int = embedding_dim
        self.separate_actor_critic: bool = separate_actor_critic
        self.action_net: nn.Module
        self.action_net2: nn.Module

        if action_mode == "action_then_node":
            self.action_func = self._sample_action_then_node
            self.action_net = nn.Linear(embedding_dim, num_actions)
            self.action_net2 = nn.Linear(embedding_dim, num_actions)
        elif action_mode == "node_then_action":
            self.action_func = self._sample_node_then_action
            self.action_net = nn.Linear(embedding_dim, 1)
            self.action_net2 = nn.Linear(embedding_dim, num_actions)
        elif action_mode == "independent":
            self.action_func = self._sample_action_and_node
            self.action_net = nn.Linear(embedding_dim, num_actions)
            self.action_net2 = nn.Linear(embedding_dim, 1)

        self.value_net = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            activation_fn(),
            nn.Linear(self.emb_size, 1),
        )

    def _get_latent(
        self, obs: dict[str, Tensor]
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        nodes, edge_index, _, batch_idx, num_graphs = self.collate(obs)

        node_embed = self.features_extractor(nodes)

        latent_nodes, latent_global = self.gnn_extractor(
            node_embed, edge_index, batch_idx, num_graphs
        )

        latent_vf = (
            self.vf_gnn_extractor(node_embed, edge_index, batch_idx, num_graphs)[1]
            if self.separate_actor_critic
            else latent_global
        )

        return latent_nodes, latent_global, batch_idx, latent_vf

    def _get_action_from_latent(
        self,
        obs: dict[str, Tensor],
        node_latent: Tensor,
        graph_latent: Tensor,
        batch_idx: Tensor,
        deterministic: bool = False,
        eval_action: Tensor | None = None,
    ):
        action_mask, node_mask = self.create_masks(obs)

        return self.action_func(
            node_latent,
            graph_latent,
            action_mask,
            node_mask,
            batch_idx,
            eval_action,
        )

    def _sample_node_then_action(
        self,
        node_latent: Tensor,
        graph_latent: Tensor,
        action_mask: Tensor,
        node_mask: Tensor,
        batch: Tensor,
        eval_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x1 = self.action_net(node_latent).squeeze(-1)
        x2 = self.action_net2(node_latent)
        # note that the action_masks are reversed here, since we now pick the node first
        return sample_node_then_action(
            x1, x2, node_mask, action_mask, batch, eval_action
        )

    def _sample_action_and_node(
        self,
        node_latent: Tensor,
        graph_latent: Tensor,
        action_mask: Tensor,
        node_mask: Tensor,
        batch: Tensor,
        eval_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x1 = self.action_net(graph_latent)
        x2 = self.action_net2(node_latent)
        return sample_action_and_node(
            x1, x2, action_mask, node_mask, batch, eval_action
        )

    def _sample_action_then_node(
        self,
        node_latent: Tensor,
        graph_latent: Tensor,
        action_mask: Tensor,
        node_mask: Tensor,
        batch: Tensor,
        eval_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x1 = self.action_net(graph_latent)
        x2 = self.action_net2(node_latent)
        return sample_action_then_node(
            x1, x2, action_mask, node_mask, batch, eval_action
        )

    def forward(self, obs: dict[str, Tensor], deterministic: bool = False):
        latent_nodes, latent_global, batch_idx, _ = self._get_latent(obs)
        return self._get_action_from_latent(obs, latent_nodes, latent_global, batch_idx)
