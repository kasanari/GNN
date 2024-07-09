from extractor import Extractor
import functional as F
import torch as th
import torch.nn as nn

from graph_net_local import LocalMultiMessagePassing


class GraphActorCriticAgent(nn.Module):
    def __init__(self):
        self.actor_gnn = Extractor()
        self.value_gnn = Extractor()

    def forward():
        latent_nodes, latent_global = self.actor_gnn(
            latent_nodes,
            latent_global,
            edge_attr,
            edge_index,
            batch_idx,
        )

        return action, logprob, value

    def _sample_node_then_action(
        self,
        node_latent,
        _,
        action_mask,
        node_mask,
        batch,
        eval_action=None,
    ):
        x1 = self.predicate_net(node_latent)
        x2 = self.object_net(node_latent)
        return F.sample_node_then_action(
            x1, x2, node_mask, action_mask, batch, eval_action
        )

    def _sample_action_and_node(
        self, node_latent, graph_latent, action_mask, node_mask, batch, eval_action=None
    ):
        x1 = self.predicate_net(graph_latent)
        x2 = self.object_net(node_latent)
        return F.sample_action_and_node(
            x1, x2, action_mask, node_mask, batch, eval_action
        )

    def _sample_action_then_node(
        self, node_latent, graph_latent, action_mask, node_mask, batch, eval_action=None
    ):
        x1 = self.predicate_net(graph_latent)
        x2 = self.object_net(node_latent)
        return F.sample_action_then_node(
            x1, x2, action_mask, node_mask, batch, eval_action
        )
