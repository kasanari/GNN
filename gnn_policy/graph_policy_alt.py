import collections
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    MultiCategoricalDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import Tensor, nn

from .functional import (
    sample_action_and_node,
    sample_action_then_node,
    sample_node,
    sample_node_then_action,
    segmented_gather,
)


class GNNPolicy(BasePolicy):
    """
    Policy class for GNN actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        mask_func,
        batch_func,
        activation_fn: Type[nn.Module],
        ortho_init: bool = True,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super(GNNPolicy, self).__init__(
            observation_space,
            action_space,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        optimizer_kwargs = optimizer_kwargs or {}
        separate_actor_critic = kwargs.pop("separate_actor_critic")
        activation_fn = activation_fn
        ortho_init = ortho_init
        gnn_steps = kwargs.pop("gnn_steps")  # message passing steps
        emb_size = features_extractor_kwargs.pop("features_dim")
        action_mode = kwargs.pop("action_mode")  # action order
        gnn_class = kwargs.pop("gnn_class")
        input_size = observation_space["nodes"].shape[-1]
        action_dist = make_proba_distribution(action_space)
        num_actions = action_space.nvec[0]

        gnn_extractor = gnn_class(
            gnn_steps,
            input_size,
            emb_size,
            emb_size,
            emb_size,
            activation_fn=activation_fn,
        )

        vf_gnn_extractor = (
            gnn_class(
                gnn_steps,
                input_size,
                emb_size,
                emb_size,
                emb_size,
                activation_fn=activation_fn,
            )
            if separate_actor_critic
            else None
        )

        action_funcs = {
            "node_then_action": (
                self._sample_node_then_action,
                lambda: nn.Linear(emb_size, 1),
                lambda: nn.Linear(emb_size, num_actions),
            ),
            "action_then_node": (
                self._sample_action_then_node,
                lambda: nn.Linear(emb_size, num_actions),
                lambda: nn.Linear(emb_size, num_actions),
            ),
            "independent": (
                self._sample_action_and_node,
                lambda: nn.Linear(emb_size, num_actions),
                lambda: nn.Linear(emb_size, 1),
            ),
        }
        value_net = nn.Sequential(nn.Linear(emb_size, 1))

        if isinstance(action_space, gym.spaces.MultiDiscrete):
            action_func, nn0, nn1 = action_funcs[action_mode]
            action_net = nn0()
            action_net2 = nn1()
        elif isinstance(action_dist, CategoricalDistribution):
            action_net = nn.Linear(emb_size, 1)
            action_net2 = None
        elif isinstance(action_dist, MultiCategoricalDistribution):
            action_net = action_dist.proba_distribution_net(latent_dim=emb_size)
            action_net2 = None
        elif isinstance(action_dist, BernoulliDistribution):
            action_net = action_dist.proba_distribution_net(latent_dim=emb_size)
            action_net2 = None
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if ortho_init:
            module_gains = {
                gnn_extractor: np.sqrt(2),
                action_net: 0.01,
                action_net2: 0.01,
                value_net: 0.01,
            }
            if separate_actor_critic:
                module_gains[vf_gnn_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.value_net: nn.Linear = value_net
        self.action_net: nn.Linear = action_net
        self.action_net2: nn.Linear = action_net2
        self.gnn_extractor: nn.Module = gnn_extractor
        self.vf_gnn_extractor: nn.Module = vf_gnn_extractor
        self.action_dist = action_dist
        self.action_func = action_func
        self.input_size = input_size
        self.create_masks = mask_func
        self.collate = batch_func
        self.action_order = action_mode
        self.separate_actor_critic = separate_actor_critic

        # Setup optimizer with initial learning rate
        self.optimizer = optimizer_class(
            self.parameters(), lr=lr_schedule(1), **optimizer_kwargs
        )

    def _get_data(self) -> Dict[str, Any]:
        data = super(BasePolicy)._get_data()

        data.update(
            dict(
                activation_fn=self.activation_fn,
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def forward(
        self, obs: Dict, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        node_embeds, graph_embeds, batch_idx, vf_embed = self._get_latent(obs)

        # Value function
        values = self.value_net(vf_embed)

        actions, log_prob, _ = self._get_action_from_latent(
            obs, node_embeds, graph_embeds, batch_idx, deterministic=deterministic
        )
        return actions, values, log_prob

    def _get_latent(
        self, obs: Dict[str, Tensor]
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """

        nodes, edge_index, edge_attr, batch_idx, num_graphs = self.collate(obs)

        latent_nodes, latent_global = self.gnn_extractor(
            nodes, edge_index, batch_idx, num_graphs
        )

        latent_vf = (
            self.vf_gnn_extractor(nodes, edge_index, batch_idx, num_graphs)[1]
            if self.separate_actor_critic
            else latent_global
        )

        return latent_nodes, latent_global, batch_idx, latent_vf

    def _get_action_from_latent(
        self,
        obs,
        node_latent: Tensor,
        graph_latent: Tensor,
        batch_idx: Tensor,
        deterministic: bool = False,
        eval_action=None,
    ):
        """
        Retrieve action distribution given the latent codes.

        :param latent_nodes: Latent code for the individual nodes
        :param latent_global: Latent code for the whole network
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        # mean_actions = self.action_net(latent_pi)

        action_mask, node_mask = self.create_masks(obs)

        if isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_func(
                node_latent,
                graph_latent,
                action_mask,
                node_mask,
                batch_idx,
                eval_action,
                deterministic=deterministic,
            )

        elif isinstance(self.action_dist, CategoricalDistribution):
            x_a1 = self.action_net(node_latent)
            a1, pa1, entropy, data_starts = sample_node(x_a1, node_mask, batch_idx)
            if eval_action is not None:
                a1 = eval_action.long()

            tot_log_prob = th.log(segmented_gather(pa1, a1, data_starts))

            return a1, tot_log_prob, entropy
        else:
            raise ValueError("Invalid action distribution")

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        base = super()._get_constructor_parameters()
        added = {
            "action_mode": self.action_order,
            "lr_schedule": self._dummy_schedule,
            "features_extractor_kwargs": {
                "features_dim": self.emb_size,
                "gnn_steps": self.gnn_steps,
            },
        }
        return {**base, **added}

    def _sample_node_then_action(
        self,
        node_latent,
        _,
        action_mask,
        node_mask,
        batch,
        eval_action=None,
        deterministic=False,
    ):
        x1 = self.action_net(node_latent)
        x2 = self.action_net2(node_latent)
        # note that the action_masks are reversed here, since we now pick the node first
        return sample_node_then_action(
            x1,
            x2,
            node_mask,
            action_mask,
            batch,
            eval_action,
            deterministic=deterministic,
        )

    def _sample_action_and_node(
        self, node_latent, graph_latent, action_mask, node_mask, batch, eval_action=None
    ):
        x1 = self.action_net(graph_latent)
        x2 = self.action_net2(node_latent)
        return sample_action_and_node(
            x1, x2, action_mask, node_mask, batch, eval_action
        )

    def _sample_action_then_node(
        self,
        node_latent,
        graph_latent,
        action_mask,
        node_mask,
        batch,
        eval_action=None,
        deterministic=False,
    ):
        x1 = self.action_net(graph_latent)
        x2 = self.action_net2(node_latent)
        return sample_action_then_node(
            x1,
            x2,
            action_mask,
            node_mask,
            batch,
            eval_action,
            deterministic=deterministic,
        )

    # REQUIRED FOR ON-POLICY ALGORITHMS (in SB3)
    def _predict(
        self, obs: Dict[str, Tensor], deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_nodes, latent_global, batch_idx, _ = self._get_latent(obs)
        actions, log_prob, entropy = self._get_action_from_latent(
            obs, latent_nodes, latent_global, batch_idx, deterministic=deterministic
        )
        return actions

    # REQUIRED FOR ON-POLICY ALGORITHMS (in SB3)
    def predict_values(self, obs: Dict[str, Tensor]) -> th.Tensor:
        _, _, _, vf_embed = self._get_latent(obs)
        values = self.value_net(vf_embed)
        return values

    # REQUIRED FOR ON-POLICY ALGORITHMS
    def evaluate_actions(
        self, obs: Dict[str, Tensor], actions: th.Tensor, action_masks: th.Tensor = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        latent_nodes, latent_global, batch_idx, vf_embed = self._get_latent(obs)
        _, log_prob, entropy = self._get_action_from_latent(
            obs, latent_nodes, latent_global, batch_idx, eval_action=actions
        )
        values = self.value_net(vf_embed)
        return values, log_prob, entropy
