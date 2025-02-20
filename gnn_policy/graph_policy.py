import collections
from functools import partial
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import Tensor, nn

from .functional import (
    eval_action_and_node,
    eval_action_then_node,
    eval_node_then_action,
    sample_action_and_node,
    sample_action_then_node,
    sample_node,
    sample_node_then_action,
    segmented_gather,
)
from .gnn_extractor import GNNExtractor
from .node_extractor import NodeExtractor


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
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
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
        net_arch: list[int | dict[str, list[int]]] | None = None,
        activation_fn: type[nn.Module] = nn.LeakyReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: list[int] | None = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = NodeExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(GNNPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        gnn_steps = int(kwargs.pop("gnn_steps"))
        emb_size = (
            features_extractor_kwargs.get("features_dim", 32)
            if features_extractor_kwargs
            else 32
        )

        separate_actor_critic = bool(kwargs.pop("separate_actor_critic"))

        action_mode = str(kwargs.pop("action_mode"))

        features_extractor = features_extractor_class(
            observation_space,
            input_dim=observation_space["nodes"].shape[-1],
            activation_fn=activation_fn,
            output_dim=emb_size,
        )  # TODO

        edge_dim = 1

        log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": sde_net_arch is not None,
            }

        gnn_class = kwargs.pop("gnn_class")
        device = self.device
        # Action distribution

        gnn_extractor = GNNExtractor(
            gnn_class,
            emb_size,
            edge_dim=edge_dim,
            activation_fn=activation_fn,
            device=device,
            steps=gnn_steps,
        )

        if separate_actor_critic:
            vf_gnn_extractor = GNNExtractor(
                gnn_class,
                emb_size,
                edge_dim=edge_dim,
                activation_fn=activation_fn,
                device=device,
                steps=gnn_steps,
            )

        latent_dim_pi = emb_size

        action_dist = make_proba_distribution(
            action_space, use_sde=use_sde, dist_kwargs=dist_kwargs
        )

        action_net: nn.Module
        action_net2: nn.Module
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            num_actions = action_space.nvec[0]
            if action_mode == "action_then_node":
                action_func = self._sample_action_then_node
                action_net = nn.Linear(emb_size, num_actions)
                action_net2 = nn.Linear(emb_size, num_actions)
            elif action_mode == "node_then_action":
                action_func = self._sample_node_then_action
                action_net = nn.Linear(emb_size, 1)
                action_net2 = nn.Linear(emb_size, num_actions)
            elif action_mode == "independent":
                action_func = self._sample_action_and_node
                action_net = nn.Linear(emb_size, num_actions)
                action_net2 = nn.Linear(emb_size, 1)

        elif isinstance(action_dist, CategoricalDistribution):
            action_net = nn.Linear(emb_size, 1)

        elif isinstance(action_dist, DiagGaussianDistribution):
            action_net, log_std = action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=log_std_init
            )
        elif isinstance(action_dist, MultiCategoricalDistribution):
            action_net = action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(action_dist, BernoulliDistribution):
            action_net = action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{action_dist}'.")

        value_net = nn.Sequential(
            nn.Linear(emb_size, emb_size), activation_fn(), nn.Linear(emb_size, 1)
        )
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if ortho_init:
            module_gains = {
                features_extractor: np.sqrt(2),
                gnn_extractor: np.sqrt(2),
                # vf_gnn_extractor: np.sqrt(2),
                action_net: 0.01,
                action_net2: 0.01,
                value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs
        self.action_order = action_mode
        self.action_dist = action_dist
        self.create_masks: Callable[[dict[str, Any]], tuple[Tensor, Tensor]] = mask_func
        self.collate: Callable[[dict[str, Any]], tuple[Tensor, Tensor]] = batch_func
        self.gnn_extractor = gnn_extractor
        self.vf_gnn_extractor = vf_gnn_extractor
        self.activation_fn = activation_fn
        self.gnn_steps = gnn_steps
        self.emb_size = emb_size
        self.separate_actor_critic = separate_actor_critic
        self.action_net = action_net
        self.action_net2 = action_net2
        self.action_func = action_func
        self.value_net = value_net
        self.features_extractor = features_extractor

        # Note: this has to be done at the end
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **optimizer_kwargs
        )

    def _get_data(self) -> dict[str, Any]:
        data = super(BasePolicy)._get_data()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                sde_net_arch=default_none_kwargs["sde_net_arch"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution
        ), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def forward(
        self, obs: dict, deterministic: bool = False, action_masks=None
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        node_embeds, graph_embeds, batch_idx, vf_embed = self._get_latent(obs)

        values = self.value_net(vf_embed)

        # Evaluate the values for the given observations
        actions, log_prob, *_ = self._get_action_from_latent(
            obs, node_embeds, graph_embeds, batch_idx, deterministic=deterministic
        )
        return actions, values, log_prob

    def _get_latent(
        self, obs: dict[str, Tensor]
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """

        nodes, edge_index, edge_attr, batch_idx, num_graphs = self.collate(obs)

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
        obs,
        node_latent: Tensor,
        graph_latent: Tensor,
        batch_idx: Tensor,
        latent_sde: th.Tensor | None = None,
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

            # # convert the actions to tuples
            # a1 = a1.cpu().numpy()
            # a2 = a2.cpu().numpy()
            # a = list(zip(a1, a2))

            return a1, tot_log_prob, entropy
        else:
            raise ValueError("Invalid action distribution")

    def _get_constructor_parameters(self) -> dict[str, Any]:
        base = super()._get_constructor_parameters()
        added = {
            "action_mode": self.action_order,
            "lr_schedule": self._dummy_schedule,
            "features_extractor_kwargs": {
                "features_dim": self.emb_size,
                "gnn_steps": self.gnn_steps,
            },
            "mask_func": self.create_masks,
            "batch_func": self.collate,
        }
        return {**base, **added}

    def _sample_node_then_action(
        self,
        node_latent: Tensor,
        _,
        action_mask: Tensor,
        node_mask: Tensor,
        batch: Tensor,
        eval_action: Tensor | None = None,
        deterministic: bool = False,
    ):
        x1 = self.action_net(node_latent).squeeze(-1)
        x2 = self.action_net2(node_latent)
        # note that the action_masks are reversed here, since we now pick the node first
        return (
            sample_node_then_action(
                x1,
                x2,
                node_mask,
                action_mask,
                batch,
                deterministic=deterministic,
            )
            if eval_action is None
            else eval_node_then_action(
                eval_action,
                x1,
                x2,
                node_mask,
                action_mask,
                batch,
            )
        )

    def _sample_action_and_node(
        self,
        node_latent: Tensor,
        graph_latent: Tensor,
        action_mask: Tensor,
        node_mask: Tensor,
        batch: Tensor,
        eval_action: Tensor | None = None,
        deterministic: bool = False,
    ):
        x1 = self.action_net(graph_latent)
        x2 = self.action_net2(node_latent)
        return (
            sample_action_and_node(
                x1,
                x2,
                action_mask,
                node_mask,
                batch,
                deterministic=deterministic,
            )
            if eval_action is None
            else eval_action_and_node(
                eval_action,
                x1,
                x2,
                action_mask,
                node_mask,
                batch,
            )
        )

    def _sample_action_then_node(
        self,
        node_latent: Tensor,
        graph_latent: Tensor,
        action_mask: Tensor,
        node_mask: Tensor,
        batch: Tensor,
        eval_action: Tensor | None = None,
        deterministic: bool = False,
    ):
        x1 = self.action_net(graph_latent)
        x2 = self.action_net2(node_latent)
        return (
            sample_action_then_node(
                x1,
                x2,
                action_mask,
                node_mask,
                batch,
                deterministic=deterministic,
            )
            if eval_action is None
            else eval_action_then_node(
                eval_action,
                x1,
                x2,
                action_mask,
                node_mask,
                batch,
            )
        )

    # REQUIRED FOR ON-POLICY ALGORITHMS (in SB3)
    def _predict(
        self, obs: dict[str, Tensor], deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_nodes, latent_global, batch_idx, _ = self._get_latent(obs)
        actions, *_ = self._get_action_from_latent(
            obs, latent_nodes, latent_global, batch_idx, deterministic=deterministic
        )
        return actions

    def get_full_prediction(self, obs: dict[str, Tensor], deterministic: bool = False):
        latent_nodes, latent_global, batch_idx, _ = self._get_latent(obs)
        return self._get_action_from_latent(
            obs, latent_nodes, latent_global, batch_idx, deterministic=deterministic
        )

    # REQUIRED FOR ON-POLICY ALGORITHMS (in SB3)
    def predict_values(self, obs: dict[str, Tensor]) -> th.Tensor:
        _, _, _, vf_embed = self._get_latent(obs)
        values = self.value_net(vf_embed)
        return values

    # REQUIRED FOR ON-POLICY ALGORITHMS
    def evaluate_actions(
        self, obs: dict[str, Tensor], actions: th.Tensor, action_masks: th.Tensor = None
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        latent_nodes, latent_global, batch_idx, vf_embed = self._get_latent(obs)
        log_prob, entropy = self._get_action_from_latent(
            obs, latent_nodes, latent_global, batch_idx, eval_action=actions
        )
        values = self.value_net(vf_embed)
        return values, log_prob, entropy
