import collections
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.distributions import (
    BernoulliDistribution, CategoricalDistribution, DiagGaussianDistribution,
    MultiCategoricalDistribution, StateDependentNoiseDistribution,
    make_proba_distribution)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor)
from stable_baselines3.common.type_aliases import Schedule
from torch import Tensor, nn

from .functional import (sample_action_and_node, sample_action_then_node,
                         sample_node, sample_node_then_action,
                         segmented_gather)
from .gnn_extractor import GNNExtractor
from .gnns import MultiMessagePassing
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
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NodeExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
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

        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.gnn_steps = kwargs.pop("gnn_steps")
        self.emb_size = (
            features_extractor_kwargs.get("features_dim", 32)
            if features_extractor_kwargs
            else 32
        )

        self.separate_actor_critic = kwargs.pop("separate_actor_critic")

        action_mode = kwargs.pop("action_mode")

        self.features_extractor = features_extractor_class(
            self.observation_space, self.observation_space['nodes'].shape[-1], **self.features_extractor_kwargs
        ) #TODO
        self.features_dim = self.features_extractor.features_dim
        self.edge_dim = 1

        self.create_masks = mask_func
        self.collate = batch_func    

        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": sde_net_arch is not None,
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs
        self.action_order = action_mode
        self.gnn_class = kwargs.pop("gnn_class")

        # Action distribution
        self.action_dist = make_proba_distribution(
            action_space, use_sde=use_sde, dist_kwargs=dist_kwargs
        )

        self._build(self.gnn_class, action_mode, lr_schedule)

    def _get_data(self) -> Dict[str, Any]:
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

    def _build(self, gnn_class, action_mode, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.gnn_extractor = GNNExtractor(
            gnn_class,
            self.emb_size,
            edge_dim=self.edge_dim,
            activation_fn=self.activation_fn,
            device=self.device,
            steps=self.gnn_steps,
        )

        self.message_embed = nn.Linear(32, self.emb_size)

        if self.separate_actor_critic:
            self.vf_gnn_extractor = GNNExtractor(
                gnn_class,
                self.emb_size,
                edge_dim=self.edge_dim,
                activation_fn=self.activation_fn,
                device=self.device,
                steps=self.gnn_steps,
            )

        emb_size = self.emb_size
        latent_dim_pi = emb_size

        self.action_net: nn.Module
        self.action_net2: nn.Module
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            num_actions = self.action_space.nvec[0]
            if action_mode == "action_then_node":
                self.action_func = self._sample_action_then_node
                self.action_net = nn.Linear(emb_size *2, num_actions)
                self.action_net2 = nn.Linear(emb_size *2, num_actions)
            elif action_mode == "node_then_action":
                self.action_func = self._sample_node_then_action
                self.action_net = nn.Linear(emb_size*2, 1)
                self.action_net2 = nn.Linear(emb_size*2, num_actions)
            elif action_mode == "independent":
                self.action_func = self._sample_action_and_node
                self.action_net = nn.Linear(emb_size*2, num_actions)
                self.action_net2 = nn.Linear(emb_size*2, 1)

            # self.action_net = nn.Linear(EMB_SIZE, 1)
            # self.action_net2 = nn.Linear(EMB_SIZE, 1)
            # self.sel_enc = nn.Sequential(nn.Linear(EMB_SIZE + 1, EMB_SIZE), nn.LeakyReLU() )
            # self.a2 = GNNExtractor(self.edge_dim,steps=2)

        # elif isinstance(self.action_space, NodeAction):
        #     self.action_net = nn.Linear(emb_size, 1)

        # elif isinstance(self.action_space, Autoregressive):
        #     self.action_net = nn.Linear(emb_size, 1)
        #     self.action_net2 = self.action_dist.proba_distribution_net(
        #         latent_dim=emb_size
        #     )
        #     self.sel_enc = nn.Sequential(
        #         nn.Linear(emb_size + 1, emb_size), nn.LeakyReLU()
        #     )
        #     self.a2 = GNNExtractor(
        #         self.edge_dim, steps=2, activation_fn=self.activation_fn
        #     )

        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = nn.Linear(emb_size, 1)

        elif isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.emb_size + self.emb_size, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.gnn_extractor: np.sqrt(2),
                #self.vf_gnn_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.action_net2: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(
        self, obs: Dict, deterministic: bool = False, action_masks=None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        node_embeds, graph_embeds, batch_idx, vf_embed, message_embed = self._get_latent(obs)
        


        values = self.value_net(th.concat([vf_embed, message_embed], dim=-1))

        # Evaluate the values for the given observations
        actions, log_prob, _ = self._get_action_from_latent(
            obs, node_embeds, graph_embeds, batch_idx, message_embed, deterministic=deterministic
        )
        return actions, values, log_prob

    def _get_latent(self, obs: Dict[str, Tensor]) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """

        message_vector = obs['messages'].float()
        message_embed = self.message_embed(message_vector)

        nodes, edge_index, edge_attr, batch_idx, num_graphs = self.collate(obs)
        
        latent_nodes = self.features_extractor(nodes)

        latent_global = th.zeros(
            (num_graphs, self.emb_size), dtype=th.float32, device=latent_nodes.device
        )

        latent_nodes, latent_global = self.gnn_extractor(
            latent_nodes,
            latent_global,
            edge_attr,
            edge_index,
            batch_idx,
        )

        latent_vf = self.vf_gnn_extractor(
            latent_nodes,
            latent_global,
            edge_attr,
            edge_index,
            batch_idx,
        )[1] if self.separate_actor_critic else latent_global

        return latent_nodes, latent_global, batch_idx, latent_vf, message_embed
    


    def _get_action_from_latent(
        self,
        obs,
        node_latent: Tensor,
        graph_latent: Tensor,
        batch_idx: Tensor,
        message_embed,
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
                message_embed,
                eval_action,
                deterministic,
            )

        elif isinstance(self.action_dist, CategoricalDistribution):
            x_a1 = self.action_net(node_latent)
            a1, pa1, data_starts, entropy = sample_node(x_a1, node_mask, batch_idx)
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
        
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        base = super()._get_constructor_parameters()
        added = {
            "action_mode": self.action_order,
            "lr_schedule": self._dummy_schedule,
            "features_extractor_kwargs": {
                "features_dim": self.emb_size,
                "gnn_steps": self.gnn_steps,
                "use_embeddings": self.use_embeddings,
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
        message_embed,
        eval_action=None,
        deterministic=False,
    ):
        x1 = self.action_net(th.cat((node_latent, message_embed), dim=-1))
        x2 = self.action_net2(th.cat((node_latent, message_embed), dim=-1))
        # note that the action_masks are reversed here, since we now pick the node first
        return sample_node_then_action(
            x1, x2, node_mask, action_mask, batch, eval_action, deterministic
        )

    def _sample_action_and_node(
        self, node_latent, graph_latent, action_mask, node_mask, batch, eval_action=None,
        deterministic=False,
    ):
        x1 = self.action_net(graph_latent)
        x2 = self.action_net2(node_latent)
        return sample_action_and_node(
            x1, x2, action_mask, node_mask, batch, eval_action, deterministic
        )

    def _sample_action_then_node(
        self, node_latent, graph_latent, action_mask, node_mask, batch, message_embed:Tensor, eval_action=None,
        deterministic=False,
    ):
        x1 = self.action_net(th.cat((graph_latent, message_embed), dim=-1))
        x2 = self.action_net2(th.cat((node_latent, message_embed[batch]), dim=-1))
        return sample_action_then_node(
            x1, x2, action_mask, node_mask, batch, eval_action, deterministic
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
        latent_nodes, latent_global, batch_idx, _, message_embed = self._get_latent(obs)
        actions, log_prob, entropy = self._get_action_from_latent(obs, latent_nodes, latent_global, batch_idx, message_embed, deterministic)
        return actions

    # REQUIRED FOR ON-POLICY ALGORITHMS (in SB3)
    def predict_values(self, obs: Dict[str, Tensor]) -> th.Tensor:
        _, _, _, vf_embed, message_embed = self._get_latent(obs)

        values = self.value_net(th.concat([vf_embed, message_embed], dim=-1))
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

        latent_nodes, latent_global, batch_idx, vf_embed, message_embed = self._get_latent(obs)
        _, log_prob, entropy = self._get_action_from_latent(obs, latent_nodes, latent_global, batch_idx, message_embed, eval_action=actions)
        values = self.value_net(th.concat([vf_embed, message_embed], dim=-1))
        return values, log_prob, entropy
