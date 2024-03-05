from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Type

import gymnasium as gym
from torch import Tensor, nn


from torch_geometric.data import Batch


class NodeExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(
        self,
        observation_space: gym.Space,
        node_dim: int,
        features_dim: int = 32,
        activation_fn: Type[nn.Module] = nn.Tanh,
    ):
        super(NodeExtractor, self).__init__(
            observation_space,
            features_dim,
        )
        self.embed_node = nn.Sequential(
            nn.Linear(node_dim, self.features_dim),
            activation_fn(),
        )

    def forward(self, x: Tensor) -> Batch:
        return self.embed_node(x)