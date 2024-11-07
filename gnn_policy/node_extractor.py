from typing import Type

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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
        input_dim: int,
        output_dim: int,
        activation_fn: type[nn.Module],
    ):
        super(NodeExtractor, self).__init__(
            observation_space,
            output_dim,
        )
        self.embed_node = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            activation_fn(),
        )

    def forward(self, x: Tensor) -> Batch:
        return self.embed_node(x)
