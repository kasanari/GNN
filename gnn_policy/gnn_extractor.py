from typing import Tuple, Type, Union

import torch as th
from stable_baselines3.common.utils import get_device
from torch import nn


class GNNExtractor(nn.Module):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(
        self,
        gnn_class,
        emb_size,
        edge_dim: int = 2,
        steps: int = 5,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        device: Union[th.device, str] = "auto",
    ):
        super(GNNExtractor, self).__init__()
        self.gnn = gnn_class(
            node_in_size=emb_size,
            node_out_size=emb_size,
            agg_size=emb_size,
            global_size=emb_size,
            edge_size=edge_dim,
            steps=steps,
            activation_fn=activation_fn,
        )
        device = get_device(device)

    def forward(  
        self,
        node_features: th.Tensor,
        global_features: th.Tensor,
        edge_features: th.Tensor,
        edge_indices: th.Tensor,
        batch_ind: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        return self.gnn(
            node_features, global_features, edge_features, edge_indices, batch_ind, 1
        )
