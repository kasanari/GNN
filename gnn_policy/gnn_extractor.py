from typing import Tuple, Type, Union

import torch as th
from torch import nn


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    From stable-baselines3.
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


class GNNExtractor(nn.Module):
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
