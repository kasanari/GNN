import numpy as np
import torch
import torch_geometric
from torch.nn import LeakyReLU, Linear, Module, ModuleList, Sequential
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import AttentionalAggregation, MaxAggregation


# def _recurse(gnns, x, edge_index, edge_attr):
#     if len(gnns) == 1:
#         y = gnns[0](x, edge_attr, edge_index)
#         return [y], y
#     else:
#         history, z = _recurse(gnns[1:], x, edge_index, edge_attr)
#         y = gnns[0](z, edge_attr, edge_index)
#         return history + [y], y

# def _recurse_global(pools, x_global, x, batch_ind):
#     if len(pools) == 1:
#         return pools[0](x_global, x[-1], batch_ind)
#     else:
#         return pools[0](_recurse_global(pools[1:], x_global, x[:1], batch_ind), x[-1], batch_ind)


# ----------------------------------------------------------------------------------------
class LocalMultiMessagePassing(Module):
    def __init__(
        self,
        steps,
        node_in_size,
        node_out_size,
        agg_size,
        global_size,
        activation_fn,
    ):
        super().__init__()

        # if node_in_size is None:
        #     node_in_size = [EMB_SIZE] * size

        self.gnns = ModuleList(
            [
                GraphNet(
                    node_in_size, node_in_size, node_out_size, activation_fn, skip=False
                )
                if i == 0
                else GraphNet(
                    node_out_size, agg_size, node_out_size, activation_fn, skip=True
                )
                for i in range(steps)
            ]
        )
        # self.pools = ModuleList(
        #     [
        #         GlobalNode(node_out_size, global_size, activation_fn)
        #         for i in range(steps)
        #     ]
        # )
        self.pool = MaxGlobalNode(node_out_size, global_size, activation_fn)
        self.hidden_size = node_out_size

        self.steps = steps

    def forward(self, x, edge_index, batch_ind, num_graphs):
        x_global = torch.zeros(num_graphs, self.hidden_size).to(x.device)
        for i in range(self.steps):
            x = self.gnns[i](x, edge_index)

        x_global = self.pool(x_global, x, batch_ind)

        return x, x_global


# ----------------------------------------------------------------------------------------
class AttentionGlobalNode(Module):
    def __init__(self, node_size, global_size, activation_fn):
        super().__init__()

        att_mask = Linear(node_size, 1)
        att_feat = Sequential(Linear(node_size, node_size), activation_fn())

        self.glob = AttentionalAggregation(att_mask, att_feat)
        self.tranform = Sequential(
            Linear(global_size * 2, global_size), activation_fn()
        )

    def forward(self, xg_old, x, batch):
        xg = self.glob(x, batch)

        xg = torch.cat([xg, xg_old], dim=1)
        xg = self.tranform(xg) + xg_old  # skip connection

        return xg

class MaxGlobalNode(Module):
    def __init__(self, node_size, global_size, activation_fn):
        super().__init__()

        self.agg = MaxAggregation()
        self.combine = Sequential(Linear(global_size * 2, global_size), activation_fn())

    def forward(self, xg_old, x, batch):
        xg = self.agg(x, batch)
        xg = torch.cat([xg, xg_old], dim=1)
        xg = self.combine(xg) + xg_old  # skip connection

        return xg



# ----------------------------------------------------------------------------------------
class GraphNet(MessagePassing):
    def __init__(
        self, node_in_size, agg_size, node_out_size, activation_fn, skip=False
    ):
        super().__init__(aggr="max")

        # self.f_mess = Sequential(Linear(node_in_size, agg_size), activation_fn())
        self.combine = Sequential(
            Linear(node_in_size + agg_size, node_out_size), activation_fn()
        )
        self.skip = skip

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # z = torch.cat([x_j, edge_attr], dim=1)
        z = x_j

        return z

    def update(self, aggr_out, x):
        z = torch.cat([x, aggr_out], dim=1)
        z = self.combine(z) + x if self.skip else self.combine(z)  # skip connection

        return z
