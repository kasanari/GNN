from typing import List, Optional, Tuple


import torch as th
from torch import Tensor, nn
import numpy as np


from torch_geometric.utils import softmax


@th.jit.script
def get_start_indices(splits):
    splits = th.roll(splits, 1)
    splits[0] = 0
    start_indices = th.cumsum(splits, 0)
    return start_indices


@th.jit.script
def masked_segmented_softmax(energies, mask, batch_ind):
    energies[~mask] = -np.inf
    probs = softmax(energies, batch_ind)

    return probs.flatten()


@th.jit.script
def masked_softmax(x: Tensor, mask: Tensor):
    x[~mask] = -np.inf
    return nn.functional.softmax(x, -1)


@th.jit.script
def segmented_sample(probs, splits: List[int]):
    probs_split = th.split(probs, splits)
    samples = [
        th.multinomial(x + 1e-5, 1) if sum(x) == 0 else th.multinomial(x, 1)
        for x in probs_split
    ]

    return th.cat(samples)


@th.jit.script
def segmented_scatter_(dest, indices, start_indices, values):
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest


@th.jit.script
def segmented_gather(src, indices, start_indices):
    real_indices = start_indices + indices
    return src[real_indices]


@th.jit.script
def gather(src, indices):
    return src.gather(1, indices.view(-1, 1)).squeeze()


@th.jit.script
def data_splits_and_starts(batch: Tensor) -> Tuple[List[int], Tensor]:
    data_splits: Tensor = th.unique(batch, return_counts=True)[1]  # nodes per graph
    data_starts = get_start_indices(data_splits)  # start index of each graph
    # lst_lens = th.tensor([len(x.mask) for x in batch.to_data_list()], device=device)
    # mask_starts = data_starts.repeat_interleave(lst_lens)
    split_list: List[int] = data_splits.tolist()
    return split_list, data_starts


@th.jit.script
def sample_action(p: Tensor):
    a = th.multinomial(p, 1)
    return a.squeeze()


@th.jit.script
def entropy(p: Tensor):
    n = p.shape[0]
    log_probs = th.log(p)
    entropy = (-p * log_probs).sum() / n
    return entropy

@th.jit.script
def masked_entropy(p: Tensor, mask: Tensor):
    """Zero probability elements are masked out"""
    unmasked_probs = p[mask]
    return entropy(unmasked_probs)


@th.jit.script
def choose_node(x_a1: Tensor, mask: Tensor, batch: Tensor):
    data_splits, data_starts = data_splits_and_starts(batch)

    p = masked_segmented_softmax(x_a1, mask, batch)
    a = segmented_sample(p, data_splits)
    h = masked_entropy(p, mask)

    return a.squeeze(), p, data_starts, h


@th.jit.script
def choose_action_given_node(
    node_embeds: Tensor, node: Tensor, mask: Tensor, batch: Tensor
):
    # only the activations for the selected nodes are kept.
    _, data_starts = data_splits_and_starts(batch)
    x = segmented_gather(node_embeds, node, data_starts)

    p = masked_softmax(x, mask)
    a = sample_action(p)
    entropy = masked_entropy(p, mask)

    return a, p, data_starts, entropy


@th.jit.script
def graph_action(x: Tensor, mask: Tensor):
    p = masked_softmax(x, mask)
    a = sample_action(p)
    entropy = masked_entropy(p, mask)
    return a, p, entropy


@th.jit.script
def choose_node_given_action(
    node_embeds: Tensor,
    action: Tensor,  # type: ignore
    batch: Tensor,
    mask: Tensor,
):
    # a single action is performed for each graph
    a_expanded = action[batch].view(-1, 1)
    # only the activations for the selected action are kept
    x_a1 = node_embeds.gather(-1, a_expanded)

    data_splits, data_starts = data_splits_and_starts(batch)

    p = masked_segmented_softmax(x_a1, mask, batch)
    a = segmented_sample(p, data_splits)
    h = masked_entropy(p, mask)

    return a, p, data_starts, h


@th.jit.script
def sample_action_and_node(
    graph_embeds: Tensor,
    node_embeds: Tensor,
    a0_mask: Tensor,
    a1_mask: Tensor,
    batch: Tensor,
    eval_action=None,
):
    a1, pa1, entropy1 = graph_action(graph_embeds, a0_mask)
    if eval_action is not None:
        a1 = eval_action[:, 0].long()
    a1_p = gather(pa1, a1)

    # x_a1 = self.action_net2(batch.x).flatten()
    a2, pa2, data_starts, entropy2 = choose_node(node_embeds, a1_mask, batch)
    if eval_action is not None:
        a2 = eval_action[:, 1].long()
    a2_p = segmented_gather(pa2, a2, data_starts)

    tot_log_prob = th.log(a1_p * a2_p)

    return th.stack((a1, a2), dim=1), tot_log_prob, entropy1 * entropy2


@th.jit.script
def sample_action_then_node(
    graph_embeds: Tensor,
    node_embeds: Tensor,
    a0_mask: Tensor,
    a1_mask: Tensor,
    batch: Tensor,
    eval_action=None,
):
    a1, pa1, entropy1 = graph_action(graph_embeds, a0_mask)
    if eval_action is not None:
        a1 = eval_action[:, 0].long()

    a2, pa2, data_starts, entropy2 = choose_node_given_action(
        node_embeds, a1, batch, a1_mask
    )
    if eval_action is not None:
        a2 = eval_action[:, 1].long()

    a1_p = gather(pa1, a1)
    a2_p = segmented_gather(pa2, a2, data_starts)
    tot_log_prob = th.log(a1_p * a2_p)

    return th.stack((a1, a2), dim=-1), tot_log_prob, entropy1 * entropy2


@th.jit.script
def sample_node_then_action(
    node_embeds: Tensor,
    node_action_embeds: Tensor,
    a0_mask: Tensor,
    a1_mask: Tensor,
    batch: Tensor,
    eval_action: Optional[Tensor] =None,
):
    a1, pa1, data_starts, entropy1 = choose_node(node_embeds, a0_mask, batch)
    if eval_action is not None:
        a1 = eval_action[:, 1].long()
    a1_p = segmented_gather(pa1, a1, data_starts)  # probabilities of the selected nodes

    # batch = self._propagate_choice(batch,a1,data_starts)
    a2, pa2, _, entropy2 = choose_action_given_node(
        node_action_embeds, a1, a1_mask, batch
    )
    if eval_action is not None:
        a2 = eval_action[:, 0].long()
    a2_p = gather(pa2, a2)

    tot_log_prob = th.log(a1_p * a2_p)

    return (
        th.stack((a2, a1), dim=-1).view(data_starts.shape[0],-1),
        tot_log_prob,
        entropy1 * entropy2,
    )


def _propagate_choice(
    self, batch, choice: th.Tensor, data_starts: th.Tensor
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Retrieve action distribution given the latent codes.

    :param latent_pi: Latent code for the actor
    :param latent_sde: Latent code for the gSDE exploration function
    :return: Action distribution    
    """
    selected_ind = th.zeros(len(batch.x), 1, device=self.device)
    segmented_scatter_(selected_ind, choice, data_starts, 1.0)

    # decode second action
    x = th.cat((batch.x, selected_ind), dim=1)
    x = self.sel_enc(x)  # 33 -> 32
    x, xg = self.a2(
        x, batch.global_features, batch.edge_attr, batch.edge_index, batch.batch
    )

    batch.x = x
    batch.global_features = xg

    # update mask for from action (depends on action specifics) TODO: generalise this or abstract out.
    # batch.mask[:,0] = True # can always move to ground
    # r = th.arange(len(choice),dtype=th.long,device=choice.device)
    # batch.mask[r,choice]=False # can't move to self

    return batch