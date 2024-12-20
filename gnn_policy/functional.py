import torch as th
from torch import Tensor, nn
from torch_geometric.utils import softmax
from torch import (
    log,
    split,
    stack,
    nonzero,
    argmax,
    where,
    roll,
    cumsum,
    unique,
    tensor,
    cat,
    prod,
    multinomial,
    ones_like,
    zeros,
)


@th.jit.script
def masked_softmax(x: Tensor, mask: Tensor) -> Tensor:
    infty = tensor(-1e9, device=x.device)
    masked_x = where(mask, x, infty)
    probs = nn.functional.softmax(masked_x, -1)
    assert not (probs.isnan()).any()
    assert not (probs.isinf()).any()
    return probs


@th.jit.script
def segmented_gather(src: Tensor, indices: Tensor, start_indices: Tensor) -> Tensor:
    real_indices = start_indices + indices.squeeze()
    return src[real_indices]


@th.jit.script
def gather(src: Tensor, indices: Tensor) -> Tensor:
    return src.gather(-1, indices).squeeze(-1)


@th.jit.script
def sample_action(p: Tensor, deterministic: bool = False) -> Tensor:
    a = multinomial(p, 1) if not deterministic else argmax(p, dim=-1).view(-1, 1)
    return a


@th.jit.script
def get_start_indices(splits: Tensor) -> Tensor:
    splits = roll(splits, 1)
    splits[0] = 0
    start_indices = cumsum(splits, 0)
    return start_indices


@th.jit.script
def data_splits_and_starts(batch: Tensor) -> tuple[list[int], Tensor]:
    data_splits: Tensor = unique(batch, return_counts=True)[1]  # nodes per graph
    data_starts = get_start_indices(data_splits)  # start index of each graph
    # lst_lens = Tensor([len(x.mask) for x in batch.to_data_list()], device=device)
    # mask_starts = data_starts.repeat_interleave(lst_lens)
    split_list: list[int] = data_splits.tolist()
    return split_list, data_starts


@th.jit.script
def masked_segmented_softmax(
    energies: Tensor, mask: Tensor, batch_ind: Tensor
) -> Tensor:
    infty = tensor(-1e9, device=energies.device)
    masked_energies = where(mask, energies, infty)
    probs = softmax(masked_energies, batch_ind)
    assert not (probs.isnan()).any()
    assert not (probs.isinf()).any()
    return probs


@th.jit.script
def node_probs(x: Tensor, mask: Tensor, batch: Tensor):
    p = masked_segmented_softmax(x, mask, batch)
    return p


@th.jit.script
def action_given_node_probs(
    node_embeds: Tensor, node: Tensor, mask: Tensor, batch: Tensor
) -> Tensor:
    _, data_starts = data_splits_and_starts(batch)
    x = segmented_gather(node_embeds, node.squeeze(), data_starts)
    p = masked_softmax(x, mask)
    return p


@th.jit.script
def node_given_action_probs(
    node_embeds: Tensor,
    action: Tensor,  # type: ignore
    batch: Tensor,
    mask: Tensor,
) -> Tensor:
    # a single action is performed for each graph
    a_expanded = action[batch].view(-1, 1)
    # only the activations for the selected action are kept
    x_a1 = node_embeds.gather(-1, a_expanded).squeeze(-1)
    p = node_probs(x_a1, mask, batch)
    return p


@th.jit.script
def action_probs(x: Tensor, mask: Tensor):
    p = masked_softmax(x, mask)
    return p


@th.jit.script
def segmented_sample(probs: Tensor, splits: list[int]) -> Tensor:
    probs_split = split(probs, splits)
    samples = [
        # randint(high=len(x.squeeze(-1)), size=(1,))
        # if x.squeeze(-1).sum() == 0 or x.squeeze(-1).sum().isnan()
        multinomial(x.squeeze(-1), 1)
        for x in probs_split
    ]

    return stack(samples)


@th.jit.script
def segmented_argmax(probs: Tensor, splits: list[int]) -> Tensor:
    probs_split = split(probs, splits)
    samples = [argmax(x.squeeze(-1), dim=-1) for x in probs_split]

    return stack(samples)


@th.jit.script
def segmented_scatter_(
    dest: Tensor, indices: Tensor, start_indices: Tensor, values: Tensor
) -> Tensor:
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest


@th.jit.script
def entropy(p: Tensor, batch_size: int) -> Tensor:
    log_probs = log(p + 1e-9)  # to avoid log(0)
    entropy = (-p * log_probs).sum() / batch_size
    return entropy


@th.jit.script
def masked_entropy(p: Tensor, mask: Tensor, batch_size: int) -> Tensor:
    """Zero probability elements are masked out"""
    unmasked_probs = p[mask]
    return entropy(unmasked_probs, batch_size)


@th.jit.script
def sample_node(
    p: Tensor, batch: Tensor, deterministic: bool = False
) -> tuple[Tensor, Tensor]:
    data_splits, data_starts = data_splits_and_starts(batch)
    a = (
        segmented_sample(p, data_splits)
        if not deterministic
        else segmented_argmax(p, data_splits)
    )

    return a, data_starts


@th.jit.script
def concat_actions(predicate_action: Tensor, object_action: Tensor) -> Tensor:
    "Action is formatted as P(x)"
    if predicate_action.dim() == 1:
        predicate_action = predicate_action.view(-1, 1)
    if object_action.dim() == 1:
        object_action = object_action.view(-1, 1)
    return cat((predicate_action, object_action), dim=-1)


@th.jit.script
def sample_action_and_node(
    graph_embeds: Tensor,
    node_logits: Tensor,
    predicate_mask: Tensor,
    node_mask: Tensor,
    batch: Tensor,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert predicate_mask.dim() == 2, "action mask must be 2D"
    assert node_mask.dim() == 1, "node mask must be 2D"
    assert node_logits.dim() == 1, "node embeddings must be 2D"
    assert graph_embeds.dim() == 2, "graph embeddings must be 2D"

    num_graphs = predicate_mask.shape[0]

    pa1 = action_probs(graph_embeds, predicate_mask)
    entropy1 = masked_entropy(pa1, predicate_mask, num_graphs)
    predicate_action = sample_action(pa1, deterministic)
    a1_p = gather(pa1, predicate_action)

    pa2 = node_probs(node_logits, node_mask, batch)
    entropy2 = masked_entropy(pa2, node_mask, num_graphs)
    a2, data_starts = sample_node(pa2, batch, deterministic)
    a2_p = segmented_gather(pa2, a2, data_starts)

    tot_log_prob = log(a1_p * a2_p)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=predicate_action, object_action=a2)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.dim() == 0, "entropy must be a scalar"
    assert a.shape[0] == predicate_mask.shape[0]
    assert a.shape[1] == 2, "action must have two components, was {}".format(a.shape)

    return (
        a,
        tot_log_prob,
        tot_entropy,
        pa1,
        pa2,
    )


@th.jit.script
def sample_action_then_node(
    action_logits: Tensor,
    node_predicate_logits: Tensor,
    predicate_mask: Tensor,
    node_mask: Tensor,
    batch: Tensor,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert predicate_mask.dim() == 2, "action mask must be 2D"
    assert node_mask.dim() == 1, "node mask must be 1D"
    assert node_predicate_logits.dim() == 2, "node embeddings must be 2D"
    assert action_logits.dim() == 2, "graph embeddings must be 2D"
    assert action_logits.shape[1] == predicate_mask.shape[1]
    num_graphs = predicate_mask.shape[0]
    pa1 = action_probs(action_logits, predicate_mask)
    entropy1 = masked_entropy(pa1, predicate_mask, num_graphs)
    predicate_action = sample_action(pa1, deterministic)

    pa2 = node_given_action_probs(
        node_predicate_logits, predicate_action, batch, node_mask
    )

    entropy2 = masked_entropy(pa2, node_mask, num_graphs)

    node_action, data_starts = sample_node(
        pa2,
        batch,
        deterministic=deterministic,
    )

    a1_p = gather(pa1, predicate_action)
    a2_p = segmented_gather(pa2, node_action, data_starts)

    assert not (a2_p == 0).any(), "node probabilities must be non-zero"

    tot_log_prob = log(a1_p * a2_p)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=predicate_action, object_action=node_action)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.dim() == 0, "entropy must be a scalar"
    assert a.shape[0] == predicate_mask.shape[0]
    assert a.shape[1] == 2, "action must have two components, was {}".format(a.shape)

    return (
        a,
        tot_log_prob,
        tot_entropy,
        pa1,
        pa2,
    )


@th.jit.script
def sample_node_then_action(
    node_predicate_embeds: Tensor,
    node_logits: Tensor,
    predicate_mask: Tensor,
    node_mask: Tensor,
    batch: Tensor,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert node_mask.dim() == 1, "node mask must be 1D"
    assert node_logits.dim() == 1, "node logits must be 1D"
    assert predicate_mask.dim() == 2, "action mask must be 2D"
    assert node_predicate_embeds.dim() == 2, "node action embeddings must be 2D"
    num_graphs = predicate_mask.shape[0]
    pa1 = node_probs(node_logits, node_mask, batch)
    entropy1 = masked_entropy(pa1, node_mask, num_graphs)
    node_action, data_starts = sample_node(pa1, batch, deterministic)

    a1_p = segmented_gather(
        pa1, node_action, data_starts
    )  # probabilities of the selected nodes

    assert not (a1_p == 0).any(), "node probabilities must be non-zero"

    pa2 = action_given_node_probs(
        node_predicate_embeds, node_action, predicate_mask, batch
    )
    entropy2 = masked_entropy(pa2, predicate_mask, num_graphs)
    predicate_action = sample_action(pa2, deterministic)

    a2_p = gather(pa2, predicate_action)

    tot_log_prob = log(a1_p * a2_p)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=predicate_action, object_action=node_action)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.dim() == 0, "entropy must be a scalar"
    assert a.shape[0] == predicate_mask.shape[0]
    assert a.shape[1] == 2, f"action must have two components, was {a.shape}"

    assert not tot_log_prob.isinf().any()

    return (a, tot_log_prob, tot_entropy, pa2, pa1)


@th.jit.script
def segmented_nonzero(tnsr: Tensor, splits: list[int]):
    x_split = split(tnsr, splits)
    x_nonzero = [nonzero(x).flatten().cpu() for x in x_split]
    return x_nonzero


@th.jit.script
def segmented_prod(tnsr: Tensor, splits: list[int]):
    x_split = split(tnsr, splits)
    x_prods = [prod(x) for x in x_split]
    x_mul = stack(x_prods)

    return x_mul


@th.jit.script
def sample_node_set(
    logits: Tensor, mask: Tensor, batch: Tensor
) -> tuple[list[Tensor], Tensor]:
    data_splits, _ = data_splits_and_starts(batch)
    a0_sel = logits.bernoulli().to(th.bool)
    af_selection = segmented_nonzero(a0_sel, data_splits)

    a0_prob = where(a0_sel, logits, 1 - logits)
    af_probs = segmented_prod(a0_prob, data_splits)
    return af_selection, af_probs


def _propagate_choice(
    self, batch: Tensor, choice: Tensor, data_starts: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Retrieve action distribution given the latent codes.

    :param latent_pi: Latent code for the actor
    :param latent_sde: Latent code for the gSDE exploration function
    :return: Action distribution
    """
    selected_ind = zeros(len(batch.x), 1, device=self.device)
    segmented_scatter_(selected_ind, choice, data_starts, 1.0)

    # decode second action
    x = cat((batch.x, selected_ind), dim=1)
    x = self.sel_enc(x)  # 33 -> 32
    x, xg = self.a2(
        x, batch.global_features, batch.edge_attr, batch.edge_index, batch.batch
    )

    batch.x = x
    batch.global_features = xg

    # update mask for from action (depends on action specifics) TODO: generalise this or abstract out.
    # batch.mask[:,0] = True # can always move to ground
    # r = arange(len(choice),dtype=th.long,device=choice.device)
    # batch.mask[r,choice]=False # can't move to self

    return batch


@th.jit.script
def eval_action_and_node(
    eval_action: Tensor,
    graph_embeds: Tensor,
    node_logits: Tensor,
    predicate_mask: Tensor,
    node_mask: Tensor,
    batch: Tensor,
) -> tuple[Tensor, Tensor]:
    assert predicate_mask.dim() == 2, "action mask must be 2D"
    assert node_mask.dim() == 1, "node mask must be 2D"
    assert node_logits.dim() == 1, "node embeddings must be 2D"
    assert graph_embeds.dim() == 2, "graph embeddings must be 2D"

    num_graphs = predicate_mask.shape[0]
    predicate_action = eval_action[:, 0].long().view(-1, 1)
    a2 = eval_action[:, 1].long().view(-1, 1)

    pa1 = action_probs(graph_embeds, predicate_mask)
    entropy1 = masked_entropy(pa1, predicate_mask, num_graphs)
    a1_p = gather(pa1, predicate_action)
    pa2 = node_probs(node_logits, node_mask, batch)
    entropy2 = masked_entropy(pa2, node_mask, num_graphs)
    _, data_starts = data_splits_and_starts(batch)
    a2_p = segmented_gather(pa2, a2, data_starts)
    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.dim() == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
    )


@th.jit.script
def eval_node_then_action(
    eval_action: Tensor,
    node_predicate_embeds: Tensor,
    node_logits: Tensor,
    predicate_mask: Tensor,
    node_mask: Tensor,
    batch: Tensor,
) -> tuple[Tensor, Tensor]:
    node_action = eval_action[:, 1].long().view(-1, 1)
    predicate_action = eval_action[:, 0].long().view(-1, 1)

    assert node_mask.dim() == 1, "node mask must be 1D"
    assert node_logits.dim() == 1, "node logits must be 1D"
    assert predicate_mask.dim() == 2, "action mask must be 2D"
    assert node_predicate_embeds.dim() == 2, "node action embeddings must be 2D"
    assert node_action.dim() == 2
    assert node_action.shape[-1] == 1
    assert predicate_action.dim() == 2
    assert predicate_action.shape[-1] == 1

    num_graphs = predicate_mask.shape[0]
    p_node = node_probs(node_logits, node_mask, batch)
    entropy1 = masked_entropy(p_node, node_mask, num_graphs)

    _, data_starts = data_splits_and_starts(batch)
    a1_p = segmented_gather(
        p_node, node_action, data_starts
    )  # probabilities of the selected nodes

    pa2 = action_given_node_probs(
        node_predicate_embeds, node_action, predicate_mask, batch
    )
    entropy2 = masked_entropy(pa2, predicate_mask, num_graphs)

    a2_p = gather(pa2, predicate_action)

    tot_log_prob = log(a1_p * a2_p)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    assert not (a1_p == 0).any(), "node probabilities must be non-zero"
    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.dim() == 0, "entropy must be a scalar"
    assert not tot_log_prob.isinf().any()

    return (tot_log_prob, tot_entropy)


@th.jit.script
def eval_action_then_node(
    eval_action: Tensor,
    graph_embeds: Tensor,
    node_predicate_embeds: Tensor,
    predicate_mask: Tensor,
    node_mask: Tensor,
    batch: Tensor,
) -> tuple[Tensor, Tensor]:
    node_action = eval_action[:, 1].long().view(-1, 1)
    predicate_action = eval_action[:, 0].long().view(-1, 1)
    assert predicate_mask.dim() == 2, "action mask must be 2D"
    assert node_mask.dim() == 1, "node mask must be 1D"
    assert node_predicate_embeds.dim() == 2, "node embeddings must be 2D"
    assert graph_embeds.dim() == 2, "graph embeddings must be 2D"
    assert node_action.dim() == 2
    assert node_action.shape[-1] == 1
    assert predicate_action.dim() == 2
    assert predicate_action.shape[-1] == 1

    num_graphs = predicate_mask.shape[0]

    pa1 = action_probs(graph_embeds, predicate_mask)
    entropy1 = masked_entropy(pa1, predicate_mask, num_graphs)

    pa2 = node_given_action_probs(
        node_predicate_embeds, predicate_action, batch, node_mask
    )

    entropy2 = masked_entropy(pa2, node_mask, num_graphs)

    _, data_starts = data_splits_and_starts(batch)

    a1_p = gather(pa1, predicate_action)
    a2_p = segmented_gather(pa2, node_action, data_starts)

    tot_log_prob = th.log(a1_p * a2_p)

    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    # assert not (a2_p == 0).any(), "node probabilities must be non-zero"
    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.dim() == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
    )
