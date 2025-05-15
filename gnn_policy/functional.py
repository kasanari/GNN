import torch as th
from torch import (
    Tensor,
    argmax,
    cat,
    cumsum,
    exp,
    isfinite,
    log,
    multinomial,
    nn,
    nonzero,
    prod,
    roll,
    split,
    stack,
    tensor,
    where,
    zeros_like,
)
from torch_scatter import scatter


def choose_masked(
    logits: Tensor,
    mask: Tensor,
    deterministic: bool = False,
):
    masked_logits = mask_logits(logits, mask)
    return choose(masked_logits, deterministic)


def choose(
    logits: Tensor,
    deterministic: bool = False,
):
    p = softmax(logits)
    action = sample_action(p, deterministic)
    h = entropy(p)
    p_action = gather(p, action)
    log_prob = log(p_action)
    return (
        action,
        log_prob,
        h,
        p,
    )


def segmented_choice_masked(
    logits: Tensor,
    mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
    deterministic: bool = False,
):
    masked_logits = mask_logits(logits, mask)
    return segmented_choice(
        masked_logits,
        batch,
        n_nodes,
        deterministic=deterministic,
    )


def segmented_choice(
    logits: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
    deterministic: bool = False,
):
    num_graphs = n_nodes.shape[0]
    p = segmented_softmax(logits, batch, num_graphs)
    action, data_starts = sample_node(p, n_nodes, deterministic=deterministic)
    p_action = segmented_gather(p, action, data_starts)
    assert not (p_action == 0).any(), "probability of action must be non-zero"
    log_prob = log(p_action)
    h = segmented_entropy(p, batch, num_graphs)
    return (
        action,
        log_prob,
        h,
        p,
    )


def evaluate_action_masked(
    action: Tensor,
    logits: Tensor,
    mask: Tensor,
):
    masked_logits = mask_logits(logits, mask)
    return evaluate_action(
        action,
        masked_logits,
    )


def evaluate_action(
    action: Tensor,
    logits: Tensor,
):
    p = softmax(logits)
    p_action = gather(p, action)
    log_prob = log(p_action)
    h = entropy(p)
    return (
        log_prob,
        h,
        p,
    )


def segmented_evaluate_action_masked(
    action: Tensor,
    logits: Tensor,
    mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
):
    masked_logits = mask_logits(logits, mask)
    return segmented_evaluate_action(
        action,
        masked_logits,
        batch,
        n_nodes,
    )


def segmented_evaluate_action(
    action: Tensor,
    logits: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
):
    num_graphs = n_nodes.shape[0]
    _, data_starts = data_splits_and_starts(n_nodes)
    p = segmented_softmax(logits, batch, num_graphs)
    p_action = segmented_gather(p, action, data_starts)
    log_prob = log(p_action)
    h = segmented_entropy(p, batch, num_graphs)
    return (
        log_prob,
        h,
        p,
    )


# @th.jit.script
def segment_sum(x: Tensor, index: Tensor, num_segments: int, dim: int = 0) -> Tensor:
    return scatter(x, index, dim, dim_size=num_segments, reduce="sum")


# @th.jit.script
def segment_mean(x: Tensor, index: Tensor, num_segments: int, dim: int = 0) -> Tensor:
    return scatter(x, index, dim, dim_size=num_segments, reduce="mean")


# @th.jit.script
def segment_max(x: Tensor, index: Tensor, num_segments: int, dim: int = 0) -> Tensor:
    return scatter(x, index, dim, dim_size=num_segments, reduce="max")


# @th.jit.script
def softmax(x: Tensor) -> Tensor:
    probs = nn.functional.softmax(x, -1)
    assert not (probs.isnan()).any()
    assert not (probs.isinf()).any()
    return probs


# @th.jit.script
def segment_softmax(
    src: Tensor,
    index: Tensor,
    num_segments: int,
    dim: int = 0,
) -> Tensor:
    src_max = scatter(src.detach(), index, dim, dim_size=num_segments, reduce="max")
    out = src - src_max.index_select(dim, index)
    out = out.exp()
    out_sum = scatter(out, index, dim, dim_size=num_segments, reduce="sum") + 1e-16
    out_sum = out_sum.index_select(dim, index)

    return out / out_sum


# @th.jit.script
def marginalize(
    ln_p_x: Tensor,
    ln_p_y__x: Tensor,
    batch_idx: Tensor,
    num_segments: int,
) -> Tensor:
    # derive p(y) from (sparse) p(x) and (sparse) p(y|x)
    # apply chain rule: p(x, y) = p(y|x) * p(x)
    # then marginalize p(x, y) over x to get p(y)
    x = segment_softmax(ln_p_x, batch_idx, num_segments)  # p(n)
    x = x.unsqueeze(-1) * softmax(ln_p_y__x)  # p(a|n) * p(n) = p(a, n)
    x = segment_sum(x, batch_idx, num_segments)  # sum over nodes to get p(a)
    return x


# @th.jit.script
def mask_logits(logits: Tensor, mask: Tensor) -> Tensor:
    infty = tensor(-1e9, device=logits.device)
    masked_logits = where(mask, logits, infty)
    return masked_logits


# @th.jit.script
def segmented_gather(src: Tensor, indices: Tensor, start_indices: Tensor) -> Tensor:
    real_indices = start_indices + indices.squeeze()
    return src[real_indices]


# @th.jit.script
def gather(src: Tensor, indices: Tensor) -> Tensor:
    return src.gather(-1, indices).squeeze(-1)


# @th.jit.script
def sample_action(p: Tensor, deterministic: bool = False) -> Tensor:
    a = multinomial(p, 1) if not deterministic else argmax(p, dim=-1).view(-1, 1)
    return a


# @th.jit.script
def get_start_indices(splits: Tensor) -> Tensor:
    splits = roll(splits, 1)
    splits[0] = 0
    start_indices = cumsum(splits, 0)
    return start_indices


# @th.jit.script
def data_splits_and_starts(n_nodes: Tensor) -> tuple[list[int], Tensor]:
    data_splits: Tensor = n_nodes  # number of nodes in each graph
    data_starts = get_start_indices(data_splits)  # start index of each graph
    # lst_lens = Tensor([len(x.mask) for x in batch.to_data_list()], device=device)
    # mask_starts = data_starts.repeat_interleave(lst_lens)
    split_list: list[int] = data_splits.tolist()
    return split_list, data_starts


# @th.jit.script
def segmented_softmax(logits: Tensor, batch_ind: Tensor, n_graphs: int) -> Tensor:
    probs = segment_softmax(logits, batch_ind, n_graphs)
    assert not (probs.isnan()).any()
    assert not (probs.isinf()).any()
    return probs


# @th.jit.script
def node_probs(x: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
    return segmented_softmax(x, batch, n_graphs)


# @th.jit.script
def action_logits_given_node(
    node_embeds: Tensor, node: Tensor, n_nodes: Tensor
) -> Tensor:
    _, data_starts = data_splits_and_starts(n_nodes)
    return segmented_gather(node_embeds, node.squeeze(), data_starts)


# @th.jit.script
def node_logits_given_action(
    node_embeds: Tensor,
    action: Tensor,  # type: ignore
    batch: Tensor,
) -> Tensor:
    # a single action is performed for each graph
    a_expanded = action[batch].view(-1, 1)
    # only the activations for the selected action are kept
    return node_embeds.gather(-1, a_expanded).squeeze(-1)


# @th.jit.script
def segmented_sample(probs: Tensor, splits: list[int]) -> Tensor:
    probs_split = split(probs, splits)
    samples = [
        # randint(high=len(x.squeeze(-1)), size=(1,))
        # if x.squeeze(-1).sum() == 0 or x.squeeze(-1).sum().isnan()
        multinomial(x.squeeze(-1), 1)
        if x.shape[0] > 1
        else th.zeros(1, device=x.device, dtype=th.long)
        for x in probs_split
    ]

    return stack(samples)


# @th.jit.script
def segmented_argmax(probs: Tensor, splits: list[int]) -> Tensor:
    probs_split = split(probs, splits)
    samples = [argmax(x.squeeze(-1), dim=-1).reshape(1) for x in probs_split]

    return stack(samples)


def segment_logsumexp(
    a: Tensor,
    segment_ids: Tensor,
    num_segments: int,
) -> Tensor:
    # take the max along the axis
    amax = segment_max(a, segment_ids, num_segments)

    # replace infs with zeros
    shift = where(isfinite(amax), amax, zeros_like(amax))
    expanded_shift = shift[segment_ids]

    # subtract the max to avoid overflow
    exp_a = exp(a - expanded_shift)

    # sum over the axis
    sumexp = segment_sum(exp_a, segment_ids, num_segments)

    # add the max back
    out = th.log(sumexp) + amax

    return out


# @th.jit.script
def segmented_scatter_(
    dest: Tensor, indices: Tensor, start_indices: Tensor, values: Tensor
) -> Tensor:
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest


# @th.jit.script
def entropy(p: Tensor) -> Tensor:
    log_probs = log(p + 1e-9)  # to avoid log(0)
    entropy = -p * log_probs
    return entropy.sum(-1)


def segmented_entropy(p: Tensor, indices: Tensor, n_graphs: int) -> Tensor:
    log_probs = log(p + 1e-9)  # to avoid log(0)
    entropy = -p * log_probs
    return scatter(entropy, indices, dim=0, dim_size=n_graphs, reduce="sum")


# @th.jit.script
def sample_node(
    p: Tensor, n_nodes: Tensor, deterministic: bool = False
) -> tuple[Tensor, Tensor]:
    data_splits, data_starts = data_splits_and_starts(n_nodes)
    a = (
        segmented_sample(p, data_splits)
        if not deterministic
        else segmented_argmax(p, data_splits)
    )

    return a, data_starts


# @th.jit.script
def concat_actions(predicate_action: Tensor, object_action: Tensor) -> Tensor:
    "Action is formatted as P(x)"
    if predicate_action.ndim == 1:
        predicate_action = predicate_action.view(-1, 1)
    if object_action.ndim == 1:
        object_action = object_action.view(-1, 1)
    return cat((predicate_action, object_action), dim=-1)


# @th.jit.script
def sample_action_and_node(
    graph_embeds: Tensor,
    node_logits: Tensor,
    predicate_mask: Tensor,
    node_mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 2D"
    assert node_logits.ndim == 1, "node embeddings must be 2D"
    assert graph_embeds.ndim == 2, "graph embeddings must be 2D"

    a_action, log_action, h_a, p_actions = choose_masked(
        graph_embeds,
        predicate_mask,
        deterministic=deterministic,
    )

    a_node, log_node, h_n, p_nodes = segmented_choice_masked(
        node_logits,
        node_mask,
        batch,
        n_nodes,
        deterministic=deterministic,
    )

    tot_log_prob = log_action + log_node  # log(p(a) * p(n))
    tot_entropy = (h_a + h_n).mean()  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=a_action, object_action=a_node)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert a.shape[0] == predicate_mask.shape[0]
    assert a.shape[1] == 2, "action must have two components, was {}".format(a.shape)

    return (
        a,
        tot_log_prob,
        tot_entropy,
        p_actions,
        p_nodes,
    )


# @th.jit.script
def sample_action_then_node(
    node_logits: Tensor,
    action_given_node_logits: Tensor,
    node_given_action_logits: Tensor,
    action_given_node_mask: Tensor,
    node_given_action_mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert node_given_action_mask.ndim == 2, "action mask must be 2D"
    assert action_given_node_mask.ndim == 2, "node mask must be 1D"
    assert node_given_action_logits.ndim == 2, "node embeddings must be 2D"
    assert node_logits.ndim == 1, f"node logits must be 1D, is {node_logits.shape}"
    num_graphs = n_nodes.shape[0]

    action_logits = segment_logsumexp(
        mask_logits(action_given_node_logits, action_given_node_mask)
        + node_logits.unsqueeze(-1),
        batch,
        num_graphs,
    )

    predicate_action, log_action, h_p, p_actions = choose(
        action_logits,
        deterministic=deterministic,
    )

    node_logits = node_logits_given_action(
        mask_logits(node_given_action_logits, node_given_action_mask),
        predicate_action,
        batch,
    )

    node_action, log_node, h_n, p_nodes = segmented_choice(
        node_logits,
        batch,
        n_nodes,
    )

    tot_log_prob = log_action + log_node  # log(p(a) * p(n))
    tot_entropy = (h_p + h_n).mean()  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=predicate_action, object_action=node_action)

    assert tot_log_prob.shape[0] == num_graphs
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert a.shape[0] == num_graphs
    assert a.shape[1] == 2, "action must have two components, was {}".format(a.shape)

    return (
        a,
        tot_log_prob,
        tot_entropy,
        p_actions,
        p_nodes,
    )


# @th.jit.script
def sample_node_then_action(
    node_predicate_embeds: Tensor,
    node_logits: Tensor,
    action_given_node_mask: Tensor,
    node_given_action_mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert node_given_action_mask.ndim == 2, "action mask must be 2D"
    assert node_logits.ndim == 1, "node logits must be 1D, was {}".format(
        node_logits.shape
    )
    assert action_given_node_mask.ndim == 2, "action mask must be 2D, was {}".format(
        action_given_node_mask.shape
    )
    assert node_predicate_embeds.ndim == 2, "node action embeddings must be 2D"

    num_graphs = n_nodes.shape[0]

    node_action, log_node, h_n, p_nodes = segmented_choice(
        node_logits,
        batch,
        n_nodes,
        deterministic=deterministic,
    )

    action_logits = action_logits_given_node(
        mask_logits(node_predicate_embeds, action_given_node_mask),
        node_action,
        n_nodes,
    )

    predicate_action, log_action, h_a, p_actions = choose(
        action_logits,
        deterministic=deterministic,
    )

    tot_log_prob = log_action + log_node
    tot_entropy = (h_n + h_a).mean()  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=predicate_action, object_action=node_action)

    assert tot_log_prob.shape[0] == num_graphs
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert a.shape[0] == num_graphs
    assert a.shape[1] == 2, f"action must have two components, was {a.shape}"

    assert not tot_log_prob.isinf().any()

    return (a, tot_log_prob, tot_entropy, p_actions, p_nodes)


# @th.jit.script
def segmented_nonzero(tnsr: Tensor, splits: list[int]):
    x_split = split(tnsr, splits)
    x_nonzero = [nonzero(x).flatten().cpu() for x in x_split]
    return x_nonzero


# @th.jit.script
def segmented_prod(tnsr: Tensor, splits: list[int]):
    x_split = split(tnsr, splits)
    x_prods = [prod(x) for x in x_split]
    x_mul = stack(x_prods)

    return x_mul


# @th.jit.script
def sample_node_set(
    logits: Tensor, mask: Tensor, n_nodes: Tensor
) -> tuple[list[Tensor], Tensor]:
    data_splits, _ = data_splits_and_starts(n_nodes)
    a0_sel = logits.bernoulli().to(th.bool)
    af_selection = segmented_nonzero(a0_sel, data_splits)

    a0_prob = where(a0_sel, logits, 1 - logits)
    af_probs = segmented_prod(a0_prob, data_splits)
    return af_selection, af_probs


# @th.jit.script
def eval_action_and_node(
    eval_action: Tensor,
    graph_embeds: Tensor,
    node_logits: Tensor,
    predicate_mask: Tensor,
    node_mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
) -> tuple[Tensor, Tensor]:
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 2D"
    assert node_logits.ndim == 1, "node embeddings must be 2D"
    assert graph_embeds.ndim == 2, "graph embeddings must be 2D"

    predicate_action = eval_action[:, 0].long().view(-1, 1)
    node_action = eval_action[:, 1].long().view(-1, 1)

    log_action, h_a, _ = evaluate_action_masked(
        predicate_action,
        graph_embeds,
        predicate_mask,
    )

    log_node, h_n, _ = segmented_evaluate_action_masked(
        node_action,
        node_logits,
        node_mask,
        batch,
        n_nodes,
    )

    tot_log_prob = log_action + log_node
    tot_entropy = (h_a + h_n).mean()  # H(X, Y) = H(X) + H(Y|X)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
    )


# @th.jit.script
def eval_node_then_action(
    eval_action: Tensor,
    node_predicate_embeds: Tensor,
    node_logits: Tensor,
    action_given_node_mask: Tensor,
    node_given_action_mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    node_action = eval_action[:, 1].long().view(-1, 1)
    predicate_action = eval_action[:, 0].long().view(-1, 1)

    assert node_given_action_mask.ndim == 2, "action mask must be 2D"
    assert node_logits.ndim == 1, "node logits must be 1D"
    assert action_given_node_mask.ndim == 2, "action mask must be 2D"
    assert node_predicate_embeds.ndim == 2, "node action embeddings must be 2D"
    assert node_action.ndim == 2
    assert node_action.shape[-1] == 1
    assert predicate_action.ndim == 2
    assert predicate_action.shape[-1] == 1

    log_node, h_n, p_nodes = segmented_evaluate_action(
        node_action,
        node_logits,
        batch,
        n_nodes,
    )

    action_logits = action_logits_given_node(
        mask_logits(node_predicate_embeds, action_given_node_mask),
        node_action,
        n_nodes,
    )

    log_action, h_a, p_actions = evaluate_action(
        predicate_action,
        action_logits,
    )

    tot_log_prob = log_action + log_node
    tot_entropy = (h_n + h_a).mean()  # H(X, Y) = H(X) + H(Y|X)

    assert tot_log_prob.shape[0] == n_nodes.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert not tot_log_prob.isinf().any()

    return (tot_log_prob, tot_entropy, p_actions, p_nodes)


# @th.jit.script
def eval_action_then_node(
    eval_action: Tensor,
    node_logits: Tensor,
    action_given_node_logits: Tensor,
    node_given_action_logits: Tensor,
    action_given_node_mask: Tensor,
    node_given_action_mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    node_action = eval_action[:, 1].long().view(-1, 1)
    predicate_action = eval_action[:, 0].long().view(-1, 1)
    assert action_given_node_mask.ndim == 2, "action mask must be 2D"
    assert node_given_action_mask.ndim == 2, "node mask must be 2D"
    assert node_given_action_logits.ndim == 2, "action|node embeddings must be 2D"
    assert node_given_action_logits.ndim == 2, "node|action embeddings must be 2D"
    assert node_action.ndim == 2
    assert node_action.shape[-1] == 1
    assert predicate_action.ndim == 2
    assert predicate_action.shape[-1] == 1

    num_graphs = n_nodes.shape[0]

    action_logits = segment_logsumexp(
        mask_logits(action_given_node_logits, action_given_node_mask)
        + node_logits.unsqueeze(-1),
        batch,
        num_graphs,
    )

    log_action, h_p, p_actions = evaluate_action(
        predicate_action,
        action_logits,
    )

    node_logits = node_logits_given_action(
        mask_logits(node_given_action_logits, node_given_action_mask),
        predicate_action,
        batch,
    )

    log_node, h_n, p_nodes = segmented_evaluate_action(
        node_action,
        node_logits,
        batch,
        n_nodes,
    )

    tot_log_prob = log_action + log_node
    tot_entropy = (h_p + h_n).mean()  # H(X, Y) = H(X) + H(Y|X)

    assert tot_log_prob.shape[0] == num_graphs
    assert tot_entropy.ndim == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
        p_actions,
        p_nodes,
    )
