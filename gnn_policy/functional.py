import torch as th
from torch import Tensor, nn
from torch_geometric.utils import softmax


@th.jit.script
def get_start_indices(splits: Tensor) -> Tensor:
    splits = th.roll(splits, 1)
    splits[0] = 0
    start_indices = th.cumsum(splits, 0)
    return start_indices


@th.jit.script
def masked_segmented_softmax(
    energies: Tensor, mask: Tensor, batch_ind: Tensor
) -> Tensor:
    infty = th.tensor(-1e9, device=energies.device)
    masked_energies = th.where(mask, energies, infty)
    probs = softmax(masked_energies, batch_ind)
    return probs


@th.jit.script
def masked_softmax(x: Tensor, mask: Tensor) -> Tensor:
    infty = th.tensor(-1e9, device=x.device)
    masked_x = th.where(mask, x, infty)
    return nn.functional.softmax(masked_x, -1)


@th.jit.script
def segmented_sample(probs: Tensor, splits: list[int]) -> Tensor:
    probs_split = th.split(probs, splits)
    samples = [
        # th.randint(high=len(x.squeeze(-1)), size=(1,))
        # if x.squeeze(-1).sum() == 0 or x.squeeze(-1).sum().isnan()
        th.multinomial(x.squeeze(-1), 1)
        for x in probs_split
    ]

    return th.stack(samples)


@th.jit.script
def segmented_argmax(probs: Tensor, splits: list[int]) -> Tensor:
    probs_split = th.split(probs, splits)
    samples = [th.argmax(x.squeeze(-1), dim=-1) for x in probs_split]

    return th.stack(samples)


@th.jit.script
def segmented_scatter_(
    dest: Tensor, indices: Tensor, start_indices: Tensor, values: Tensor
) -> Tensor:
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest


@th.jit.script
def segmented_gather(src: Tensor, indices: Tensor, start_indices: Tensor) -> Tensor:
    real_indices = start_indices + indices.squeeze()
    return src[real_indices]


@th.jit.script
def gather(src: Tensor, indices: Tensor) -> Tensor:
    return src.gather(-1, indices).squeeze(-1)


@th.jit.script
def data_splits_and_starts(batch: Tensor) -> tuple[list[int], Tensor]:
    data_splits: Tensor = th.unique(batch, return_counts=True)[1]  # nodes per graph
    data_starts = get_start_indices(data_splits)  # start index of each graph
    # lst_lens = Tensor([len(x.mask) for x in batch.to_data_list()], device=device)
    # mask_starts = data_starts.repeat_interleave(lst_lens)
    split_list: list[int] = data_splits.tolist()
    return split_list, data_starts


@th.jit.script
def sample_action(p: Tensor):
    a = th.multinomial(p, 1)
    return a


@th.jit.script
def entropy(p: Tensor, batch_size: int) -> Tensor:
    log_probs = th.log(p + 1e-9)  # to avoid log(0)
    entropy = (-p * log_probs).sum() / batch_size
    return entropy


@th.jit.script
def masked_entropy(p: Tensor, mask: Tensor, batch_size: int) -> Tensor:
    """Zero probability elements are masked out"""
    unmasked_probs = p[mask]
    return entropy(unmasked_probs, batch_size)


@th.jit.script
def sample_node(
    x: Tensor, mask: Tensor, batch: Tensor, deterministic: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    data_splits, data_starts = data_splits_and_starts(batch)
    p = masked_segmented_softmax(x, mask, batch)
    a = (
        segmented_sample(p, data_splits)
        if not deterministic
        else segmented_argmax(p, data_splits)
    )
    h = masked_entropy(p, mask, a.shape[0])
    return a, p, data_starts, h


@th.jit.script
def sample_action_given_node(
    node_embeds: Tensor,
    node: Tensor,
    mask: Tensor,
    batch: Tensor,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # only the activations for the selected nodes are kept.
    _, data_starts = data_splits_and_starts(batch)
    x = segmented_gather(node_embeds, node.squeeze(), data_starts)

    p = masked_softmax(x, mask)
    a = sample_action(p) if not deterministic else th.argmax(p, dim=-1).view(-1, 1)
    entropy = masked_entropy(p, mask, a.shape[0])

    return a, p, data_starts, entropy


@th.jit.script
def graph_action(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    p = masked_softmax(x, mask)
    a = sample_action(p)
    entropy = masked_entropy(p, mask, a.shape[0])
    return a, p, entropy


@th.jit.script
def sample_node_given_action(
    node_embeds: Tensor,
    action: Tensor,  # type: ignore
    batch: Tensor,
    mask: Tensor,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # a single action is performed for each graph
    a_expanded = action[batch].view(-1, 1)
    # only the activations for the selected action are kept
    x_a1 = node_embeds.gather(-1, a_expanded).squeeze(-1)

    return sample_node(x_a1, mask, batch, deterministic)


@th.jit.script
def concat_actions(predicate_action: Tensor, object_action: Tensor) -> Tensor:
    "Action is formatted as P(x)"
    if predicate_action.dim() == 1:
        predicate_action = predicate_action.view(-1, 1)
    if object_action.dim() == 1:
        object_action = object_action.view(-1, 1)
    return th.cat((predicate_action, object_action), dim=-1)


@th.jit.script
def sample_action_and_node(
    graph_embeds: Tensor,
    node_logits: Tensor,
    predicate_mask: Tensor,
    node_mask: Tensor,
    batch: Tensor,
    eval_action: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    assert predicate_mask.dim() == 2, "action mask must be 2D"
    assert node_mask.dim() == 1, "node mask must be 2D"
    assert node_logits.dim() == 1, "node embeddings must be 2D"
    assert graph_embeds.dim() == 2, "graph embeddings must be 2D"

    predicate_action, pa1, entropy1 = graph_action(graph_embeds, predicate_mask)
    if eval_action is not None:
        predicate_action = eval_action[:, 0].long().view(-1, 1)
    a1_p = gather(pa1, predicate_action)

    # x_a1 = self.action_net2(batch.x).flatten()
    a2, pa2, data_starts, entropy2 = sample_node(node_logits, node_mask, batch)
    if eval_action is not None:
        a2 = eval_action[:, 1].long().view(-1, 1)
    a2_p = segmented_gather(pa2, a2, data_starts)

    tot_log_prob = th.log(a1_p * a2_p)
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
    )


@th.jit.script
def sample_action_then_node(
    graph_embeds: Tensor,
    node_predicate_embeds: Tensor,
    predicate_mask: Tensor,
    node_mask: Tensor,
    batch: Tensor,
    eval_action: Tensor | None = None,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert predicate_mask.dim() == 2, "action mask must be 2D"
    assert node_mask.dim() == 1, "node mask must be 2D"
    assert node_predicate_embeds.dim() == 2, "node embeddings must be 2D"
    assert graph_embeds.dim() == 2, "graph embeddings must be 2D"

    predicate_action, pa1, entropy1 = graph_action(graph_embeds, predicate_mask)
    if eval_action is not None:
        predicate_action = eval_action[:, 0].long().view(-1, 1)
        assert predicate_action.dim() == 2
        assert predicate_action.shape[-1] == 1

    node_action, pa2, data_starts, entropy2 = sample_node_given_action(
        node_predicate_embeds,
        predicate_action,
        batch,
        node_mask,
        deterministic=deterministic,
    )
    if eval_action is not None:
        node_action = eval_action[:, 1].long().view(-1, 1)
        assert node_action.dim() == 2
        assert node_action.shape[-1] == 1

    a1_p = gather(pa1, predicate_action)
    a2_p = segmented_gather(pa2, node_action, data_starts)
    tot_log_prob = th.log(a1_p * a2_p)
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
    node_logits: Tensor,
    node_predicate_embeds: Tensor,
    node_mask: Tensor,
    predicate_mask: Tensor,
    batch: Tensor,
    eval_action: Tensor | None = None,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert node_mask.dim() == 1, "node mask must be 1D"
    assert node_logits.dim() == 1, "node logits must be 1D"
    assert predicate_mask.dim() == 2, "action mask must be 2D"
    assert node_predicate_embeds.dim() == 2, "node action embeddings must be 2D"

    node_action, pa1, data_starts, entropy1 = sample_node(
        node_logits, node_mask, batch, deterministic
    )
    if eval_action is not None:
        node_action = eval_action[:, 1].long().view(-1, 1)
        assert node_action.dim() == 2
        assert node_action.shape[-1] == 1

    a1_p = segmented_gather(
        pa1, node_action, data_starts
    )  # probabilities of the selected nodes

    # batch = self._propagate_choice(batch,a1,data_starts)
    predicate_action, pa2, _, entropy2 = sample_action_given_node(
        node_predicate_embeds, node_action, predicate_mask, batch, deterministic
    )
    if eval_action is not None:
        predicate_action = eval_action[:, 0].long().view(-1, 1)
        assert predicate_action.dim() == 2
        assert predicate_action.shape[-1] == 1

    a2_p = gather(pa2, predicate_action)

    tot_log_prob = th.log(a1_p * a2_p)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=predicate_action, object_action=node_action)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.dim() == 0, "entropy must be a scalar"
    assert a.shape[0] == predicate_mask.shape[0]
    assert a.shape[1] == 2, f"action must have two components, was {a.shape}"

    return (a, tot_log_prob, tot_entropy, pa1, pa2)


@th.jit.script
def segmented_nonzero(tnsr: Tensor, splits: list[int]):
    x_split = th.split(tnsr, splits)
    x_nonzero = [th.nonzero(x).flatten().cpu() for x in x_split]
    return x_nonzero


@th.jit.script
def segmented_prod(tnsr: Tensor, splits: list[int]):
    x_split = th.split(tnsr, splits)
    x_prods = [th.prod(x) for x in x_split]
    x_mul = th.stack(x_prods)

    return x_mul


@th.jit.script
def sample_node_set(
    logits: Tensor, mask: Tensor, batch: Tensor
) -> tuple[list[Tensor], Tensor]:
    data_splits, _ = data_splits_and_starts(batch)
    a0_sel = logits.bernoulli().to(th.bool)
    af_selection = segmented_nonzero(a0_sel, data_splits)

    a0_prob = th.where(a0_sel, logits, 1 - logits)
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
