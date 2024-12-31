from collections.abc import Callable
from functools import partial
import jax.numpy as jnp
from jax.numpy import (
    log,
    stack,
    nonzero,
    flatnonzero,
    argmax,
    where,
    roll,
    cumsum,
    unique,
    concatenate,
    prod,
    ones_like,
    zeros,
    isinf,
    any,
    sum,
    max,
)
from jax.lax import split
from jax.random import categorical
from jax import random
import jax.nn as nn
from jax import Array, vmap, jit
from jraph import segment_softmax
import jax
from typing import TypeAlias


BatchIdx: TypeAlias = Array
Mask: TypeAlias = Array


def segmented_gather(src: Array):
    def g(start_indices: Array):
        def f(indices: Array) -> Array:
            real_indices = start_indices + indices.squeeze()
            return src[real_indices]

        return f

    return g


gather: Callable[[Array, Array], Array] = vmap(lambda x, y: x[y], in_axes=(0, 0))


def sample_action(logits: Array) -> Callable[[Array], tuple[Array, Array]]:
    def f(key: Array):
        new_key, subkey = random.split(key)
        a = categorical(subkey, logits, 1)
        return a, new_key

    return f


def select_max_action(logits: Array) -> Array:
    return argmax(logits, axis=-1)


def get_start_indices(splits: Array) -> Array:
    splits = roll(splits, 1)
    splits = splits.at[0].set(0)
    start_indices = cumsum(splits, 0)
    return start_indices


def data_splits(batch: BatchIdx) -> Array:
    return unique(batch, return_counts=True)[1]


def data_starts(batch: BatchIdx) -> Array:
    return get_start_indices(data_splits(batch))


def split_list(ds: Array) -> list[int]:
    return ds.tolist()


def mask_logits(mask: Mask) -> Callable[[Array], Array]:
    return lambda x: where(mask, x, -1e9)


def node_probs(batch: BatchIdx) -> Callable[[Array], Array]:
    return lambda x: segment_softmax(x, batch)


def action_logits_given_node(node_embeds: Array):
    x = segmented_gather(node_embeds)

    def f(ds: Array):
        y = x(ds)

        def h(node: Array) -> Array:
            logits = y(node)
            return logits

        return h

    return f


def node_logits_given_action(batch: BatchIdx):
    def f(node_embeds: Array):
        def g(action: Array) -> Array:
            a_expanded = action[batch]  # .reshape(-1, 1)
            # a single action is performed for each graph
            x_a1 = gather(node_embeds, a_expanded)  # .squeeze(-1)
            return x_a1

        return g

    return f


def softmax(x: Array) -> Array:
    return nn.softmax(x, axis=-1)  # type: ignore


def segmented_sample(logits: Array):
    def f(splits: list[int]):
        def g(key: Array):
            probs_split = split(logits, splits)
            new_key, *subkeys = random.split(key, num=len(probs_split) + 1)
            samples = [
                # th.randint(high=len(x.squeeze(-1)), size=(1,))
                # if x.squeeze(-1).sum() == 0 or x.squeeze(-1).sum().isnan()
                categorical(k, x).reshape(1)
                for k, x in zip(subkeys, probs_split)
            ]
            return concatenate(samples), new_key

        return g

    return f


def segmented_argmax(probs: Array) -> Callable[[list[int]], Array]:
    def f(splits: list[int]) -> Array:
        probs_split = split(probs, splits)
        samples = [argmax(x, axis=0).reshape(1) for x in probs_split]
        return concatenate(samples)

    return f


def segmented_scatter_(
    dest: Array, indices: Array, start_indices: Array, values: Array
) -> Array:
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest


def entropy(batch_size: Array) -> Callable[[Array], Array]:
    def f(p: Array) -> Array:
        log_probs = log(p + 1e-9)  # to avoid log(0)
        entropy = sum(-p * log_probs) / batch_size
        return entropy

    return f


def masked_entropy(batch_size: Array):
    """Zero probability elements are masked out"""
    x = entropy(batch_size)

    def g(mask: Mask):
        def h(p: Array) -> Array:
            unmasked_probs = where(mask, p, 1.0)
            return x(unmasked_probs)

        return h

    return g


def sample_node(batch: BatchIdx):
    ds = split_list(data_splits(batch))

    def f(logits: Array) -> Callable[[Array], tuple[Array, Array]]:
        def g(key: Array) -> tuple[Array, Array]:
            a, new_key = segmented_sample(logits)(ds)(key)
            return a, new_key

        return g

    return f


def select_node(batch: BatchIdx):
    ds = split_list(data_splits(batch))

    def g(logits: Array) -> Array:
        a = segmented_argmax(logits)(ds)
        return a

    return g


def concat_actions(predicate_action: Array, object_action: Array) -> Array:
    "Action is formatted as P(x)"
    if predicate_action.ndim == 1:
        predicate_action = predicate_action.reshape(-1, 1)
    if object_action.ndim == 1:
        object_action = object_action.reshape(-1, 1)
    return concatenate((predicate_action, object_action), axis=-1)


def sample_action_and_node(
    key: Array,
    graph_embeds: Array,
    node_logits: Array,
    predicate_mask: Mask,
    node_mask: Mask,
    batch: BatchIdx,
    deterministic: bool = False,
) -> tuple[Array, Array, Array, Array, Array]:
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 2D"
    assert node_logits.ndim == 1, "node embeddings must be 2D"
    assert graph_embeds.ndim == 2, "graph embeddings must be 2D"

    num_graphs = max(batch) + 1
    entropy = masked_entropy(num_graphs)
    node_prob = node_probs(batch)
    sample_node_ = sample_node(batch)
    select_node_ = select_node(batch)
    ds = data_starts(batch)

    p_ml = mask_logits(predicate_mask)
    p_me = entropy(predicate_mask)

    n_ml = mask_logits(node_mask)
    n_me = entropy(node_mask)

    masked_action_logits = p_ml(graph_embeds)
    pa1 = softmax(masked_action_logits)
    entropy1 = p_me(pa1)
    predicate_action, key = (
        sample_action(masked_action_logits)(key)
        if not deterministic
        else select_max_action(masked_action_logits)
    )
    a1_p = gather(pa1, predicate_action)

    masked_node_logits = n_ml(node_logits)
    pa2 = node_prob(masked_node_logits)
    entropy2 = n_me(pa2)
    a2, key = (
        sample_node_(masked_node_logits)(key)
        if not deterministic
        else select_node_(masked_node_logits)
    )

    a2_p = segmented_gather(pa2)(ds)(a2)
    a2_p = where(
        predicate_action.squeeze() == 0, ones_like(a2_p), a2_p
    )  # Assume 0 means no action, therefore parameter prob is 1

    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=predicate_action, object_action=a2)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert a.shape[0] == predicate_mask.shape[0]
    assert a.shape[1] == 2, "action must have two components, was {}".format(a.shape)

    return (
        a,
        tot_log_prob,
        tot_entropy,
        pa1,
        pa2,
    )


def sample(
    key: Array,
    action_sample_func: Callable[[Array], tuple[Array, Array]],
    node_sample_func: Callable[[Array], Callable[[Array], tuple[Array, Array]]],
):
    predicate_action, key = action_sample_func(key)
    node_action, key = node_sample_func(key)(predicate_action)
    return predicate_action, node_action, key


def eval_action_then_node(
    eval_action: Array,
    graph_embeds: Array,
    node_predicate_embeds: Array,
    predicate_mask: Mask,
    node_mask: Mask,
    batch: BatchIdx,
) -> tuple[Array, Array]:
    node_action = eval_action[:, 1]  # .long().view(-1, 1)
    predicate_action = eval_action[:, 0]  # .long().view(-1, 1)
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_predicate_embeds.ndim == 2, "node embeddings must be 2D"
    assert graph_embeds.ndim == 2, "graph embeddings must be 2D"
    assert node_action.ndim == 1
    assert node_action.ndim == 1
    assert predicate_action.ndim == 1
    assert predicate_action.ndim == 1

    ds = data_starts(batch)

    entropy = masked_entropy(max(batch) + 1)

    action_logits = mask_logits(predicate_mask)(graph_embeds)

    pa1 = softmax(action_logits)
    a1_p = gather(pa1, predicate_action)
    entropy1 = entropy(predicate_mask)(pa1)

    node_logits = node_logits_given_action(batch)(node_predicate_embeds)(
        predicate_action
    )
    masked_node_logits = mask_logits(node_mask)(node_logits)
    pa2 = node_probs(batch)(masked_node_logits)

    entropy2 = entropy(node_mask)(pa2)

    a2_p = segmented_gather(pa2)(ds)(node_action)
    a2_p = where(
        predicate_action.squeeze() == 0, ones_like(a2_p), a2_p
    )  # Assume 0 means no action, therefore parameter prob is 1

    assert not any(a2_p == 0), "node probabilities must be non-zero"
    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
    )


def graph_masking(predicate_mask: Mask, entropy: Callable[[Array], Array]):
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    p_ml = mask_logits(predicate_mask)
    p_me = entropy(predicate_mask)
    return p_ml, p_me


def graph_logits(graph_embeds: Array, masking_func: Callable[[Array], Array]):
    assert graph_embeds.ndim == 2, "graph embeddings must be 2D"
    pa1 = softmax(masking_func(graph_embeds))
    return pa1


def node_masking(node_mask: Mask, entropy: Callable[[Array], Array]):
    assert node_mask.ndim == 1, "node mask must be 1D"
    n_ml = mask_logits(node_mask)
    n_me = entropy(node_mask)
    return n_ml, n_me


def node_logits(
    node_predicate_logits: Array,
    n_ml: Callable[[Array], Array],
    node_logits_f: Callable[[Array], Callable[[Array], Array]],
) -> Callable[[Array], Array]:
    assert node_predicate_logits.ndim == 2, "node embeddings must be 2D"

    return lambda predicate_action: n_ml(
        node_logits_f(node_predicate_logits)(predicate_action)
    )


def sample_action_then_node(
    key: Array,
    action_logits: Array,
    node_predicate_logits: Array,
    predicate_mask: Mask,
    node_mask: Mask,
    batch: BatchIdx,
    deterministic: bool = False,
) -> tuple[Array, Array, Array, Array, Array]:
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_predicate_logits.ndim == 2, "node embeddings must be 2D"
    assert action_logits.ndim == 2, "graph embeddings must be 2D"
    assert action_logits.shape[1] == predicate_mask.shape[1]

    num_graphs = max(batch) + 1
    entropy = masked_entropy(num_graphs)
    p_me = entropy(predicate_mask)
    n_me = entropy(node_mask)
    ds = data_starts(batch)

    masked_action_logits = mask_logits(predicate_mask)(action_logits)
    pa1 = softmax(masked_action_logits)
    entropy1 = p_me(pa1)
    predicate_action, key = sample_action(masked_action_logits)(key)

    node_logits = node_logits_given_action(batch)(node_predicate_logits)(
        predicate_action
    )
    masked_node_logits = mask_logits(node_mask)(node_logits)
    pa2 = node_probs(batch)(masked_node_logits)

    entropy2 = n_me(pa2)

    node_action, key = sample_node(batch)(masked_node_logits)(key)

    a1_p = gather(pa1, predicate_action)
    a2_p = segmented_gather(pa2)(ds)(node_action)
    assert a2_p.shape == node_action.shape
    a2_p = where(predicate_action.squeeze() == 0, ones_like(a2_p), a2_p)  # Assume a
    assert a2_p.shape == node_action.shape

    assert not any(a2_p == 0), "node probabilities must be non-zero"

    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=predicate_action, object_action=node_action)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert a.shape[0] == predicate_mask.shape[0]
    assert a.shape[1] == 2, "action must have two components, was {}".format(a.shape)

    return (
        a,
        tot_log_prob,
        tot_entropy,
        pa1,
        pa2,
    )


def sample_node_then_action(
    key: Array,
    node_predicate_embeds: Array,
    node_logits: Array,
    predicate_mask: Mask,
    node_mask: Mask,
    batch: BatchIdx,
    deterministic: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_logits.ndim == 1, "node logits must be 1D"
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_predicate_embeds.ndim == 2, "node action embeddings must be 2D"
    num_graphs = max(batch) + 1

    entropy = masked_entropy(num_graphs)
    node_probs_f = node_probs(batch)

    masked_node_logits = mask_logits(node_mask)(node_logits)
    pa1 = node_probs_f(masked_node_logits)
    entropy1 = entropy(node_mask)(pa1)
    node_action, key = (
        sample_node(batch)(masked_node_logits)(key)
        if not deterministic
        else select_node(batch)(masked_node_logits)
    )

    ds = data_starts(batch)
    a1_p = segmented_gather(pa1)(ds)(node_action)
    # probabilities of the selected nodes

    assert not any(a1_p == 0), "node probabilities must be non-zero."

    action_logits = action_logits_given_node(node_predicate_embeds)(ds)(node_action)

    masked_action_logits = mask_logits(predicate_mask)(action_logits)
    pa2 = softmax(masked_action_logits)

    entropy2 = entropy(predicate_mask)(pa2)
    predicate_action, key = sample_action(masked_action_logits)(key)

    a2_p = gather(pa2, predicate_action)

    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=predicate_action, object_action=node_action)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert a.shape[0] == predicate_mask.shape[0]
    assert a.shape[1] == 2, f"action must have two components, was {a.shape}"

    assert not any(isinf(tot_log_prob))

    return (a, tot_log_prob, tot_entropy, pa2, pa1, key)


def segmented_nonzero(tnsr: Array, splits: list[int]):
    x_split = split(tnsr, splits)
    x_nonzero = [flatnonzero(x) for x in x_split]
    return x_nonzero


def segmented_prod(tnsr: Array, splits: list[int]):
    x_split = split(tnsr, splits)
    x_prods = [prod(x) for x in x_split]
    x_mul = stack(x_prods)

    return x_mul


def eval_action_and_node(
    eval_action: Array,
    graph_embeds: Array,
    node_logits: Array,
    predicate_mask: Mask,
    node_mask: Mask,
    batch: BatchIdx,
) -> tuple[Array, Array]:
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 2D"
    assert node_logits.ndim == 1, "node embeddings must be 2D"
    assert graph_embeds.ndim == 2, "graph embeddings must be 2D"

    num_graphs = max(batch) + 1
    predicate_action = eval_action[:, 0]
    a2 = eval_action[:, 1]

    entropy = masked_entropy(num_graphs)

    p_ml = mask_logits(predicate_mask)
    p_me = entropy(predicate_mask)

    n_ml = mask_logits(node_mask)
    n_me = entropy(node_mask)

    node_prob = node_probs(batch)

    masked_action_logits = p_ml(graph_embeds)

    pa1 = softmax(masked_action_logits)
    entropy1 = p_me(pa1)
    a1_p = gather(pa1, predicate_action)
    masked_node_logits = n_ml(node_logits)
    pa2 = node_prob(masked_node_logits)
    entropy2 = n_me(pa2)
    ds = data_starts(batch)
    a2_p = segmented_gather(pa2)(ds)(a2)
    a2_p = where(
        predicate_action.squeeze() == 0, ones_like(a2_p), a2_p
    )  # Assume 0 means no action, therefore parameter prob is 1
    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
    )


def eval_node_then_action(
    eval_action: Array,
    node_predicate_embeds: Array,
    node_logits: Array,
    predicate_mask: Mask,
    node_mask: Mask,
    batch: BatchIdx,
) -> tuple[Array, Array]:
    node_action = eval_action[:, 1]  # .reshape(-1, 1)
    predicate_action = eval_action[:, 0]  # .reshape(-1, 1)

    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_logits.ndim == 1, "node logits must be 1D"
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_predicate_embeds.ndim == 2, "node action embeddings must be 2D"
    assert node_action.ndim == 1
    assert predicate_action.ndim == 1
    # assert predicate_action.shape[-1] == 1

    num_graphs = max(batch) + 1
    entropy = masked_entropy(num_graphs)

    p_me = entropy(predicate_mask)

    n_ml = mask_logits(node_mask)
    n_me = entropy(node_mask)

    node_prob = node_probs(batch)

    masked_node_logits = n_ml(node_logits)
    p_node = node_prob(masked_node_logits)
    entropy1 = n_me(p_node)

    ds = data_starts(batch)
    a1_p = segmented_gather(p_node)(ds)(node_action)
    # probabilities of the selected nodes

    p_ml = mask_logits(predicate_mask)
    action_logits_f = action_logits_given_node(node_predicate_embeds)(ds)
    pa2 = softmax(p_ml(action_logits_f(node_action)))

    entropy2 = p_me(pa2)

    a2_p = gather(pa2, predicate_action)

    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    assert not any((a1_p == 0)), "node probabilities must be non-zero"
    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert not any(isinf(tot_log_prob))

    return (tot_log_prob, tot_entropy)
