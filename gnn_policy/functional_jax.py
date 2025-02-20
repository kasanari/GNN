from collections.abc import Callable
from functools import partial
from typing import TypeAlias

import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, random, vmap
from jax.numpy import (
    arange,
    argmax,
    asarray,
    cumsum,
    exp,
    expand_dims,
    isinf,
    log,
    roll,
    sum,
    where,
)
from jax.ops import segment_max, segment_sum
from jax.random import categorical, gumbel

Tensor = Array

BatchIdx: TypeAlias = Tensor
Mask: TypeAlias = Tensor


def _categorical(
    key: Tensor,
    logits: Tensor,
    axis: int = -1,
) -> Tensor:
    """Sample random values from categorical distributions.

    Args:
      key: a PRNG key used as the random key.
      logits: Unnormalized log probabilities of the categorical distribution(s) to sample from,
        so that `softmax(logits, axis)` gives the corresponding probabilities.
      axis: Axis along which logits belong to the same categorical distribution.
      shape: Optional, a tuple of nonnegative integers representing the result shape.
        Must be broadcast-compatible with ``np.delete(logits.shape, axis)``.
        The default (None) produces a result shape equal to ``np.delete(logits.shape, axis)``.

    Returns:
      A random array with int dtype and shape given by ``shape`` if ``shape``
      is not None, or else ``np.delete(logits.shape, axis)``.
    """
    # key, _ = _check_prng_key("categorical", key)
    # check_arraylike("categorical", logits)
    # logits_arr = jnp.asarray(logits)

    if axis >= 0:
        axis -= len(logits.shape)

    shape = tuple(np.delete(logits.shape, axis))

    shape_prefix = shape[: len(shape) - len(shape)]
    logits_shape = list(shape[len(shape) - len(shape) :])
    logits_shape.insert(axis % len(logits.shape), logits.shape[axis])
    return gumbel(key, (*shape_prefix, *logits_shape), logits.dtype) + lax.expand_dims(
        logits, tuple(range(len(shape_prefix)))
    )


@partial(jit, static_argnums=(2, 3, 4))
def segment_softmax(
    logits: Tensor,
    segment_ids: Tensor,
    num_segments: int,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> Tensor:
    """Computes a segment-wise softmax.

    For a given tree of logits that can be divded into segments, computes a
    softmax over the segments.

      logits = jnp.ndarray([1.0, 2.0, 3.0, 1.0, 2.0])
      segment_ids = jnp.ndarray([0, 0, 0, 1, 1])
      segment_softmax(logits, segments)
      >> DeviceArray([0.09003057, 0.24472848, 0.66524094, 0.26894142, 0.7310586],
      >> dtype=float32)

    Args:
      logits: an array of logits to be segment softmaxed.
      segment_ids: an array with integer dtype that indicates the segments of
        `data` (along its leading axis) to be maxed over. Values can be repeated
        and need not be sorted. Values outside of the range [0, num_segments) are
        dropped and do not contribute to the result.
      num_segments: optional, an int with positive value indicating the number of
        segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
        jnp.max(-segment_ids))`` but since ``num_segments`` determines the size of
        the output, a static value must be provided to use ``segment_sum`` in a
        ``jit``-compiled function.
      indices_are_sorted: whether ``segment_ids`` is known to be sorted
      unique_indices: whether ``segment_ids`` is known to be free of duplicates

    Returns:
      The segment softmax-ed ``logits``.
    """
    # First, subtract the segment max for numerical stability
    maxs = segment_max(
        logits,
        segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    logits = logits - maxs[segment_ids]
    # Then take the exp
    logits = exp(logits)
    # Then calculate the normalizers
    normalizers = segment_sum(
        logits,
        segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
    normalizers = normalizers[segment_ids]
    softmax = logits / normalizers
    return softmax


@partial(jit, static_argnums=(3,))
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
    x = expand_dims(x, 1) * softmax(ln_p_y__x)  # p(a|n) * p(n) = p(a, n), A X N
    x = segment_sum(x, batch_idx, num_segments)  # sum over nodes to get p(a)
    return x


@partial(jit, static_argnums=(2,))
def logsumexp(
    a: Tensor,
    segment_ids: Tensor,
    num_segments: int,
) -> Tensor:
    # take the max along the axis
    amax = segment_max(a, segment_ids, num_segments)

    # replace infs with zeros
    amax = lax.stop_gradient(
        lax.select(jnp.isfinite(amax), amax, lax.full_like(amax, 0))
    )
    expanded_amax = amax[segment_ids]

    # subtract the max to avoid overflow
    exp_a = lax.exp(lax.sub(a, expanded_amax))

    # sum over the axis
    sumexp = segment_sum(exp_a, segment_ids, num_segments)

    # add the max back
    out = lax.add(lax.log(sumexp), amax)

    return out


@partial(jit, static_argnums=(3,))
def marginalize_logits(
    ln_p_x: Tensor,  # 1 x N
    ln_p_y__x: Tensor,  # N x A
    batch_idx: Tensor,
    num_segments: int,
) -> Tensor:  # 1 x A
    # use logsumexp trick to marginalize over x
    return logsumexp(expand_dims(ln_p_x, 1) + ln_p_y__x, batch_idx, num_segments)


def segmented_gather(src: Tensor, indices: Tensor, start_indices: Tensor) -> Tensor:
    real_indices = start_indices + indices.squeeze()
    return src[real_indices]


gather: Callable[[Tensor, Tensor], Tensor] = vmap(lambda x, y: x[y], in_axes=(0, 0))


@partial(vmap, in_axes=(0, 0, None))
def sample_action(keys: Tensor, logits: Tensor, deterministic: bool) -> Tensor:
    a = categorical(keys, logits, 1) if not deterministic else argmax(logits, axis=-1)
    return a


def get_start_indices(splits: Tensor) -> Tensor:
    splits = roll(splits, 1)
    splits = splits.at[0].set(0)
    start_indices = cumsum(splits, 0)
    return start_indices


def data_starts(n_nodes: Tensor) -> Tensor:
    return get_start_indices(n_nodes)


def mask_logits(
    x: Tensor,
    mask: Mask,
) -> Tensor:
    assert mask.dtype == jnp.bool_, "mask must be boolean"
    return where(mask, x, -1e9)


@partial(jit, static_argnums=(2,))
def segmented_softmax(energies: Tensor, batch_ind: Tensor, n_segments: int) -> Tensor:
    # infty = jnp.array(-1e9, device=energies.device)
    # masked_energies = where(mask, energies, infty)
    probs = segment_softmax(energies, batch_ind, n_segments, indices_are_sorted=True)
    # assert not (probs.isnan()).any()
    # assert not (probs.isinf()).any()
    return probs


@partial(jit, static_argnums=(2,))
def node_probs(logits: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
    assert batch.dtype == jnp.int32, "batch must be int32"
    p = segmented_softmax(logits, batch, n_graphs)
    return p


def action_logits_given_node(node_embeds: Tensor, node: Tensor, ds: Tensor) -> Tensor:
    return segmented_gather(node_embeds, node.squeeze(), ds)


def node_logits_given_action(
    node_embeds: Tensor,
    action: Tensor,  # type: ignore
    batch: Tensor,
) -> Tensor:
    # a single action is performed for each graph
    a_expanded = action[batch]  # .reshape(-1, 1)
    x_a1 = gather(node_embeds, a_expanded)  # .squeeze(-1)
    # only the activations for the selected action are kept
    return x_a1


#
def softmax(x: Tensor) -> Tensor:
    return nn.softmax(x, axis=-1)  # type: ignore


@partial(jit, static_argnums=(3,))
def segmented_sample(
    keys: Tensor, logits: Tensor, batch: Tensor, num_segments: int
) -> tuple[Tensor, Tensor]:
    # probs_split = split(logits, splits)

    probs = _categorical(keys, logits, -1)
    samples = segmented_argmax(probs, batch, num_segments)

    return samples


@partial(jit, static_argnums=(2,))
def segmented_argmax(
    data: Tensor,
    segment_ids: Tensor,
    num_segments: int,
):
    # if num_segments is None:
    #    num_segments = np.max(segment_ids) + 1
    # num_segments = int(num_segments)

    def f(i: Tensor) -> Tensor:
        return where(
            i == segment_ids, data, -jnp.inf
        )  # Everything but the segment is masked

    indices = jax.vmap(f)(arange(num_segments)).argmax(1)  # map over all segments
    return indices


def segmented_scatter_(
    dest: Tensor, indices: Tensor, start_indices: Tensor, values: Tensor
) -> Tensor:
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest


def entropy(p: Tensor) -> Tensor:
    bs = p.shape[0] if p.ndim > 1 else 1
    log_probs = log(p + 1e-9)  # to avoid log(0)
    entropy = -p * log_probs
    return sum(entropy) / bs


def segmented_entropy(p: Tensor, indices: Tensor, n_nodes: Tensor) -> Tensor:
    log_probs = log(p + 1e-9)  # to avoid log(0)
    entropy = -p * log_probs
    return jax.ops.segment_sum(entropy, indices, num_segments=n_nodes.shape[0]) * (
        1 / n_nodes
    )


@partial(
    jit,
    static_argnums=(
        3,
        4,
    ),
)
def sample_node(
    keys: Tensor,
    logits: Tensor,
    batch: Tensor,
    n_segments: int,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        segmented_sample(keys, logits, batch, n_segments)
        if not deterministic
        else segmented_argmax(logits, batch, n_segments)
    )


@vmap
def concat_actions(predicate_action: Tensor, object_action: Tensor) -> Tensor:
    "Action is formatted as P(x)"
    return asarray([predicate_action, object_action])


def sample_action_and_node(
    key: Tensor,
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

    num_graphs = n_nodes.shape[0]

    masked_action_logits = mask_logits(graph_embeds, predicate_mask)
    p_actions = softmax(masked_action_logits)
    a_action = sample_action(key, p_actions, deterministic)
    p_action = gather(p_actions, a_action)

    masked_node_logits = mask_logits(node_mask, node_logits)
    p_nodes = node_probs(batch, masked_node_logits, num_graphs)
    a_node, data_starts = sample_node(p_nodes, n_nodes, deterministic)
    p_node = segmented_gather(p_nodes, a_node, data_starts)

    tot_log_prob = log(p_action * p_node)
    tot_entropy = masked_entropy(
        p_actions, predicate_mask, num_graphs
    ) + masked_entropy(p_nodes, node_mask, num_graphs)  # H(X, Y) = H(X) + H(Y|X)

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


@partial(jit, static_argnames=("deterministic",))
def sample_action_then_node(
    key: Tensor,
    node_logits: Tensor,
    action_given_node_logits: Tensor,
    node_given_action_logits: Tensor,
    predicate_mask: Tensor,
    action_given_node_mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert action_given_node_mask.ndim == 2, "node mask must be 1D"
    assert node_given_action_logits.ndim == 2, "node embeddings must be 2D"
    assert node_logits.ndim == 1, "graph embeddings must be 2D"

    num_graphs = n_nodes.shape[0]
    ds = data_starts(n_nodes)

    action_logits = marginalize_logits(
        node_logits,
        action_given_node_logits,
        batch,
        num_graphs,
    )
    p_actions = softmax(action_logits)
    p_actions = p_actions * predicate_mask

    key, *subkeys = random.split(key, num_graphs + 1)
    predicate_action = sample_action(asarray(subkeys), action_logits, deterministic)

    node_given_action_logits = node_logits_given_action(
        mask_logits(node_given_action_logits, action_given_node_mask),
        predicate_action,
        batch,
    )

    p_nodes = node_probs(node_given_action_logits, batch, num_graphs)

    node_action = sample_node(
        key,
        node_given_action_logits,
        batch,
        num_graphs,
        deterministic=deterministic,
    )

    p_action = gather(p_actions, predicate_action)
    p_node = p_nodes[node_action]

    assert not (p_node == 0).any(), "node probabilities must be non-zero"

    h_p = entropy(p_actions)
    h_n = segmented_entropy(p_nodes, batch, n_nodes)

    tot_log_prob = log(p_action * p_node + 1e-9)
    tot_entropy = (h_p + h_n).mean()  # H(X, Y) = H(X) + H(Y|X)

    real_nodes = node_action - ds  # unbatched nodes indices for each graph

    a = concat_actions(predicate_action=predicate_action, object_action=real_nodes)

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


@partial(jit, static_argnames=("deterministic",))
def sample_node_then_action(
    key: Tensor,
    node_predicate_embeds: Tensor,
    node_logits: Tensor,
    node_mask: Tensor,
    action_given_node_mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_logits.ndim == 1, "node logits must be 1D, was {}".format(
        node_logits.shape
    )
    assert action_given_node_mask.ndim == 2, "action mask must be 2D, was {}".format(
        action_given_node_mask.shape
    )
    assert node_predicate_embeds.ndim == 2, "node action embeddings must be 2D"
    num_graphs = n_nodes.shape[0]
    masked_node_logits = mask_logits(node_logits, node_mask)
    p_nodes = node_probs(masked_node_logits, batch, num_graphs)
    key, subkey = random.split(key)
    a_node = sample_node(subkey, masked_node_logits, batch, num_graphs, deterministic)

    p_node = p_nodes[a_node]

    assert not (p_node == 0).any(), "node probabilities must be non-zero"
    ds = data_starts(n_nodes)
    masked_action_logits = action_logits_given_node(
        mask_logits(node_predicate_embeds, action_given_node_mask),
        a_node,
        ds,
    )
    p_actions = softmax(masked_action_logits)

    _, *subkeys = random.split(key, num_graphs + 1)
    a_action = sample_action(asarray(subkeys), masked_action_logits, deterministic)
    p_action = gather(p_actions, a_action)

    # p_node = where(a_action.squeeze() == 0, ones_like(p_node), p_node)

    tot_log_prob = log(p_node * p_action + 1e-9)
    h_n = segmented_entropy(p_nodes, batch, n_nodes)
    h_a = entropy(p_actions)
    tot_entropy = (h_n + h_a).mean()  # H(X, Y) = H(X) + H(Y|X)

    real_nodes = a_node - ds  # unbatched nodes indices for each graph

    a = concat_actions(predicate_action=a_action, object_action=real_nodes)

    assert tot_log_prob.shape[0] == num_graphs
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert a.shape[0] == num_graphs
    assert a.shape[1] == 2, f"action must have two components, was {a.shape}"

    assert not isinf(tot_log_prob).any()

    return (a, tot_log_prob, tot_entropy, p_actions, p_nodes)


@jit
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

    num_graphs = n_nodes.shape[0]
    predicate_action = eval_action[:, 0]
    a2 = eval_action[:, 1]

    p_actions = softmax(mask_logits(graph_embeds, predicate_mask))
    p_action = gather(p_actions, predicate_action)
    p_nodes = node_probs(mask_logits(node_logits, node_mask), batch, num_graphs)
    _, data_starts = data_splits_and_starts(n_nodes)
    p_node = segmented_gather(p_nodes, a2, data_starts)
    tot_log_prob = log(p_action * p_node + 1e-9)
    tot_entropy = masked_entropy(
        p_actions, predicate_mask, num_graphs
    ) + masked_entropy(p_nodes, node_mask, num_graphs)  # H(X, Y) = H(X) + H(Y|X)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
    )


@jit
def eval_node_then_action(
    eval_action: Tensor,
    node_predicate_embeds: Tensor,
    node_logits: Tensor,
    node_mask: Tensor,
    action_given_node_mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ds = data_starts(n_nodes)
    node_action = eval_action[:, 1]
    node_action = ds + node_action
    predicate_action = eval_action[:, 0]

    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_logits.ndim == 1, "node logits must be 1D"
    assert action_given_node_mask.ndim == 2, "action mask must be 2D"
    assert node_predicate_embeds.ndim == 2, "node action embeddings must be 2D"
    num_graphs = n_nodes.shape[0]
    masked_node_logits = mask_logits(node_logits, node_mask)
    p_nodes = node_probs(masked_node_logits, batch, num_graphs)
    p_node = p_nodes[node_action]

    p_actions = softmax(
        action_logits_given_node(
            mask_logits(node_predicate_embeds, action_given_node_mask),
            node_action,
            ds,
        ),
    )

    p_action = gather(p_actions, predicate_action)

    tot_log_prob = log(p_node * p_action + 1e-9)
    h_n = segmented_entropy(p_nodes, batch, n_nodes)
    h_a = entropy(p_actions)
    tot_entropy = (h_n + h_a).mean()  # H(X, Y) = H(X) + H(Y|X)

    assert not (p_node == 0).any(), "node probabilities must be non-zero"
    assert tot_log_prob.shape[0] == n_nodes.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert not isinf(tot_log_prob).any()

    return (tot_log_prob, tot_entropy, p_actions, p_nodes)


@jit
def eval_action_then_node(
    eval_action: Tensor,
    node_logits: Tensor,
    action_given_node_logits: Tensor,
    node_given_action_logits: Tensor,
    predicate_mask: Tensor,
    node_given_action_mask: Tensor,
    batch: Tensor,
    n_nodes: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    predicate_action = eval_action[:, 0]  # .long().view(-1, 1)
    # The node actions are locally indexed, so we need to convert them to global indices
    ds = data_starts(n_nodes)
    node_action = ds + eval_action[:, 1]  # .long().view(-1, 1)
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_given_action_mask.ndim == 2, "node mask must be 1D"
    assert node_given_action_logits.ndim == 2, "action|node embeddings must be 2D"
    assert node_given_action_logits.ndim == 2, "node|action embeddings must be 2D"
    assert node_action.ndim == 1

    assert predicate_action.ndim == 1
    assert predicate_action.ndim == 1

    num_graphs = n_nodes.shape[0]

    p_actions = marginalize(
        node_logits,
        action_given_node_logits,
        batch,
        num_graphs,
    )
    p_actions = p_actions * predicate_mask

    p_nodes = node_probs(
        node_logits_given_action(
            mask_logits(node_given_action_logits, node_given_action_mask),
            predicate_action,
            batch,
        ),
        batch,
        num_graphs,
    )

    p_action = gather(p_actions, predicate_action)
    p_node = p_nodes[node_action]

    tot_log_prob = log(p_action * p_node + 1e-9)
    h_p = entropy(p_actions)
    h_n = segmented_entropy(p_nodes, batch, n_nodes)
    tot_entropy = (h_p + h_n).mean()  # H(X, Y) = H(X) + H(Y|X)

    # assert not (a2_p == 0).any(), "node probabilities must be non-zero"
    assert tot_log_prob.shape[0] == num_graphs
    assert tot_entropy.ndim == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
        p_actions,
        p_nodes,
    )
