from collections.abc import Callable
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
    concatenate,
    prod,
    ones_like,
    zeros,
    isinf,
    array,
    any,
    sum,
    max,
    arange,
    asarray,
    exp,
)
from jax.lax import split
from jax.random import gumbel, categorical
from jax import random
import jax.nn as nn
from jax import Array, vmap, jit
from jax.ops import segment_max, segment_sum
from jax import tree
from jax import lax
from functools import partial
from typing import TypeAlias
import jax
import numpy as np
import jax.numpy as jnp

BatchIdx: TypeAlias = Array
Mask: TypeAlias = Array


def _categorical(
    key: Array,
    logits: Array,
    axis: int = -1,
) -> Array:
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
    logits: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> Array:
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


def masked_softmax(x: Array, mask: Array) -> Array:
    infty = jnp.asarray(-1e9, device=x.device)
    masked_x = where(mask, x, infty)
    probs = nn.softmax(masked_x, -1)
    # assert not (probs.isnan()).any()
    # assert not (probs.isinf()).any()
    return probs


gather: Callable[[Array, Array], Array] = vmap(lambda x, y: x[y], in_axes=(0, 0))


@partial(vmap, in_axes=(0, 0, None))
def sample_action(keys: Array, logits: Array, deterministic: bool) -> Array:
    # keys = jnp.asarray(keys)
    a = categorical(keys, logits, 1) if not deterministic else argmax(logits, axis=-1)
    return a


@jit
def get_start_indices(splits: Array) -> Array:
    splits = roll(splits, 1)
    splits = splits.at[0].set(0)
    start_indices = cumsum(splits, 0)
    return start_indices


@jit
def data_starts(n_nodes: Array) -> Array:
    return get_start_indices(n_nodes)


def mask_logits(mask: Mask, x: Array) -> Array:
    return where(mask, x, -1e9)


@partial(jit, static_argnums=(2,))
def segmented_softmax(energies: Array, batch_ind: Array, n_segments: int) -> Array:
    # infty = jnp.array(-1e9, device=energies.device)
    # masked_energies = where(mask, energies, infty)
    probs = segment_softmax(energies, batch_ind, n_segments, indices_are_sorted=True)
    # assert not (probs.isnan()).any()
    # assert not (probs.isinf()).any()
    return probs


@partial(jit, static_argnums=(2,))
def node_probs(batch: Array, logits: Array, n_graphs: int) -> Array:
    p = segmented_softmax(logits, batch, n_graphs)
    return p


@jit
def action_given_node_probs(
    node_embeds: Array, node: Array, mask: Array, n_nodes: Array
) -> tuple[Array, Array]:
    # _data_starts = data_starts(n_nodes)
    # logits = segmented_gather(node_embeds, node, _data_starts)
    logits = node_embeds[node]
    masked_logits = where(mask, logits, -1e9)
    p = nn.softmax(masked_logits, -1)  # masked_softmax(masked_logits, mask)
    return p, masked_logits


@jit
def node_logits_given_action(
    node_embeds: Array,
    action: Array,  # type: ignore
    batch: Array,
) -> Array:
    # a single action is performed for each graph
    a_expanded = action[batch]  # .reshape(-1, 1)
    x_a1 = gather(node_embeds, a_expanded)  # .squeeze(-1)
    # only the activations for the selected action are kept
    return x_a1


# @jit
def softmax(x: Array) -> Array:
    return nn.softmax(x, axis=-1)  # type: ignore


@partial(jit, static_argnums=(3,))
def segmented_sample(
    keys: Array, logits: Array, batch: Array, num_segments: int
) -> tuple[Array, Array]:
    # probs_split = split(logits, splits)

    probs = _categorical(keys, logits, -1)
    samples = segmented_argmax(probs, batch, num_segments)

    return samples


@partial(jit, static_argnums=(2,))
def segmented_argmax(
    data: Array,
    segment_ids: Array,
    num_segments: int,
):
    # if num_segments is None:
    #    num_segments = np.max(segment_ids) + 1
    # num_segments = int(num_segments)

    def f(i: Array) -> Array:
        return where(
            i == segment_ids, data, -jnp.inf
        )  # Everything but the segment is masked

    indices = jax.vmap(f)(arange(num_segments)).argmax(1)  # map over all segments
    return indices


def segmented_scatter_(
    dest: Array, indices: Array, start_indices: Array, values: Array
) -> Array:
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest


def entropy(p: Array, batch_size: int) -> Array:
    log_probs = log(p + 1e-9)  # to avoid log(0)
    entropy = sum(-p * log_probs) / batch_size
    return entropy


def masked_entropy(p: Array, mask: Array, batch_size: int) -> Array:
    """Zero probability elements are masked out"""
    unmasked_probs = where(mask, p, 1.0)
    return entropy(unmasked_probs, batch_size)


@partial(
    jit,
    static_argnums=(
        3,
        4,
    ),
)
def sample_node(
    keys: Array,
    logits: Array,
    batch: Array,
    n_segments: int,
    deterministic: bool,
) -> tuple[Array, Array, Array]:
    return (
        segmented_sample(keys, logits, batch, n_segments)
        if not deterministic
        else segmented_argmax(logits, batch, n_segments)
    )


@vmap
def concat_actions(predicate_action: Array, object_action: Array) -> Array:
    "Action is formatted as P(x)"
    return array([predicate_action, object_action])


def sample_action_and_node(
    key: Array,
    graph_embeds: Array,
    node_logits: Array,
    predicate_mask: Array,
    node_mask: Array,
    batch: Array,
    deterministic: bool = False,
) -> tuple[Array, Array, Array, Array, Array]:
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 2D"
    assert node_logits.ndim == 1, "node embeddings must be 2D"
    assert graph_embeds.ndim == 2, "graph embeddings must be 2D"

    num_graphs = predicate_mask.shape[0]

    pa1, masked_action_logits = action_probs(graph_embeds, predicate_mask)
    entropy1 = masked_entropy(pa1, predicate_mask, num_graphs)
    predicate_action, key = sample_action(key, masked_action_logits, deterministic)
    a1_p = gather(pa1, predicate_action)

    pa2, masked_node_logits = node_probs(node_logits, node_mask, batch)
    entropy2 = masked_entropy(pa2, node_mask, num_graphs)
    a2, data_starts, key = sample_node(key, masked_node_logits, batch, deterministic)
    a2_p = pa2[a2]  # segmented_gather(pa2, a2, data_starts)

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


@partial(jit, static_argnames=("n_graphs",))
def eval_action_then_node(
    eval_action: Array,
    graph_embeds: Array,
    node_predicate_embeds: Array,
    predicate_mask: Array,
    node_mask: Array,
    batch: Array,
    n_nodes: Array,
    n_graphs: int,
) -> tuple[Array, Array]:
    predicate_action = eval_action[:, 0]  # .long().view(-1, 1)
    # The node actions are locally indexed, so we need to convert them to global indices
    ds = data_starts(n_nodes)
    node_action = ds + eval_action[:, 1]  # .long().view(-1, 1)
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_predicate_embeds.ndim == 2, "node embeddings must be 2D"
    assert graph_embeds.ndim == 2, "graph embeddings must be 2D"
    assert node_action.ndim == 1
    assert node_action.ndim == 1
    assert predicate_action.ndim == 1
    assert predicate_action.ndim == 1

    # num_graphs = predicate_mask.shape[0]
    masked_action_logits = mask_logits(predicate_mask, graph_embeds)
    pa1 = softmax(masked_action_logits)
    a1_p = gather(pa1, predicate_action)
    entropy1 = masked_entropy(pa1, predicate_mask, n_graphs)

    node_logits = node_logits_given_action(
        node_predicate_embeds, predicate_action, batch
    )
    masked_node_logits = mask_logits(node_mask, node_logits)
    pa2 = node_probs(batch, masked_node_logits, n_graphs)

    entropy2 = masked_entropy(pa2, node_mask, n_graphs)

    a2_p = pa2[node_action]
    a2_p = where(
        predicate_action.squeeze() == 0, ones_like(a2_p), a2_p
    )  # Assume 0 means no action, therefore parameter prob is 1

    # assert not any(a2_p == 0), "node probabilities must be non-zero"
    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
    )


@partial(
    jit,
    static_argnames=(
        "n_graphs",
        "deterministic",
    ),
)
def sample_action_then_node(
    key: Array,
    action_logits: Array,
    node_predicate_logits: Array,
    predicate_mask: Array,
    node_mask: Array,
    batch: Array,
    n_nodes: Array,
    n_graphs: int,
    deterministic: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_predicate_logits.ndim == 2, "node embeddings must be 2D"
    assert action_logits.ndim == 2, "graph embeddings must be 2D"
    assert action_logits.shape[1] == predicate_mask.shape[1]
    num_graphs = predicate_mask.shape[0]
    ds = data_starts(n_nodes)
    masked_action_logits = mask_logits(predicate_mask, action_logits)
    pa1 = softmax(masked_action_logits)
    entropy1 = masked_entropy(pa1, predicate_mask, num_graphs)

    key, *subkeys = random.split(key, n_graphs + 1)
    predicate_action = sample_action(
        asarray(subkeys), masked_action_logits, deterministic
    )

    node_logits = node_logits_given_action(
        node_predicate_logits, predicate_action, batch
    )
    masked_node_logits = mask_logits(node_mask, node_logits)
    pa2 = node_probs(batch, masked_node_logits, n_graphs)

    entropy2 = masked_entropy(pa2, node_mask, num_graphs)
    key, subkeys = random.split(key)
    node_action = sample_node(
        subkeys,
        masked_node_logits,
        batch,
        n_graphs,
        deterministic=deterministic,
    )

    a1_p = gather(pa1, predicate_action)
    a2_p = pa2[node_action]
    assert a2_p.shape == node_action.shape
    a2_p = where(predicate_action.squeeze() == 0, ones_like(a2_p), a2_p)  # Assume a
    assert a2_p.shape == node_action.shape

    # assert not any(a2_p == 0), "node probabilities must be non-zero"

    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    real_nodes = node_action - ds  # unbatched nodes indices for each graph

    a = concat_actions(predicate_action=predicate_action, object_action=real_nodes)

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
    predicate_mask: Array,
    node_mask: Array,
    batch: Array,
    n_nodes: Array,
    n_graphs: int,
    deterministic: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_logits.ndim == 1, "node logits must be 1D"
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_predicate_embeds.ndim == 2, "node action embeddings must be 2D"
    num_graphs = predicate_mask.shape[0]

    pa1, masked_node_logits = node_probs(node_logits, node_mask, batch)
    entropy1 = masked_entropy(pa1, node_mask, num_graphs)
    node_action, data_starts, key = sample_node(
        key, masked_node_logits, batch, deterministic
    )

    a1_p = pa1[node_action]
    # segmented_gather(
    #    pa1, node_action, data_starts
    # )  # probabilities of the selected nodes

    assert not any(a1_p == 0), "node probabilities must be non-zero."

    pa2, action_logits = action_given_node_probs(
        node_predicate_embeds, node_action, predicate_mask, n_nodes
    )

    masked_action_logits = where(predicate_mask, action_logits, -1e9)

    entropy2 = masked_entropy(pa2, predicate_mask, num_graphs)
    predicate_action, key = sample_action(key, masked_action_logits, deterministic)

    a2_p = gather(pa2, predicate_action)

    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    a = concat_actions(predicate_action=predicate_action, object_action=node_action)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert a.shape[0] == predicate_mask.shape[0]
    assert a.shape[1] == 2, f"action must have two components, was {a.shape}"

    assert not isinf(tot_log_prob).any()

    return (a, tot_log_prob, tot_entropy, pa2, pa1, key)


def eval_action_and_node(
    eval_action: Array,
    graph_embeds: Array,
    node_logits: Array,
    predicate_mask: Array,
    node_mask: Array,
    batch: Array,
) -> tuple[Array, Array]:
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 2D"
    assert node_logits.ndim == 1, "node embeddings must be 2D"
    assert graph_embeds.ndim == 2, "graph embeddings must be 2D"

    num_graphs = predicate_mask.shape[0]
    predicate_action = eval_action[:, 0]
    a2 = eval_action[:, 1]

    pa1, _ = action_probs(graph_embeds, predicate_mask)
    entropy1 = masked_entropy(pa1, predicate_mask, num_graphs)
    a1_p = gather(pa1, predicate_action)
    pa2, _ = node_probs(node_logits, node_mask, batch)
    entropy2 = masked_entropy(pa2, node_mask, num_graphs)
    _, ds = data_starts(batch)
    a2_p = pa2[a2]  # segmented_gather(pa2, a2, ds)
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

    num_graphs = predicate_mask.shape[0]
    p_node, _ = node_probs(node_logits, node_mask, batch)
    entropy1 = masked_entropy(p_node, node_mask, num_graphs)

    ds = data_starts(batch)
    a1_p = p_node[node_action]
    # segmented_gather(
    #    p_node, node_action, ds
    # )  # probabilities of the selected nodes

    pa2, _ = action_given_node_probs(
        node_predicate_embeds, node_action, predicate_mask, n_nodes
    )
    entropy2 = masked_entropy(pa2, predicate_mask, num_graphs)

    a2_p = gather(pa2, predicate_action)

    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    assert not (a1_p == 0).any(), "node probabilities must be non-zero"
    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"
    assert not isinf(tot_log_prob).any()

    return (tot_log_prob, tot_entropy)
