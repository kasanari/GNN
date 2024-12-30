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
    unique,
    concatenate,
    prod,
    ones_like,
    zeros,
    isinf,
    sum,
)
from jax.lax import split
from jax.random import categorical
from jax import random
import jax.nn as nn
from jax import Array, vmap, jit
from jraph import segment_softmax
import jax


@jit
def masked_softmax(x: Array, mask: Array) -> Array:
    infty = jnp.array(-1e9, device=x.device)
    masked_x = where(mask, x, infty)
    probs = nn.softmax(masked_x, -1)
    # assert not (probs.isnan()).any()
    # assert not (probs.isinf()).any()
    return probs


@jit
def real_indices(start_indices: Array, indices: Array) -> Array:
    return start_indices + indices.squeeze()


@jit
def segmented_gather(src: Array, indices: Array, start_indices: Array) -> Array:
    real_indices = start_indices + indices.squeeze()
    return src[real_indices]


gather: Callable[[Array, Array], Array] = vmap(lambda x, y: x[y], in_axes=(0, 0))


@jit
def sample_action(logits: Array) -> Callable[[Array], tuple[Array, Array]]:
    def f(key: Array):
        new_key, subkey = random.split(key)
        a = categorical(subkey, logits, 1)
        return a, new_key

    return f


@jit
def select_max_action(logits: Array) -> Array:
    return argmax(logits, axis=-1)


@jit
def get_start_indices(splits: Array) -> Array:
    splits = roll(splits, 1)
    splits = splits.at[0].set(0)
    start_indices = cumsum(splits, 0)
    return start_indices


@jit
def data_splits(batch: Array) -> Array:
    return unique(batch, return_counts=True)[1]


@jit
def data_starts(batch: Array) -> Array:
    return get_start_indices(data_splits(batch))


@jit
def split_list(ds: Array) -> list[int]:
    return ds.tolist()


@jit
def mask_logits(mask: Array) -> Callable[[Array], Array]:
    return lambda x: where(mask, x, -1e9)


@jit
def node_probs(batch: Array) -> Callable[[Array], Array]:
    return lambda x: segment_softmax(x, batch)


@jit
def action_given_node_probs(node_embeds: Array) -> Callable[[Array], Array]:
    def h(node: Array) -> Callable[[Array], Callable[[Array], Array]]:
        def f(ds: Array) -> Callable[[Array], Array]:
            logits = segmented_gather(node_embeds, node.squeeze(), ds)

            def g(mask: Array) -> Array:
                masked_logits = mask_logits(mask)(logits)
                p = nn.softmax(masked_logits, -1)  # masked_softmax(masked_logits, mask)
                return p

            return g

        return f

    return h


@jit
def node_given_action_probs(
    node_embeds: Array,
) -> Array:
    def g(batch: Array):
        def f(action: Array):
            # a single action is performed for each graph
            a_expanded = action[batch]  # .reshape(-1, 1)
            x_a1 = gather(node_embeds, a_expanded)  # .squeeze(-1)
            # only the activations for the selected action are kept
            p = node_probs(batch)(x_a1)
            return p

        return f

    return g


@jit
def action_probs(x: Array) -> Array:
    return nn.softmax(x, -1)


@jit
def segmented_sample(
    key: Array, logits: Array, splits: list[int]
) -> tuple[Array, Array]:
    probs_split = split(logits, splits)
    new_key, *subkeys = random.split(key, num=len(probs_split) + 1)
    samples = [
        # th.randint(high=len(x.squeeze(-1)), size=(1,))
        # if x.squeeze(-1).sum() == 0 or x.squeeze(-1).sum().isnan()
        categorical(k, x)
        for k, x in zip(subkeys, probs_split)
    ]

    return concatenate(samples), new_key


@jit
def segmented_argmax(probs: Array) -> Callable[[list[int]], Array]:
    def f(splits: list[int]) -> Array:
        probs_split = split(probs, splits)
        samples = [argmax(x, axis=0).reshape(1) for x in probs_split]
        return concatenate(samples)

    return f


@jit
def segmented_scatter_(
    dest: Array, indices: Array, start_indices: Array, values: Array
) -> Array:
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest


@jit
def entropy(batch_size: int) -> Callable[[Array], Array]:
    def f(p: Array) -> Array:
        log_probs = log(p + 1e-9)  # to avoid log(0)
        entropy = sum(-p * log_probs) / batch_size
        return entropy

    return f


@jit
def masked_entropy(mask: Array) -> Callable[[Array], Array]:
    """Zero probability elements are masked out"""

    def g(p: Array) -> Callable[[int], Array]:
        unmasked_probs = where(mask, p, 1.0)

        def h(batch_size: int) -> Array:
            return entropy(batch_size)(unmasked_probs)

        return h

    return g


@jit
def sample_node(key: Array, logits: Array, batch: Array) -> tuple[Array, Array, Array]:
    ds = data_splits(batch)
    a, new_key = segmented_sample(key, logits, ds)
    return a, new_key


@jit
def select_node(key: Array, logits: Array, batch: Array) -> tuple[Array, Array, Array]:
    data_splits, data_starts = data_splits_and_starts(batch)
    a, new_key = segmented_argmax(key, logits, data_splits)
    return a, data_starts, new_key


@jit
def concat_actions(predicate_action: Array, object_action: Array) -> Array:
    "Action is formatted as P(x)"
    if predicate_action.ndim == 1:
        predicate_action = predicate_action.reshape(-1, 1)
    if object_action.ndim == 1:
        object_action = object_action.reshape(-1, 1)
    return concatenate((predicate_action, object_action), axis=-1)


@jit
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

    masked_action_logits = mask_logits(predicate_mask)(graph_embeds)
    pa1 = action_probs(masked_action_logits)
    entropy1 = masked_entropy(pa1, predicate_mask, num_graphs)
    predicate_action, key = (
        sample_action(key, masked_action_logits)
        if not deterministic
        else select_max_action(key, masked_action_logits)
    )
    a1_p = gather(pa1, predicate_action)

    pa2, masked_node_logits = node_probs(node_logits, node_mask, batch)
    entropy2 = masked_entropy(pa2, node_mask, num_graphs)
    a2, data_starts, key = (
        sample_node(key, masked_node_logits, batch)
        if not deterministic
        else select_node(key, masked_node_logits, batch)
    )
    a2_p = segmented_gather(pa2, a2, data_starts)

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


@jit
def sample_action_then_node(
    key: Array,
    action_logits: Array,
    node_predicate_logits: Array,
    predicate_mask: Array,
    node_mask: Array,
    batch: Array,
    deterministic: bool = False,
) -> tuple[Array, Array, Array, Array, Array]:
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_predicate_logits.ndim == 2, "node embeddings must be 2D"
    assert action_logits.ndim == 2, "graph embeddings must be 2D"
    assert action_logits.shape[1] == predicate_mask.shape[1]
    num_graphs = predicate_mask.shape[0]
    masked_action_logits = mask_logits(predicate_mask)(action_logits)
    pa1 = action_probs(masked_action_logits)
    entropy1 = masked_entropy(predicate_mask)(pa1)(num_graphs)
    predicate_action, key = (
        sample_action(key, masked_action_logits)
        if not deterministic
        else select_max_action(key, masked_action_logits)
    )

    pa2, masked_node_logits = node_given_action_probs(
        node_predicate_logits, predicate_action, batch, node_mask
    )

    entropy2 = masked_entropy(pa2, node_mask, num_graphs)

    node_action, data_starts, key = (
        sample_node(
            key,
            masked_node_logits,
            batch,
        )
        if not deterministic
        else select_node(key, masked_node_logits, batch)
    )

    a1_p = gather(pa1, predicate_action)
    a2_p = segmented_gather(pa2, node_action, data_starts)
    assert a2_p.shape == node_action.shape
    a2_p = where(predicate_action.squeeze() == 0, ones_like(a2_p), a2_p)  # Assume a
    assert a2_p.shape == node_action.shape

    assert not (a2_p == 0).any(), "node probabilities must be non-zero"

    tot_log_prob = log(a1_p * a2_p)
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


@jit
def sample_node_then_action(
    key: Array,
    node_predicate_embeds: Array,
    node_logits: Array,
    predicate_mask: Array,
    node_mask: Array,
    batch: Array,
    deterministic: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    assert node_mask.ndim == 1, "node mask must be 1D"
    assert node_logits.ndim == 1, "node logits must be 1D"
    assert predicate_mask.ndim == 2, "action mask must be 2D"
    assert node_predicate_embeds.ndim == 2, "node action embeddings must be 2D"
    num_graphs = predicate_mask.shape[0]

    masked_node_logits = mask_logits(node_mask)(node_logits)
    pa1 = node_probs(batch)(masked_node_logits)
    entropy1 = masked_entropy(pa1, node_mask, num_graphs)
    node_action, data_starts, key = (
        sample_node(key, masked_node_logits, batch)
        if not deterministic
        else select_node(key, masked_node_logits, batch)
    )

    a1_p = segmented_gather(
        pa1, node_action, data_starts
    )  # probabilities of the selected nodes

    assert not (a1_p == 0).any(), "node probabilities must be non-zero."

    pa2, action_logits = action_given_node_probs(
        node_predicate_embeds, node_action, predicate_mask, batch
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


@jit
def segmented_nonzero(tnsr: Array, splits: list[int]):
    x_split = split(tnsr, splits)
    x_nonzero = [flatnonzero(x) for x in x_split]
    return x_nonzero


@jit
def segmented_prod(tnsr: Array, splits: list[int]):
    x_split = split(tnsr, splits)
    x_prods = [prod(x) for x in x_split]
    x_mul = stack(x_prods)

    return x_mul


@jit
def sample_node_set(
    key: Array, logits: Array, mask: Array, batch: Array
) -> tuple[list[Array], Array]:
    data_splits, _ = data_splits_and_starts(batch)
    new_key, subkey = random.split(key)
    a0_sel = random.bernoulli(subkey, logits)
    af_selection = segmented_nonzero(a0_sel, data_splits)

    a0_prob = where(a0_sel, logits, 1 - logits)
    af_probs = segmented_prod(a0_prob, data_splits)
    return af_selection, af_probs, new_key


@jit
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
    _, data_starts = data_splits_and_starts(batch)
    a2_p = segmented_gather(pa2, a2, data_starts)
    tot_log_prob = log(a1_p * a2_p + 1e-9)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
    )


@jit
def eval_node_then_action(
    eval_action: Array,
    node_predicate_embeds: Array,
    node_logits: Array,
    predicate_mask: Array,
    node_mask: Array,
    batch: Array,
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

    _, data_starts = data_splits_and_starts(batch)
    a1_p = segmented_gather(
        p_node, node_action, data_starts
    )  # probabilities of the selected nodes

    pa2, _ = action_given_node_probs(
        node_predicate_embeds, node_action, predicate_mask, batch
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


@jit
def eval_action_then_node(
    eval_action: Array,
    graph_embeds: Array,
    node_predicate_embeds: Array,
    predicate_mask: Array,
    node_mask: Array,
    batch: Array,
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

    _, data_starts = data_splits_and_starts(batch)

    num_graphs = predicate_mask.shape[0]

    pa1, _ = action_probs(graph_embeds, predicate_mask)
    a1_p = gather(pa1, predicate_action)
    entropy1 = masked_entropy(pa1, predicate_mask, num_graphs)

    pa2, _ = node_given_action_probs(
        node_predicate_embeds, predicate_action, batch, node_mask
    )

    entropy2 = masked_entropy(pa2, node_mask, num_graphs)

    a2_p = segmented_gather(pa2, node_action, data_starts)
    a2_p = where(
        predicate_action.squeeze() == 0, ones_like(a2_p), a2_p
    )  # Assume 0 means no action, therefore parameter prob is 1

    assert not (a2_p == 0).any(), "node probabilities must be non-zero"
    tot_log_prob = log(a1_p * a2_p)
    tot_entropy = entropy1 + entropy2  # H(X, Y) = H(X) + H(Y|X)

    assert tot_log_prob.shape[0] == predicate_mask.shape[0]
    assert tot_entropy.ndim == 0, "entropy must be a scalar"

    return (
        tot_log_prob,
        tot_entropy,
    )
