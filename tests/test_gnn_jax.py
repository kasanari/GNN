import pytest
from jax import random
from jax.numpy import all, array, asarray, float32, isclose, log, where

import gnn_policy.functional_jax as F


def data_splits_and_starts():
    batch_idx = array([0, 0, 0, 1, 1, 1])
    n_graphs = array([3, 3])
    starts = F.data_starts(n_graphs)

    assert (starts == array([0, 3])).all()
    assert len(n_graphs) == len(starts)
    assert len(n_graphs) == 2


def test_masked_segmented_softmax():
    logits = array([50, 50, 20, 100, 20])  # 100 will be masked
    mask = array([True, True, True, False, True])
    batch_ind = array([0, 0, 1, 1, 1])
    masked_logits = where(mask, logits, -1e9)
    n_segments = 2
    probs = F.segmented_softmax(masked_logits, batch_ind, n_segments)
    assert all(isclose(probs, array([0.5, 0.5, 0.5, 0.0, 0.5])))


def test_segmented_argmax():
    probs = array([0.1, 0.0, 1.0, 0.0, 0.0])
    # splits = [2, 3]
    batch = array([0, 0, 1, 1, 1])
    n_nodes = array([2, 3])
    num_segments = 2
    ds = F.data_starts(n_nodes)
    samples = F.segmented_argmax(probs, batch, num_segments)
    assert samples.shape == (2,)
    assert all((samples - ds) == array([0, 0]))


def test_segmented_sample():
    probs = array([10.0, 0.0, 10.0, 0.0, 0.0])
    # splits = [2, 3]
    batch = array([0, 0, 1, 1, 1])
    n_nodes = array([2, 3])
    ds = F.data_starts(n_nodes)
    num_segments = 2
    key = random.key(42)
    samples = F.segmented_sample(key, probs, batch, num_segments)
    assert samples.shape == (2,)
    assert ((samples - ds) == array([[0, 0]])).all()


def test_segmented_gather():
    src = array([1, 2, 3, 4, 5, 6])
    indices = array([0, 1])
    start_indices = array([0, 3])
    values = src[
        start_indices + indices
    ]  # F.segmented_gather(src, indices, start_indices)
    assert (values == array([1, 5])).all()


def test_graph_action():
    logits = array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [100.0, 0.0, 50.0]])
    mask = array([[True, True, True], [True, True, True], [False, True, True]])
    logits = where(mask, logits, -1e9)
    key = random.key(42)
    keys = random.split(key, 3)
    # p = F.action_probs(logits, mask)
    a = F.sample_action(keys, logits, False)
    assert a.shape == (3,)
    assert (a == array([[0, 1, 2]])).all()
    assert isclose(
        logits, array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [-1e9, 0.0, 50.0]])
    ).all()
    pass


def test_sample_node():
    logits = array([50, 0, 0, 50, 100], dtype="float32")
    mask = array([True, True, True, True, False])
    logits = where(mask, logits, -1e9)
    batch_idx = array([0, 0, 1, 1, 1])
    n_nodes = array([2, 3])
    n_graphs = 2
    ds = F.data_starts(n_nodes)
    key = random.key(42)
    # p = F.node_probs(logits, mask, batch_idx)
    a = F.sample_node(key, logits, batch_idx, n_graphs, False)
    assert (a - ds == array([0, 1])).all()
    assert isclose(logits, array([[50.0, 0.0, 0.0, 50.0, -1e9]])).all()


def test_entropy():
    p = array([0.5, 0.5])
    e = F.entropy(p)
    e = e * 1 / log(array(2.0))
    assert e == array(1.0)

    p = array([[1.0, 0.0], [1 / 2, 1 / 2]])
    e = F.entropy(p)
    e = e * 1 / log(array(2.0))
    assert e == array(0.5)


def test_masked_entropy():
    p = array([[0.5, 0.5]])
    mask = array([[True, True]])
    p = p * mask
    e = F.entropy(p)
    e = e * 1 / log(array(2.0))
    assert e.item() == 1.0

    p = array([[1 / 2, 0.0, 0.0], [0.0, 1 / 2, 1.0]])
    mask = array([[True, False, False], [False, True, False]])
    p = p * mask
    e = F.entropy(p)
    e = e * 1 / log(array(2.0))
    assert e.item() == 0.5


def test_sample_action_given_node():
    x = array([[10, 0], [0, 10], [10, 1]], dtype="float32")
    action_mask = array([True, True])
    node = array([1])
    n_nodes = array([3])
    ds = F.data_starts(n_nodes)
    key = random.key(42)
    n_graphs = 1
    action_logits = F.action_logits_given_node(x, node, ds)
    masked_action_logits = F.mask_logits(action_logits, action_mask)
    key, *keys = random.split(key, n_graphs + 1)
    a = F.sample_action(asarray(keys), masked_action_logits, False)
    assert all(a == array([1]))

    x = array([[10, 0], [0, 10], [100, 10]], dtype="float32")
    action_mask = array([[True, True], [False, True]])
    node = array([1, 0])
    n_nodes = array([2, 1])
    data_starts = F.data_starts(n_nodes)
    n_graphs = 2
    action_logits = F.action_logits_given_node(x, node, data_starts)
    masked_action_logits = F.mask_logits(action_logits, action_mask)
    _, *keys = random.split(key, n_graphs + 1)
    a = F.sample_action(asarray(keys), masked_action_logits, False)
    assert all(a == array([1, 1]))


def test_sample_node_given_action():
    x = array([[10, 0], [0, 10], [0, 0]])
    node_mask = array([True, True, False])
    action = array([1])
    batch = array([0, 0, 0])
    n_graphs = 1
    key = random.key(42)
    node_logits = F.node_logits_given_action(x, action, batch)
    masked_logits = F.mask_logits(node_logits, node_mask)

    a = F.sample_node(
        key,
        masked_logits,
        batch,
        n_graphs,
        False,
    )
    assert all(a == array([1]))

    x = array([[10, 0], [0, 10], [10, 0], [0, 10], [100, 0]])
    node_mask = array([True, True, True, True, False])
    action = array([1, 0])
    batch = array([0, 0, 1, 1, 1])
    ds = F.data_starts(array([2, 3]))
    n_graphs = 2

    node_logits = F.node_logits_given_action(x, action, batch)
    masked_logits = F.mask_logits(node_logits, node_mask)
    key = random.key(42)

    a = F.sample_node(
        key,
        masked_logits,
        batch,
        n_graphs,
        False,
    )
    assert all(a - ds == array([1, 0]))


def test_sample_action_and_node():
    node_logits = array([100, 0, 10, 0, 100], dtype=float32)  # ln(p(n))

    node_given_action_logits = array(
        [[70, 100], [60, 0], [50, 10], [0, 500], [100, 10]], dtype=float32
    )  # ln(p(n | a))
    action_given_node_logits = array(
        [[10, 100], [0, 0], [0, 0], [0, 100], [100, 0]], dtype=float32
    )  # ln(p(a | n))
    mask1 = array([[True, True]])
    mask2 = array([True, True, False])
    batch = array([0, 0, 0])
    key = random.key(42)

    a, logprob, *_ = F.sample_action_and_node(
        key,
        x1,
        x2,
        mask1,
        mask2,
        batch,
    )
    assert (a == array([[0, 0]])).all()
    assert logprob.shape == (1,)

    eval_logprob, _ = F.eval_action_and_node(
        a,
        x1,
        x2,
        mask1,
        mask2,
        batch,
    )

    assert (a == array([[0, 0]])).all()
    assert eval_logprob.shape == (1,)
    assert eval_logprob == logprob


def test_sample_action_then_node():
    action_logits = array([[10, 100], [100, 0]], dtype=float32)  # ln(p(a))
    node_logits = array([100, 0, 10, 0, 100], dtype=float32)  # ln(p(n))

    node_given_action_logits = array(
        [[70, 100], [60, 0], [50, 10], [0, 500], [100, 10]], dtype=float32
    )  # ln(p(n | a))
    action_given_node_logits = array(
        [[10, 100], [0, 0], [0, 0], [0, 100], [100, 0]], dtype=float32
    )  # ln(p(a | n))

    action_mask = array([[True, True], [True, True]])
    node_given_action_mask = array([[True, True, True, True, True]] * 2).squeeze().T
    batch = array([0, 0, 0, 1, 1])
    key = random.key(42)
    n_nodes = array([3, 2])
    n_graphs = 2

    a, logprob, h, *_ = F.sample_action_then_node(
        key,
        node_logits,
        action_given_node_logits,
        node_given_action_logits,
        action_mask,
        node_given_action_mask,
        batch,
        n_nodes,
        False,
    )
    assert all(
        a == array([[1, 0], [0, 1]])
    ), "expected 0, 0 and 1, 0 but got {a}".format(a=a)
    assert logprob.shape == (n_graphs,)

    eval_logprob, h, *_ = F.eval_action_then_node(
        a,
        node_logits,
        action_given_node_logits,
        node_given_action_logits,
        action_mask,
        node_given_action_mask,
        batch,
        n_nodes,
    )

    assert eval_logprob.shape == (n_graphs,)
    assert all(isclose(eval_logprob, logprob)), "expected {l1} but got {l2}".format(
        l1=logprob, l2=eval_logprob
    )


def test_sample_node_then_action():
    # 10, 100, 0 | 100, 0

    node_logits = array([10, 100, 0, 100, 0], dtype="float32")
    action_logits = array(
        [[10, 100], [0, 100], [0, 100], [100, 10], [100, 0]], dtype="float32"
    )
    n_nodes = array([3, 2])
    # expect nodes 0 and 0 to be sampled

    # the 100 in the first graph is masked
    node_mask = array([True, False, True, True, True])

    # action 1 is masked in the first graph
    # action 0 is masked in the second graph

    action_given_node_mask = array(
        [[[True, False]] * 3 + [[False, True]] * 2]
    ).squeeze()
    batch = array([0, 0, 0, 1, 1])
    key = random.key(42)

    a, logprob, h, p_a__n, p_n = F.sample_node_then_action(
        key,
        action_logits,
        node_logits,
        node_mask,
        action_given_node_mask,
        batch,
        n_nodes,
        False,
    )
    assert (a == array([[0, 0], [1, 0]])).all()
    assert logprob.shape == (2,)

    eval_logprob, h, *_ = F.eval_node_then_action(
        a,
        action_logits,
        node_logits,
        node_mask,
        action_given_node_mask,
        batch,
        n_nodes,
    )

    assert (a == array([[0, 0], [1, 0]])).all()
    assert eval_logprob.shape == (2,)
    assert isclose(eval_logprob, logprob).all()


@pytest.mark.skip(reason="Not implemented")
def test_sample_node_set():
    x = array([1.0, 1.0, 0.0])
    mask = array([True, True, False])
    batch = array([0, 0, 0])
    key = random.key(42)
    a, logprob, _ = F.sample_node_set(key, x, mask, batch)
    assert (a[0] == array([0, 1])).all()
    assert logprob.shape == (1,)

    pass


if __name__ == "__main__":

    test_entropy()
    test_sample_action_given_node()
    test_sample_node()
    # test_sample_node_set()
    test_sample_node_then_action()
    test_sample_action_then_node()
    # test_sample_action_and_node()
    test_sample_node_given_action()
    test_masked_entropy()
    test_graph_action()
    test_segmented_gather()
    test_segmented_sample()
    test_masked_segmented_softmax()
    data_splits_and_starts()
    pass
