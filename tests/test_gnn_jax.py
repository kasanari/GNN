from jax.numpy import array, log, where, isclose, asarray, all
from jax import random
import gnn_policy.functional_jax as F
import pytest


def data_splits_and_starts():
    batch_idx = array([0, 0, 0, 1, 1, 1])
    splits, starts = F.data_splits_and_starts(batch_idx)
    assert splits == [3, 3]
    assert (starts == array([0, 3])).all()
    assert len(splits) == len(starts)
    assert len(splits) == 2


def test_masked_segmented_softmax():
    logits = array([50, 50, 20, 100, 20])  # 100 will be masked
    mask = array([True, True, True, False, True])
    batch_ind = array([0, 0, 1, 1, 1])
    masked_logits = where(mask, logits, -1e9)
    n_segments = 2
    probs = F.segmented_softmax(masked_logits, batch_ind, n_segments)
    assert all(isclose(probs, array([0.5, 0.5, 0.5, 0.0, 0.5])))


def test_masked_softmax():
    x = array([50, 0, 50])
    mask = array([True, False, True])
    probs = F.masked_softmax(x, mask)
    assert (probs == array([0.5, 0.0, 0.5])).all()

    x = array([[50, 0, 50], [0, 50, 0], [20, 20, 20]])
    mask = array([[True, False, True], [False, True, False], [True, True, True]])
    probs = F.masked_softmax(x, mask)
    assert (
        probs == array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    ).all()


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
    e = F.entropy(p, 1)
    e = e * 1 / log(array(2.0))
    assert e == array(1.0)

    p = array([[1.0, 0.0], [1 / 2, 1 / 2]])
    e = F.entropy(p, 2)
    e = e * 1 / log(array(2.0))
    assert e == array(0.5)


def test_masked_entropy():
    p = array([[0.5, 0.5]])
    mask = array([[True, True]])
    e = F.masked_entropy(p, mask, 1)
    e = e * 1 / log(array(2.0))
    assert e.item() == 1.0

    p = array([[1 / 2, 0.0, 0.0], [0.0, 1 / 2, 1.0]])
    mask = array([[True, False, False], [False, True, False]])
    e = F.masked_entropy(p, mask, 2)
    e = e * 1 / log(array(2.0))
    assert e.item() == 0.5


def test_sample_action_given_node():
    x = array([[10, 0], [0, 10], [10, 1]], dtype="float32")
    action_mask = array([True, True])
    node = array([1])
    n_nodes = array([3])
    key = random.key(42)
    n_graphs = 1
    _, masked_logits = F.action_given_node_probs(x, node, action_mask, n_nodes)
    key, *keys = random.split(key, n_graphs + 1)
    a = F.sample_action(asarray(keys), masked_logits, False)
    assert all(a == array([1]))

    x = array([[10, 0], [0, 10], [10, 1]], dtype="float32")
    action_mask = array([[True, True], [False, True]])
    node = array([1, 0])
    n_nodes = array([2, 1])
    n_graphs = 2
    _, masked_logits = F.action_given_node_probs(x, node, action_mask, n_nodes)
    _, *keys = random.split(key, n_graphs + 1)
    a = F.sample_action(asarray(keys), masked_logits, False)
    assert all(a == array([1, 1]))


def test_sample_node_given_action():
    x = array([[10, 0], [0, 10], [0, 0]])
    node_mask = array([True, True, False])
    action = array([1])
    batch = array([0, 0, 0])
    n_graphs = 1
    key = random.key(42)
    _, masked_logits = F.node_logits_given_action(x, action, batch, node_mask, n_graphs)

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

    _, masked_logits = F.node_logits_given_action(x, action, batch, node_mask, n_graphs)
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
    x1 = array([[10, 0]])
    x2 = array([10, 0, 0])
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
    action_logits = array([[10, 100], [10, 100]])
    node_action_logits = array([[10, 0], [0, 10], [100, 0], [0, 100], [100, 0]])
    action_mask = array([[True, False], [True, True]])
    node_mask = array([True, True, False, True, True])
    batch = array([0, 0, 0, 1, 1])
    key = random.key(42)
    n_nodes = array([3, 2])
    n_graphs = 2

    a, logprob, h, *_ = F.sample_action_then_node(
        key,
        action_logits,
        node_action_logits,
        action_mask,
        node_mask,
        batch,
        n_nodes,
        n_graphs,
        False,
    )
    assert all(
        a == array([[0, 0], [1, 0]])
    ), "expected 0, 0 and 1, 0 but got {a}".format(a=a)
    assert logprob.shape == (n_graphs,)

    eval_logprob, h = F.eval_action_then_node(
        a,
        action_logits,
        node_action_logits,
        action_mask,
        node_mask,
        batch,
        n_nodes,
        n_graphs,
    )

    assert eval_logprob.shape == (n_graphs,)
    assert all(isclose(eval_logprob, logprob)), "expected {l1} but got {l2}".format(
        l1=logprob, l2=eval_logprob
    )


def test_sample_node_then_action():
    node_logits = array([10, 100, 0, 100, 0], dtype="float32")
    action_logits = array(
        [[10, 100], [0, 100], [0, 100], [100, 10], [100, 0]], dtype="float32"
    )
    # expect 0, 0
    node_mask = array([True, False, True, True, True])
    action_mask = array([[True, False], [False, True]])
    batch = array([0, 0, 0, 1, 1])
    key = random.key(42)

    a, logprob, h, *_ = F.sample_node_then_action(
        key,
        action_logits,
        node_logits,
        action_mask,
        node_mask,
        batch,
    )
    assert (a == array([[0, 0], [1, 0]])).all()
    assert logprob.shape == (2,)

    eval_logprob, h = F.eval_node_then_action(
        a,
        action_logits,
        node_logits,
        action_mask,
        node_mask,
        batch,
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
    import jax.profiler

    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        test_sample_action_given_node()
    test_sample_node()
    test_sample_node_set()
    test_sample_node_then_action()
    test_sample_action_then_node()
    test_sample_action_and_node()
    test_sample_node_given_action()
    test_entropy()
    test_masked_entropy()
    test_graph_action()
    test_segmented_gather()
    test_segmented_sample()
    test_masked_softmax()
    test_masked_segmented_softmax()
    data_splits_and_starts()
    pass
