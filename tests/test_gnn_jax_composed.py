from jax.numpy import array, log, where, isclose, all
from jax import random
import gnn_policy.kinda_composed as F
import pytest


def data_splits_and_starts():
    n_nodes = array([3, 3])
    starts = F.data_starts(n_nodes)
    assert all(starts == array([0, 3]))


def test_masked_segmented_softmax():
    logits = array([50, 50, 20, 100, 20])  # 100 will be masked
    mask = array([True, True, True, False, True])
    batch_ind = array([0, 0, 1, 1, 1])
    num_graphs = max(batch_ind) + 1
    masked_logits = where(mask, logits, -1e9)
    probs = F.node_probs(batch_ind, num_graphs)(masked_logits)
    assert all(isclose(probs, array([0.5, 0.5, 0.5, 0.0, 0.5])))


def test_masked_softmax():
    x = array([50, 0, 50])
    mask = array([True, False, True])
    masked_logits = F.mask_logits(mask)(x)
    probs = F.softmax(masked_logits)
    assert all(probs == array([0.5, 0.0, 0.5]))

    x = array([[50, 0, 50], [0, 50, 0], [20, 20, 20]])
    mask = array([[True, False, True], [False, True, False], [True, True, True]])
    masked_logits = F.mask_logits(mask)(x)
    probs = F.softmax(masked_logits)
    assert all(
        isclose(probs, array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [1 / 3, 1 / 3, 1 / 3]]))
    )


def test_segmented_sample():
    probs = array([10.0, 0.0, 10.0, 0.0, 0.0])
    n_nodes = array([2, 3])
    key = random.key(42)  # type: ignore
    samples, _ = F.segmented_sample(n_nodes)(probs)(key)
    assert samples.shape == (2,)
    assert all(samples == array([[0, 0]]))


def test_segmented_gather():
    src = array([1, 2, 3, 4, 5, 6])
    indices = array([0, 1])
    start_indices = array([0, 3])
    values = F.segmented_gather(src)(start_indices, indices)
    assert all(values == array([1, 5]))


def test_graph_action():
    logits = array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [100.0, 0.0, 50.0]])
    mask = array([[True, True, True], [True, True, True], [False, True, True]])
    logits = where(mask, logits, -1e9)
    key = random.key(42)  # type: ignore
    # p = F.action_probs(logits, mask)
    a, _ = F.sample_action(key)(logits)
    assert a.shape == (3,)
    assert all(a == array([[0, 1, 2]]))
    assert all(
        isclose(logits, array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [-1e9, 0.0, 50.0]]))
    )
    pass


def test_sample_node():
    logits = array([50, 0, 0, 50, 100], dtype="float32")
    mask = array([True, True, True, True, False])
    logits = where(mask, logits, -1e9)
    n_nodes = array([2, 3])
    key = random.key(42)  # type: ignore
    # p = F.node_probs(logits, mask, batch_idx)
    a, key = F.segmented_sample(n_nodes)(logits)(key)
    assert all(a == array([0, 1]))
    assert all(isclose(logits, array([[50.0, 0.0, 0.0, 50.0, -1e9]])))


def test_entropy():
    p = array([[0.5, 0.5]])
    e = F.entropy(1)(p)
    e = e * 1 / log(array(2.0))
    assert e.item() == 1.0

    p = array([[1.0, 0.0], [1 / 2, 1 / 2]])
    e = F.entropy(2)(p)
    e = e * 1 / log(array(2.0))
    assert e.item() == 0.5


def test_masked_entropy():
    p = array([[0.5, 0.5]])
    mask = array([[True, True]])
    e = F.masked_entropy(1)(mask)(p)
    e = e * 1 / log(array(2.0))
    assert e.item() == 1.0

    p = array([[1 / 2, 0.0, 0.0], [0.0, 1 / 2, 1.0]])
    mask = array([[True, False, False], [False, True, False]])
    e = F.masked_entropy(2)(mask)(p)
    e = e * 1 / log(array(2.0))
    assert e.item() == 0.5


def test_sample_action_given_node():
    x = array([[10, 0], [0, 10], [10, 1]], dtype="float32")
    action_mask = array([[True, True]])
    node = array([1])
    ds = F.data_starts(array([3]))
    key = random.key(42)  # type: ignore
    logits = F.segmented_gather(x)(ds, node)
    masked_logits = F.mask_logits(action_mask)(logits)
    a, key = F.sample_action(key)(masked_logits)
    assert all(a == array([1]))

    x = array([[10, 0], [0, 10], [10, 1]], dtype="float32")
    action_mask = array([[True, True], [False, True]])
    node = array([1, 0])
    n_nodes = array([2, 1])
    ds = F.data_starts(n_nodes)
    logits = F.segmented_gather(x)(ds, node)
    masked_logits = F.mask_logits(action_mask)(logits)
    a, _ = F.sample_action(key)(masked_logits)
    assert all(a == array([1, 1]))


def test_sample_node_given_action():
    x = array([[10, 0], [0, 10], [0, 0]])
    node_mask = array([True, True, False])
    action = array([1])
    batch = array([0, 0, 0])
    n_nodes = array([3])
    key = random.key(42)  # type: ignore
    logits = F.node_logits_given_action(batch)(x)(action)
    masked_logits = F.mask_logits(node_mask)(logits)

    a, key = F.segmented_sample(n_nodes)(masked_logits)(key)
    assert all(a == array([[1]]))

    x = array([[10, 0], [0, 10], [10, 0], [0, 10], [100, 0]])
    node_mask = array([True, True, True, True, False])
    action = array([1, 0])
    batch = array([0, 0, 1, 1, 1])
    n_nodes = array([2, 3])

    logits = F.node_logits_given_action(batch)(x)(action)
    masked_logits = F.mask_logits(node_mask)(logits)

    a, _ = F.segmented_sample(n_nodes)(masked_logits)(key)

    assert all(a == array([1, 0]))


def test_sample_action_and_node():
    x1 = array([[10, 0]])
    x2 = array([10, 0, 0])
    mask1 = array([[True, True]])
    mask2 = array([True, True, False])
    batch = array([0, 0, 0])
    n_nodes = array([3])
    key = random.key(42)  # type: ignore

    a, logprob, *_ = F.sample_action_and_node(
        key,
        x1,
        x2,
        mask1,
        mask2,
        batch,
        n_nodes,
        num_graphs=1,
    )
    assert all(a == array([[0, 0]]))
    assert logprob.shape == (1,)

    eval_logprob, _ = F.eval_action_and_node(
        a,
        x1,
        x2,
        mask1,
        mask2,
        batch,
        n_nodes,
    )

    assert all(a == array([[0, 0]]))
    assert eval_logprob.shape == (1,)
    assert isclose(eval_logprob, logprob)


def test_sample_action_then_node():
    x1 = array([[10, 100]])
    x2 = array([[10, 0], [0, 10], [100, 0]])
    mask1 = array([[True, False]])
    mask2 = array([True, True, False])
    batch = array([0, 0, 0])
    n_nodes = array([3])
    key = random.key(42)  # type: ignore

    a, logprob, _, *_ = F.sample_action_then_node(
        key,
        x1,
        x2,
        mask1,
        mask2,
        batch,
        n_nodes,
    )
    assert all(a == array([[0, 0]]))
    assert logprob.shape == (1,)

    eval_logprob, _ = F.eval_action_then_node(
        a,
        x1,
        x2,
        mask1,
        mask2,
        batch,
        n_nodes,
    )

    assert all(a == array([[0, 0]]))
    assert eval_logprob.shape == (1,)
    assert all(isclose(eval_logprob, logprob))


def test_sample_node_then_action():
    node_logits = array([10, 100, 0, 100, 0], dtype="float32")
    action_logits = array(
        [[10, 100], [0, 100], [0, 100], [100, 10], [100, 0]], dtype="float32"
    )
    # expect 0, 0
    node_mask = array([True, False, True, True, True])
    action_mask = array([[True, False], [False, True]])
    batch = array([0, 0, 0, 1, 1])
    n_nodes = array([3, 2])
    key = random.key(42)  # type: ignore

    a, logprob, _, *_ = F.sample_node_then_action(
        key,
        action_logits,
        node_logits,
        action_mask,
        node_mask,
        batch,
        n_nodes,
    )
    assert all(a == array([[0, 0], [1, 0]]))
    assert logprob.shape == (2,)

    eval_logprob, _ = F.eval_node_then_action(
        a,
        action_logits,
        node_logits,
        action_mask,
        node_mask,
        batch,
        n_nodes,
    )

    assert all(a == array([[0, 0], [1, 0]]))
    assert eval_logprob.shape == (2,)
    assert all(isclose(eval_logprob, logprob))


@pytest.mark.skip(reason="Not implemented")
def test_sample_node_set():
    x = array([1.0, 1.0, 0.0])
    mask = array([True, True, False])
    batch = array([0, 0, 0])
    key = random.key(42)  # type: ignore
    a, logprob, _ = F.sample_node_set()
    assert (a[0] == array([0, 1])).all()
    assert logprob.shape == (1,)

    pass


if __name__ == "__main__":
    import jax.profiler

    with jax.profiler.trace("jax-trace", create_perfetto_link=True):
        test_sample_action_then_node()
    # test_sample_node()

    # data_splits_and_starts()
    # test_sample_node_then_action()
    # test_sample_node_given_action()
    # test_sample_action_given_node()
    # test_entropy()
    # test_masked_entropy()
    # test_graph_action()
    # test_segmented_gather()
    # test_segmented_sample()
    # test_masked_softmax()
    # test_masked_segmented_softmax()
    # # test_sample_node_set()
    pass
