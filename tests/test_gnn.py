import pytest
import torch as th
from torch import all

import gnn_policy.functional as F


def data_splits_and_starts():
    batch_idx = th.tensor([0, 0, 0, 1, 1, 1])
    splits, starts = F.data_splits_and_starts(batch_idx)
    assert splits == [3, 3]
    assert all(starts == th.tensor([0, 3]))
    assert len(splits) == len(starts)
    assert len(splits) == 2


def test_masked_segmented_softmax():
    logits = th.tensor([50, 50, 20, 100, 20])  # 100 will be masked
    mask = th.tensor([True, True, True, False, True])
    batch_ind = th.tensor([0, 0, 1, 1, 1])
    n_graphs = 2
    masked_logits = F.mask_logits(logits, mask)
    probs = F.segmented_softmax(masked_logits, batch_ind, n_graphs)
    assert all(probs == th.tensor([0.5, 0.5, 0.5, 0.0, 0.5]))


def test_masked_softmax():
    x = th.tensor([50, 0, 50])
    mask = th.tensor([True, False, True])
    masked_x = F.mask_logits(x, mask)
    probs = F.softmax(masked_x)
    assert (probs == th.tensor([0.5, 0.0, 0.5])).all()

    x = th.tensor([[50, 0, 50], [0, 50, 0], [20, 20, 20]])
    mask = th.tensor([[True, False, True], [False, True, False], [True, True, True]])
    masked_x = F.mask_logits(x, mask)
    probs = F.softmax(masked_x)
    assert (
        probs == th.tensor([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    ).all()


def test_segmented_logsumexp():
    logits = th.tensor([50, 0, 50])
    mask = th.tensor([True, False, True])
    batch_ind = th.tensor([0, 0, 0])
    masked_logits = F.mask_logits(logits, mask)
    logsumexp = F.segment_logsumexp(masked_logits, batch_ind, 1)
    assert logsumexp.shape == (1,)
    assert logsumexp.item() == 50.0

    logits = th.tensor([[50, 0, 50], [0, 50, 0], [20, 20, 20]])
    mask = th.tensor([[True, False, True], [False, True, False], [True, True, True]])
    masked_logits = F.mask_logits(logits, mask)
    logsumexp = F.segment_logsumexp(masked_logits)
    assert logsumexp.shape == (3,)
    assert (logsumexp == th.tensor([50.0, 50.0, 60.0])).all()


def test_segmented_sample():
    probs = th.tensor([0.1, 0.0, 1.0, 0.0, 0.0])
    splits = [2, 3]
    samples = F.segmented_sample(probs, splits)
    assert samples.shape == (2, 1)
    assert (samples == th.tensor([[0], [0]])).all()


def test_segmented_argmax():
    probs = th.tensor([0.1, 0.0, 1.0, 0.0, 0.0])
    splits = [2, 3]
    samples = F.segmented_argmax(probs, splits)
    assert samples.shape == (2, 1)
    assert (samples == th.tensor([[0], [0]])).all()


def test_segmented_gather():
    src = th.tensor([1, 2, 3, 4, 5, 6])
    indices = th.tensor([0, 1])
    start_indices = th.tensor([0, 3])
    values = F.segmented_gather(src, indices, start_indices)
    assert (values == th.tensor([1, 5])).all()


def test_graph_action():
    logits = th.tensor([[50, 0, 0], [0, 50, 0], [100, 0, 50]])
    mask = th.tensor([[True, True, True], [True, True, True], [False, True, True]])
    masked_logits = F.mask_logits(logits, mask)
    p = F.softmax(masked_logits)
    a = F.sample_action(p)
    assert a.shape == (3, 1)
    assert (a == th.tensor([[0], [1], [2]])).all()
    assert (
        th.trunc(p) == th.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ).all()
    pass


def test_sample_node():
    energies = th.tensor([50, 0, 0, 0, 50, 100])
    mask = th.tensor([True, True, True, True, True, False])
    batch_idx = th.tensor([0, 0, 0, 1, 1, 1])
    n_nodes = th.tensor([3, 3])
    n_graphs = 2
    masked_logits = F.mask_logits(energies, mask)
    p = F.node_probs(masked_logits, batch_idx, n_graphs)
    a, data_starts = F.sample_node(p, n_nodes)
    assert (data_starts == th.tensor([0, 3])).all()
    assert (a == th.tensor([[0], [1]])).all()
    assert (th.trunc(p) == th.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])).all()


def test_entropy():
    p = th.tensor([[0.5, 0.5]])
    e = F.entropy(p, 1)
    e = e * 1 / th.log(th.tensor(2.0))
    assert e.item() == 1.0

    p = th.tensor([[1.0, 0.0], [1 / 2, 1 / 2]])
    e = F.entropy(p, 2)
    e = e * 1 / th.log(th.tensor(2.0))
    assert e.item() == 0.5


def test_masked_entropy():
    p = th.tensor([[0.5, 0.5]])
    mask = th.tensor([[True, True]])
    e = F.masked_entropy(p, mask, 1)
    e = e * 1 / th.log(th.tensor(2.0))
    assert e.item() == 1.0

    p = th.tensor([[1 / 2, 0.0, 0.0], [0.0, 1 / 2, 1.0]])
    mask = th.tensor([[True, False, False], [False, True, False]])
    e = F.masked_entropy(p, mask, 2)
    e = e * 1 / th.log(th.tensor(2.0))
    assert e.item() == 0.5


def test_sample_action_given_node():
    x = th.tensor([[10, 0], [0, 10], [10, 1]])
    action_mask = th.tensor([[True, True]])
    node = th.tensor([[1]])
    batch = th.tensor([0, 0, 0])
    n_nodes = th.tensor([3])
    action_logits = F.action_logits_given_node(x, node, n_nodes)
    masked_action_logits = F.mask_logits(action_logits, action_mask)
    p = F.softmax(masked_action_logits)
    a = F.sample_action(p)
    assert (a == th.tensor([[1]])).all()

    x = th.tensor([[10, 0], [0, 10], [10, 1]])
    action_mask = th.tensor([[True, True], [False, True]])
    node = th.tensor([[1, 0]])
    batch = th.tensor([0, 0, 1])
    n_nodes = th.tensor([2, 1])
    action_logits = F.action_logits_given_node(x, node, n_nodes)
    masked_action_logits = F.mask_logits(action_logits, action_mask)
    p = F.softmax(masked_action_logits)
    a = F.sample_action(p)
    assert (a == th.tensor([[1], [1]])).all()


def test_sample_node_given_action():
    x = th.tensor([[10, 0], [0, 10], [0, 0]])
    node_mask = th.tensor([True, True, False])
    action = th.tensor([[1]])
    batch = th.tensor([0, 0, 0])
    n_nodes = th.tensor([3])
    n_graphs = 1

    node_logits = F.node_logits_given_action(x, action, batch)
    masked_node_logits = F.mask_logits(node_logits, node_mask)
    p = F.segmented_softmax(masked_node_logits, batch, n_graphs)

    a, _ = F.sample_node(
        p,
        n_nodes,
    )
    assert (a == th.tensor([[1]])).all()

    x = th.tensor([[10, 0], [0, 10], [10, 0], [0, 10], [100, 0]])
    node_mask = th.tensor([True, True, True, True, False])
    action = th.tensor([[1], [0]])
    batch = th.tensor([0, 0, 1, 1, 1])
    n_nodes = th.tensor([2, 3])
    n_graphs = 2

    node_logits = F.node_logits_given_action(x, action, batch)
    masked_node_logits = F.mask_logits(node_logits, node_mask)
    p = F.segmented_softmax(masked_node_logits, batch, n_graphs)

    a, _ = F.sample_node(
        p,
        n_nodes,
    )
    assert (a == th.tensor([[1], [0]])).all()


def test_sample_action_and_node():
    x1 = th.tensor([[10, 0]])
    x2 = th.tensor([10, 0, 0])
    mask1 = th.tensor([[True, True]])
    mask2 = th.tensor([True, True, False])
    batch = th.tensor([0, 0, 0])
    n_nodes = th.tensor([3])

    a, logprob, *_ = F.sample_action_and_node(
        x1,
        x2,
        mask1,
        mask2,
        batch,
        n_nodes,
    )
    assert (a == th.tensor([[0, 0]])).all()
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

    assert (a == th.tensor([[0, 0]])).all()
    assert eval_logprob.shape == (1,)
    assert eval_logprob == logprob


@pytest.mark.parametrize("deterministic", [True, False])
def test_sample_action_then_node(deterministic: bool):
    action_logits = th.tensor([[10, 100]], dtype=th.float32)  # ln(p(a))
    node_given_action_logits = th.tensor(
        [[20, 0], [0, 10], [100, 0]], dtype=th.float32
    )  # ln(p(n | a))
    node_logits = (action_logits + node_given_action_logits).sum(-1)
    action_given_node_logits = th.tensor(
        [[10, 100], [0, 0], [0, 0]], dtype=th.float32
    )  # ln(p(a | n))

    mask1 = th.tensor([[True, False]])
    mask2 = th.tensor([True, True, False])
    batch = th.tensor([0, 0, 0])
    n_nodes = th.tensor([3])

    a, logprob, h, *_ = F.sample_action_then_node(
        node_logits,
        action_given_node_logits,
        node_given_action_logits,
        mask1,
        mask2,
        batch,
        n_nodes,
        deterministic=deterministic,
    )
    assert (a == th.tensor([[0, 0]])).all()
    assert logprob.shape == (1,)

    eval_logprob, h, *_ = F.eval_action_then_node(
        a,
        node_logits,
        action_given_node_logits,
        node_given_action_logits,
        mask1,
        mask2,
        batch,
        n_nodes,
    )

    assert (a == th.tensor([[0, 0]])).all()
    assert eval_logprob.shape == (1,)
    assert eval_logprob == logprob


@pytest.mark.parametrize("deterministic", [True, False])
def test_sample_node_then_action(deterministic: bool):
    node_logits = th.tensor([10, 100, 0])
    action_given_node_logits = th.tensor([[10, 100], [0, 10], [100, 0]])
    mask1 = th.tensor([True, False, True])
    mask2 = th.tensor([[True, False]])
    batch = th.tensor([0, 0, 0])
    n_nodes = th.tensor([3])

    a, logprob, h, *_ = F.sample_node_then_action(
        action_given_node_logits,
        node_logits,
        mask2,
        mask1,
        batch,
        n_nodes,
        deterministic=deterministic,
    )
    assert (a == th.tensor([[0, 0]])).all()
    assert logprob.shape == (1,)

    eval_logprob, h, *_ = F.eval_node_then_action(
        a,
        action_given_node_logits,
        node_logits,
        mask2,
        mask1,
        batch,
        n_nodes,
    )

    assert (a == th.tensor([[0, 0]])).all()
    assert eval_logprob.shape == (1,)
    assert eval_logprob == logprob


@pytest.mark.skip(reason="Not implemented")
def test_sample_node_set():
    x = th.tensor([1.0, 1.0, 0.0])
    mask = th.tensor([True, True, False])
    batch = th.tensor([0, 0, 0])
    n_nodes = th.tensor([3])
    a, logprob = F.sample_node_set(x, mask, n_nodes)
    assert (a[0] == th.tensor([0, 1])).all()
    assert logprob.shape == (1,)

    pass
