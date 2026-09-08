"""Regression tests for bounded-memory Node2Vec training."""

import numpy as np

import collatz_graph.node2vec as node2vec
from collatz_graph.bounded import build_inverse_graph_up_to
from collatz_graph.node2vec import Node2VecConfig, iter_node2vec_walks, train_node2vec


def test_streaming_pair_batches_are_bounded() -> None:
    graph = build_inverse_graph_up_to(10, show_progress=False).graph
    config = Node2VecConfig(
        dimensions=8,
        walk_length=6,
        walks_per_node=2,
        context_size=2,
        epochs=1,
        seed=7,
        backend="numpy",
        batch_size=3,
    )
    node_ids = np.asarray(sorted(graph.nodes), dtype=np.int64)
    node_to_index = {int(node): index for index, node in enumerate(node_ids)}
    batches = list(
        node2vec._iter_training_batches(
            iter_node2vec_walks(graph, config),
            node_to_index,
            config.context_size,
            config.batch_size,
        )
    )

    assert batches
    assert max(len(targets) for targets, _ in batches) <= config.batch_size
    assert sum(len(targets) for targets, _ in batches) == sum(
        len(contexts) for _, contexts in batches
    )


def test_training_does_not_materialize_all_pairs(monkeypatch) -> None:
    graph = build_inverse_graph_up_to(10, show_progress=False).graph
    config = Node2VecConfig(
        dimensions=8,
        walk_length=6,
        walks_per_node=1,
        epochs=1,
        seed=7,
        backend="numpy",
        batch_size=4,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("legacy all-pairs materializer was called")

    monkeypatch.setattr(node2vec, "_training_pairs", fail_if_called)
    result = train_node2vec(graph, config, show_progress=False)

    assert result.embeddings.shape == (10, 8)
