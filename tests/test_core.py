"""Correctness tests for exact inverse Collatz graph construction."""

import networkx as nx
import numpy as np
import pytest

from collatz_graph import InverseGraphConfig, build_inverse_graph, collatz_successor, inverse_predecessors
from collatz_graph.bounded import build_inverse_graph_up_to, save_graph_artifacts
from collatz_graph.features import compute_node_features
from collatz_graph.statistics import compute_graph_statistics
from collatz_graph.node2vec import Node2VecConfig, generate_node2vec_walks, save_node2vec_result, train_node2vec


def test_forward_map() -> None:
    assert collatz_successor(8) == 4
    assert collatz_successor(7) == 22


def test_inverse_branches() -> None:
    assert inverse_predecessors(1) == (2,)
    assert inverse_predecessors(4) == (1, 8)
    assert inverse_predecessors(10) == (3, 20)


def test_invalid_values() -> None:
    with pytest.raises(ValueError):
        collatz_successor(0)
    with pytest.raises(TypeError):
        inverse_predecessors(True)


def test_config_validation() -> None:
    with pytest.raises(TypeError):
        InverseGraphConfig(roots=[1])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        InverseGraphConfig(max_nodes=10.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        InverseGraphConfig(max_nodes=0)
    with pytest.raises(ValueError):
        InverseGraphConfig(roots=(1, 2), max_nodes=1)


def test_edges_obey_forward_map() -> None:
    graph = build_inverse_graph(InverseGraphConfig(roots=(1,), max_nodes=200, show_progress=False))
    assert isinstance(graph, nx.DiGraph)
    assert all(collatz_successor(source) == target for source, target in graph.edges)
    assert graph.number_of_nodes() <= 200


def test_budget_is_a_hard_limit_without_early_termination() -> None:
    graph = build_inverse_graph(
        InverseGraphConfig(roots=(1,), max_nodes=3, show_progress=False)
    )
    assert list(graph.nodes) == [1, 2, 4]
    assert set(graph.edges) == {(2, 1), (4, 2)}
    assert graph.number_of_nodes() == 3


def test_expansion_is_deterministic() -> None:
    config = InverseGraphConfig(roots=(1,), max_nodes=100, show_progress=False)
    first = build_inverse_graph(config)
    second = build_inverse_graph(config)
    assert list(first.nodes(data=True)) == list(second.nodes(data=True))
    assert list(first.edges) == list(second.edges)


def test_bounded_graph_contains_only_in_domain_edges(tmp_path) -> None:
    result = build_inverse_graph_up_to(10, show_progress=False)
    assert set(result.graph.nodes) == set(range(1, 11))
    assert all(target <= 10 and collatz_successor(source) == target for source, target in result.graph.edges)
    assert result.graph.graph["node_count"] == 10
    assert result.graph.graph["edge_count"] == result.graph.number_of_edges()

    paths = save_graph_artifacts(result, tmp_path)
    assert all(path.exists() for path in paths.values())
    assert paths["nodes"].read_text(encoding="utf-8").splitlines()[0] == "node"
    assert paths["edges"].read_text(encoding="utf-8").splitlines()[0] == "source,target"
    assert '"domain_maximum": 10' in paths["metadata"].read_text(encoding="utf-8")


def test_bounded_graph_boundary_and_input_validation() -> None:
    result = build_inverse_graph_up_to(1, show_progress=False)
    assert list(result.graph.nodes) == [1]
    assert result.graph.number_of_edges() == 0
    with pytest.raises(ValueError):
        build_inverse_graph_up_to(0, show_progress=False)
    with pytest.raises(TypeError):
        build_inverse_graph_up_to(True, show_progress=False)


def test_node_features_on_small_graph() -> None:
    graph = build_inverse_graph_up_to(10, show_progress=False).graph
    features = compute_node_features(graph, show_progress=False).data.set_index("node")
    assert features.loc[1, "total_stopping_time"] == 0
    assert features.loc[7, "total_stopping_time"] == 16
    assert features.loc[7, "maximum_excursion"] == 52
    assert features.loc[7, "binary_length"] == 3
    assert features.loc[8, "stopping_time"] == 1
    assert features.loc[1, "branching_factor"] == features.loc[1, "in_degree"]
    assert features.loc[7, "distance_from_root"] == -1


def test_graph_statistics_on_small_graph() -> None:
    graph = build_inverse_graph_up_to(10, show_progress=False).graph
    result = compute_graph_statistics(graph, exact_diameter_limit=100, show_progress=False)
    assert result.summary.loc[0, "nodes"] == 10
    assert set(result.centrality["node"]) == set(range(1, 11))
    assert result.degree_distribution["node_count"].sum() == 10
    assert result.summary.loc[0, "weak_components"] >= 1
    assert set(result.spectral["statistic"]) >= {"adjacency_spectral_radius_estimate", "adjacency_frobenius_norm"}


def test_node2vec_reproducibility_and_shape() -> None:
    graph = build_inverse_graph_up_to(10, show_progress=False).graph
    config = Node2VecConfig(dimensions=8, walk_length=6, walks_per_node=1, epochs=1, seed=7, backend="numpy")
    first = train_node2vec(graph, config, show_progress=False)
    second = train_node2vec(graph, config, show_progress=False)
    assert first.embeddings.shape == (10, 8)
    assert np.array_equal(first.node_ids, np.arange(1, 11))
    assert np.allclose(first.embeddings, second.embeddings)
    assert len(generate_node2vec_walks(graph, config)[1]) == 10


def test_node2vec_saved_artifacts_preserve_row_mapping(tmp_path) -> None:
    graph = build_inverse_graph_up_to(10, show_progress=False).graph
    result = train_node2vec(graph, Node2VecConfig(dimensions=8, walk_length=6, walks_per_node=1, epochs=1, seed=7, backend="numpy"), show_progress=False)
    paths = save_node2vec_result(result, tmp_path, extra_metadata={"graph": {"domain_maximum": 10}})
    assert all(path.exists() for path in paths.values())
    assert np.array_equal(np.load(paths["embeddings"]), result.embeddings)
    metadata = paths["metadata"].read_text(encoding="utf-8")
    assert '"domain_maximum": 10' in metadata
