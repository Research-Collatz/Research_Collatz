"""Hand-checkable tests for geometric descriptors."""

import math

import networkx as nx
import numpy as np

import collatz_graph.features as feature_module
from collatz_graph.bounded import build_inverse_graph_up_to
from collatz_graph.features import compute_node_features
from collatz_graph.geometric import (
    add_geometric_quantities,
    ancestor_density,
    branching_entropy,
    flow_energy,
    local_potential,
    neighborhood_overlap_curvature,
)


def test_geometric_descriptors_on_two_branch_graph() -> None:
    graph = nx.DiGraph([(2, 1), (3, 1)])
    ancestor_counts = {1: 3, 2: 1, 3: 1}

    overlap = neighborhood_overlap_curvature(graph)
    entropy = branching_entropy(graph, ancestor_counts)
    density = ancestor_density(graph, ancestor_counts, radius=1)
    potential = local_potential(graph, ancestor_counts)
    energy = flow_energy(graph, potential)

    assert np.isclose(overlap[0], 0.8)
    assert np.isclose(overlap[1], 0.8)
    assert np.isclose(overlap[2], 0.8)
    assert np.isclose(entropy[0], math.log(2))
    assert np.array_equal(entropy[1:], np.zeros(2))
    assert np.isclose(density[0], 1.0)
    assert np.array_equal(density[1:], np.ones(2))
    assert np.isclose(potential[0], 0.0)
    assert np.allclose(potential[1:], math.log(2))
    assert np.isclose(energy[0], 2 * math.log(2) ** 2)
    assert np.array_equal(energy[1:], np.zeros(2))


def test_feature_table_records_convergence_status() -> None:
    graph = nx.DiGraph([(2, 1), (3, 1)])
    graph.add_nodes_from([1, 2, 3])
    features = compute_node_features(graph, show_progress=False).data

    assert set(features["trajectory_status"]) == {"reached_1"}
    assert features["converged_to_1"].all()


def test_feature_table_marks_trajectory_limit(monkeypatch) -> None:
    monkeypatch.setattr(feature_module, "TRAJECTORY_STEP_LIMIT", 0)
    graph = build_inverse_graph_up_to(2, show_progress=False).graph

    features = compute_node_features(graph, show_progress=False).data.set_index("node")

    assert features.loc[1, "trajectory_status"] == "reached_1"
    assert features.loc[2, "trajectory_status"] == "trajectory_limit_reached"
    assert not features.loc[2, "converged_to_1"]
    assert features.loc[2, "total_stopping_time"] == -1


def test_add_geometric_quantities_uses_explicit_overlap_name() -> None:
    graph = nx.DiGraph([(2, 1), (3, 1)])
    graph.add_nodes_from([1, 2, 3])
    features = compute_node_features(graph, show_progress=False).data

    result = add_geometric_quantities(graph, features)

    assert "neighborhood_overlap_curvature" in result
    assert "ollivier_ricci" not in result
