"""Derived geometric descriptors for finite inverse Collatz graphs.

The overlap descriptor in this module is intentionally not called
Ollivier-Ricci curvature: it compares closed-neighborhood overlap and does not
compute the Wasserstein distance required by Ollivier-Ricci curvature.
"""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np
import pandas as pd


def _ancestor_count_array(
    graph: nx.DiGraph, ancestor_counts: Mapping[int, float] | np.ndarray
) -> np.ndarray:
    """Return ancestor counts indexed by node value for nodes ``1..N``."""
    n = graph.number_of_nodes()
    if isinstance(ancestor_counts, np.ndarray):
        if ancestor_counts.shape != (n,):
            raise ValueError("ancestor_counts must have one value per graph node")
        return np.asarray(ancestor_counts, dtype=np.float64)
    values = np.zeros(n, dtype=np.float64)
    for node in graph.nodes:
        if node not in ancestor_counts:
            raise KeyError(f"missing ancestor count for node {node}")
        values[node - 1] = float(ancestor_counts[node])
    return values


def neighborhood_overlap_curvature(graph: nx.DiGraph) -> np.ndarray:
    """Compute closed-neighborhood overlap for each node.

    For an undirected edge ``(u, v)``, the edge score is
    ``2*|N[u] intersect N[v]| / (|N[u]| + |N[v]|)``. The node value is the
    mean score over incident edges. This is a similarity descriptor, not
    Ollivier-Ricci curvature.
    """
    undirected = graph.to_undirected()
    n = graph.number_of_nodes()
    neighborhoods = {
        node: set(undirected.neighbors(node)) | {node} for node in undirected.nodes
    }
    values = np.zeros(n, dtype=np.float64)
    for node in undirected.nodes:
        scores = []
        for neighbor in undirected.neighbors(node):
            left = neighborhoods[node]
            right = neighborhoods[neighbor]
            denominator = len(left) + len(right)
            scores.append(2.0 * len(left & right) / denominator if denominator else 0.0)
        values[node - 1] = float(np.mean(scores)) if scores else 0.0
    return values


def branching_entropy(
    graph: nx.DiGraph, ancestor_counts: Mapping[int, float] | np.ndarray
) -> np.ndarray:
    """Compute entropy of predecessor ancestor-count proportions."""
    counts = _ancestor_count_array(graph, ancestor_counts)
    values = np.zeros(graph.number_of_nodes(), dtype=np.float64)
    for node in graph.nodes:
        predecessors = list(graph.predecessors(node))
        if len(predecessors) <= 1:
            continue
        sizes = np.asarray([counts[pred - 1] for pred in predecessors], dtype=np.float64)
        probabilities = sizes / sizes.sum()
        values[node - 1] = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    return values


def ancestor_density(
    graph: nx.DiGraph,
    ancestor_counts: Mapping[int, float] | np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """Compute ancestor count divided by the reverse neighborhood volume."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    counts = _ancestor_count_array(graph, ancestor_counts)
    values = np.zeros(graph.number_of_nodes(), dtype=np.float64)
    for node in graph.nodes:
        visited = {node}
        frontier = {node}
        for _ in range(radius):
            next_frontier = set()
            for current in frontier:
                next_frontier.update(graph.predecessors(current))
            frontier = next_frontier - visited
            visited.update(frontier)
        values[node - 1] = counts[node - 1] / len(visited) if visited else 0.0
    return values


def local_potential(
    graph: nx.DiGraph,
    ancestor_counts: Mapping[int, float] | np.ndarray,
    root: int = 1,
) -> np.ndarray:
    """Compute ``u(node) = sum(log(ancestor_count + 1))`` to the root.

    The forward graph has at most one successor per node. Potentials are
    memoized as paths are resolved, so shared trajectories are traversed once.
    The root is anchored at zero, matching the existing descriptor definition.
    """
    if root not in graph:
        raise ValueError("root must be a node in the graph")
    counts = _ancestor_count_array(graph, ancestor_counts)
    successor = {
        source: target for source, target in graph.edges if source in graph and target in graph
    }
    weights = np.log1p(counts)
    memo: dict[int, float] = {root: 0.0}
    for start in graph.nodes:
        if start in memo:
            continue
        path: list[int] = []
        positions: set[int] = set()
        current = start
        while current not in memo and current in successor and current not in positions:
            positions.add(current)
            path.append(current)
            current = successor[current]
        running = memo.get(current, 0.0)
        for node in reversed(path):
            running += float(weights[node - 1])
            memo[node] = running
    return np.asarray([memo.get(node, 0.0) for node in range(1, graph.number_of_nodes() + 1)])


def flow_energy(graph: nx.DiGraph, potential: np.ndarray) -> np.ndarray:
    """Compute squared potential differences across incoming edges."""
    if potential.shape != (graph.number_of_nodes(),):
        raise ValueError("potential must have one value per graph node")
    values = np.zeros(graph.number_of_nodes(), dtype=np.float64)
    for node in graph.nodes:
        node_potential = potential[node - 1]
        values[node - 1] = sum(
            (node_potential - potential[pred - 1]) ** 2
            for pred in graph.predecessors(node)
        )
    return values


def add_geometric_quantities(
    graph: nx.DiGraph, feature_data: pd.DataFrame, *, root: int = 1
) -> pd.DataFrame:
    """Return a copy of node features augmented with geometric descriptors."""
    if "node" not in feature_data or "ancestor_count" not in feature_data:
        raise ValueError("feature_data must contain node and ancestor_count columns")
    result = feature_data.copy()
    ancestor_counts = {
        int(row.node): float(row.ancestor_count)
        for row in feature_data[["node", "ancestor_count"]].itertuples(index=False)
    }
    overlap = neighborhood_overlap_curvature(graph)
    entropy = branching_entropy(graph, ancestor_counts)
    density = ancestor_density(graph, ancestor_counts)
    potential = local_potential(graph, ancestor_counts, root=root)
    energy = flow_energy(graph, potential)
    result["neighborhood_overlap_curvature"] = overlap
    result["branching_entropy"] = entropy
    result["ancestor_density_k2"] = density
    result["local_potential"] = potential
    result["flow_energy"] = energy
    return result
