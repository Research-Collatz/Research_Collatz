#!/usr/bin/env python3
"""Compute geometric quantities on the inverse Collatz graph.

This script computes five quantities on the bounded inverse Collatz graph
on the integer domain ``1, ..., N``:

    1. ``ego_overlap_local``     - mean Jaccard overlap of 1-hop ego
                                    neighbourhoods across incident edges
                                    (proxy for local curvature; this is not
                                    Ollivier-Ricci curvature).
    2. ``branching_entropy``     - Shannon entropy (bits) of the
                                    distribution over inverse-branch sizes
                                    at each node.
    3. ``ancestor_density_k2``   - ``ancestor_count / |k=2 inverse ball|``.
    4. ``local_potential``       - sum of ``log2(ancestor_count + 1)`` along
                                    the forward Collatz orbit from a node
                                    down to (but not including) the root.
    5. ``flow_energy``           - sum of squared potential differences
                                    between a node and its inverse
                                    predecessors.

Run from the repository root:

    python compute_geometric_quantities.py --max-node 10000 \\
        --output outputs/scaling/features_with_geometry.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from collatz_graph.bounded import build_inverse_graph_up_to
from collatz_graph.core import collatz_successor
from collatz_graph.features import compute_node_features

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bfs_k_hop_predecessors(
    graph: nx.DiGraph, source: int, k: int
) -> set[int]:
    """Return the inverse ``k``-hop neighbourhood of ``source``.

    The returned set contains ``source`` itself, so a 0-hop BFS returns
    ``{source}`` and a 1-hop BFS returns ``{source} ∪ predecessors(source)``.
    The walk is breadth-first on the reverse graph to mirror inverse
    Collatz dynamics.
    """
    visited: set[int] = {source}
    frontier: set[int] = {source}
    for _ in range(k):
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(graph.predecessors(node))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def _progress(iterable: Iterable, *, desc: str, total: int) -> tqdm:
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


# ---------------------------------------------------------------------------
# Quantity 1: ego-graph overlap (proxy for local curvature)
# ---------------------------------------------------------------------------


def _compute_ego_overlap(
    graph: nx.DiGraph, undirected: nx.Graph, nodes: list[int]
) -> np.ndarray:
    """Mean Jaccard overlap of 1-hop ego neighbourhoods across edges.

    The result is **not** Ollivier-Ricci curvature: that requires the lazy
    random-walk measure, not an indicator measure. We expose it under an
    honest name.
    """
    n = len(nodes)
    overlap = np.zeros(n, dtype=np.float64)
    ego_cache: dict[int, set[int]] = {}

    for node in _progress(nodes, desc="ego overlap", total=n):
        ego = ego_cache.get(node)
        if ego is None:
            ego = set(undirected.neighbors(node)) | {node}
            ego_cache[node] = ego
        neighbours = list(undirected.neighbors(node))
        if not neighbours:
            overlap[node - 1] = 0.0
            continue
        edge_values: list[float] = []
        for nbr in neighbours:
            other = ego_cache.get(nbr)
            if other is None:
                other = set(undirected.neighbors(nbr)) | {nbr}
                ego_cache[nbr] = other
            union = len(ego) + len(other)
            if union == 0:
                edge_values.append(0.0)
                continue
            inter = len(ego & other)
            edge_values.append(2.0 * inter / union)
        overlap[node - 1] = float(np.mean(edge_values))
    return overlap


# ---------------------------------------------------------------------------
# Quantity 2: branching entropy
# ---------------------------------------------------------------------------


def _compute_branching_entropy(
    graph: nx.DiGraph, nodes: list[int], ancestor_count: dict[int, int]
) -> np.ndarray:
    """Shannon entropy (bits) of the inverse-branch size distribution."""
    n = len(nodes)
    entropy = np.zeros(n, dtype=np.float64)

    for node in _progress(nodes, desc="branching entropy", total=n):
        predecessors = list(graph.predecessors(node))
        if len(predecessors) <= 1:
            entropy[node - 1] = 0.0
            continue
        sizes = np.asarray(
            [ancestor_count[predecessor] for predecessor in predecessors],
            dtype=np.float64,
        )
        if sizes.sum() <= 0:
            entropy[node - 1] = 0.0
            continue
        probs = sizes / sizes.sum()
        entropy[node - 1] = float(-np.sum(probs * np.log2(probs + 1e-12)))
    return entropy


# ---------------------------------------------------------------------------
# Quantity 3: ancestor density at k=2
# ---------------------------------------------------------------------------


def _compute_ancestor_density_k2(
    graph: nx.DiGraph, nodes: list[int], ancestor_count: dict[int, int]
) -> np.ndarray:
    n = len(nodes)
    density = np.zeros(n, dtype=np.float64)
    for node in _progress(nodes, desc="ancestor density k=2", total=n):
        ball = bfs_k_hop_predecessors(graph, node, k=2)
        volume = len(ball)
        if volume == 0:
            density[node - 1] = 0.0
            continue
        density[node - 1] = ancestor_count[node] / volume
    return density


# ---------------------------------------------------------------------------
# Quantity 4: local potential
# ---------------------------------------------------------------------------


def _compute_local_potential(
    graph: nx.DiGraph, nodes: list[int], ancestor_count: dict[int, int]
) -> np.ndarray:
    """Telescoping sum of ``log2(ancestor_count + 1)`` along the forward orbit.

    The walk follows the canonical forward Collatz successor, not the
    NetworkX edge set, so it is robust to extra edges in the graph.
    """
    n = len(nodes)
    f = {node: float(np.log2(ancestor_count[node] + 1)) for node in nodes}
    potential = np.zeros(n, dtype=np.float64)
    in_domain = set(nodes)
    for node in _progress(nodes, desc="local potential", total=n):
        if node == 1:
            potential[node - 1] = 0.0
            continue
        current = node
        accumulator = 0.0
        seen: set[int] = set()
        while current in in_domain and current not in seen and current != 1:
            seen.add(current)
            accumulator += f[current]
            current = collatz_successor(current)
        potential[node - 1] = accumulator
    return potential


# ---------------------------------------------------------------------------
# Quantity 5: flow energy
# ---------------------------------------------------------------------------


def _compute_flow_energy(
    graph: nx.DiGraph, nodes: list[int], potential_by_node: dict[int, float]
) -> np.ndarray:
    n = len(nodes)
    energy = np.zeros(n, dtype=np.float64)
    for node in _progress(nodes, desc="flow energy", total=n):
        u_value = potential_by_node[node]
        predecessors = list(graph.predecessors(node))
        if not predecessors:
            energy[node - 1] = 0.0
            continue
        diffs = [(u_value - potential_by_node[predecessor]) ** 2 for predecessor in predecessors]
        energy[node - 1] = float(np.sum(diffs))
    return energy


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def compute_geometric_quantities(max_node: int) -> pd.DataFrame:
    """Compute all five geometric quantities for the graph on ``1..max_node``."""
    LOGGER.info("Building bounded graph for 1..%d", max_node)
    build_result = build_inverse_graph_up_to(max_node, show_progress=True)
    graph = build_result.graph

    LOGGER.info("Computing base node features")
    feature_result = compute_node_features(graph, root=1, show_progress=True)
    features = feature_result.data

    nodes = sorted(graph.nodes())
    n = len(nodes)
    if features["node"].min() != 1 or features["node"].max() != n:
        raise ValueError(
            f"Feature table node range ({features['node'].min()}.."
            f"{features['node'].max()}) does not match graph domain (1..{n})"
        )

    ancestor_count: dict[int, int] = dict(
        zip(features["node"].to_numpy(), features["ancestor_count"].to_numpy())
    )

    undirected = graph.to_undirected(as_view=False)

    ego_overlap = _compute_ego_overlap(graph, undirected, nodes)
    branching = _compute_branching_entropy(graph, nodes, ancestor_count)
    density = _compute_ancestor_density_k2(graph, nodes, ancestor_count)
    potential = _compute_local_potential(graph, nodes, ancestor_count)
    potential_by_node = {node: float(potential[index]) for index, node in enumerate(nodes)}
    energy = _compute_flow_energy(graph, nodes, potential_by_node)

    enriched = features.copy()
    enriched["ego_overlap_local"] = ego_overlap
    enriched["branching_entropy"] = branching
    enriched["ancestor_density_k2"] = density
    enriched["local_potential"] = potential
    enriched["flow_energy"] = energy

    LOGGER.info(
        "Computed geometric quantities: ego_overlap mean=%.4f, "
        "branching_entropy mean=%.4f, ancestor_density mean=%.4f, "
        "local_potential mean=%.4f, flow_energy mean=%.4f",
        float(ego_overlap.mean()),
        float(branching.mean()),
        float(density.mean()),
        float(potential.mean()),
        float(energy.mean()),
    )
    return enriched


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-node",
        type=int,
        default=10_000,
        help="Inclusive upper bound N for the finite graph (default: 10000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/scaling/features_with_geometry.csv"),
        help="Destination CSV path (default: outputs/scaling/features_with_geometry.csv).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    enriched = compute_geometric_quantities(args.max_node)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    LOGGER.info("Wrote %d rows to %s", len(enriched), args.output.resolve())


if __name__ == "__main__":
    main()
