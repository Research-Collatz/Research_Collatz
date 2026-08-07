"""Descriptive statistics and publication tables for finite Collatz graphs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GraphStatisticsResult:
    """Publication-ready graph statistic tables and runtime metadata."""

    summary: pd.DataFrame
    degree_distribution: pd.DataFrame
    centrality: pd.DataFrame
    components: pd.DataFrame
    spectral: pd.DataFrame
    runtime_seconds: float
    betweenness_method: str
    diameter_method: str


def _pagerank_numpy(graph: nx.DiGraph, alpha: float = 0.85, tolerance: float = 1e-10, max_iter: int = 200) -> dict[int, float]:
    """Compute PageRank with an edge-streamed NumPy power iteration."""
    nodes = list(graph.nodes)
    index = {node: position for position, node in enumerate(nodes)}
    n_nodes = len(nodes)
    rank = np.full(n_nodes, 1.0 / n_nodes, dtype=np.float64)
    out_degree = np.fromiter((graph.out_degree(node) for node in nodes), dtype=np.int64, count=n_nodes)
    for _ in range(max_iter):
        next_rank = np.zeros(n_nodes, dtype=np.float64)
        dangling_mass = float(rank[out_degree == 0].sum())
        for source, target in graph.edges:
            next_rank[index[target]] += rank[index[source]] / out_degree[index[source]]
        next_rank = alpha * (next_rank + dangling_mass / n_nodes) + (1.0 - alpha) / n_nodes
        if float(np.abs(next_rank - rank).sum()) < tolerance:
            rank = next_rank
            break
        rank = next_rank
    return {node: float(rank[index[node]]) for node in nodes}


def _inverse_tree_depth(graph: nx.DiGraph, root: int) -> dict[int, int]:
    """Return shortest inverse distance from root, or -1 if unreachable."""
    depths = {node: -1 for node in graph}
    if root in graph:
        depths.update(nx.single_source_shortest_path_length(graph.reverse(copy=False), root))
    return depths


def _spectral_statistics(graph: nx.DiGraph, iterations: int, seed: int) -> pd.DataFrame:
    """Estimate directed adjacency spectral statistics without densifying."""
    nodes = list(graph.nodes)
    index = {node: position for position, node in enumerate(nodes)}
    vector = np.random.default_rng(seed).random(len(nodes))
    vector /= np.linalg.norm(vector)
    estimate = float("nan")
    for _ in range(iterations):
        next_vector = np.zeros_like(vector)
        for source, target in graph.edges:
            next_vector[index[target]] += vector[index[source]]
        norm = float(np.linalg.norm(next_vector))
        if norm == 0:
            estimate = 0.0
            break
        estimate = norm
        vector = next_vector / norm
    degrees = np.fromiter((graph.out_degree(node) for node in nodes), dtype=np.float64)
    frobenius_norm = float(np.sqrt(graph.number_of_edges()))
    return pd.DataFrame(
        [
            {
                "statistic": "adjacency_spectral_radius_estimate",
                "value": estimate,
                "method": f"{iterations}-iteration edge-wise power iteration",
            },
            {
                "statistic": "adjacency_frobenius_norm",
                "value": frobenius_norm,
                "method": "sqrt(number of directed edges)",
            },
            {
                "statistic": "maximum_out_degree",
                "value": float(degrees.max()) if degrees.size else 0.0,
                "method": "exact",
            },
        ]
    )


def compute_graph_statistics(
    graph: nx.DiGraph,
    *,
    root: int = 1,
    betweenness_k: int | None = None,
    exact_diameter_limit: int = 5_000,
    spectral_iterations: int = 100,
    random_seed: int = 20260722,
    show_progress: bool = True,
) -> GraphStatisticsResult:
    """Compute descriptive statistics for a finite forward-oriented graph.

    Degree distribution is exact. PageRank, directed closeness, and degree
    centralities are computed with NetworkX. Betweenness is exact when
    ``betweenness_k`` is ``None`` and the graph has at most 2,000 nodes;
    otherwise a reproducible node sample is used. The returned method label
    makes this distinction explicit in tables.

    Diameter is the exact undirected diameter of the largest weakly connected
    component when that component has at most ``exact_diameter_limit`` nodes.
    A directed graph may have infinite all-pairs distance, so this statistic
    is deliberately identified as a weak-component diameter.

    The spectral table avoids a dense adjacency matrix. Its radius is an
    estimate from edge-wise power iteration and should not be reported as an
    exact eigenvalue unless independently verified on a smaller graph.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one node")
    if root not in graph:
        raise ValueError("root must be present in graph")
    if exact_diameter_limit < 1 or spectral_iterations < 1:
        raise ValueError("limits and iteration counts must be positive")

    start = perf_counter()
    nodes = list(graph.nodes)
    n_nodes = len(nodes)
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    degree_distribution = (
        pd.DataFrame({"in_degree": pd.Series(in_degrees), "out_degree": pd.Series(out_degrees)})
        .groupby(["in_degree", "out_degree"], as_index=False)
        .size()
        .rename(columns={"size": "node_count"})
    )
    degree_distribution["proportion"] = degree_distribution["node_count"] / n_nodes

    if betweenness_k is None and n_nodes <= 2_000:
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        betweenness_method = "exact"
    else:
        sample_size = min(betweenness_k or 512, n_nodes)
        betweenness = nx.betweenness_centrality(graph, k=sample_size, normalized=True, seed=random_seed)
        betweenness_method = f"approximate, k={sample_size}, seed={random_seed}"
    pagerank = _pagerank_numpy(graph)
    closeness = nx.closeness_centrality(graph)
    in_degree_centrality = nx.in_degree_centrality(graph)
    out_degree_centrality = nx.out_degree_centrality(graph)
    centrality = pd.DataFrame(
        {
            "node": nodes,
            "in_degree_centrality": [in_degree_centrality[node] for node in nodes],
            "out_degree_centrality": [out_degree_centrality[node] for node in nodes],
            "betweenness_centrality": [betweenness[node] for node in nodes],
            "closeness_centrality": [closeness[node] for node in nodes],
            "pagerank": [pagerank[node] for node in nodes],
        }
    )

    weak_components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    strong_components = list(nx.strongly_connected_components(graph))
    component_rows = []
    for component_id, component in enumerate(weak_components):
        subgraph = graph.subgraph(component)
        component_rows.append(
            {
                "component_id": component_id,
                "component_type": "weak",
                "size": len(component),
                "edges": subgraph.number_of_edges(),
                "root_present": root in component,
            }
        )
    for component_id, component in enumerate(sorted(strong_components, key=len, reverse=True)):
        component_rows.append(
            {
                "component_id": component_id,
                "component_type": "strong",
                "size": len(component),
                "edges": graph.subgraph(component).number_of_edges(),
                "root_present": root in component,
            }
        )
    components = pd.DataFrame(component_rows)

    depths = _inverse_tree_depth(graph, root)
    depth_values = np.asarray(list(depths.values()), dtype=np.int64)
    largest_weak = graph.subgraph(weak_components[0]).copy()
    if len(largest_weak) <= exact_diameter_limit:
        diameter = nx.diameter(largest_weak.to_undirected())
        diameter_method = f"exact undirected diameter of largest weak component (n={len(largest_weak)})"
    else:
        diameter = np.nan
        diameter_method = f"not computed: largest weak component has {len(largest_weak)} nodes"

    summary = pd.DataFrame(
        [
            {
                "nodes": n_nodes,
                "edges": graph.number_of_edges(),
                "weak_components": len(weak_components),
                "strong_components": len(strong_components),
                "largest_weak_component": len(weak_components[0]),
                "largest_strong_component": max(map(len, strong_components)),
                "tree_depth_max": int(depth_values.max()),
                "tree_depth_mean_reachable": float(depth_values[depth_values >= 0].mean()),
                "average_branching_factor": float(np.mean(list(in_degrees.values()))),
                "weak_component_diameter": diameter,
                "diameter_method": diameter_method,
                "betweenness_method": betweenness_method,
            }
        ]
    )
    spectral = _spectral_statistics(graph, spectral_iterations, random_seed)
    runtime_seconds = perf_counter() - start
    LOGGER.info("Computed graph statistics for %d nodes in %.6f s", n_nodes, runtime_seconds)
    return GraphStatisticsResult(
        summary=summary,
        degree_distribution=degree_distribution,
        centrality=centrality,
        components=components,
        spectral=spectral,
        runtime_seconds=runtime_seconds,
        betweenness_method=betweenness_method,
        diameter_method=diameter_method,
    )


def save_statistics_tables(result: GraphStatisticsResult, output_directory: str | Path, stem: str = "graph") -> dict[str, Path]:
    """Save all publication tables as CSV files."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary": result.summary,
        "degree_distribution": result.degree_distribution,
        "centrality": result.centrality,
        "components": result.components,
        "spectral": result.spectral,
    }
    paths = {}
    for name, table in tables.items():
        path = output / f"{stem}_{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def save_publication_figures(
    graph: nx.DiGraph,
    result: GraphStatisticsResult,
    output_directory: str | Path,
    *,
    dpi: int = 300,
) -> dict[str, Path]:
    """Save degree, centrality, and component figures as publication PNGs."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12})
    paths: dict[str, Path] = {}

    degree = result.degree_distribution
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.bar(degree["in_degree"].astype(str), degree["node_count"], color="#2166ac", label="in-degree")
    ax.set_xlabel("In-degree")
    ax.set_ylabel("Number of nodes")
    ax.set_title("Inverse predecessor degree distribution")
    ax.legend(frameon=False)
    paths["degree_distribution"] = output / "degree_distribution.png"
    fig.savefig(paths["degree_distribution"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    axes[0].hist(result.centrality["pagerank"], bins=40, color="#1b9e77")
    axes[0].set_title("PageRank")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Nodes")
    axes[1].hist(result.centrality["betweenness_centrality"], bins=40, color="#d95f02")
    axes[1].set_title("Betweenness")
    axes[1].set_xlabel("Centrality")
    axes[1].set_ylabel("Nodes")
    paths["centrality"] = output / "centrality_distributions.png"
    fig.savefig(paths["centrality"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    weak = result.components[result.components["component_type"] == "weak"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.bar(np.arange(len(weak)), weak["size"], color="#762a83")
    ax.set_xlabel("Weak component rank")
    ax.set_ylabel("Nodes")
    ax.set_title("Weakly connected component sizes")
    paths["components"] = output / "weak_component_sizes.png"
    fig.savefig(paths["components"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved publication figures to %s", output.resolve())
    return paths