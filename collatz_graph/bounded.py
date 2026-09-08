"""Finite inverse Collatz graphs on the integer domain ``1, ..., N``.

The graph uses forward edge orientation: ``u -> T(u)``. Calling it an
inverse graph refers to the fact that its construction and analysis support
the inverse Collatz relation; the edge direction is intentionally not
reversed so that every retained edge can be checked directly with
``collatz_successor``.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
from tqdm.auto import tqdm

from .core import collatz_successor

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BoundedGraphBuildResult:
    """Result of constructing and optionally exporting a bounded graph."""

    graph: nx.DiGraph
    runtime_seconds: float
    n: int


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be at least 1")


def build_inverse_graph_up_to(n: int, *, show_progress: bool = True) -> BoundedGraphBuildResult:
    """Construct the finite graph induced by ``{1, ..., n}``.

    Nodes are all integers from 1 through ``n``. For each node ``u``, the
    edge ``u -> T(u)`` is retained exactly when ``T(u) <= n``. Iteration is
    over integers directly, so no edge or node list is materialized before
    NetworkX receives it. The resulting NetworkX graph itself necessarily
    stores every requested node and retained edge in memory.

    Args:
        n: Inclusive upper bound for the finite integer domain.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        A graph, measured construction runtime, and the validated bound.

    Complexity:
        Construction performs one Collatz evaluation per node, so the time
        complexity is ``O(N)``. NetworkX storage is ``O(N + E_N)``, where
        ``E_N`` is the number of retained edges and ``E_N <= N``.
    """
    _validate_n(n)
    start = perf_counter()
    graph = nx.DiGraph(name=f"Finite inverse Collatz graph on 1..{n}")
    graph.add_nodes_from(range(1, n + 1))

    iterator = tqdm(range(1, n + 1), desc="Building bounded graph", disable=not show_progress)
    for source in iterator:
        target = collatz_successor(source)
        if target <= n:
            graph.add_edge(source, target)

    runtime_seconds = perf_counter() - start
    graph.graph.update(
        {
            "construction": "finite_domain",
            "domain_minimum": 1,
            "domain_maximum": n,
            "edge_orientation": "forward",
            "runtime_seconds": runtime_seconds,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        }
    )
    LOGGER.info(
        "Built bounded graph for 1..%d: %d nodes, %d edges in %.6f s",
        n,
        graph.number_of_nodes(),
        graph.number_of_edges(),
        runtime_seconds,
    )
    return BoundedGraphBuildResult(graph=graph, runtime_seconds=runtime_seconds, n=n)


def _write_nodes(graph: nx.DiGraph, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("node",))
        for node in graph.nodes:
            writer.writerow((node,))


def _write_edges(graph: nx.DiGraph, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source", "target"))
        for source, target in graph.edges:
            writer.writerow((source, target))


def _git_commit_sha() -> str | None:
    """Return the current Git commit SHA when running inside a Git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = completed.stdout.strip()
    return sha or None


def _metadata(graph: nx.DiGraph, result: BoundedGraphBuildResult) -> dict[str, Any]:
    """Create JSON-serializable metadata for a saved graph."""
    return {
        **graph.graph,
        "python_version": platform.python_version(),
        "networkx_version": nx.__version__,
        "runtime_seconds": result.runtime_seconds,
        "platform": platform.platform(),
        "git_commit_sha": _git_commit_sha(),
    }


def save_graph_artifacts(
    result: BoundedGraphBuildResult, output_directory: str | Path
) -> dict[str, Path]:
    """Stream node, edge, and metadata files to ``output_directory``.

    Files are named ``nodes_N.csv``, ``edges_N.csv``, and ``metadata_N.json``.
    CSV writing uses iterators over NetworkX views, avoiding duplicate Python
    lists and keeping peak export memory proportional to NetworkX storage.
    """
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    n = result.n
    paths = {
        "nodes": output / f"nodes_{n}.csv",
        "edges": output / f"edges_{n}.csv",
        "metadata": output / f"metadata_{n}.json",
    }
    _write_nodes(result.graph, paths["nodes"])
    _write_edges(result.graph, paths["edges"])
    paths["metadata"].write_text(
        json.dumps(_metadata(result.graph, result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved graph artifacts to %s", output.resolve())
    return paths


def build_and_save_inverse_graph(
    n: int,
    output_directory: str | Path,
    *,
    show_progress: bool = True,
) -> BoundedGraphBuildResult:
    """Build a finite graph and save all reproducibility artifacts."""
    result = build_inverse_graph_up_to(n, show_progress=show_progress)
    save_graph_artifacts(result, output_directory)
    return result