"""Arithmetic and structural features for finite Collatz graphs.

The functions in this module target the bounded graph produced by
``build_inverse_graph_up_to``. Nodes are assumed to be exactly ``1..N`` and
each node has at most one forward edge. This functional-graph structure makes
distance and ancestor counts computable with compact NumPy arrays instead of
one NetworkX traversal per node.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)
INT64_MAX = int(np.iinfo(np.int64).max)
TRAJECTORY_STEP_LIMIT = 1_000_000


@dataclass(frozen=True, slots=True)
class FeatureComputationResult:
    """A feature table and measured computation time."""

    data: pd.DataFrame
    runtime_seconds: float


def _validate_graph(graph: nx.DiGraph, root: int) -> int:
    nodes = graph.nodes
    if not nodes:
        raise ValueError("graph must contain nodes")
    minimum = min(nodes)
    maximum = max(nodes)
    if minimum != 1 or len(nodes) != maximum:
        raise ValueError("graph nodes must be exactly the integers 1..N")
    if root < 1 or root > maximum:
        raise ValueError("root must be a node in the graph")
    if any(graph.out_degree(node) > 1 for node in nodes):
        raise ValueError("graph must have at most one forward edge per node")
    return maximum


def _successor_array(graph: nx.DiGraph, n: int) -> np.ndarray:
    successor = np.full(n + 1, -1, dtype=np.int64)
    for source, target in graph.edges:
        if not isinstance(target, int) or target > INT64_MAX:
            raise OverflowError("node values must fit in signed 64-bit integers")
        successor[source] = target
    return successor


def _arithmetic_features(
    n: int, *, show_progress: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stopping, total stopping, excursion, and parity lengths.

    A compact cache is retained for values in ``1..N``. Values above ``N``
    are kept only in a sparse dictionary because Collatz trajectories can
    temporarily leave the finite graph domain.
    """
    total = np.full(n + 1, -1, dtype=np.int64)
    excursion = np.zeros(n + 1, dtype=np.int64)
    total[1] = 0
    excursion[1] = 1
    stopping = np.zeros(n + 1, dtype=np.int64)
    parity_length = np.zeros(n + 1, dtype=np.int64)
    converged = np.zeros(n + 1, dtype=bool)
    trajectory_status = np.full(n + 1, "unknown", dtype="U24")
    converged[1] = True
    trajectory_status[1] = "reached_1"
    external_cache: dict[int, tuple[int, int]] = {}

    for start in tqdm(
        range(1, n + 1), desc="Computing arithmetic features", disable=not show_progress
    ):
        value = start
        steps = 0
        maximum = start
        first_lower = 0 if start == 1 else -1
        path: list[tuple[int, int]] = []
        reached_known_value = False
        hit_limit = False
        while True:
            if value < start and first_lower == -1:
                first_lower = steps
            if first_lower >= 0 and value <= n and total[value] >= 0:
                known_total = int(total[value])
                known_excursion = int(excursion[value])
                reached_known_value = True
                break
            if first_lower >= 0 and value > n and value in external_cache:
                known_total, known_excursion = external_cache[value]
                reached_known_value = True
                break
            if steps >= TRAJECTORY_STEP_LIMIT:
                hit_limit = True
                break
            path.append((value, steps))
            if value > maximum:
                maximum = value
            value = value // 2 if value % 2 == 0 else 3 * value + 1
            if value > INT64_MAX:
                raise OverflowError("Collatz trajectory exceeded signed 64-bit range")
            steps += 1

        if reached_known_value:
            for path_value, _path_steps in reversed(path):
                known_total += 1
                known_excursion = max(known_excursion, path_value)
                if path_value <= n:
                    total[path_value] = known_total
                    excursion[path_value] = known_excursion
                else:
                    external_cache[path_value] = (known_total, known_excursion)
        if first_lower == -1:
            first_lower = steps
        stopping[start] = first_lower
        if reached_known_value:
            converged[start] = True
            trajectory_status[start] = "reached_1"
            parity_length[start] = int(total[start])
        elif hit_limit:
            trajectory_status[start] = "trajectory_limit_reached"
            parity_length[start] = -1
        if maximum > excursion[start]:
            excursion[start] = maximum

    return (
        stopping[1:],
        total[1:],
        excursion[1:],
        parity_length[1:],
        converged[1:],
        trajectory_status[1:],
    )


def _distance_to_root(successor: np.ndarray, root: int) -> np.ndarray:
    """Find shortest forward distances to root in a functional graph."""
    n = len(successor) - 1
    distances = np.full(n + 1, -1, dtype=np.int64)
    distances[root] = 0
    for start in range(1, n + 1):
        if distances[start] >= 0:
            continue
        path: list[int] = []
        positions: set[int] = set()
        current = start
        while current != -1 and distances[current] < 0 and current not in positions:
            positions.add(current)
            path.append(current)
            current = int(successor[current])
        base = int(distances[current]) if current != -1 and distances[current] >= 0 else -1
        for node in reversed(path):
            if base < 0:
                distances[node] = -1
            else:
                base += 1
                distances[node] = base
    return distances[1:]


def _ancestor_counts(successor: np.ndarray) -> np.ndarray:
    """Count all nodes whose forward orbit reaches each node.

    A reverse Kahn peel removes non-cycle nodes. Subtree sizes are accumulated
    in reverse peel order. Each cycle receives the total size of its connected
    functional component, which is the correct ancestor set for a directed
    cycle.
    """
    n = len(successor) - 1
    indegree = np.zeros(n + 1, dtype=np.int64)
    for node in range(1, n + 1):
        target = int(successor[node])
        if target != -1:
            indegree[target] += 1
    queue = np.empty(n, dtype=np.int64)
    head = tail = 0
    for node in range(1, n + 1):
        if indegree[node] == 0:
            queue[tail] = node
            tail += 1
    peeled = np.empty(n, dtype=np.int64)
    peeled_count = 0
    while head < tail:
        node = int(queue[head])
        head += 1
        peeled[peeled_count] = node
        peeled_count += 1
        target = int(successor[node])
        if target != -1:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue[tail] = target
                tail += 1

    subtree = np.ones(n + 1, dtype=np.int64)
    for index in range(peeled_count - 1, -1, -1):
        node = int(peeled[index])
        target = int(successor[node])
        if target != -1:
            subtree[target] += subtree[node]

    counts = subtree.copy()
    cycle_nodes = np.flatnonzero(indegree[1:]) + 1
    visited: set[int] = set()
    for cycle_start in cycle_nodes:
        cycle_start = int(cycle_start)
        if cycle_start in visited:
            continue
        component: list[int] = []
        current = cycle_start
        while current not in visited:
            visited.add(current)
            component.append(current)
            current = int(successor[current])
        component_count = int(sum(subtree[node] for node in component))
        for node in component:
            counts[node] = component_count
    return counts[1:]


def compute_node_features(
    graph: nx.DiGraph,
    *,
    root: int = 1,
    show_progress: bool = True,
) -> FeatureComputationResult:
    """Return arithmetic and graph features for every node in ``graph``.

    ``stopping_time`` is the first positive ``k`` for which ``T^k(n) < n``.
    ``total_stopping_time`` and ``parity_vector_length`` count transitions to
    1 and are ``-1`` when the trajectory limit is reached.
    ``maximum_excursion`` is the largest value visited, including ``n``.
    ``converged_to_1`` and ``trajectory_status`` distinguish evaluated
    convergence from the computational step limit.
    ``distance_from_root`` is the shortest in-domain forward distance to
    ``root``; ``-1`` means the finite graph contains no such path.
    ``ancestor_count`` includes the node itself and every node whose forward
    orbit reaches it. ``branching_factor`` is the number of inverse children,
    equal to in-degree under the forward edge convention.

    The arithmetic and structural passes are linear in ``N + E`` apart from
    the length of the evaluated Collatz trajectories. Temporary path storage
    is proportional to the longest trajectory, while the principal arrays
    use fixed-width NumPy integers. The returned DataFrame necessarily costs
    additional memory proportional to ``N``.
    """
    n = _validate_graph(graph, root)
    start = perf_counter()
    successor = _successor_array(graph, n)
    (
        stopping,
        total,
        excursion,
        parity_length,
        converged,
        trajectory_status,
    ) = _arithmetic_features(n, show_progress=show_progress)
    distances = _distance_to_root(successor, root)
    ancestors = _ancestor_counts(successor)
    nodes = np.arange(1, n + 1, dtype=np.int64)
    in_degree = np.fromiter((graph.in_degree(node) for node in nodes), dtype=np.int64, count=n)
    out_degree = np.fromiter((graph.out_degree(node) for node in nodes), dtype=np.int64, count=n)
    data = pd.DataFrame(
        {
            "node": nodes,
            "stopping_time": stopping,
            "total_stopping_time": total,
            "maximum_excursion": excursion,
            "binary_length": np.floor(np.log2(nodes)).astype(np.int64) + 1,
            "parity_vector_length": parity_length,
            "converged_to_1": converged,
            "trajectory_status": trajectory_status,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "distance_from_root": distances,
            "ancestor_count": ancestors,
            "branching_factor": in_degree,
        }
    )
    runtime_seconds = perf_counter() - start
    LOGGER.info("Computed %d node feature rows in %.6f s", n, runtime_seconds)
    return FeatureComputationResult(data=data, runtime_seconds=runtime_seconds)


def save_node_features(result: FeatureComputationResult, path: str | Path) -> Path:
    """Write a feature table to CSV and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(destination, index=False)
    LOGGER.info("Saved node features to %s", destination.resolve())
    return destination
