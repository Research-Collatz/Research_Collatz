"""Exact-arithmetic construction of bounded inverse Collatz graphs."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import networkx as nx
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def collatz_successor(value: int) -> int:
    """Return the forward Collatz successor of a positive integer.

    Raises:
        ValueError: If ``value`` is not a positive integer.
        TypeError: If ``value`` is not an integer.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("value must be an integer")
    if value <= 0:
        raise ValueError("value must be positive")
    return value // 2 if value % 2 == 0 else 3 * value + 1


def inverse_predecessors(value: int) -> tuple[int, ...]:
    """Return all positive integer predecessors of ``value`` under ``T``.

    The result is sorted and contains no duplicates. The odd inverse branch
    exists exactly when ``value = 1 (mod 3)`` and its candidate is odd.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("value must be an integer")
    if value <= 0:
        raise ValueError("value must be positive")

    predecessors = [2 * value]
    if value % 3 == 1:
        candidate = (value - 1) // 3
        if candidate > 0 and candidate % 2 == 1:
            predecessors.append(candidate)
    return tuple(sorted(predecessors))


@dataclass(frozen=True, slots=True)
class InverseGraphConfig:
    """Parameters controlling a bounded inverse graph expansion."""

    roots: tuple[int, ...] = (1,)
    max_nodes: int = 10_000
    show_progress: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.roots, tuple):
            raise TypeError("roots must be a tuple of positive integers")
        if not self.roots:
            raise ValueError("roots must contain at least one value")
        if any(
            not isinstance(root, int) or isinstance(root, bool) or root <= 0
            for root in self.roots
        ):
            raise ValueError("roots must contain positive integers")
        if not isinstance(self.max_nodes, int) or isinstance(self.max_nodes, bool):
            raise TypeError("max_nodes must be an integer")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if self.max_nodes < len(set(self.roots)):
            raise ValueError("max_nodes must include every distinct root")


def build_inverse_graph(config: InverseGraphConfig) -> nx.DiGraph:
    """Build a node-bounded inverse Collatz graph.

    Nodes are discovered breadth-first from ``config.roots``. Once the node
    budget is reached, undiscovered predecessors are skipped rather than
    terminating the entire expansion. This keeps the node budget hard while
    allowing all already-admitted work to finish.
    """
    graph = nx.DiGraph()
    queue: deque[int] = deque()
    for root in dict.fromkeys(config.roots):
        graph.add_node(root, root=True, inverse_depth=0)
        queue.append(root)

    progress = tqdm(desc="Expanding inverse graph", disable=not config.show_progress)
    try:
        while queue:
            target = queue.popleft()
            progress.update(1)
            target_depth = graph.nodes[target]["inverse_depth"]
            for predecessor in inverse_predecessors(target):
                if predecessor not in graph:
                    if len(graph) >= config.max_nodes:
                        LOGGER.debug(
                            "Node budget %d reached; skipping predecessor %d",
                            config.max_nodes,
                            predecessor,
                        )
                        continue
                    graph.add_node(
                        predecessor,
                        root=False,
                        inverse_depth=target_depth + 1,
                    )
                    queue.append(predecessor)

                graph.add_edge(predecessor, target)
    finally:
        progress.close()

    LOGGER.info(
        "Built graph with %d nodes and %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph
