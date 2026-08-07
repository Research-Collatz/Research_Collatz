"""Matplotlib visualizations for bounded inverse Collatz graphs."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import networkx as nx

LOGGER = logging.getLogger(__name__)


def plot_inverse_graph(
    graph: nx.DiGraph,
    *,
    ax: Axes | None = None,
    layout_seed: int = 20260722,
    node_size: float = 24.0,
) -> Axes:
    """Draw a stable spring-layout view and return the Matplotlib axes.

    Layout randomness is controlled for reproducibility, but coordinates are
    a visualization choice and should not be treated as graph measurements.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    positions = nx.spring_layout(graph, seed=layout_seed)
    node_colors = ["#d95f02" if graph.nodes[node].get("root") else "#1b9e77" for node in graph]
    nx.draw_networkx_nodes(graph, positions, node_size=node_size, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(graph, positions, arrows=True, arrowsize=7, width=0.5, alpha=0.45, ax=ax)
    ax.set_axis_off()
    ax.set_title("Bounded inverse Collatz graph")
    return ax
