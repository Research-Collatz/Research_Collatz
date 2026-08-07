"""Tabular summaries for inverse Collatz graph experiments."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def graph_node_table(graph: nx.DiGraph) -> pd.DataFrame:
    """Return one deterministic row per graph node.

    The table includes structural features only; it does not infer properties
    for nodes excluded by the graph's finite expansion budget.
    """
    records = [
        {
            "node": int(node),
            "is_root": bool(data.get("root", False)),
            "inverse_depth": int(data.get("inverse_depth", -1)),
            "in_degree": int(graph.in_degree(node)),
            "out_degree": int(graph.out_degree(node)),
        }
        for node, data in graph.nodes(data=True)
    ]
    table = pd.DataFrame.from_records(records)
    if table.empty:
        return pd.DataFrame(columns=["node", "is_root", "inverse_depth", "in_degree", "out_degree"])
    return table.sort_values(["inverse_depth", "node"], ignore_index=True)


def graph_summary(graph: nx.DiGraph) -> pd.Series:
    """Compute reproducible scalar summary statistics for a graph."""
    nodes = np.asarray(list(graph.nodes), dtype=np.int64)
    depths = [data.get("inverse_depth") for _, data in graph.nodes(data=True)]
    return pd.Series(
        {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "roots": sum(bool(data.get("root", False)) for _, data in graph.nodes(data=True)),
            "maximum_node": int(nodes.max()) if nodes.size else np.nan,
            "maximum_inverse_depth": int(max(depths)) if depths else np.nan,
            "mean_in_degree": float(np.mean([degree for _, degree in graph.in_degree()])) if nodes.size else np.nan,
        },
        dtype=object,
    )
