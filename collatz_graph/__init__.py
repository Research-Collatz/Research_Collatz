"""Reusable algorithms for bounded inverse Collatz graph experiments."""

from .bounded import (
    BoundedGraphBuildResult,
    build_and_save_inverse_graph,
    build_inverse_graph_up_to,
    save_graph_artifacts,
)
from .core import InverseGraphConfig, build_inverse_graph, collatz_successor, inverse_predecessors
from .features import FeatureComputationResult, compute_node_features, save_node_features
from .geometric import (
    add_geometric_quantities,
    ancestor_density,
    branching_entropy,
    flow_energy,
    local_potential,
    neighborhood_overlap_curvature,
)
from .node2vec import Node2VecConfig, Node2VecResult, save_node2vec_result, train_node2vec
from .statistics import (
    GraphStatisticsResult,
    compute_graph_statistics,
    save_publication_figures,
    save_statistics_tables,
)

__all__ = [
    "InverseGraphConfig",
    "build_inverse_graph",
    "collatz_successor",
    "inverse_predecessors",
    "BoundedGraphBuildResult",
    "build_inverse_graph_up_to",
    "save_graph_artifacts",
    "build_and_save_inverse_graph",
    "FeatureComputationResult",
    "compute_node_features",
    "save_node_features",
    "add_geometric_quantities",
    "ancestor_density",
    "branching_entropy",
    "flow_energy",
    "local_potential",
    "neighborhood_overlap_curvature",
    "GraphStatisticsResult",
    "compute_graph_statistics",
    "save_statistics_tables",
    "save_publication_figures",
    "Node2VecConfig",
    "Node2VecResult",
    "train_node2vec",
    "save_node2vec_result",
]
