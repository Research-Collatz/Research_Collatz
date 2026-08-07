"""Reusable algorithms for bounded inverse Collatz graph experiments."""

from .core import InverseGraphConfig, build_inverse_graph, collatz_successor, inverse_predecessors
from .bounded import BoundedGraphBuildResult, build_and_save_inverse_graph, build_inverse_graph_up_to, save_graph_artifacts
from .features import FeatureComputationResult, compute_node_features, save_node_features
from .statistics import GraphStatisticsResult, compute_graph_statistics, save_publication_figures, save_statistics_tables
from .node2vec import Node2VecConfig, Node2VecResult, save_node2vec_result, train_node2vec

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
    "GraphStatisticsResult",
    "compute_graph_statistics",
    "save_statistics_tables",
    "save_publication_figures",
    "Node2VecConfig",
    "Node2VecResult",
    "train_node2vec",
    "save_node2vec_result",
]
