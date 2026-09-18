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
from .node2vec import (
    Node2VecConfig,
    Node2VecResult,
    iter_node2vec_walks,
    save_node2vec_result,
    train_node2vec,
)
from .reproducibility import (
    get_environment_metadata,
    get_git_commit_sha,
    write_experiment_manifest,
)
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
    "iter_node2vec_walks",
    "train_node2vec",
    "save_node2vec_result",
    "get_environment_metadata",
    "get_git_commit_sha",
    "write_experiment_manifest",
]
