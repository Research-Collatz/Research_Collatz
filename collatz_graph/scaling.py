"""Cross-scale convergence analysis for inverse-Collatz Node2Vec embeddings.

The bounded domains are nested: ``V(10_000) subset V(50_000) subset V(100_000)``.
Every cross-scale comparison in this module is therefore restricted to a *common
core* of nodes present in all compared runs, with rows reindexed to a shared node
order.  Without that restriction the embedding matrices index different node sets
and are not comparable.

Node2Vec identifies an embedding only up to an orthogonal transform, so three of
the four pairwise similarity measures here (``knn_overlap``, ``linear_cka``,
``distance_spearman``) are rotation invariant.  ``procrustes_residual`` reports
the residual that remains after the optimal orthogonal alignment is applied.

A cross-scale difference is only interpretable against a noise floor: how much the
embedding moves between random seeds at *fixed* ``N``.  Both comparison families
are computed on the same common core and the same candidate pool size, so the
numbers can be divided.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

LOGGER = logging.getLogger(__name__)

SCALES: tuple[int, ...] = (10_000, 50_000, 100_000)
SEEDS: tuple[int, ...] = (20260722, 20260723, 20260724)
ANALYSIS_SEED = 20260722

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}

SCALE_COLOURS = {10_000: "#2166ac", 50_000: "#f4a261", 100_000: "#762a83"}


@dataclass(frozen=True, slots=True)
class RunReference:
    """Locator for one saved Node2Vec run in the scaling study."""

    max_node: int
    seed: int
    directory: Path

    @property
    def label(self) -> str:
        """Return a short human-readable identifier such as ``N=50000/s2``."""
        return f"N={self.max_node}/seed{self.seed}"


def run_directory(output_root: str | Path, max_node: int, seed: int) -> Path:
    """Return the canonical directory for one scaling-study Node2Vec run."""
    return Path(output_root) / "embeddings" / f"node2vec_N{max_node}_seed{seed}"


def enumerate_runs(
    output_root: str | Path,
    scales: tuple[int, ...] = SCALES,
    seeds: tuple[int, ...] = SEEDS,
) -> list[RunReference]:
    """List every (scale, seed) run reference for the study, in a stable order."""
    return [
        RunReference(max_node=n, seed=s, directory=run_directory(output_root, n, s))
        for n in scales
        for s in seeds
    ]


def load_embedding_matrix(directory: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(node_ids, embeddings)`` without recomputing the feature table.

    ``embedding_plots.load_embedding_features`` rebuilds the graph and the full
    Collatz feature table on every call.  The geometry measures below need only
    the matrix and its node order, so this cheaper loader is used for them.
    """
    path = Path(directory)
    embeddings = np.load(path / "node2vec_embeddings.npy")
    node_ids = pd.read_csv(path / "node2vec_nodes.csv")["node"].to_numpy(dtype=np.int64)
    if embeddings.ndim != 2 or len(node_ids) != len(embeddings):
        raise ValueError(f"embedding matrix and node mapping disagree in {path}")
    return node_ids, embeddings


def common_core(node_id_arrays: list[np.ndarray]) -> np.ndarray:
    """Return the sorted node IDs present in every supplied run."""
    if not node_id_arrays:
        raise ValueError("at least one node ID array is required")
    shared = node_id_arrays[0]
    for other in node_id_arrays[1:]:
        shared = np.intersect1d(shared, other, assume_unique=True)
    if shared.size == 0:
        raise ValueError("runs share no nodes; cross-scale comparison is undefined")
    return shared


def assert_nested_domains(node_id_arrays: dict[int, np.ndarray]) -> None:
    """Verify that smaller domains are subsets of larger ones.

    Every cross-scale comparison assumes ``V(N1) subset V(N2)`` for ``N1 < N2``.
    That holds for ``1..N`` domains by construction, but it is checked rather than
    assumed because a silent violation would make the common core smaller than
    expected without any visible error.
    """
    scales = sorted(node_id_arrays)
    for smaller, larger in zip(scales, scales[1:]):
        missing = np.setdiff1d(node_id_arrays[smaller], node_id_arrays[larger], assume_unique=True)
        if missing.size:
            raise ValueError(
                f"domain N={smaller} is not nested in N={larger}: {missing.size} nodes missing"
            )


def restrict_to_nodes(
    node_ids: np.ndarray, embeddings: np.ndarray, wanted: np.ndarray
) -> np.ndarray:
    """Return embedding rows for ``wanted``, in ``wanted`` order.

    ``node_ids`` is sorted ascending because ``generate_node2vec_walks`` sorts the
    graph nodes, so positions are recovered with a binary search.
    """
    positions = np.searchsorted(node_ids, wanted)
    if np.any(positions >= len(node_ids)) or not np.array_equal(node_ids[positions], wanted):
        raise ValueError("requested nodes are not all present in the run")
    return embeddings[positions]


def _row_normalise(matrix: np.ndarray) -> np.ndarray:
    """Return the matrix with every row scaled to unit Euclidean norm."""
    as_float = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(as_float, axis=1, keepdims=True)
    return as_float / np.maximum(norms, 1e-12)


def top_k_neighbours(embeddings: np.ndarray, k: int, block_size: int = 2048) -> np.ndarray:
    """Return the ``k`` nearest cosine neighbours of every row, excluding itself.

    Similarities are computed in row blocks so peak memory stays at
    ``block_size x n`` rather than ``n x n``.
    """
    normalised = _row_normalise(embeddings)
    n_rows = len(normalised)
    if k >= n_rows:
        raise ValueError("k must be smaller than the number of rows")
    neighbours = np.empty((n_rows, k), dtype=np.int32)
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        similarities = normalised[start:stop] @ normalised.T
        similarities[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        partitioned = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
        neighbours[start:stop] = partitioned
    return neighbours


def knn_overlap(embeddings_a: np.ndarray, embeddings_b: np.ndarray, k: int) -> float:
    """Return the mean Jaccard overlap of top-``k`` cosine neighbourhoods.

    This is the primary stability measure: it depends only on the neighbourhood
    structure of the latent space, so it is invariant to rotation, reflection, and
    global rescaling.  Both sets have size ``k``, so the Jaccard index reduces to
    ``i / (2k - i)`` for an intersection of size ``i``.
    """
    if len(embeddings_a) != len(embeddings_b):
        raise ValueError("embeddings must describe the same rows")
    neighbours_a = np.sort(top_k_neighbours(embeddings_a, k), axis=1)
    neighbours_b = np.sort(top_k_neighbours(embeddings_b, k), axis=1)
    overlaps = np.empty(len(neighbours_a), dtype=np.float64)
    for row in range(len(neighbours_a)):
        intersection = np.intersect1d(neighbours_a[row], neighbours_b[row], assume_unique=True).size
        overlaps[row] = intersection / (2 * k - intersection)
    return float(overlaps.mean())


def procrustes_residual(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    """Return the relative residual after optimal orthogonal alignment.

    Both matrices are column-centred and scaled to unit Frobenius norm, so the
    result is a scale-free number in ``[0, sqrt(2)]``: ``0`` means the two latent
    spaces agree exactly up to rotation, and ``sqrt(2)`` means they are maximally
    misaligned.
    """
    left = np.asarray(embeddings_a, dtype=np.float64)
    right = np.asarray(embeddings_b, dtype=np.float64)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    left /= max(float(np.linalg.norm(left)), 1e-12)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    singular_values = np.linalg.svd(left.T @ right, compute_uv=False)
    # ||L R_opt - R||_F^2 = ||L||^2 + ||R||^2 - 2 sum(sigma) = 2 - 2 sum(sigma).
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * float(singular_values.sum()))))


def linear_cka(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    """Return linear centred kernel alignment in ``[0, 1]``.

    Computed through the ``d x d`` cross-covariance rather than the ``n x n`` Gram
    matrices, so the cost is ``O(n d^2)`` instead of ``O(n^2 d)``.  Invariant to
    rotation and isotropic scaling; ``1`` means the two representations are
    identical up to such a transform.
    """
    left = np.asarray(embeddings_a, dtype=np.float64)
    right = np.asarray(embeddings_b, dtype=np.float64)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(left.T @ right, ord="fro") ** 2)
    left_norm = float(np.linalg.norm(left.T @ left, ord="fro"))
    right_norm = float(np.linalg.norm(right.T @ right, ord="fro"))
    denominator = left_norm * right_norm
    return float(cross / denominator) if denominator > 0 else float("nan")


def distance_spearman(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    sample_size: int = 2_000,
    seed: int = ANALYSIS_SEED,
) -> float:
    """Return the Spearman correlation of pairwise cosine distances (Mantel style).

    All pairwise distances of a fixed random node sample are compared, which tests
    whether the two embeddings induce the same *global* distance ordering rather
    than only the same local neighbourhoods.
    """
    if len(embeddings_a) != len(embeddings_b):
        raise ValueError("embeddings must describe the same rows")
    rng = np.random.default_rng(seed)
    size = min(sample_size, len(embeddings_a))
    sample = rng.choice(len(embeddings_a), size=size, replace=False)
    distances_a = pdist(_row_normalise(embeddings_a[sample]), metric="cosine")
    distances_b = pdist(_row_normalise(embeddings_b[sample]), metric="cosine")
    correlation = stats.spearmanr(distances_a, distances_b).statistic
    return float(correlation)


def geometry_descriptors(
    embeddings: np.ndarray, sample_size: int = 5_000, seed: int = ANALYSIS_SEED
) -> dict[str, float]:
    """Describe the shape of one embedding cloud, independent of any pairing.

    ``effective_rank`` is the exponential of the spectral entropy of the
    eigenvalue distribution and ``participation_ratio`` is
    ``(sum lambda)^2 / sum lambda^2``; both estimate how many directions the cloud
    genuinely occupies.  If the latent geometry stabilises, these should settle to
    a constant as ``N`` grows rather than tracking the node count.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = (centred.T @ centred) / max(1, len(centred) - 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)[::-1]
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("embedding has zero variance")
    proportions = eigenvalues / total
    positive = proportions[proportions > 0]
    entropy = float(-(positive * np.log(positive)).sum())

    rng = np.random.default_rng(seed)
    size = min(sample_size, len(matrix))
    sample = _row_normalise(matrix[rng.choice(len(matrix), size=size, replace=False)])
    cosines = pdist(sample, metric="cosine")
    return {
        "n_rows": int(len(matrix)),
        "effective_rank": float(np.exp(entropy)),
        "participation_ratio": float(total**2 / float((eigenvalues**2).sum())),
        "top1_variance_ratio": float(proportions[0]),
        "top10_variance_ratio": float(proportions[:10].sum()),
        "top32_variance_ratio": float(proportions[:32].sum()),
        "mean_pairwise_cosine": float(1.0 - cosines.mean()),
        "std_pairwise_cosine": float(cosines.std()),
    }


def double_sweep_diameter_bound(graph: nx.DiGraph, sweeps: int = 4, seed: int = ANALYSIS_SEED) -> int:
    """Return a lower bound on the undirected diameter of the largest weak component.

    ``compute_graph_statistics`` only reports an exact diameter when the largest
    weak component is small (default limit 5,000 nodes), because exact
    ``nx.diameter`` is ``O(V E)``.  Repeated double sweeps give a cheap and always
    valid lower bound, so depth structure stays comparable across all scales.
    """
    components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    undirected = graph.subgraph(components[0]).to_undirected()
    rng = np.random.default_rng(seed)
    nodes = np.asarray(sorted(undirected.nodes), dtype=np.int64)
    best = 0
    for _ in range(sweeps):
        source = int(rng.choice(nodes))
        for _ in range(2):
            lengths = nx.single_source_shortest_path_length(undirected, source)
            source, distance = max(lengths.items(), key=lambda item: item[1])
            best = max(best, int(distance))
    return best


def _pair_metrics(
    left: np.ndarray, right: np.ndarray, neighbour_sizes: tuple[int, ...] = (10, 50)
) -> dict[str, float]:
    """Compute every pairwise similarity measure for one aligned matrix pair."""
    metrics: dict[str, float] = {}
    for k in neighbour_sizes:
        metrics[f"knn_overlap_k{k}"] = knn_overlap(left, right, k)
    metrics["procrustes_residual"] = procrustes_residual(left, right)
    metrics["linear_cka"] = linear_cka(left, right)
    metrics["distance_spearman"] = distance_spearman(left, right)
    return metrics


def compute_embedding_stability(
    runs: list[RunReference], neighbour_sizes: tuple[int, ...] = (10, 50)
) -> pd.DataFrame:
    """Compare every within-scale and seed-matched cross-scale pair of runs.

    All comparisons are restricted to the common core shared by *all* runs, so the
    candidate pool for the neighbourhood measures has one fixed size.  Comparing a
    top-10 neighbourhood drawn from 10,000 candidates against one drawn from
    100,000 would otherwise conflate stability with pool difficulty.
    """
    loaded = {run.label: load_embedding_matrix(run.directory) for run in runs}
    assert_nested_domains({run.max_node: loaded[run.label][0] for run in runs})
    core = common_core([node_ids for node_ids, _ in loaded.values()])
    LOGGER.info("Common core spans %d nodes across %d runs", len(core), len(runs))
    aligned = {
        label: restrict_to_nodes(node_ids, embeddings, core)
        for label, (node_ids, embeddings) in loaded.items()
    }

    records: list[dict[str, object]] = []
    by_scale: dict[int, list[RunReference]] = {}
    for run in runs:
        by_scale.setdefault(run.max_node, []).append(run)

    for scale, scale_runs in by_scale.items():
        for first, second in combinations(scale_runs, 2):
            LOGGER.info("Within-scale pair %s vs %s", first.label, second.label)
            records.append(
                {
                    "comparison_type": "within_scale",
                    "max_node_a": first.max_node,
                    "seed_a": first.seed,
                    "max_node_b": second.max_node,
                    "seed_b": second.seed,
                    "scale_label": f"N={scale}",
                    "n_common_core": int(len(core)),
                    **_pair_metrics(aligned[first.label], aligned[second.label], neighbour_sizes),
                }
            )

    scales = sorted(by_scale)
    for smaller, larger in combinations(scales, 2):
        for first, second in zip(by_scale[smaller], by_scale[larger]):
            if first.seed != second.seed:
                raise ValueError("cross-scale comparison expects seed-matched runs")
            LOGGER.info("Cross-scale pair %s vs %s", first.label, second.label)
            records.append(
                {
                    "comparison_type": "cross_scale",
                    "max_node_a": smaller,
                    "seed_a": first.seed,
                    "max_node_b": larger,
                    "seed_b": second.seed,
                    "scale_label": f"{smaller}->{larger}",
                    "n_common_core": int(len(core)),
                    **_pair_metrics(aligned[first.label], aligned[second.label], neighbour_sizes),
                }
            )
    return pd.DataFrame.from_records(records)


def compute_pool_sensitivity(
    runs: list[RunReference],
    pool_sizes: tuple[int, ...] = (2_000, 5_000, 10_000, 20_000),
    k: int = 10,
    seed: int = ANALYSIS_SEED,
) -> pd.DataFrame:
    """Measure how the neighbourhood overlap depends on the candidate pool size.

    ``compute_embedding_stability`` fixes the pool at the common core.  This table
    justifies that choice by showing the same within-scale seed pair scored at
    several pool sizes, so a reader can see the pool effect is not mistaken for a
    scale effect.
    """
    records: list[dict[str, object]] = []
    by_scale: dict[int, list[RunReference]] = {}
    for run in runs:
        by_scale.setdefault(run.max_node, []).append(run)

    rng = np.random.default_rng(seed)
    for scale, scale_runs in sorted(by_scale.items()):
        if len(scale_runs) < 2:
            continue
        first, second = scale_runs[0], scale_runs[1]
        nodes_a, matrix_a = load_embedding_matrix(first.directory)
        nodes_b, matrix_b = load_embedding_matrix(second.directory)
        shared = np.intersect1d(nodes_a, nodes_b, assume_unique=True)
        for pool in pool_sizes:
            if pool > len(shared):
                continue
            sample = np.sort(rng.choice(len(shared), size=pool, replace=False))
            subset = shared[sample]
            records.append(
                {
                    "max_node": scale,
                    "seed_a": first.seed,
                    "seed_b": second.seed,
                    "pool_size": int(pool),
                    "k": k,
                    "knn_overlap": knn_overlap(
                        restrict_to_nodes(nodes_a, matrix_a, subset),
                        restrict_to_nodes(nodes_b, matrix_b, subset),
                        k,
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def compute_geometry_table(runs: list[RunReference]) -> pd.DataFrame:
    """Tabulate latent-geometry descriptors for every run, on the common core.

    Descriptors are computed twice: on the run's native node set (how the whole
    cloud looks) and on the common core (how the *same* 10,000 nodes are arranged
    at each scale).  Only the second is comparable across ``N``.
    """
    loaded = {run.label: load_embedding_matrix(run.directory) for run in runs}
    core = common_core([node_ids for node_ids, _ in loaded.values()])
    records: list[dict[str, object]] = []
    for run in runs:
        node_ids, embeddings = loaded[run.label]
        records.append(
            {
                "scope": "native",
                "max_node": run.max_node,
                "seed": run.seed,
                **geometry_descriptors(embeddings),
            }
        )
        records.append(
            {
                "scope": "common_core",
                "max_node": run.max_node,
                "seed": run.seed,
                **geometry_descriptors(restrict_to_nodes(node_ids, embeddings, core)),
            }
        )
    return pd.DataFrame.from_records(records)


def compute_cluster_stability(
    runs: list[RunReference],
    label_paths: dict[str, Path],
    reference_k: int = 8,
) -> pd.DataFrame:
    """Compare KMeans partitions across seeds and scales on the common core.

    KMeans is the only algorithm in the pipeline fitted on every row at every
    scale, so it is the only one whose labels are directly comparable.  A single
    ``reference_k`` is used for the headline comparison because agreement between
    partitions with different cluster counts is not interpretable.
    """
    loaded = {run.label: load_embedding_matrix(run.directory) for run in runs}
    core = common_core([node_ids for node_ids, _ in loaded.values()])
    label_sets: dict[str, np.ndarray] = {}
    for run in runs:
        with np.load(label_paths[run.label]) as archive:
            labels = archive[f"k{reference_k}"]
        node_ids, _ = loaded[run.label]
        positions = np.searchsorted(node_ids, core)
        label_sets[run.label] = labels[positions]

    records: list[dict[str, object]] = []
    by_scale: dict[int, list[RunReference]] = {}
    for run in runs:
        by_scale.setdefault(run.max_node, []).append(run)

    for scale, scale_runs in sorted(by_scale.items()):
        for first, second in combinations(scale_runs, 2):
            records.append(
                {
                    "comparison_type": "within_scale",
                    "max_node_a": first.max_node,
                    "seed_a": first.seed,
                    "max_node_b": second.max_node,
                    "seed_b": second.seed,
                    "scale_label": f"N={scale}",
                    "reference_k": reference_k,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(label_sets[first.label], label_sets[second.label])
                    ),
                    "adjusted_mutual_info": float(
                        adjusted_mutual_info_score(label_sets[first.label], label_sets[second.label])
                    ),
                }
            )

    for smaller, larger in combinations(sorted(by_scale), 2):
        for first, second in zip(by_scale[smaller], by_scale[larger]):
            records.append(
                {
                    "comparison_type": "cross_scale",
                    "max_node_a": smaller,
                    "seed_a": first.seed,
                    "max_node_b": larger,
                    "seed_b": second.seed,
                    "scale_label": f"{smaller}->{larger}",
                    "reference_k": reference_k,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(label_sets[first.label], label_sets[second.label])
                    ),
                    "adjusted_mutual_info": float(
                        adjusted_mutual_info_score(label_sets[first.label], label_sets[second.label])
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def summarise_convergence(
    stability: pd.DataFrame, cluster_stability: pd.DataFrame
) -> pd.DataFrame:
    """Express each cross-scale similarity relative to the same-``N`` noise floor.

    A ratio near ``1`` means enlarging the domain perturbs the representation no
    more than changing the random seed does, which is the operational definition
    of a stabilised latent geometry used in the report.  For
    ``procrustes_residual`` (lower is better) the ratio is inverted so that values
    near ``1`` always mean "indistinguishable from reseeding".
    """
    similarity_metrics = ["knn_overlap_k10", "knn_overlap_k50", "linear_cka", "distance_spearman"]
    records: list[dict[str, object]] = []

    within = stability[stability["comparison_type"] == "within_scale"]
    cross = stability[stability["comparison_type"] == "cross_scale"]

    for metric in similarity_metrics + ["procrustes_residual"]:
        floor = float(within[metric].mean())
        for scale_label, group in cross.groupby("scale_label", sort=True):
            value = float(group[metric].mean())
            ratio = (floor / value) if metric == "procrustes_residual" else (value / floor)
            records.append(
                {
                    "quantity": metric,
                    "source": "embedding",
                    "scale_label": scale_label,
                    "within_scale_mean": floor,
                    "within_scale_std": float(within[metric].std(ddof=0)),
                    "cross_scale_mean": value,
                    "cross_scale_std": float(group[metric].std(ddof=0)),
                    "ratio_to_noise_floor": float(ratio),
                }
            )

    within_cluster = cluster_stability[cluster_stability["comparison_type"] == "within_scale"]
    cross_cluster = cluster_stability[cluster_stability["comparison_type"] == "cross_scale"]
    for metric in ["adjusted_rand_index", "adjusted_mutual_info"]:
        floor = float(within_cluster[metric].mean())
        for scale_label, group in cross_cluster.groupby("scale_label", sort=True):
            value = float(group[metric].mean())
            records.append(
                {
                    "quantity": metric,
                    "source": "clustering",
                    "scale_label": scale_label,
                    "within_scale_mean": floor,
                    "within_scale_std": float(within_cluster[metric].std(ddof=0)),
                    "cross_scale_mean": value,
                    "cross_scale_std": float(group[metric].std(ddof=0)),
                    "ratio_to_noise_floor": float(value / floor) if floor else float("nan"),
                }
            )
    return pd.DataFrame.from_records(records)


def compute_graph_scaling_table(
    statistics_directories: dict[int, Path], diameter_bounds: dict[int, int]
) -> pd.DataFrame:
    """Collect saved per-scale summary tables into one comparison table.

    Derived ratios (edges per node, largest-component share, components per node)
    are what reveal whether the *shape* of the graph is scale invariant, as
    opposed to raw counts which necessarily grow with ``N``.
    """
    records: list[dict[str, object]] = []
    for max_node in sorted(statistics_directories):
        directory = statistics_directories[max_node]
        summary = pd.read_csv(directory / f"graph_{max_node}_summary.csv").iloc[0]
        spectral = pd.read_csv(directory / f"graph_{max_node}_spectral.csv").set_index("statistic")
        nodes = float(summary["nodes"])
        records.append(
            {
                "max_node": max_node,
                "nodes": int(summary["nodes"]),
                "edges": int(summary["edges"]),
                "edges_per_node": float(summary["edges"]) / nodes,
                "weak_components": int(summary["weak_components"]),
                "weak_components_per_node": float(summary["weak_components"]) / nodes,
                "strong_components": int(summary["strong_components"]),
                "largest_weak_component": int(summary["largest_weak_component"]),
                "largest_weak_component_share": float(summary["largest_weak_component"]) / nodes,
                "tree_depth_max": int(summary["tree_depth_max"]),
                "tree_depth_mean_reachable": float(summary["tree_depth_mean_reachable"]),
                "average_branching_factor": float(summary["average_branching_factor"]),
                "weak_component_diameter_exact": float(summary["weak_component_diameter"]),
                "weak_component_diameter_lower_bound": int(diameter_bounds[max_node]),
                "adjacency_spectral_radius_estimate": float(
                    spectral.loc["adjacency_spectral_radius_estimate", "value"]
                ),
                "maximum_out_degree": float(spectral.loc["maximum_out_degree", "value"]),
                "diameter_method": str(summary["diameter_method"]),
            }
        )
    return pd.DataFrame.from_records(records)


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Write one figure as both a raster PNG and a vector PDF."""
    figure.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_graph_scaling(table: pd.DataFrame, output_path: Path) -> None:
    """Plot raw and normalised graph statistics against the domain bound."""
    panels = [
        ("edges", "Edges", True),
        ("edges_per_node", "Edges per node", False),
        ("weak_components_per_node", "Weak components per node", False),
        ("largest_weak_component_share", "Largest weak component / N", False),
        ("tree_depth_max", "Maximum inverse-tree depth", False),
        ("tree_depth_mean_reachable", "Mean reachable depth", False),
    ]
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), constrained_layout=True)
        for axis, (column, label, logarithmic) in zip(axes.flatten(), panels):
            axis.plot(table["max_node"], table[column], "o-", color="#2166ac", linewidth=1.4)
            axis.set_xscale("log")
            if logarithmic:
                axis.set_yscale("log")
            axis.set_xlabel("Domain bound N")
            axis.set_ylabel(label)
            axis.set_title(label, fontsize=11)
            axis.grid(alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle("Inverse Collatz graph statistics across domain scales", fontsize=14)
        _save_figure(figure, output_path)


def plot_embedding_stability(stability: pd.DataFrame, output_path: Path) -> None:
    """Plot cross-scale similarity against the within-scale seed noise floor."""
    metrics = [
        ("knn_overlap_k10", "kNN overlap (k=10)"),
        ("knn_overlap_k50", "kNN overlap (k=50)"),
        ("linear_cka", "Linear CKA"),
        ("distance_spearman", "Distance Spearman"),
        ("procrustes_residual", "Procrustes residual (lower = better)"),
    ]
    within = stability[stability["comparison_type"] == "within_scale"]
    cross = stability[stability["comparison_type"] == "cross_scale"]
    cross_labels = sorted(cross["scale_label"].unique())

    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 5, figsize=(19.0, 4.2), constrained_layout=True)
        for axis, (metric, label) in zip(axes, metrics):
            floor_mean = float(within[metric].mean())
            floor_std = float(within[metric].std(ddof=0))
            axis.axhspan(
                floor_mean - floor_std,
                floor_mean + floor_std,
                color="#bdbdbd",
                alpha=0.45,
                label="Within-scale seed noise (±1 sd)",
            )
            axis.axhline(floor_mean, color="#525252", linestyle="--", linewidth=1.1)
            positions = np.arange(len(cross_labels))
            means = [float(cross[cross["scale_label"] == name][metric].mean()) for name in cross_labels]
            errors = [float(cross[cross["scale_label"] == name][metric].std(ddof=0)) for name in cross_labels]
            axis.errorbar(
                positions, means, yerr=errors, fmt="o", color="#762a83", capsize=4, markersize=7,
                label="Cross-scale (seed matched)",
            )
            axis.set_xticks(positions)
            axis.set_xticklabels(cross_labels, rotation=20, ha="right")
            axis.set_title(label, fontsize=10)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25, axis="y")
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8, loc="best")
        figure.suptitle(
            "Embedding stability: cross-scale agreement versus the same-N reseeding noise floor",
            fontsize=13,
        )
        _save_figure(figure, output_path)


def plot_geometry_spectrum(
    runs: list[RunReference], geometry: pd.DataFrame, output_path: Path
) -> None:
    """Plot eigenvalue spectra and the effective rank of the latent cloud."""
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)

        seen: set[int] = set()
        for run in runs:
            _, embeddings = load_embedding_matrix(run.directory)
            centred = embeddings - embeddings.mean(axis=0, keepdims=True)
            covariance = (centred.T @ centred) / max(1, len(centred) - 1)
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 1e-18, None)[::-1]
            proportions = eigenvalues / eigenvalues.sum()
            axes[0].plot(
                np.arange(1, len(proportions) + 1),
                proportions,
                color=SCALE_COLOURS[run.max_node],
                alpha=0.75,
                linewidth=1.2,
                label=f"N={run.max_node}" if run.max_node not in seen else None,
            )
            seen.add(run.max_node)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Component index")
        axes[0].set_ylabel("Explained variance ratio")
        axes[0].set_title("Eigenvalue spectrum (native node set)", fontsize=11)
        axes[0].legend(frameon=False)

        core = geometry[geometry["scope"] == "common_core"]
        for axis, column, label in (
            (axes[1], "effective_rank", "Effective rank (exp spectral entropy)"),
            (axes[2], "participation_ratio", "Participation ratio"),
        ):
            grouped = core.groupby("max_node")[column]
            scales = sorted(grouped.groups)
            means = [float(grouped.get_group(s).mean()) for s in scales]
            errors = [float(grouped.get_group(s).std(ddof=0)) for s in scales]
            axis.errorbar(scales, means, yerr=errors, fmt="o-", color="#762a83", capsize=4)
            axis.set_xscale("log")
            axis.set_xlabel("Domain bound N")
            axis.set_ylabel(label)
            axis.set_title(f"{label}\n(common core, mean ± sd over seeds)", fontsize=10)
            axis.grid(alpha=0.25)
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle("Latent geometry dimensionality across domain scales", fontsize=13)
        _save_figure(figure, output_path)


def plot_cluster_stability(
    cluster_stability: pd.DataFrame, kmeans_metrics: pd.DataFrame, output_path: Path
) -> None:
    """Plot the ARI comparison matrix and the silhouette curve for each scale."""
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)

        for axis, metric, label in (
            (axes[0], "adjusted_rand_index", "Adjusted Rand Index"),
            (axes[1], "adjusted_mutual_info", "Adjusted Mutual Information"),
        ):
            within = cluster_stability[cluster_stability["comparison_type"] == "within_scale"]
            cross = cluster_stability[cluster_stability["comparison_type"] == "cross_scale"]
            floor_mean = float(within[metric].mean())
            floor_std = float(within[metric].std(ddof=0))
            axis.axhspan(
                floor_mean - floor_std, floor_mean + floor_std,
                color="#bdbdbd", alpha=0.45, label="Within-scale seed noise (±1 sd)",
            )
            axis.axhline(floor_mean, color="#525252", linestyle="--", linewidth=1.1)
            labels = sorted(cross["scale_label"].unique())
            positions = np.arange(len(labels))
            means = [float(cross[cross["scale_label"] == name][metric].mean()) for name in labels]
            errors = [float(cross[cross["scale_label"] == name][metric].std(ddof=0)) for name in labels]
            axis.errorbar(positions, means, yerr=errors, fmt="o", color="#762a83", capsize=4, markersize=7,
                          label="Cross-scale (seed matched)")
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=20, ha="right")
            axis.set_ylabel(label)
            axis.set_title(f"KMeans partition agreement\n{label}", fontsize=10)
            axis.grid(alpha=0.25, axis="y")
        axes[0].legend(frameon=False, fontsize=8)

        axis = axes[2]
        for max_node, group in kmeans_metrics.groupby("max_node"):
            curve = group.groupby("n_clusters")["silhouette_score"]
            ks = sorted(curve.groups)
            means = [float(curve.get_group(k).mean()) for k in ks]
            axis.plot(ks, means, "o-", color=SCALE_COLOURS[max_node], label=f"N={max_node}", linewidth=1.4)
        axis.set_xlabel("Number of clusters k")
        axis.set_ylabel("Silhouette score")
        axis.set_title("KMeans silhouette curve by scale\n(mean over seeds)", fontsize=10)
        axis.legend(frameon=False)
        axis.grid(alpha=0.25)
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle("Cluster stability across seeds and domain scales", fontsize=13)
        _save_figure(figure, output_path)


def plot_prediction_scaling(prediction: pd.DataFrame, output_path: Path) -> None:
    """Plot predictive R² against the domain bound for both evaluation scopes."""
    targets = sorted(prediction["target_property"].unique())
    scopes = ["full_n", "matched_core"]
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(2, len(targets), figsize=(4.0 * len(targets), 7.4), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for row, scope in enumerate(scopes):
            subset = prediction[prediction["evaluation_scope"] == scope]
            for column, target in enumerate(targets):
                axis = axes[row, column]
                target_rows = subset[subset["target_property"] == target]
                for model, group in target_rows.groupby("model_name"):
                    ordered = group.sort_values("max_node")
                    axis.plot(ordered["max_node"], ordered["r2_oof"], "o-", label=model, linewidth=1.4)
                axis.set_xscale("log")
                axis.set_xlabel("Domain bound N")
                axis.set_ylabel("Out-of-fold R²")
                scope_label = "full node set" if scope == "full_n" else "matched common core"
                axis.set_title(f"{target}\n({scope_label})", fontsize=10)
                axis.grid(alpha=0.25)
                axis.spines[["top", "right"]].set_visible(False)
                if row == 0 and column == 0:
                    axis.legend(frameon=False, fontsize=8)
        figure.suptitle(
            "Predictive performance across scales: raw node set versus distribution-matched core",
            fontsize=13,
        )
        _save_figure(figure, output_path)


def plot_convergence_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot every cross-scale quantity as a ratio to its same-``N`` noise floor."""
    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
        quantities = list(dict.fromkeys(summary["quantity"]))
        labels = sorted(summary["scale_label"].unique())
        width = 0.8 / len(labels)
        positions = np.arange(len(quantities))
        for index, label in enumerate(labels):
            values = [
                float(
                    summary[(summary["quantity"] == q) & (summary["scale_label"] == label)][
                        "ratio_to_noise_floor"
                    ].iloc[0]
                )
                for q in quantities
            ]
            axis.bar(positions + (index - len(labels) / 2 + 0.5) * width, values, width, label=label, alpha=0.9)
        axis.axhline(1.0, color="#252525", linestyle="--", linewidth=1.2)
        axis.text(
            len(quantities) - 0.4, 1.02, "parity with reseeding", fontsize=8, color="#252525", ha="right"
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(quantities, rotation=25, ha="right")
        axis.set_ylabel("Cross-scale value ÷ within-scale noise floor")
        axis.set_title(
            "Convergence summary: 1.0 means enlarging N perturbs the result\n"
            "no more than changing the random seed",
            fontsize=12,
        )
        axis.legend(frameon=False, title="Scale transition")
        axis.grid(alpha=0.25, axis="y")
        axis.spines[["top", "right"]].set_visible(False)
        _save_figure(figure, output_path)


def plot_umap_grid(umap_directories: dict[int, Path], output_path: Path) -> None:
    """Assemble saved UMAP coordinates into one comparable per-scale panel grid."""
    columns = [("level_set", "inverse-tree level set"), ("stopping_time", "stopping time")]
    scales = sorted(umap_directories)
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            len(scales), len(columns), figsize=(6.0 * len(columns), 4.8 * len(scales)),
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        for row, max_node in enumerate(scales):
            table = pd.read_csv(umap_directories[max_node] / "umap_features.csv")
            for column, (field, label) in enumerate(columns):
                axis = axes[row, column]
                values = table[field].to_numpy(dtype=float)
                valid = values >= 0
                if np.any(~valid):
                    axis.scatter(
                        table.loc[~valid, "umap_1"], table.loc[~valid, "umap_2"],
                        s=3, c="#d9d9d9", alpha=0.5, linewidths=0, rasterized=True,
                    )
                points = axis.scatter(
                    table.loc[valid, "umap_1"], table.loc[valid, "umap_2"],
                    s=3, c=values[valid], cmap="viridis", alpha=0.8, linewidths=0, rasterized=True,
                )
                colorbar = figure.colorbar(points, ax=axis, pad=0.02)
                colorbar.set_label(label)
                axis.set_title(f"N={max_node} — {label}", fontsize=11)
                axis.set_xlabel("UMAP 1")
                axis.set_ylabel("UMAP 2")
                axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle("Node2Vec UMAP layouts across domain scales", fontsize=14)
        _save_figure(figure, output_path)


def compute_scale_summary_markdown(tables: dict[str, pd.DataFrame]) -> str:
    """Generate a markdown summary of all comparison tables."""
    lines = ["# Cross-Scale Convergence Summary", ""]
    for name, table in tables.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(table.to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def save_comparison_artifacts(tables: dict[str, pd.DataFrame], output_directory: str | Path) -> dict[str, Path]:
    """Write every comparison table as CSV and return the written paths."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
        LOGGER.info("Saved %s (%d rows) to %s", name, len(table), path)
    (output / "comparison_manifest.json").write_text(
        json.dumps({name: str(path) for name, path in paths.items()}, indent=2), encoding="utf-8"
    )
    return paths
