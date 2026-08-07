"""Orchestrate the multi-scale Node2Vec convergence study.

Runs the complete embedding pipeline at several domain bounds ``N`` and several
random seeds, then compares the results across scales.  Every stage is resumable:
an already-written artifact is detected and skipped, so an interrupted study
never restarts from the beginning.

Stages
------
``embed``       Node2Vec for each (scale, seed) pair.
``statistics``  Descriptive graph statistics and figures for each scale.
``umap``        UMAP layouts for the reference seed of each scale.
``cluster``     KMeans/DBSCAN/HDBSCAN sweeps; the full pipeline runs for the
                reference seed and a KMeans-only sweep for the remaining seeds,
                which is all the partition-agreement analysis needs.
``supervised``  Property prediction on the full node set and on the
                distribution-matched common core.
``analysis``    Cross-scale tables and figures.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .bounded import build_inverse_graph_up_to
from .clustering import run_kmeans_sweep
from .clustering_cli import execute_clustering_pipeline
from .embedding_plots import create_umap_figures, load_embedding_features
from .node2vec import Node2VecConfig, save_node2vec_result, train_node2vec
from .scaling import (
    SCALES,
    SEEDS,
    RunReference,
    common_core,
    compute_cluster_stability,
    compute_embedding_stability,
    compute_geometry_table,
    compute_graph_scaling_table,
    compute_pool_sensitivity,
    compute_scale_summary_markdown,
    double_sweep_diameter_bound,
    enumerate_runs,
    load_embedding_matrix,
    plot_cluster_stability,
    plot_convergence_summary,
    plot_embedding_stability,
    plot_geometry_spectrum,
    plot_graph_scaling,
    plot_prediction_scaling,
    plot_umap_grid,
    restrict_to_nodes,
    save_comparison_artifacts,
    summarise_convergence,
)
from .statistics import compute_graph_statistics, save_publication_figures, save_statistics_tables
from .supervised import run_all_supervised_experiments
from .supervised_cli import execute_supervised_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

#: Rows used for the two quadratic clustering stages.  Held fixed across scales
#: so silhouette scores and density fits stay comparable; at N=10,000 it exceeds
#: the node count and the exhaustive code path runs unchanged.
CLUSTER_SUBSAMPLE = 10_000

#: Cluster count used for every cross-run partition-agreement comparison.  A
#: single k is required because agreement between partitions with different
#: cluster counts is not interpretable.
REFERENCE_K = 8

TARGET_PROPERTIES: list[tuple[str, bool]] = [
    ("stopping_time", False),
    ("maximum_excursion", True),
    ("binary_length", False),
    ("level_set", False),
]

STAGES = ("embed", "statistics", "umap", "cluster", "supervised", "analysis")


def _statistics_directory(root: Path, max_node: int) -> Path:
    return root / f"statistics_N{max_node}"


def _clustering_directory(root: Path, max_node: int, seed: int) -> Path:
    return root / f"clustering_N{max_node}_seed{seed}"


def _supervised_directory(root: Path, max_node: int) -> Path:
    return root / f"supervised_N{max_node}"


def stage_embed(root: Path, runs: list[RunReference], force: bool) -> None:
    """Train and persist one Node2Vec embedding per (scale, seed) pair."""
    for run in runs:
        target = run.directory / "node2vec_embeddings.npy"
        if target.exists() and not force:
            LOGGER.info("[embed] skip %s (already present)", run.label)
            continue
        LOGGER.info("[embed] training %s", run.label)
        start = perf_counter()
        graph = build_inverse_graph_up_to(run.max_node, show_progress=False).graph
        config = Node2VecConfig(seed=run.seed, backend="torch")
        result = train_node2vec(graph, config, show_progress=False)
        save_node2vec_result(
            result,
            run.directory,
            extra_metadata={
                "graph": {
                    "domain_maximum": run.max_node,
                    "edge_orientation": "forward Collatz u -> T(u)",
                    "walk_orientation": "inverse",
                },
                "study": "multi-scale convergence",
            },
        )
        LOGGER.info("[embed] %s finished in %.1f s", run.label, perf_counter() - start)


def stage_statistics(root: Path, scales: tuple[int, ...], force: bool) -> None:
    """Compute descriptive graph statistics, figures, and a diameter bound per scale."""
    bounds_path = root / "diameter_bounds.json"
    bounds: dict[str, int] = {}
    if bounds_path.exists():
        bounds = json.loads(bounds_path.read_text(encoding="utf-8"))

    for max_node in scales:
        directory = _statistics_directory(root, max_node)
        summary_path = directory / f"graph_{max_node}_summary.csv"
        if summary_path.exists() and str(max_node) in bounds and not force:
            LOGGER.info("[statistics] skip N=%d (already present)", max_node)
            continue
        LOGGER.info("[statistics] computing N=%d", max_node)
        graph = build_inverse_graph_up_to(max_node, show_progress=False).graph
        statistics = compute_graph_statistics(graph, show_progress=False)
        save_statistics_tables(statistics, directory, stem=f"graph_{max_node}")
        save_publication_figures(graph, statistics, directory)
        bounds[str(max_node)] = double_sweep_diameter_bound(graph)
        bounds_path.write_text(json.dumps(bounds, indent=2, sort_keys=True), encoding="utf-8")
        LOGGER.info(
            "[statistics] N=%d done; diameter lower bound %d", max_node, bounds[str(max_node)]
        )


def stage_umap(root: Path, runs: list[RunReference], reference_seed: int, force: bool) -> None:
    """Fit one UMAP layout per scale, using the reference seed's embedding."""
    for run in runs:
        if run.seed != reference_seed:
            continue
        target = run.directory / "umap" / "umap_features.csv"
        if target.exists() and not force:
            LOGGER.info("[umap] skip %s (already present)", run.label)
            continue
        LOGGER.info("[umap] fitting %s", run.label)
        start = perf_counter()
        create_umap_figures(run.directory)
        LOGGER.info("[umap] %s finished in %.1f s", run.label, perf_counter() - start)


def _kmeans_only_sweep(run: RunReference, output_directory: Path, seed: int) -> None:
    """Sweep KMeans alone and persist labels, for seeds that skip the full pipeline."""
    output_directory.mkdir(parents=True, exist_ok=True)
    _, embeddings = load_embedding_matrix(run.directory)
    metrics, labels = run_kmeans_sweep(
        embeddings, range(2, 16), random_state=seed, silhouette_sample_size=CLUSTER_SUBSAMPLE
    )
    metrics.to_csv(output_directory / "clustering_metrics.csv", index=False)
    np.savez_compressed(
        output_directory / "kmeans_labels.npz",
        **{f"k{k}": value.astype(np.int16) for k, value in labels.items()},
    )


def stage_cluster(root: Path, runs: list[RunReference], reference_seed: int, force: bool) -> None:
    """Cluster every run; the full sweep for the reference seed, KMeans for the rest."""
    for run in runs:
        directory = _clustering_directory(root, run.max_node, run.seed)
        marker = directory / ("clustering_manifest.json" if run.seed == reference_seed else "kmeans_labels.npz")
        if marker.exists() and not force:
            LOGGER.info("[cluster] skip %s (already present)", run.label)
            continue
        start = perf_counter()
        if run.seed == reference_seed:
            LOGGER.info("[cluster] full pipeline for %s", run.label)
            execute_clustering_pipeline(
                run.directory,
                directory,
                seed=42,
                silhouette_sample_size=CLUSTER_SUBSAMPLE,
                density_subsample=CLUSTER_SUBSAMPLE,
                save_kmeans_labels=True,
            )
        else:
            LOGGER.info("[cluster] KMeans-only sweep for %s", run.label)
            _kmeans_only_sweep(run, directory, seed=42)
        LOGGER.info("[cluster] %s finished in %.1f s", run.label, perf_counter() - start)


def _matched_core_predictions(
    runs: list[RunReference], reference_seed: int, random_state: int = 42
) -> pd.DataFrame:
    """Evaluate property prediction on the node set shared by every scale.

    Raw cross-scale R² is confounded: the target distributions themselves widen
    with ``N`` (``binary_length`` and ``level_set`` both gain range), and the
    sample size changes tenfold.  Restricting every scale to the same 10,000
    nodes holds both fixed, so a difference in score reflects the embedding.
    """
    reference_runs = [run for run in runs if run.seed == reference_seed]
    core = common_core([load_embedding_matrix(run.directory)[0] for run in reference_runs])
    LOGGER.info("[supervised] matched core spans %d nodes", len(core))

    frames: list[pd.DataFrame] = []
    for run in reference_runs:
        LOGGER.info("[supervised] matched-core evaluation for %s", run.label)
        embeddings, features, _ = load_embedding_features(run.directory)
        node_ids = features["node"].to_numpy(dtype=np.int64)
        positions = np.searchsorted(node_ids, core)
        if not np.array_equal(node_ids[positions], core):
            raise ValueError("common core nodes missing from run feature table")
        metrics, _, _ = run_all_supervised_experiments(
            embeddings=embeddings[positions],
            features_df=features.iloc[positions].reset_index(drop=True),
            target_properties=TARGET_PROPERTIES,
            n_splits=3,
            random_state=random_state,
        )
        metrics.insert(0, "max_node", run.max_node)
        metrics.insert(1, "evaluation_scope", "matched_core")
        metrics.insert(2, "n_rows", len(core))
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def stage_supervised(root: Path, runs: list[RunReference], reference_seed: int, force: bool) -> None:
    """Run property prediction on the full node set and on the matched common core."""
    for run in runs:
        if run.seed != reference_seed:
            continue
        directory = _supervised_directory(root, run.max_node)
        if (directory / "supervised_metrics.csv").exists() and not force:
            LOGGER.info("[supervised] skip full-N %s (already present)", run.label)
            continue
        LOGGER.info("[supervised] full-N pipeline for %s", run.label)
        start = perf_counter()
        execute_supervised_pipeline(run.directory, directory, seed=42)
        LOGGER.info("[supervised] %s finished in %.1f s", run.label, perf_counter() - start)

    matched_path = root / "supervised_matched_core.csv"
    if matched_path.exists() and not force:
        LOGGER.info("[supervised] skip matched-core evaluation (already present)")
        return
    _matched_core_predictions(runs, reference_seed).to_csv(matched_path, index=False)


def _collect_prediction_table(root: Path, scales: tuple[int, ...]) -> pd.DataFrame:
    """Merge the full-N and matched-core prediction metrics into one table."""
    frames: list[pd.DataFrame] = []
    for max_node in scales:
        path = _supervised_directory(root, max_node) / "supervised_metrics.csv"
        table = pd.read_csv(path)
        table.insert(0, "max_node", max_node)
        table.insert(1, "evaluation_scope", "full_n")
        table.insert(2, "n_rows", max_node)
        frames.append(table)
    frames.append(pd.read_csv(root / "supervised_matched_core.csv"))
    return pd.concat(frames, ignore_index=True)


def _collect_kmeans_metrics(root: Path, runs: list[RunReference]) -> pd.DataFrame:
    """Gather the KMeans sweep metrics of every run into one table."""
    frames: list[pd.DataFrame] = []
    for run in runs:
        directory = _clustering_directory(root, run.max_node, run.seed)
        table = pd.read_csv(directory / "clustering_metrics.csv")
        table = table[table["algorithm"] == "KMeans"].copy()
        table.insert(0, "max_node", run.max_node)
        table.insert(1, "seed", run.seed)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def stage_analysis(root: Path, runs: list[RunReference], reference_seed: int) -> None:
    """Build every cross-scale comparison table and figure."""
    comparison = root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    scales = tuple(sorted({run.max_node for run in runs}))

    LOGGER.info("[analysis] embedding stability")
    stability = compute_embedding_stability(runs)

    LOGGER.info("[analysis] neighbourhood pool sensitivity")
    pool_sensitivity = compute_pool_sensitivity(runs)

    LOGGER.info("[analysis] latent geometry descriptors")
    geometry = compute_geometry_table(runs)

    LOGGER.info("[analysis] cluster stability")
    label_paths = {
        run.label: _clustering_directory(root, run.max_node, run.seed) / "kmeans_labels.npz"
        for run in runs
    }
    cluster_stability = compute_cluster_stability(runs, label_paths, reference_k=REFERENCE_K)
    kmeans_metrics = _collect_kmeans_metrics(root, runs)

    LOGGER.info("[analysis] graph statistics scaling")
    bounds = json.loads((root / "diameter_bounds.json").read_text(encoding="utf-8"))
    graph_scaling = compute_graph_scaling_table(
        {n: _statistics_directory(root, n) for n in scales},
        {n: int(bounds[str(n)]) for n in scales},
    )

    LOGGER.info("[analysis] prediction scaling")
    prediction = _collect_prediction_table(root, scales)

    LOGGER.info("[analysis] convergence summary")
    summary = summarise_convergence(stability, cluster_stability)

    tables = {
        "graph_statistics_scaling": graph_scaling,
        "embedding_stability": stability,
        "knn_pool_sensitivity": pool_sensitivity,
        "embedding_geometry": geometry,
        "cluster_stability": cluster_stability,
        "kmeans_sweep_metrics": kmeans_metrics,
        "prediction_scaling": prediction,
        "convergence_summary": summary,
    }
    save_comparison_artifacts(tables, comparison)

    LOGGER.info("[analysis] figures")
    plot_graph_scaling(graph_scaling, comparison / "fig_graph_scaling")
    plot_embedding_stability(stability, comparison / "fig_embedding_stability")
    plot_geometry_spectrum(runs, geometry, comparison / "fig_geometry_spectrum")
    plot_cluster_stability(cluster_stability, kmeans_metrics, comparison / "fig_cluster_stability")
    plot_prediction_scaling(prediction, comparison / "fig_prediction_scaling")
    plot_convergence_summary(summary, comparison / "fig_convergence_summary")
    plot_umap_grid(
        {run.max_node: run.directory / "umap" for run in runs if run.seed == reference_seed},
        comparison / "fig_umap_grid",
    )

    (comparison / "tables.md").write_text(
        compute_scale_summary_markdown(tables), encoding="utf-8"
    )
    LOGGER.info("[analysis] wrote comparison artifacts to %s", comparison)


def build_parser() -> argparse.ArgumentParser:
    """Construct the documented command-line interface."""
    parser = argparse.ArgumentParser(description="Run the multi-scale Node2Vec convergence study.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/scaling"), help="Directory holding every artifact of the study.")
    parser.add_argument("--scales", type=int, nargs="+", default=list(SCALES), help="Domain bounds N to evaluate.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="Random seeds trained at every scale.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES), help="Subset of pipeline stages to execute.")
    parser.add_argument("--force", action="store_true", help="Recompute stages even when their artifacts already exist.")
    return parser


def main() -> None:
    """Execute the requested stages of the multi-scale study."""
    args = build_parser().parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    scales = tuple(sorted(args.scales))
    seeds = tuple(args.seeds)
    reference_seed = seeds[0]
    runs = enumerate_runs(root, scales, seeds)

    if "embed" in args.stages:
        stage_embed(root, runs, args.force)
    if "statistics" in args.stages:
        stage_statistics(root, scales, args.force)
    if "umap" in args.stages:
        stage_umap(root, runs, reference_seed, args.force)
    if "cluster" in args.stages:
        stage_cluster(root, runs, reference_seed, args.force)
    if "supervised" in args.stages:
        stage_supervised(root, runs, reference_seed, args.force)
    if "analysis" in args.stages:
        stage_analysis(root, runs, reference_seed)


if __name__ == "__main__":
    main()
