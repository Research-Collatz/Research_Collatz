"""Command-line interface for Node2Vec embedding clustering and Collatz property evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .clustering import (
    compute_cluster_property_summary,
    compute_property_association_tests,
    plot_metrics_comparison,
    plot_property_distributions,
    plot_property_heatmap,
    plot_umap_clusters,
    run_dbscan_sweep,
    run_hdbscan_sweep,
    run_kmeans_sweep,
)
from .embedding_plots import fit_umap, load_embedding_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def execute_clustering_pipeline(
    run_directory: str | Path,
    output_directory: str | Path,
    seed: int = 42,
    *,
    silhouette_sample_size: int | None = None,
    density_subsample: int | None = None,
    save_kmeans_labels: bool = False,
) -> dict[str, str]:
    """Run full clustering, metric evaluation, statistical test, and plot generation pipeline.

    ``silhouette_sample_size`` and ``density_subsample`` bound the two quadratic
    stages so the pipeline stays tractable at 50,000 and 100,000 nodes; leaving
    both ``None`` reproduces the original exhaustive behaviour.  Setting
    ``save_kmeans_labels`` also writes every swept KMeans labelling to
    ``kmeans_labels.npz`` for later cross-run partition-agreement analysis.
    """
    run_path = Path(run_directory)
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading Node2Vec embeddings and Collatz feature table from %s", run_path)
    embeddings, features_df, metadata = load_embedding_features(run_path)

    # Calculate or load UMAP 2D coordinates for cluster visualizations
    umap_cache = run_path / "umap" / "umap_coordinates.npy"
    if umap_cache.exists():
        LOGGER.info("Loading pre-computed UMAP coordinates from %s", umap_cache)
        umap_coords = np.load(umap_cache)
    else:
        LOGGER.info("Fitting 2D UMAP projection on %d embeddings...", len(embeddings))
        umap_coords = fit_umap(embeddings, seed=seed)

    LOGGER.info("1/3 Running KMeans hyperparameter grid search (k=2..15)...")
    kmeans_df, kmeans_labels_dict = run_kmeans_sweep(
        embeddings, range(2, 16), random_state=seed, silhouette_sample_size=silhouette_sample_size
    )
    if save_kmeans_labels:
        np.savez_compressed(
            out_dir / "kmeans_labels.npz",
            **{f"k{k}": labels.astype(np.int16) for k, labels in kmeans_labels_dict.items()},
        )

    LOGGER.info("2/3 Running DBSCAN hyperparameter grid search...")
    dbscan_df, dbscan_labels_dict = run_dbscan_sweep(
        embeddings, fit_subsample=density_subsample, subsample_random_state=seed
    )

    LOGGER.info("3/3 Running HDBSCAN hyperparameter grid search...")
    hdbscan_df, hdbscan_labels_dict = run_hdbscan_sweep(
        embeddings, fit_subsample=density_subsample, subsample_random_state=seed
    )

    # Save comprehensive evaluation metrics table
    all_metrics_df = pd.concat([kmeans_df, dbscan_df, hdbscan_df], ignore_index=True)
    metrics_csv_path = out_dir / "clustering_metrics.csv"
    all_metrics_df.to_csv(metrics_csv_path, index=False)
    LOGGER.info("Saved clustering quality evaluation metrics to %s", metrics_csv_path)

    # Select representative/best models for each algorithm
    # Best KMeans by Silhouette Score
    best_k_idx = kmeans_df["silhouette_score"].idxmax()
    best_k = int(kmeans_df.loc[best_k_idx, "n_clusters"])
    kmeans_best_labels = kmeans_labels_dict[best_k]

    # Best DBSCAN (non-trivial noise < 50%, highest silhouette score among valid)
    valid_db = dbscan_df[(dbscan_df["n_clusters"] > 1) & (dbscan_df["noise_ratio"] < 0.5)]
    if not valid_db.empty:
        best_db_row = valid_db.loc[valid_db["silhouette_score"].idxmax()]
        best_db_key = str(best_db_row["param_value"])
        dbscan_best_labels = dbscan_labels_dict[best_db_key]
    else:
        best_db_key = list(dbscan_labels_dict.keys())[0]
        dbscan_best_labels = dbscan_labels_dict[best_db_key]

    # Best HDBSCAN (highest silhouette score with noise < 50%)
    valid_hdb = hdbscan_df[(hdbscan_df["n_clusters"] > 1) & (hdbscan_df["noise_ratio"] < 0.5)]
    if not valid_hdb.empty:
        best_hdb_row = valid_hdb.loc[valid_hdb["silhouette_score"].idxmax()]
        best_mcs = int(best_hdb_row["param_value"])
        hdbscan_best_labels = hdbscan_labels_dict[best_mcs]
    else:
        best_mcs = 30
        hdbscan_best_labels = hdbscan_labels_dict[best_mcs]

    # Save assignments table
    assignments_df = features_df[["node"]].copy()
    assignments_df[f"kmeans_k{best_k}"] = kmeans_best_labels
    assignments_df[f"dbscan_{best_db_key}"] = dbscan_best_labels
    assignments_df[f"hdbscan_mcs{best_mcs}"] = hdbscan_best_labels
    assignments_path = out_dir / "cluster_assignments.csv"
    assignments_df.to_csv(assignments_path, index=False)

    # Compute Collatz property statistics per cluster
    properties = [
        "stopping_time",
        "total_stopping_time",
        "maximum_excursion",
        "binary_length",
        "level_set",
        "distance_from_root",
        "in_degree",
        "ancestor_count",
    ]

    summary_kmeans = compute_cluster_property_summary(features_df, kmeans_best_labels, properties)
    summary_dbscan = compute_cluster_property_summary(features_df, dbscan_best_labels, properties)
    summary_hdbscan = compute_cluster_property_summary(features_df, hdbscan_best_labels, properties)

    summary_kmeans.to_csv(out_dir / "cluster_summary_kmeans.csv", index=False)
    summary_dbscan.to_csv(out_dir / "cluster_summary_dbscan.csv", index=False)
    summary_hdbscan.to_csv(out_dir / "cluster_summary_hdbscan.csv", index=False)

    # Compute property association tests (ANOVA, Kruskal-Wallis, Mutual Information)
    assoc_kmeans = compute_property_association_tests(features_df, kmeans_best_labels, properties)
    assoc_kmeans["algorithm"] = f"KMeans (k={best_k})"

    assoc_hdbscan = compute_property_association_tests(features_df, hdbscan_best_labels, properties)
    assoc_hdbscan["algorithm"] = f"HDBSCAN (mcs={best_mcs})"

    assoc_df = pd.concat([assoc_kmeans, assoc_hdbscan], ignore_index=True)
    assoc_df.to_csv(out_dir / "property_associations.csv", index=False)

    LOGGER.info("Generating plots and figures...")

    # Plot metrics comparison curve
    plot_metrics_comparison(kmeans_df, dbscan_df, hdbscan_df, out_dir / "cluster_evaluation_metrics")

    # Plot UMAP cluster scatter plots
    plot_umap_clusters(umap_coords, kmeans_best_labels, f"KMeans Clusters (k={best_k})", out_dir / "umap_kmeans_clusters")
    plot_umap_clusters(umap_coords, dbscan_best_labels, f"DBSCAN Clusters ({best_db_key})", out_dir / "umap_dbscan_clusters")
    plot_umap_clusters(umap_coords, hdbscan_best_labels, f"HDBSCAN Clusters (mcs={best_mcs})", out_dir / "umap_hdbscan_clusters")

    # Plot property distributions
    prop_specs = [
        ("stopping_time", "Stopping Time", False),
        ("total_stopping_time", "Total Stopping Time", False),
        ("maximum_excursion", "Maximum Excursion", True),
        ("binary_length", "Binary Length", False),
        ("level_set", "Inverse-Tree Level Set", False),
        ("distance_from_root", "Distance from Root", False),
    ]

    plot_property_distributions(features_df, kmeans_best_labels, prop_specs, f"KMeans (k={best_k})", out_dir / "property_distributions_kmeans")
    plot_property_distributions(features_df, hdbscan_best_labels, prop_specs, f"HDBSCAN (mcs={best_mcs})", out_dir / "property_distributions_hdbscan")

    # Plot property heatmaps
    plot_property_heatmap(summary_kmeans, f"KMeans k={best_k}", out_dir / "property_heatmap_kmeans")
    plot_property_heatmap(summary_hdbscan, f"HDBSCAN mcs={best_mcs}", out_dir / "property_heatmap_hdbscan")

    manifest = {
        "silhouette_sample_size": silhouette_sample_size,
        "density_subsample": density_subsample,
        "density_label_conventions": {
            "-1": "noise, examined by the clusterer and left unassigned",
            "-2": "not evaluated: row lay outside the density fit subsample",
        },
        "metrics_table": str(out_dir / "clustering_metrics.csv"),
        "cluster_assignments": str(out_dir / "cluster_assignments.csv"),
        "property_summary_kmeans": str(out_dir / "cluster_summary_kmeans.csv"),
        "property_summary_hdbscan": str(out_dir / "cluster_summary_hdbscan.csv"),
        "property_associations": str(out_dir / "property_associations.csv"),
        "umap_kmeans_figure": str(out_dir / "umap_kmeans_clusters.png"),
        "umap_dbscan_figure": str(out_dir / "umap_dbscan_clusters.png"),
        "umap_hdbscan_figure": str(out_dir / "umap_hdbscan_clusters.png"),
        "property_distributions_kmeans": str(out_dir / "property_distributions_kmeans.png"),
        "property_heatmap_kmeans": str(out_dir / "property_heatmap_kmeans.png"),
        "cluster_evaluation_metrics": str(out_dir / "cluster_evaluation_metrics.png"),
    }

    manifest_path = out_dir / "clustering_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Pipeline completed successfully. Artifact manifest written to %s", manifest_path)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster Node2Vec embeddings and evaluate Collatz property alignments.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing node2vec artifacts.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save clustering artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--silhouette-sample-size", type=int, default=None, help="Rows sampled for the O(n^2) silhouette score; None evaluates every row.")
    parser.add_argument("--density-subsample", type=int, default=None, help="Rows used to fit DBSCAN/HDBSCAN; None fits every row.")
    parser.add_argument("--save-kmeans-labels", action="store_true", help="Also write every swept KMeans labelling to kmeans_labels.npz.")
    args = parser.parse_args()

    manifest = execute_clustering_pipeline(
        args.run_dir,
        args.output_dir,
        seed=args.seed,
        silhouette_sample_size=args.silhouette_sample_size,
        density_subsample=args.density_subsample,
        save_kmeans_labels=args.save_kmeans_labels,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
