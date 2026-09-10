"""Clustering analysis for Collatz Node2Vec embeddings.

This module provides tools to cluster high-dimensional Node2Vec embeddings using
KMeans, DBSCAN, and HDBSCAN, compute intrinsic cluster validity indices, correlate
clusters with Collatz arithmetic/graph properties (stopping time, maximum excursion,
binary length, level set, etc.), and generate structured tables and visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from scipy import stats
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

LOGGER = logging.getLogger(__name__)

#: Label written for rows that were never passed to a density-based clusterer
#: because the fit was restricted to a subsample.  Kept distinct from DBSCAN's
#: own ``-1`` noise label so a subsampled run can never be misread as a run in
#: which those points were examined and rejected.
NOT_EVALUATED_LABEL = -2

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


def _subsample_indices(
    n_samples: int, subsample: int | None, random_state: int
) -> np.ndarray | None:
    """Return sorted row indices for a reproducible subsample, or None for all rows."""
    if subsample is None or subsample >= n_samples:
        return None
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_samples, size=subsample, replace=False))


def run_kmeans_sweep(
    embeddings: np.ndarray,
    k_range: range | list[int] = range(2, 16),
    random_state: int = 42,
    *,
    silhouette_sample_size: int | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Evaluate KMeans clustering over a range of cluster counts k.

    KMeans itself is always fitted on every row: its cost is ``O(n k d)`` and
    stays practical at 100,000 nodes.  The silhouette score, by contrast, is
    ``O(n^2 d)`` and becomes infeasible there, so ``silhouette_sample_size``
    evaluates it on a reproducible random subsample.  Calinski-Harabasz and
    Davies-Bouldin are ``O(n k d)`` and remain exact on the full matrix.
    """
    results: list[dict[str, object]] = []
    labels_dict: dict[int, np.ndarray] = {}
    n_samples = len(embeddings)
    n_metric_rows = min(silhouette_sample_size, n_samples) if silhouette_sample_size else n_samples

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(embeddings)
        labels_dict[k] = labels

        # Intrinsic metrics
        if silhouette_sample_size is not None and silhouette_sample_size < n_samples:
            sil = float(
                silhouette_score(
                    embeddings,
                    labels,
                    sample_size=silhouette_sample_size,
                    random_state=random_state,
                )
            )
        else:
            sil = float(silhouette_score(embeddings, labels))
        ch = float(calinski_harabasz_score(embeddings, labels))
        db = float(davies_bouldin_score(embeddings, labels))

        results.append(
            {
                "algorithm": "KMeans",
                "param_name": "n_clusters",
                "param_value": k,
                "n_clusters": k,
                "n_noise": 0,
                "noise_ratio": 0.0,
                "silhouette_score": sil,
                "calinski_harabasz_score": ch,
                "davies_bouldin_score": db,
                "n_fitted": n_samples,
                "n_metric_rows": int(n_metric_rows),
            }
        )

    df_results = pd.DataFrame.from_records(results)
    return df_results, labels_dict


def run_dbscan_sweep(
    embeddings: np.ndarray,
    eps_list: list[float] | None = None,
    min_samples_list: list[int] | None = None,
    metric: str = "cosine",
    *,
    fit_subsample: int | None = None,
    subsample_random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate DBSCAN clustering over a grid of eps and min_samples.

    Cosine DBSCAN falls back to brute-force pairwise distances, which is
    ``O(n^2)`` and impractical beyond a few tens of thousands of rows.  Setting
    ``fit_subsample`` fits the grid on a reproducible subsample; rows outside it
    receive :data:`NOT_EVALUATED_LABEL` so downstream tables never present a
    partial clustering as a complete one.  The returned ``noise_ratio`` is always
    relative to the rows actually clustered.
    """
    if eps_list is None:
        eps_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    if min_samples_list is None:
        min_samples_list = [5, 10, 15, 30]

    results: list[dict[str, object]] = []
    labels_dict: dict[str, np.ndarray] = {}
    selected = _subsample_indices(len(embeddings), fit_subsample, subsample_random_state)
    fitted_embeddings = embeddings if selected is None else embeddings[selected]

    for eps in eps_list:
        for min_samples in min_samples_list:
            key = f"eps_{eps}_ms_{min_samples}"
            model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            fitted_labels = model.fit_predict(fitted_embeddings)

            if selected is None:
                labels = fitted_labels
            else:
                labels = np.full(len(embeddings), NOT_EVALUATED_LABEL, dtype=np.int64)
                labels[selected] = fitted_labels
            labels_dict[key] = labels

            n_samples = len(fitted_labels)
            n_noise = int(np.sum(fitted_labels == -1))
            noise_ratio = float(n_noise / n_samples)
            unique_clusters = set(fitted_labels) - {-1}
            n_clusters = len(unique_clusters)

            sil, ch, db = np.nan, np.nan, np.nan
            if n_clusters > 1 and (n_samples - n_noise) > n_clusters:
                valid_mask = fitted_labels != -1
                valid_embeddings = fitted_embeddings[valid_mask]
                valid_labels = fitted_labels[valid_mask]
                if len(set(valid_labels)) > 1:
                    sil = float(silhouette_score(valid_embeddings, valid_labels, metric=metric))
                    ch = float(calinski_harabasz_score(valid_embeddings, valid_labels))
                    db = float(davies_bouldin_score(valid_embeddings, valid_labels))

            results.append(
                {
                    "algorithm": "DBSCAN",
                    "param_name": "eps_min_samples",
                    "param_value": key,
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_ratio": noise_ratio,
                    "silhouette_score": sil,
                    "calinski_harabasz_score": ch,
                    "davies_bouldin_score": db,
                    "n_fitted": n_samples,
                    "n_metric_rows": n_samples,
                }
            )

    df_results = pd.DataFrame.from_records(results)
    return df_results, labels_dict


def run_hdbscan_sweep(
    embeddings: np.ndarray,
    min_cluster_size_list: list[int] | None = None,
    metric: str = "cosine",
    *,
    fit_subsample: int | None = None,
    subsample_random_state: int = 42,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Evaluate HDBSCAN clustering over min_cluster_size values.

    As with :func:`run_dbscan_sweep`, a cosine metric forces brute-force pairwise
    distances, so ``fit_subsample`` bounds the cost at large ``N`` and rows
    outside the subsample carry :data:`NOT_EVALUATED_LABEL`.
    """
    if min_cluster_size_list is None:
        min_cluster_size_list = [5, 10, 15, 30, 50, 100, 200]

    results: list[dict[str, object]] = []
    labels_dict: dict[int, np.ndarray] = {}
    selected = _subsample_indices(len(embeddings), fit_subsample, subsample_random_state)
    fitted_embeddings = embeddings if selected is None else embeddings[selected]

    for mcs in min_cluster_size_list:
        model = HDBSCAN(min_cluster_size=mcs, metric=metric)
        fitted_labels = model.fit_predict(fitted_embeddings)

        if selected is None:
            labels = fitted_labels
        else:
            labels = np.full(len(embeddings), NOT_EVALUATED_LABEL, dtype=np.int64)
            labels[selected] = fitted_labels
        labels_dict[mcs] = labels

        n_samples = len(fitted_labels)
        n_noise = int(np.sum(fitted_labels == -1))
        noise_ratio = float(n_noise / n_samples)
        unique_clusters = set(fitted_labels) - {-1}
        n_clusters = len(unique_clusters)

        sil, ch, db = np.nan, np.nan, np.nan
        if n_clusters > 1 and (n_samples - n_noise) > n_clusters:
            valid_mask = fitted_labels != -1
            valid_embeddings = fitted_embeddings[valid_mask]
            valid_labels = fitted_labels[valid_mask]
            if len(set(valid_labels)) > 1:
                sil = float(silhouette_score(valid_embeddings, valid_labels, metric=metric))
                ch = float(calinski_harabasz_score(valid_embeddings, valid_labels))
                db = float(davies_bouldin_score(valid_embeddings, valid_labels))

        results.append(
            {
                "algorithm": "HDBSCAN",
                "param_name": "min_cluster_size",
                "param_value": mcs,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": noise_ratio,
                "silhouette_score": sil,
                "calinski_harabasz_score": ch,
                "davies_bouldin_score": db,
                "n_fitted": n_samples,
                "n_metric_rows": n_samples,
            }
        )

    df_results = pd.DataFrame.from_records(results)
    return df_results, labels_dict


def compute_cluster_property_summary(
    features_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    properties: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-cluster mean, std, median, IQR, min, and max for Collatz properties."""
    if properties is None:
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

    df = features_df.copy()
    df["cluster"] = cluster_labels
    # Rows a subsampled density fit never examined carry no cluster information.
    df = df[df["cluster"] != NOT_EVALUATED_LABEL]

    records: list[dict[str, object]] = []
    clusters = sorted(df["cluster"].unique())

    for cid in clusters:
        cluster_data = df[df["cluster"] == cid]
        size = len(cluster_data)

        for prop in properties:
            if prop not in cluster_data.columns:
                continue
            vals = cluster_data[prop].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            p25, p75 = np.percentile(vals, [25, 75])
            records.append(
                {
                    "cluster": cid,
                    "cluster_size": size,
                    "property": prop,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "iqr": float(p75 - p25),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )

    return pd.DataFrame.from_records(records)


def compute_property_association_tests(
    features_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    properties: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute ANOVA F-test, Kruskal-Wallis H-test, and Mutual Information across clusters."""
    if properties is None:
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

    df = features_df.copy()
    df["cluster"] = cluster_labels

    # Exclude noise points (-1) and unexamined rows (-2) from statistical tests
    valid_df = df[~df["cluster"].isin([-1, NOT_EVALUATED_LABEL])]
    if len(valid_df["cluster"].unique()) < 2:
        return pd.DataFrame()

    results: list[dict[str, object]] = []
    labels = valid_df["cluster"].to_numpy()

    for prop in properties:
        if prop not in valid_df.columns:
            continue
        vals = valid_df[prop].to_numpy(dtype=float)

        # Group data by cluster
        groups = [
            group[prop].dropna().to_numpy(dtype=float) for _, group in valid_df.groupby("cluster")
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            continue

        # ANOVA F-test
        f_stat, f_pval = stats.f_oneway(*groups)

        # Kruskal-Wallis H-test (non-parametric)
        h_stat, h_pval = stats.kruskal(*groups)

        # Mutual Information
        X = vals.reshape(-1, 1)
        mi = float(
            mutual_info_classif(X, labels, discrete_features=False, random_state=random_state)[0]
        )

        results.append(
            {
                "property": prop,
                "anova_f_stat": float(f_stat),
                "anova_p_val": float(f_pval),
                "kruskal_h_stat": float(h_stat),
                "kruskal_p_val": float(h_pval),
                "mutual_info": mi,
            }
        )

    return pd.DataFrame.from_records(results)


def plot_umap_clusters(
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Draw a 2D UMAP scatter plot colored by cluster assignments."""
    unique_labels = sorted(set(labels))
    has_noise = -1 in unique_labels
    has_unevaluated = NOT_EVALUATED_LABEL in unique_labels

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)

        if has_unevaluated:
            unevaluated_mask = labels == NOT_EVALUATED_LABEL
            ax.scatter(
                coordinates[unevaluated_mask, 0],
                coordinates[unevaluated_mask, 1],
                s=4,
                c="#ededed",
                alpha=0.35,
                linewidths=0,
                rasterized=True,
                label="Outside density fit subsample (-2)",
            )

        if has_noise:
            noise_mask = labels == -1
            ax.scatter(
                coordinates[noise_mask, 0],
                coordinates[noise_mask, 1],
                s=6,
                c="#c0c0c0",
                alpha=0.45,
                linewidths=0,
                rasterized=True,
                label="Noise (-1)",
            )

        cluster_mask = labels >= 0
        if np.any(cluster_mask):
            c_labels = labels[cluster_mask]
            unique_cids = sorted(set(c_labels))
            n_clusters = len(unique_cids)

            if n_clusters <= 10:
                cmap = plt.get_cmap("tab10", n_clusters)
            elif n_clusters <= 20:
                cmap = plt.get_cmap("tab20", n_clusters)
            else:
                cmap = plt.get_cmap("turbo", n_clusters)

            norm = BoundaryNorm(np.arange(-0.5, n_clusters + 0.5), ncolors=n_clusters)

            # Map cluster IDs to contiguous 0..n_clusters-1 for qualitative colormap
            cluster_id_map = {cid: idx for idx, cid in enumerate(unique_cids)}
            mapped_labels = np.array([cluster_id_map[cid] for cid in c_labels])

            points = ax.scatter(
                coordinates[cluster_mask, 0],
                coordinates[cluster_mask, 1],
                s=8,
                c=mapped_labels,
                cmap=cmap,
                norm=norm,
                alpha=0.85,
                linewidths=0,
                rasterized=True,
            )

            if n_clusters <= 20:
                ticks = np.arange(n_clusters)
                tick_labels = [str(cid) for cid in unique_cids]
                cbar = fig.colorbar(points, ax=ax, pad=0.02, ticks=ticks)
                cbar.ax.set_yticklabels(tick_labels)
            else:
                cbar = fig.colorbar(points, ax=ax, pad=0.02)

            cbar.set_label("Cluster ID")

        ax.set_title(title, pad=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines[["top", "right"]].set_visible(False)
        if has_noise or has_unevaluated:
            ax.legend(frameon=False, loc="best", markerscale=2)

        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_property_distributions(
    features_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    properties: list[tuple[str, str, bool]],
    algorithm_name: str,
    output_path: Path,
) -> None:
    """Plot box plots of Collatz property distributions across clusters."""
    df = features_df.copy()
    df["cluster"] = cluster_labels
    df = df[df["cluster"] != NOT_EVALUATED_LABEL]

    n_props = len(properties)
    n_cols = 2
    n_rows = (n_props + 1) // n_cols

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), constrained_layout=True)
        axes = axes.flatten()

        clusters = sorted(df["cluster"].unique())

        for idx, (prop, label, logarithmic) in enumerate(properties):
            ax = axes[idx]
            data_by_cluster = []
            for cid in clusters:
                vals = df[df["cluster"] == cid][prop].dropna().to_numpy(dtype=float)
                if logarithmic:
                    vals = np.log10(np.maximum(vals, 1.0))
                data_by_cluster.append(vals)

            bp = ax.boxplot(
                data_by_cluster,
                tick_labels=[str(c) for c in clusters],
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "red",
                    "markeredgecolor": "red",
                    "markersize": 4,
                },
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
            )
            for box in bp["boxes"]:
                box.set(facecolor="#4C72B0", alpha=0.7, linewidth=0.8)

            ax.set_title(f"{label} {'(log10)' if logarithmic else ''} by Cluster", fontsize=11)
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel(f"log10({label})" if logarithmic else label)
            ax.tick_params(
                axis="x",
                rotation=90 if len(clusters) > 15 else 0,
                labelsize=7 if len(clusters) > 20 else 9,
            )
            ax.spines[["top", "right"]].set_visible(False)

        # Hide empty axes if n_props is odd
        for idx in range(n_props, len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(f"Collatz Property Distributions for {algorithm_name} Clusters", fontsize=14)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_metrics_comparison(
    kmeans_df: pd.DataFrame,
    dbscan_df: pd.DataFrame,
    hdbscan_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) across models."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

        # Silhouette score comparison
        ax = axes[0]
        ax.plot(
            kmeans_df["n_clusters"],
            kmeans_df["silhouette_score"],
            "o-",
            label="KMeans",
            color="#1f77b4",
        )
        if not hdbscan_df.empty:
            valid_h = hdbscan_df.dropna(subset=["silhouette_score"])
            ax.scatter(
                valid_h["n_clusters"],
                valid_h["silhouette_score"],
                c="#2ca02c",
                label="HDBSCAN",
                marker="s",
                s=40,
            )
        if not dbscan_df.empty:
            valid_d = dbscan_df.dropna(subset=["silhouette_score"])
            ax.scatter(
                valid_d["n_clusters"],
                valid_d["silhouette_score"],
                c="#ff7f0e",
                label="DBSCAN",
                marker="^",
                s=30,
                alpha=0.7,
            )

        ax.set_title("Silhouette Score (Higher is better)")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        # Calinski-Harabasz score comparison
        ax = axes[1]
        ax.plot(
            kmeans_df["n_clusters"],
            kmeans_df["calinski_harabasz_score"],
            "o-",
            label="KMeans",
            color="#1f77b4",
        )
        if not hdbscan_df.empty:
            valid_h = hdbscan_df.dropna(subset=["calinski_harabasz_score"])
            ax.scatter(
                valid_h["n_clusters"],
                valid_h["calinski_harabasz_score"],
                c="#2ca02c",
                label="HDBSCAN",
                marker="s",
                s=40,
            )
        ax.set_title("Calinski-Harabasz Score (Higher is better)")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("CH Index")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        # Davies-Bouldin score comparison
        ax = axes[2]
        ax.plot(
            kmeans_df["n_clusters"],
            kmeans_df["davies_bouldin_score"],
            "o-",
            label="KMeans",
            color="#1f77b4",
        )
        if not hdbscan_df.empty:
            valid_h = hdbscan_df.dropna(subset=["davies_bouldin_score"])
            ax.scatter(
                valid_h["n_clusters"],
                valid_h["davies_bouldin_score"],
                c="#2ca02c",
                label="HDBSCAN",
                marker="s",
                s=40,
            )
        ax.set_title("Davies-Bouldin Score (Lower is better)")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("DB Index")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_property_heatmap(
    summary_df: pd.DataFrame,
    algorithm_name: str,
    output_path: Path,
) -> None:
    """Plot a standardized heatmap of mean Collatz property values across clusters."""
    # Filter out noise cluster if exists
    clean_df = summary_df[summary_df["cluster"] != -1]
    if clean_df.empty:
        return

    pivot = clean_df.pivot(index="cluster", columns="property", values="mean")

    # Standardize (z-score normalize) properties across clusters for fair visual comparison
    z_pivot = (pivot - pivot.mean()) / (pivot.std().replace(0, 1.0))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8.5, max(4, 0.6 * len(pivot))), constrained_layout=True)
        cax = ax.imshow(z_pivot.to_numpy(), cmap="coolwarm", aspect="auto")

        ax.set_xticks(np.arange(len(z_pivot.columns)))
        ax.set_xticklabels(z_pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(z_pivot.index)))
        ax.set_yticklabels([f"Cluster {c}" for c in z_pivot.index])

        # Add text annotations inside cells
        for i in range(len(z_pivot.index)):
            for j in range(len(z_pivot.columns)):
                val = z_pivot.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    color="black" if abs(val) < 1.5 else "white",
                    fontsize=8,
                )

        cbar = fig.colorbar(cax, ax=ax, pad=0.02)
        cbar.set_label("Standardized Feature Z-Score")

        ax.set_title(f"Standardized Collatz Property Profiles ({algorithm_name})", pad=12)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
