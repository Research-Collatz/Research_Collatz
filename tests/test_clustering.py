"""Unit tests for the Node2Vec embedding clustering and evaluation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from collatz_graph.clustering import (
    compute_cluster_property_summary,
    compute_property_association_tests,
    run_dbscan_sweep,
    run_hdbscan_sweep,
    run_kmeans_sweep,
)


@pytest.fixture
def synthetic_embeddings_and_features():
    """Generate synthetic embeddings and matched feature dataframe for testing."""
    np.random.seed(42)
    n_samples = 100
    n_dim = 16

    # 2 synthetic cluster centers
    c1 = np.random.randn(n_samples // 2, n_dim) + 3.0
    c2 = np.random.randn(n_samples // 2, n_dim) - 3.0
    embeddings = np.vstack([c1, c2])

    nodes = np.arange(1, n_samples + 1)
    features_df = pd.DataFrame(
        {
            "node": nodes,
            "stopping_time": np.random.randint(1, 50, size=n_samples),
            "total_stopping_time": np.random.randint(5, 100, size=n_samples),
            "maximum_excursion": np.random.randint(10, 10000, size=n_samples),
            "binary_length": np.floor(np.log2(nodes)).astype(int) + 1,
            "level_set": np.random.randint(0, 20, size=n_samples),
            "distance_from_root": np.random.randint(0, 20, size=n_samples),
            "in_degree": np.random.choice([0, 1, 2], size=n_samples),
            "ancestor_count": np.random.randint(1, 100, size=n_samples),
        }
    )

    return embeddings, features_df


def test_run_kmeans_sweep(synthetic_embeddings_and_features):
    embeddings, _ = synthetic_embeddings_and_features
    df_results, labels_dict = run_kmeans_sweep(embeddings, k_range=range(2, 5))

    assert len(df_results) == 3
    assert set(df_results["algorithm"]) == {"KMeans"}
    assert "silhouette_score" in df_results.columns
    assert 2 in labels_dict
    assert len(labels_dict[2]) == len(embeddings)


def test_run_dbscan_sweep(synthetic_embeddings_and_features):
    embeddings, _ = synthetic_embeddings_and_features
    df_results, labels_dict = run_dbscan_sweep(embeddings, eps_list=[0.5, 1.0], min_samples_list=[5])

    assert len(df_results) == 2
    assert set(df_results["algorithm"]) == {"DBSCAN"}
    assert "noise_ratio" in df_results.columns


def test_run_hdbscan_sweep(synthetic_embeddings_and_features):
    embeddings, _ = synthetic_embeddings_and_features
    df_results, labels_dict = run_hdbscan_sweep(embeddings, min_cluster_size_list=[10, 20])

    assert len(df_results) == 2
    assert set(df_results["algorithm"]) == {"HDBSCAN"}
    assert 10 in labels_dict


def test_compute_cluster_property_summary(synthetic_embeddings_and_features):
    _, features_df = synthetic_embeddings_and_features
    cluster_labels = np.array([0] * 50 + [1] * 50)

    summary_df = compute_cluster_property_summary(features_df, cluster_labels)
    assert not summary_df.empty
    assert "cluster" in summary_df.columns
    assert "mean" in summary_df.columns
    assert "median" in summary_df.columns
    assert set(summary_df["cluster"].unique()) == {0, 1}


def test_compute_property_association_tests(synthetic_embeddings_and_features):
    _, features_df = synthetic_embeddings_and_features
    cluster_labels = np.array([0] * 50 + [1] * 50)

    assoc_df = compute_property_association_tests(features_df, cluster_labels)
    assert not assoc_df.empty
    assert "anova_f_stat" in assoc_df.columns
    assert "kruskal_h_stat" in assoc_df.columns
    assert "mutual_info" in assoc_df.columns
