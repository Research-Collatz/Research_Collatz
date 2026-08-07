"""Unit tests for supervised predictive modeling from Node2Vec embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from collatz_graph.supervised import (
    evaluate_property_prediction,
    instantiate_model,
    run_all_supervised_experiments,
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic embeddings and matched feature dataframe for testing."""
    np.random.seed(42)
    n_samples = 60
    n_dim = 16

    embeddings = np.random.randn(n_samples, n_dim)
    nodes = np.arange(1, n_samples + 1)

    features_df = pd.DataFrame(
        {
            "node": nodes,
            "stopping_time": np.random.randint(1, 50, size=n_samples),
            "maximum_excursion": np.random.randint(10, 10000, size=n_samples),
            "binary_length": np.floor(np.log2(nodes)).astype(int) + 1,
            "level_set": np.random.randint(0, 20, size=n_samples),
        }
    )

    return embeddings, features_df


def test_instantiate_model():
    rf = instantiate_model("RandomForest")
    assert rf is not None

    try:
        xgb = instantiate_model("XGBoost")
        assert xgb is not None
    except ImportError:
        pass

    try:
        lgbm = instantiate_model("LightGBM")
        assert lgbm is not None
    except ImportError:
        pass


def test_evaluate_property_prediction(synthetic_data):
    embeddings, features_df = synthetic_data

    metrics, oof_preds, importances = evaluate_property_prediction(
        embeddings=embeddings,
        features_df=features_df,
        target_property="stopping_time",
        model_name="RandomForest",
        n_splits=3,
        random_state=42,
        log_transform=False,
    )

    assert "r2_mean" in metrics
    assert "mae_mean" in metrics
    assert len(oof_preds) == len(embeddings)
    assert len(importances) == embeddings.shape[1]


def test_run_all_supervised_experiments(synthetic_data):
    embeddings, features_df = synthetic_data

    metrics_df, importance_df, oof_preds_dict = run_all_supervised_experiments(
        embeddings=embeddings,
        features_df=features_df,
        target_properties=[("stopping_time", False), ("binary_length", False)],
        models=["RandomForest"],
        n_splits=3,
        random_state=42,
    )

    assert not metrics_df.empty
    assert "r2_oof" in metrics_df.columns
    assert not importance_df.empty
    assert "stopping_time__RandomForest" in oof_preds_dict
