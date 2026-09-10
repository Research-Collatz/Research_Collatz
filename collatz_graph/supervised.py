"""Supervised predictive modeling of Collatz properties from Node2Vec embeddings.

This module provides tools to train and evaluate Random Forest, XGBoost, and LightGBM
regressors on 128-dimensional Node2Vec embeddings to predict Collatz arithmetic and
topological properties (stopping_time, maximum_excursion, binary_length, level_set),
compute 5-fold cross-validation metrics, feature importances, and plot comparisons.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

LOGGER = logging.getLogger(__name__)

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


def instantiate_model(model_name: str, random_state: int = 42) -> Any:
    """Instantiate a regressor model based on model_name."""
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost package is required for XGBoost model.")
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        return XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device,
            random_state=random_state,
        )
    elif model_name == "LightGBM":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm package is required for LightGBM model.")
        return LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=10,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def compute_accuracy_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, is_log_scale: bool = False
) -> dict[str, float]:
    """Compute exact and tolerance accuracy metrics."""
    if is_log_scale:
        # 0.1 in log10 space is roughly 25% ratio; 0.0414 is roughly 10%.
        within_10pct = float(np.mean(np.abs(y_true - y_pred) <= 0.0414))
        within_25pct = float(np.mean(np.abs(y_true - y_pred) <= 0.1))
        return {
            "accuracy_exact": within_10pct,
            "accuracy_pm1": within_10pct,
            "accuracy_pm2": within_25pct,
        }
    else:
        diff = np.abs(y_true - y_pred)
        exact = float(np.mean(np.round(y_pred) == y_true))
        pm1 = float(np.mean(diff <= 1.0))
        pm2 = float(np.mean(diff <= 2.0))
        return {
            "accuracy_exact": exact,
            "accuracy_pm1": pm1,
            "accuracy_pm2": pm2,
        }


def evaluate_property_prediction(
    embeddings: np.ndarray,
    features_df: pd.DataFrame,
    target_property: str,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    log_transform: bool = False,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Train and evaluate a model family on a target property using K-Fold cross-validation."""
    if target_property not in features_df.columns:
        raise ValueError(f"Target property '{target_property}' not found in feature dataframe.")

    y = features_df[target_property].to_numpy(dtype=float)
    if log_transform:
        y = np.log10(np.maximum(y, 1.0))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_predictions = np.zeros(len(embeddings), dtype=float)
    feature_importances_list: list[np.ndarray] = []

    r2_folds: list[float] = []
    mae_folds: list[float] = []
    rmse_folds: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
        LOGGER.info(
            "Starting Fold %d/%d (%s)",
            fold + 1,
            n_splits,
            model_name,
        )

        fold_start = perf_counter()

        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = instantiate_model(
            model_name,
            random_state=random_state + fold,
        )

        LOGGER.info(
            "Training %s...",
            model_name,
        )
        model.fit(X_train, y_train)

        LOGGER.info(
            "Predicting...",
        )
        y_pred = model.predict(X_test)
        oof_predictions[test_idx] = y_pred

        r2_folds.append(float(r2_score(y_test, y_pred)))
        mae_folds.append(float(mean_absolute_error(y_test, y_pred)))
        rmse_folds.append(float(root_mean_squared_error(y_test, y_pred)))

        if hasattr(model, "feature_importances_"):
            feature_importances_list.append(model.feature_importances_)

        LOGGER.info(
            "Finished Fold %d in %.2f sec",
            fold + 1,
            perf_counter() - fold_start,
        )

    # Aggregate overall out-of-fold metrics
    overall_r2 = float(r2_score(y, oof_predictions))
    overall_mae = float(mean_absolute_error(y, oof_predictions))
    overall_rmse = float(root_mean_squared_error(y, oof_predictions))
    acc_metrics = compute_accuracy_metrics(y, oof_predictions, is_log_scale=log_transform)

    mean_importances = (
        np.mean(feature_importances_list, axis=0)
        if feature_importances_list
        else np.zeros(embeddings.shape[1])
    )

    metrics = {
        "target_property": target_property,
        "model_name": model_name,
        "is_log_transformed": log_transform,
        "r2_mean": float(np.mean(r2_folds)),
        "r2_std": float(np.std(r2_folds)),
        "r2_oof": overall_r2,
        "mae_mean": float(np.mean(mae_folds)),
        "mae_std": float(np.std(mae_folds)),
        "mae_oof": overall_mae,
        "rmse_mean": float(np.mean(rmse_folds)),
        "rmse_std": float(np.std(rmse_folds)),
        "rmse_oof": overall_rmse,
        "accuracy_exact": acc_metrics["accuracy_exact"],
        "accuracy_pm1": acc_metrics["accuracy_pm1"],
        "accuracy_pm2": acc_metrics["accuracy_pm2"],
    }

    return metrics, oof_predictions, mean_importances


def run_all_supervised_experiments(
    embeddings: np.ndarray,
    features_df: pd.DataFrame,
    target_properties: list[tuple[str, bool]] | None = None,
    models: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Execute 5-fold CV across all combinations of models and Collatz properties."""
    if target_properties is None:
        target_properties = [
            ("stopping_time", False),
            ("maximum_excursion", True),  # Log-transformed for stability
            ("binary_length", False),
            ("level_set", False),
        ]

    if models is None:
        models = ["RandomForest"]
        if HAS_XGBOOST:
            models.append("XGBoost")
        if HAS_LIGHTGBM:
            models.append("LightGBM")

    all_metrics: list[dict[str, Any]] = []
    importance_records: list[dict[str, Any]] = []
    oof_predictions_dict: dict[str, np.ndarray] = {}

    for prop, log_transform in target_properties:
        for model_name in models:
            experiment_start = perf_counter()

            LOGGER.info("Evaluating %s on target '%s' (log=%s)...", model_name, prop, log_transform)
            metrics, oof_preds, importances = evaluate_property_prediction(
                embeddings=embeddings,
                features_df=features_df,
                target_property=prop,
                model_name=model_name,
                n_splits=n_splits,
                random_state=random_state,
                log_transform=log_transform,
            )
            all_metrics.append(metrics)
            LOGGER.info(
                "%s finished in %.2f minutes",
                model_name,
                (perf_counter() - experiment_start) / 60,
            )
            key = f"{prop}__{model_name}"
            oof_predictions_dict[key] = oof_preds

            for dim_idx, imp_val in enumerate(importances):
                importance_records.append(
                    {
                        "target_property": prop,
                        "model_name": model_name,
                        "dimension": dim_idx,
                        "importance": float(imp_val),
                    }
                )

    metrics_df = pd.DataFrame.from_records(all_metrics)
    importance_df = pd.DataFrame.from_records(importance_records)
    return metrics_df, importance_df, oof_predictions_dict


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Draw comparative bar charts of R2 score and MAE across models and target properties."""
    targets = metrics_df["target_property"].unique()
    models = metrics_df["model_name"].unique()

    n_targets = len(targets)
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        x = np.arange(n_targets)
        width = 0.25

        # R2 score comparison plot
        ax = axes[0]
        for i, model in enumerate(models):
            m_df = metrics_df[metrics_df["model_name"] == model]
            r2_vals = [m_df[m_df["target_property"] == t]["r2_oof"].values[0] for t in targets]
            ax.bar(x + (i - len(models) / 2 + 0.5) * width, r2_vals, width, label=model, alpha=0.85)

        ax.set_title("Coefficient of Determination (R² Score) by Target", fontsize=11)
        ax.set_ylabel("R² Score")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=15)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        # MAE comparison plot
        ax = axes[1]
        for i, model in enumerate(models):
            m_df = metrics_df[metrics_df["model_name"] == model]
            mae_vals = [m_df[m_df["target_property"] == t]["mae_oof"].values[0] for t in targets]
            ax.bar(
                x + (i - len(models) / 2 + 0.5) * width, mae_vals, width, label=model, alpha=0.85
            )

        ax.set_title("Mean Absolute Error (MAE) by Target", fontsize=11)
        ax.set_ylabel("MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=15)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_prediction_scatters(
    features_df: pd.DataFrame,
    oof_predictions_dict: dict[str, np.ndarray],
    target_properties: list[tuple[str, bool]],
    model_name: str,
    output_path: Path,
) -> None:
    """Plot actual vs predicted scatter plots for out-of-fold predictions."""
    n_targets = len(target_properties)
    n_cols = 2
    n_rows = (n_targets + 1) // n_cols

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(11, 4.5 * n_rows), constrained_layout=True
        )
        axes = axes.flatten()

        for idx, (prop, log_transform) in enumerate(target_properties):
            ax = axes[idx]
            key = f"{prop}__{model_name}"
            if key not in oof_predictions_dict:
                continue

            y_actual = features_df[prop].to_numpy(dtype=float)
            if log_transform:
                y_actual = np.log10(np.maximum(y_actual, 1.0))

            y_pred = oof_predictions_dict[key]

            ax.scatter(
                y_actual, y_pred, s=8, alpha=0.4, c="#1f77b4", edgecolors="none", rasterized=True
            )

            # Identity line
            min_val = min(np.min(y_actual), np.min(y_pred))
            max_val = max(np.max(y_actual), np.max(y_pred))
            ax.plot(
                [min_val, max_val], [min_val, max_val], "r--", linewidth=1.2, label="Ideal (y=x)"
            )

            title_suffix = " (log10)" if log_transform else ""
            ax.set_title(f"Actual vs Predicted: {prop}{title_suffix} ({model_name})", fontsize=11)
            ax.set_xlabel(f"Actual {prop}{title_suffix}")
            ax.set_ylabel(f"Predicted {prop}{title_suffix}")
            ax.legend(frameon=False)
            ax.spines[["top", "right"]].set_visible(False)

        for idx in range(n_targets, len(axes)):
            fig.delaxes(axes[idx])

        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_feature_importance_top(
    importance_df: pd.DataFrame, output_path: Path, top_n: int = 10
) -> None:
    """Plot top embedding dimensions by feature importance across targets."""
    targets = importance_df["target_property"].unique()
    models = importance_df["model_name"].unique()

    model_name = models[0]  # Primary model family
    sub_df = importance_df[importance_df["model_name"] == model_name]

    n_targets = len(targets)
    n_cols = 2
    n_rows = (n_targets + 1) // n_cols

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4 * n_rows), constrained_layout=True)
        axes = axes.flatten()

        for idx, prop in enumerate(targets):
            ax = axes[idx]
            prop_df = (
                sub_df[sub_df["target_property"] == prop]
                .sort_values("importance", ascending=False)
                .head(top_n)
            )

            y_pos = np.arange(len(prop_df))
            ax.barh(y_pos, prop_df["importance"], align="center", color="#2ca02c", alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"Dim {d}" for d in prop_df["dimension"]])
            ax.invert_yaxis()
            ax.set_xlabel("Relative Feature Importance")
            ax.set_title(f"Top {top_n} Node2Vec Dimensions for {prop}", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)

        for idx in range(n_targets, len(axes)):
            fig.delaxes(axes[idx])

        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
