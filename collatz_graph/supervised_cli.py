"""CLI for supervised prediction from Node2Vec embeddings."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .embedding_plots import load_embedding_features
from .supervised import (
    plot_feature_importance_top,
    plot_model_comparison,
    plot_prediction_scatters,
    run_all_supervised_experiments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def execute_supervised_pipeline(
    run_directory: str | Path,
    output_directory: str | Path,
    seed: int = 42,
) -> dict[str, str]:
    """Run supervised training, cross-validation, feature importance, and visualization pipeline."""
    run_path = Path(run_directory)
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading Node2Vec embeddings and feature table from %s", run_path)
    embeddings, features_df, _ = load_embedding_features(run_path)

    target_properties = [
        ("stopping_time", False),
        ("maximum_excursion", True),  # Log10-transformed
        ("binary_length", False),
        ("level_set", False),
    ]

    LOGGER.info(
        "Executing 3-Fold Cross-Validation across models (RandomForest, XGBoost, LightGBM)..."
    )
    metrics_df, importance_df, oof_preds_dict = run_all_supervised_experiments(
        embeddings=embeddings,
        features_df=features_df,
        target_properties=target_properties,
        n_splits=3,
        random_state=seed,
    )

    metrics_path = out_dir / "supervised_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    LOGGER.info("Saved predictive metrics table to %s", metrics_path)

    importance_path = out_dir / "feature_importances.csv"
    importance_df.to_csv(importance_path, index=False)
    LOGGER.info("Saved feature importances table to %s", importance_path)

    LOGGER.info("Generating comparative figures and scatter plots...")

    plot_model_comparison(metrics_df, out_dir / "model_performance_comparison")
    plot_feature_importance_top(importance_df, out_dir / "feature_importance_top")

    # Plot actual vs predicted scatter plots for primary models
    models = metrics_df["model_name"].unique()
    for model_name in models:
        plot_prediction_scatters(
            features_df,
            oof_preds_dict,
            target_properties,
            model_name,
            out_dir / f"prediction_scatter_{model_name.lower()}",
        )

    manifest = {
        "metrics_table": str(metrics_path),
        "feature_importances": str(importance_path),
        "model_performance_comparison": str(out_dir / "model_performance_comparison.png"),
        "feature_importance_top": str(out_dir / "feature_importance_top.png"),
    }
    for model_name in models:
        manifest[f"prediction_scatter_{model_name.lower()}"] = str(
            out_dir / f"prediction_scatter_{model_name.lower()}.png"
        )

    manifest_path = out_dir / "supervised_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Supervised pipeline completed successfully. Manifest written to %s", manifest_path)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Collatz properties from Node2Vec embeddings using supervised ML."
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Directory containing node2vec artifacts."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to save output artifacts."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    manifest = execute_supervised_pipeline(args.run_dir, args.output_dir, seed=args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
