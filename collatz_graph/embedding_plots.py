"""UMAP visualizations of saved inverse-Collatz Node2Vec embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import networkx as nx
import numpy as np
import pandas as pd

from .bounded import build_inverse_graph_up_to
from .features import compute_node_features

UMAP_SEED = 20260722
PLOT_STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12, "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 450}


def _inverse_levels(graph: nx.DiGraph, root: int = 1) -> dict[int, int]:
    """Return breadth-first inverse-tree levels for nodes reachable from root."""
    return nx.single_source_shortest_path_length(graph.reverse(copy=False), root)


def load_embedding_features(run_directory: str | Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    """Load a saved matrix and align finite-graph features to its row order."""
    run = Path(run_directory)
    embeddings = np.load(run / "node2vec_embeddings.npy")
    nodes = pd.read_csv(run / "node2vec_nodes.csv")
    metadata = json.loads((run / "node2vec_metadata.json").read_text(encoding="utf-8"))
    if embeddings.ndim != 2 or len(nodes) != len(embeddings):
        raise ValueError("embedding matrix and node mapping have incompatible shapes")
    graph_info = metadata.get("graph")
    if not isinstance(graph_info, dict) or "domain_maximum" not in graph_info:
        raise ValueError("metadata must record graph.domain_maximum")
    graph = build_inverse_graph_up_to(int(graph_info["domain_maximum"]), show_progress=False).graph
    features = compute_node_features(graph, show_progress=False).data.set_index("node")
    aligned = features.loc[nodes["node"].to_numpy()].reset_index()
    levels = _inverse_levels(graph)
    aligned["level_set"] = aligned["node"].map(levels).fillna(-1).astype(np.int64)
    return embeddings, aligned, metadata


def fit_umap(embeddings: np.ndarray, *, seed: int = UMAP_SEED) -> np.ndarray:
    """Produce a reproducible two-dimensional cosine-UMAP representation."""
    try:
        import umap
    except ImportError as error:
        raise ImportError("UMAP plots require `pip install umap-learn`.") from error
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.15, metric="cosine", random_state=seed, transform_seed=seed, n_jobs=1)
    return reducer.fit_transform(embeddings)


def _plot_one(coordinates: np.ndarray, values: pd.Series, label: str, output_path: Path, *, logarithmic: bool = False, categorical: bool = False) -> None:
    """Draw one consistently styled, colourbar-annotated UMAP scatter plot."""
    raw = values.to_numpy(dtype=float)
    valid = raw >= 0
    colour_values = np.log10(raw[valid]) if logarithmic else raw[valid]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
        if np.any(~valid):
            ax.scatter(coordinates[~valid, 0], coordinates[~valid, 1], s=5, c="#d9d9d9", alpha=0.55, linewidths=0, rasterized=True, label="Outside finite root component")
        arguments: dict[str, object] = {"s": 6, "c": colour_values, "cmap": "viridis", "alpha": 0.82, "linewidths": 0, "rasterized": True}
        if categorical:
            arguments["norm"] = BoundaryNorm(np.arange(-0.5, int(np.nanmax(colour_values)) + 1.5), plt.get_cmap("viridis").N)
        points = ax.scatter(coordinates[valid, 0], coordinates[valid, 1], **arguments)
        colorbar = fig.colorbar(points, ax=ax, pad=0.02)
        colorbar.set_label(f"log10({label})" if logarithmic else label)
        ax.set_title(f"Node2Vec UMAP coloured by {label}", pad=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.tick_params(direction="out", length=3)
        ax.spines[["top", "right"]].set_visible(False)
        if np.any(~valid):
            ax.legend(frameon=False, loc="best", markerscale=2)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def create_umap_figures(run_directory: str | Path, output_directory: str | Path | None = None) -> dict[str, Path]:
    """Save one matched UMAP figure for each requested Collatz quantity."""
    run = Path(run_directory)
    output = Path(output_directory) if output_directory else run / "umap"
    output.mkdir(parents=True, exist_ok=True)
    embeddings, features, _ = load_embedding_features(run)
    coordinates = fit_umap(embeddings)
    np.save(output / "umap_coordinates.npy", coordinates.astype(np.float32))
    features.assign(umap_1=coordinates[:, 0], umap_2=coordinates[:, 1]).to_csv(output / "umap_features.csv", index=False)
    specifications = {"stopping_time": ("stopping time", False, False), "maximum_excursion": ("maximum excursion", True, False), "binary_length": ("binary length", False, True), "distance_from_root": ("distance from root", False, True), "level_set": ("inverse-tree level set", False, True)}
    paths: dict[str, Path] = {"coordinates": output / "umap_coordinates.npy", "features": output / "umap_features.csv"}
    for column, (label, logarithmic, categorical) in specifications.items():
        base = output / f"umap_{column}"
        _plot_one(coordinates, features[column], label, base, logarithmic=logarithmic, categorical=categorical)
        paths[column] = base.with_suffix(".png")
        paths[f"{column}_pdf"] = base.with_suffix(".pdf")
    return paths
