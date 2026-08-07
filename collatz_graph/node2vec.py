"""Reproducible Node2Vec embeddings for finite inverse Collatz graphs.

The graph construction code stores edges in the forward Collatz direction
(``u -> T(u)``).  Node2Vec walks use the reversed edges by default, so a walk
traverses the inverse Collatz graph from a value to its retained predecessors.
NumPy is the required CPU backend; PyTorch is selected automatically when a
CUDA device is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Node2VecConfig:
    """Hyperparameters for reproducible Node2Vec training."""

    dimensions: int = 128
    walk_length: int = 40
    walks_per_node: int = 2
    context_size: int = 5
    p: float = 1.0
    q: float = 1.0
    negative_samples: int = 5
    epochs: int = 3
    learning_rate: float = 0.025
    seed: int = 20260722
    backend: str = "auto"
    follow_reverse: bool = True
    batch_size: int = 512

    def __post_init__(self) -> None:
        if self.dimensions < 1 or self.walk_length < 2 or self.walks_per_node < 1:
            raise ValueError("dimensions, walk_length, and walks_per_node must be positive")
        if self.context_size < 1 or self.negative_samples < 1 or self.epochs < 1:
            raise ValueError("context_size, negative_samples, and epochs must be positive")
        if self.p <= 0 or self.q <= 0 or self.learning_rate <= 0:
            raise ValueError("p, q, and learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.backend not in {"auto", "numpy", "torch"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'torch'")


@dataclass(frozen=True, slots=True)
class Node2VecResult:
    """Trained embedding matrix and reproducibility records."""

    node_ids: np.ndarray
    embeddings: np.ndarray
    history: pd.DataFrame
    config: Node2VecConfig
    backend: str
    device: str
    runtime_seconds: float


def _select_backend(config: Node2VecConfig) -> tuple[str, str]:
    if config.backend == "numpy":
        return "numpy", "cpu"
    try:
        import torch
    except ImportError:
        if config.backend == "torch":
            raise ImportError("backend='torch' requires PyTorch")
        return "numpy", "cpu"
    if config.backend == "torch" or torch.cuda.is_available():
        return "torch", "cuda" if torch.cuda.is_available() else "cpu"
    return "numpy", "cpu"


def _transition_probabilities(graph: nx.DiGraph, current: int, previous: int | None, config: Node2VecConfig) -> tuple[np.ndarray, np.ndarray]:
    neighbors = np.asarray(list(graph.successors(current)), dtype=np.int64)
    if neighbors.size == 0:
        return neighbors, np.empty(0, dtype=np.float64)
    if previous is None:
        weights = np.ones(neighbors.size, dtype=np.float64)
    else:
        previous_neighbors = set(graph.successors(previous))
        weights = np.asarray(
            [1.0 / config.p if neighbor == previous else 1.0 if neighbor in previous_neighbors else 1.0 / config.q for neighbor in neighbors],
            dtype=np.float64,
        )
    return neighbors, weights / weights.sum()


def generate_node2vec_walks(graph: nx.DiGraph, config: Node2VecConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate deterministic biased random walks and return node indexing."""
    if not graph:
        raise ValueError("graph must contain nodes")
    node_ids = np.asarray(sorted(graph.nodes), dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    walk_graph = graph.reverse(copy=False) if config.follow_reverse else graph
    walks: list[np.ndarray] = []
    for _ in range(config.walks_per_node):
        starts = node_ids.copy()
        rng.shuffle(starts)
        for start in starts:
            walk = [int(start)]
            previous: int | None = None
            current = int(start)
            for _ in range(config.walk_length - 1):
                neighbors, probabilities = _transition_probabilities(walk_graph, current, previous, config)
                if neighbors.size == 0:
                    break
                next_node = int(rng.choice(neighbors, p=probabilities))
                previous, current = current, next_node
                walk.append(current)
            walks.append(np.asarray(walk, dtype=np.int64))
    return node_ids, walks


def _training_pairs(walks: list[np.ndarray], node_to_index: dict[int, int], context_size: int) -> tuple[np.ndarray, np.ndarray]:
    targets: list[int] = []
    contexts: list[int] = []
    for walk in walks:
        indices = [node_to_index[int(node)] for node in walk]
        for center, target in enumerate(indices):
            left = max(0, center - context_size)
            right = min(len(indices), center + context_size + 1)
            for context in indices[left:center] + indices[center + 1:right]:
                targets.append(target)
                contexts.append(context)
    return np.asarray(targets, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


def _negative_sampling_distribution(
    graph: nx.DiGraph, node_ids: np.ndarray, config: Node2VecConfig
) -> np.ndarray:
    """Return the Node2Vec/word2vec degree-to-the-three-quarters distribution."""
    walk_graph = graph.reverse(copy=False) if config.follow_reverse else graph
    degrees = np.fromiter(
        (walk_graph.in_degree(int(node)) + walk_graph.out_degree(int(node)) for node in node_ids),
        dtype=np.float64,
        count=len(node_ids),
    )
    weights = np.power(np.maximum(degrees, 1.0), 0.75)
    return weights / weights.sum()


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + np.exp(-clipped))


def _train_numpy(targets: np.ndarray, contexts: np.ndarray, node_count: int, negative_distribution: np.ndarray, config: Node2VecConfig, rng: np.random.Generator, show_progress: bool) -> tuple[np.ndarray, pd.DataFrame]:
    input_embeddings = rng.normal(0.0, 1.0 / config.dimensions, (node_count, config.dimensions)).astype(np.float32)
    output_embeddings = np.zeros_like(input_embeddings)
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        order = rng.permutation(len(targets))
        loss_sum = 0.0
        for position in tqdm(range(0, len(order), config.batch_size), desc=f"Node2Vec epoch {epoch + 1}", disable=not show_progress):
            batch = order[position:position + config.batch_size]
            for target, context in zip(targets[batch], contexts[batch]):
                negatives = rng.choice(node_count, size=config.negative_samples, p=negative_distribution)
                input_before = input_embeddings[target].copy()
                positive_probability = _sigmoid(float(np.dot(input_before, output_embeddings[context])))
                positive_gradient = config.learning_rate * (1.0 - positive_probability)
                input_embeddings[target] += positive_gradient * output_embeddings[context]
                output_embeddings[context] += positive_gradient * input_before
                loss_sum -= np.log(max(positive_probability, 1e-12))
                for negative in negatives:
                    negative_probability = _sigmoid(float(np.dot(input_before, output_embeddings[negative])))
                    negative_gradient = -config.learning_rate * negative_probability
                    input_embeddings[target] += negative_gradient * output_embeddings[negative]
                    output_embeddings[negative] += negative_gradient * input_before
                    loss_sum -= np.log(max(1.0 - negative_probability, 1e-12))
        history.append({"epoch": epoch + 1, "loss": loss_sum / max(1, len(targets)), "pairs": len(targets), "learning_rate": config.learning_rate})
    embeddings = input_embeddings + output_embeddings
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    return embeddings.astype(np.float32), pd.DataFrame(history)


def _train_torch(targets: np.ndarray, contexts: np.ndarray, node_count: int, negative_distribution: np.ndarray, config: Node2VecConfig, device: str, show_progress: bool) -> tuple[np.ndarray, pd.DataFrame]:
    """Train skip-gram negative sampling on CPU or CUDA with PyTorch."""
    import torch
    from torch import nn

    torch.manual_seed(config.seed + 1)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed + 1)
        # CuDNN's benchmark can choose different convolution kernels between
        # runs. It is irrelevant to embeddings, but disabling it makes the
        # reproducibility policy explicit for GPU-backed experiments.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    input_embedding = nn.Embedding(node_count, config.dimensions, device=device)
    output_embedding = nn.Embedding(node_count, config.dimensions, device=device)
    nn.init.normal_(input_embedding.weight, mean=0.0, std=1.0 / config.dimensions)
    nn.init.zeros_(output_embedding.weight)
    optimizer = torch.optim.SGD(list(input_embedding.parameters()) + list(output_embedding.parameters()), lr=config.learning_rate)
    negative_probabilities = torch.as_tensor(negative_distribution, dtype=torch.float32, device=device)
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        order = np.random.default_rng(config.seed + 1000 + epoch).permutation(len(targets))
        loss_sum = 0.0
        for position in tqdm(range(0, len(order), config.batch_size), desc=f"Node2Vec epoch {epoch + 1}", disable=not show_progress):
            batch = order[position:position + config.batch_size]
            target_tensor = torch.as_tensor(targets[batch], dtype=torch.long, device=device)
            context_tensor = torch.as_tensor(contexts[batch], dtype=torch.long, device=device)
            negative_tensor = torch.multinomial(
                negative_probabilities,
                num_samples=len(batch) * config.negative_samples,
                replacement=True,
            ).reshape(len(batch), config.negative_samples)
            positive_score = (input_embedding(target_tensor) * output_embedding(context_tensor)).sum(dim=1)
            negative_score = (input_embedding(target_tensor).unsqueeze(1) * output_embedding(negative_tensor)).sum(dim=2)
            loss = torch.nn.functional.softplus(-positive_score).mean() + torch.nn.functional.softplus(negative_score).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch)
        history.append({"epoch": epoch + 1, "loss": loss_sum / max(1, len(targets)), "pairs": len(targets), "learning_rate": config.learning_rate})
    embeddings = (input_embedding.weight.detach() + output_embedding.weight.detach()).cpu().numpy()
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    return embeddings.astype(np.float32), pd.DataFrame(history)


def train_node2vec(graph: nx.DiGraph, config: Node2VecConfig | None = None, *, show_progress: bool = True) -> Node2VecResult:
    """Train a reproducible Node2Vec embedding, 128-dimensional by default."""
    config = config or Node2VecConfig()
    backend, device = _select_backend(config)
    start = perf_counter()
    node_ids, walks = generate_node2vec_walks(graph, config)
    node_to_index = {int(node): index for index, node in enumerate(node_ids)}
    targets, contexts = _training_pairs(walks, node_to_index, config.context_size)
    if len(targets) == 0:
        raise ValueError("walk configuration produced no skip-gram training pairs")
    negative_distribution = _negative_sampling_distribution(graph, node_ids, config)
    if backend == "torch":
        embeddings, history = _train_torch(targets, contexts, len(node_ids), negative_distribution, config, device, show_progress)
    else:
        embeddings, history = _train_numpy(targets, contexts, len(node_ids), negative_distribution, config, np.random.default_rng(config.seed + 1), show_progress)
    runtime_seconds = perf_counter() - start
    LOGGER.info("Trained Node2Vec embeddings for %d nodes in %.6f s", len(node_ids), runtime_seconds)
    return Node2VecResult(node_ids, embeddings, history, config, backend, device, runtime_seconds)


def save_node2vec_result(
    result: Node2VecResult,
    output_directory: str | Path,
    stem: str = "node2vec",
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist embeddings, node order, loss history, metadata, and a plot."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"embeddings": output / f"{stem}_embeddings.npy", "nodes": output / f"{stem}_nodes.csv", "history": output / f"{stem}_training.csv", "metadata": output / f"{stem}_metadata.json", "training_plot": output / f"{stem}_training_loss.png"}
    np.save(paths["embeddings"], result.embeddings)
    pd.DataFrame({"node": result.node_ids}).to_csv(paths["nodes"], index=False)
    result.history.to_csv(paths["history"], index=False)
    metadata: dict[str, Any] = {"config": asdict(result.config), "backend": result.backend, "device": result.device, "runtime_seconds": result.runtime_seconds, "node_count": int(len(result.node_ids)), "embedding_dimensions": int(result.embeddings.shape[1]), "python_version": platform.python_version(), "numpy_version": np.__version__, "negative_sampling_distribution": "(total walk-graph degree)^0.75"}
    if result.backend == "torch":
        import torch
        metadata["torch_version"] = torch.__version__
    if extra_metadata:
        metadata.update(extra_metadata)
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot(result.history["epoch"], result.history["loss"], marker="o", color="#2166ac")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean negative-sampling loss")
    ax.set_title("Node2Vec training convergence")
    ax.grid(alpha=0.25)
    fig.savefig(paths["training_plot"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths
