"""Command-line entry point for reproducible inverse-Collatz Node2Vec runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bounded import build_inverse_graph_up_to
from .node2vec import Node2VecConfig, save_node2vec_result, train_node2vec


def build_parser() -> argparse.ArgumentParser:
    """Construct the documented command-line interface."""
    parser = argparse.ArgumentParser(description="Train Node2Vec on a bounded inverse Collatz graph.")
    parser.add_argument("--max-node", type=int, default=10_000, help="Inclusive upper bound N for the graph 1..N.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory used to store experiment artifacts.")
    parser.add_argument("--seed", type=int, default=20260722, help="Random seed for walks, shuffling, and initialization.")
    parser.add_argument("--dimensions", type=int, default=128, help="Embedding coordinates per node.")
    parser.add_argument("--walk-length", type=int, default=40, help="Maximum nodes in each biased random walk.")
    parser.add_argument("--walks-per-node", type=int, default=2, help="Random walks generated from each graph node.")
    parser.add_argument("--context-size", type=int, default=5, help="Skip-gram window radius around a walk position.")
    parser.add_argument("--p", type=float, default=1.0, help="Node2Vec return parameter.")
    parser.add_argument("--q", type=float, default=1.0, help="Node2Vec in/out exploration parameter.")
    parser.add_argument("--negative-samples", type=int, default=5, help="Negative contexts per positive pair.")
    parser.add_argument("--epochs", type=int, default=3, help="Passes over the generated skip-gram pairs.")
    parser.add_argument("--learning-rate", type=float, default=0.025, help="SGD step size.")
    parser.add_argument("--batch-size", type=int, default=512, help="Pairs processed per optimization batch.")
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto", help="auto selects CUDA PyTorch when available.")
    parser.add_argument("--forward-walks", action="store_true", help="Follow stored forward Collatz edges instead of inverse edges.")
    return parser


def main() -> None:
    """Build graph, train, persist artifacts, and print the matrix location."""
    args = build_parser().parse_args()
    graph_result = build_inverse_graph_up_to(args.max_node, show_progress=True)
    config = Node2VecConfig(
        dimensions=args.dimensions, walk_length=args.walk_length, walks_per_node=args.walks_per_node,
        context_size=args.context_size, p=args.p, q=args.q, negative_samples=args.negative_samples,
        epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed, backend=args.backend,
        follow_reverse=not args.forward_walks, batch_size=args.batch_size,
    )
    result = train_node2vec(graph_result.graph, config, show_progress=True)
    run_directory = args.output_dir / f"node2vec_N{args.max_node}_seed{args.seed}"
    paths = save_node2vec_result(
        result, run_directory,
        extra_metadata={"graph": {"domain_maximum": args.max_node, "edge_orientation": "forward Collatz u -> T(u)", "walk_orientation": "inverse" if config.follow_reverse else "forward"}},
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    print(f"embedding_matrix_shape={result.embeddings.shape}")


if __name__ == "__main__":
    main()
