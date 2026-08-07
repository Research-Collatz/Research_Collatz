"""CLI for publication-quality UMAP plots of a saved Node2Vec run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .embedding_plots import create_umap_figures

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved inverse-Collatz Node2Vec embedding with UMAP.")
    parser.add_argument("run_directory", type=Path, help="Directory containing Node2Vec artifacts.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to RUN_DIRECTORY/umap.")
    args = parser.parse_args()
    print(json.dumps({name: str(path) for name, path in create_umap_figures(args.run_directory, args.output_dir).items()}, indent=2))

if __name__ == "__main__":
    main()
