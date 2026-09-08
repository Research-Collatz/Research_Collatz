#!/usr/bin/env python3
"""
Complete Reproducible Pipeline for Inverse Collatz Graph Analysis
==================================================================

This script reproduces the entire analysis from scratch:
1. Build inverse Collatz graphs at multiple scales
2. Train Node2Vec embeddings with multiple seeds
3. Compute geometric quantities
4. Cross-scale convergence analysis
5. Supervised prediction and clustering
6. Generate all figures and tables

Usage:
    python run_all.py

Requirements:
    pip install -r requirements.txt -r requirements-ml.txt
"""

import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], description: str) -> bool:
    """Run a command and report status."""
    print(f'\n{"="*60}')
    print(f'STEP: {description}')
    print(f'CMD: {shlex.join(cmd)}')
    print(f'{"="*60}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'ERROR: {result.stderr}')
        return False
    print(result.stdout)
    return True

def main():
    # Ensure output directories exist
    for d in ['outputs/scaling', 'Figures', 'Tables', 'Results', 'Embeddings', 'Graphs']:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Build graphs and compute features at all scales
    scales = [10000, 50000, 100000]
    seeds = [20260722, 20260723, 20260724]
    
    for scale in scales:
        print(f'\n{"#"*60}')
        print(f'# PROCESSING SCALE N={scale}')
        print(f'{"#"*60}')
        
        # Build graph and compute features
        cmd = [
            sys.executable,
            "-m",
            "collatz_graph.bounded_cli",
            "--max-node",
            str(scale),
            "--output-dir",
            "outputs/scaling",
        ]
        if not run_cmd(cmd, f'Build graph N={scale}'):
            return False
        
        # Compute node features
        feature_code = (
            "from collatz_graph.bounded import build_inverse_graph_up_to; "
            "from collatz_graph.features import compute_node_features, save_node_features; "
            f"g = build_inverse_graph_up_to({scale}, show_progress=True); "
            "f = compute_node_features(g.graph, root=1, show_progress=True); "
            f"save_node_features(f, 'outputs/scaling/features_N{scale}')"
        )
        cmd = [sys.executable, "-c", feature_code]
        if not run_cmd(cmd, f'Compute features N={scale}'):
            return False
        
        # Compute graph statistics
        cmd = [
            sys.executable,
            "-m",
            "collatz_graph.statistics_cli",
            "--max-node",
            str(scale),
            "--output-dir",
            f"outputs/scaling/statistics_N{scale}",
        ]
        if not run_cmd(cmd, f'Graph statistics N={scale}'):
            return False
    
    # Step 2: Train Node2Vec embeddings for all scale/seed combinations
    print(f'\n{"#"*60}')
    print('# TRAINING NODE2VEC EMBEDDINGS')
    print(f'{"#"*60}')
    
    for scale in scales:
        for seed in seeds:
            cmd = [
                sys.executable,
                "-m",
                "collatz_graph.node2vec_cli",
                "--max-node",
                str(scale),
                "--seed",
                str(seed),
                "--backend",
                "torch",
                "--output-dir",
                "outputs/scaling",
            ]
            if not run_cmd(cmd, f'Node2Vec N={scale} seed={seed}'):
                return False
    
    # Step 3: Run full scaling study
    print(f'\n{"#"*60}')
    print('# RUNNING SCALING STUDY')
    print(f'{"#"*60}')
    
    cmd = [
        sys.executable,
        "-m",
        "collatz_graph.scaling_cli",
        "--output-root",
        "outputs/scaling",
        "--stages",
        "embed",
        "statistics",
        "umap",
        "cluster",
        "supervised",
        "analysis",
        "--scales",
        "10000",
        "50000",
        "100000",
        "--seeds",
        "20260722",
        "20260723",
        "20260724",
    ]
    if not run_cmd(cmd, 'Full scaling study'):
        return False
    
    # Step 4: Compute geometric quantities on N=10000 reference
    print(f'\n{"#"*60}')
    print('# COMPUTING GEOMETRIC QUANTITIES')
    print(f'{"#"*60}')
    
    cmd = [sys.executable, "compute_geometric_quantities.py"]
    if not run_cmd(cmd, 'Geometric quantities'):
        return False
    
    # Step 5: Generate all figures and tables
    print(f'\n{"#"*60}')
    print('# GENERATING FIGURES AND TABLES')
    print(f'{"#"*60}')
    
    cmd = [sys.executable, "generate_all_figures.py"]
    if not run_cmd(cmd, 'Generate figures'):
        return False
    
    cmd = [sys.executable, "generate_all_tables.py"]
    if not run_cmd(cmd, 'Generate tables'):
        return False
    
    # Step 6: Organize outputs
    print(f'\n{"#"*60}')
    print('# ORGANIZING OUTPUTS')
    print(f'{"#"*60}')
    
    cmd = [sys.executable, "organize_outputs.py"]
    if not run_cmd(cmd, 'Organize outputs'):
        return False
    
    print(f'\n{"#"*60}')
    print('PIPELINE COMPLETED SUCCESSFULLY!')
    print(f'{"#"*60}')
    print('Outputs available in:')
    print('  Figures/   - All SVG + PDF figures')
    print('  Tables/    - All CSV + LaTeX tables')
    print('  Results/   - Computed results')
    print('  Embeddings/ - Node2Vec embeddings')
    print('  Graphs/    - Graph statistics')
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)