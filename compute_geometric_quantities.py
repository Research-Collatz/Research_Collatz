from __future__ import annotations

import argparse
from pathlib import Path

from collatz_graph.bounded import build_inverse_graph_up_to
from collatz_graph.features import compute_node_features
from collatz_graph.geometric import add_geometric_quantities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute geometric descriptors for a bounded inverse Collatz graph."
    )
    parser.add_argument("--n", type=int, default=10_000, help="Inclusive graph domain maximum.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/scaling/features_with_geometry.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    print("Building graph and computing base features...")
    graph_result = build_inverse_graph_up_to(args.n, show_progress=True)
    features = compute_node_features(graph_result.graph, root=1, show_progress=True)
    feature_data = add_geometric_quantities(graph_result.graph, features.data)

    print(
        f"Graph: {graph_result.graph.number_of_nodes()} nodes, "
        f"{graph_result.graph.number_of_edges()} edges"
    )
    print(f"Features: {feature_data.shape}")
    for column in (
        "neighborhood_overlap_curvature",
        "branching_entropy",
        "ancestor_density_k2",
        "local_potential",
        "flow_energy",
    ):
        values = feature_data[column]
        print(f"  {column}: mean={values.mean():.4f}, std={values.std():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_data.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(f"Columns: {feature_data.columns.tolist()}")


if __name__ == "__main__":
    main()