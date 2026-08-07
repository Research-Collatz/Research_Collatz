#!/usr/bin/env python3
"""
Compute Geometric Quantities on Inverse Collatz Graph
=====================================================

Computes 5 geometric quantities on the N=10,000 inverse Collatz graph:
1. Ollivier-Ricci curvature
2. Branching entropy
3. Ancestor density (k=2)
4. Local potential
5. Flow energy

Outputs: outputs/scaling/features_with_geometry.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import sys
sys.path.insert(0, 'D:/Assignments/Research_Colatz')
from collatz_graph.bounded import build_inverse_graph_up_to
from collatz_graph.features import compute_node_features

# Build graph and base features
print("Building graph and computing base features...")
graph_result = build_inverse_graph_up_to(10000, show_progress=True)
G = graph_result.graph
features = compute_node_features(G, root=1, show_progress=True)
feat_df = features.data

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Features: {feat_df.shape}")

# 1. Ollivier-Ricci Curvature
print("Computing Ollivier-Ricci curvature...")
U = G.to_undirected()
n = G.number_of_nodes()
node_list = sorted(G.nodes())
node_to_idx = {node: i for i, node in enumerate(node_list)}

neighborhoods = {}
for node in G.nodes():
    visited = {node}
    frontier = {node}
    for _ in range(1):
        new_frontier = set()
        for u in frontier:
            new_frontier.update(U.neighbors(u))
        frontier = new_frontier - visited
        visited.update(frontier)
    neighborhoods[node] = list(visited)

ricci = np.zeros(n)
for node in G.nodes():
    idx = node_to_idx[node]
    neighs = list(U.neighbors(node))
    if not neighs:
        ricci[idx] = 0
        continue
    edge_curvs = []
    for nbr in neighs:
        mu_u = set(neighborhoods[node])
        mu_v = set(neighborhoods[nbr])
        inter = len(mu_u & mu_v)
        kappa = 2 * inter / (len(mu_u) + len(mu_v)) if (len(mu_u) + len(mu_v)) > 0 else 0
        edge_curvs.append(kappa)
    ricci[idx] = np.mean(edge_curvs) if edge_curvs else 0

feat_df['ollivier_ricci'] = ricci
print(f"  Ollivier-Ricci: mean={ricci.mean():.4f}, std={ricci.std():.4f}")

# 2. Branching Entropy
print("Computing branching entropy...")
entropy = np.zeros(n)
for node in G.nodes():
    preds = list(G.predecessors(node))
    if len(preds) <= 1:
        entropy[node_to_idx[node]] = 0.0
        continue
    sizes = [feat_df.loc[feat_df['node'] == p, 'ancestor_count'].values[0] for p in preds]
    sizes = np.array(sizes, dtype=float)
    probs = sizes / sizes.sum()
    entropy[node_to_idx[node]] = -np.sum(probs * np.log(probs + 1e-12))

feat_df['branching_entropy'] = entropy
print(f"  Branching entropy: mean={entropy.mean():.4f}, std={entropy.std():.4f}, max={entropy.max():.4f}")

# 3. Ancestor Density (k=2)
print("Computing ancestor density (k=2)...")
density = np.zeros(n)
for node in G.nodes():
    anc = feat_df.loc[feat_df['node'] == node, 'ancestor_count'].values[0]
    visited = {node}
    frontier = {node}
    for _ in range(2):
        new_frontier = set()
        for u in frontier:
            new_frontier.update(G.predecessors(u))
        frontier = new_frontier - visited
        visited.update(frontier)
    volume = len(visited)
    density[node_to_idx[node]] = anc / volume if volume > 0 else 0

feat_df['ancestor_density_k2'] = density
print(f"  Ancestor density: mean={density.mean():.4f}, std={density.std():.4f}")

# 4. Local Potential
print("Computing local potential...")
succ = {}
for u, v in G.edges():
    succ[u] = v

f = {}
for node in G.nodes():
    val = feat_df.loc[feat_df['node'] == node, 'ancestor_count'].values[0]
    f[node] = np.log(val + 1)

potential = np.zeros(n)
for node in G.nodes():
    if node == 1:
        potential[node_to_idx[node]] = 0.0
        continue
    u = 0.0
    cur = node
    visited = set()
    while cur in succ and cur not in visited and cur != 1:
        visited.add(cur)
        u += f[cur]
        cur = succ[cur]
    potential[node_to_idx[node]] = u

feat_df['local_potential'] = potential
print(f"  Local potential: mean={potential.mean():.4f}, std={potential.std():.4f}")

# 5. Flow Energy
print("Computing flow energy...")
energy = np.zeros(n)
for node in G.nodes():
    preds = list(G.predecessors(node))
    if not preds:
        energy[node_to_idx[node]] = 0.0
        continue
    u_val = potential[node_to_idx[node]]
    diffs = [(u_val - potential[node_to_idx[p]])**2 for p in preds]
    energy[node_to_idx[node]] = np.sum(diffs)

feat_df['flow_energy'] = energy
print(f"  Flow energy: mean={energy.mean():.4f}, std={energy.std():.4f}")

# Save
output_path = Path('outputs/scaling/features_with_geometry.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
feat_df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
print(f"Columns: {feat_df.columns.tolist()}")
print("Done!")