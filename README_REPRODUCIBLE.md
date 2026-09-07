# Inverse Collatz Graph Analysis - Reproducible Pipeline

## Project Structure
`
Figures/          # All publication-quality figures (SVG + PDF)
Tables/           # All tables (CSV + LaTeX)
Results/          # Computed results and intermediate data
Embeddings/       # Node2Vec embeddings for all scales/seeds
Graphs/           # Graph statistics for all scales
`

## Key Findings

### Embedding Geometry
- 128-dimensional Node2Vec embeddings are ISOTROPIC (participation ratio = 126.4/128)
- Effective rank = 127.2/128 - no dimensional collapse
- Eigenvalue spectrum nearly flat (max/min ratio = 1.54)

### Cross-Scale Convergence
- Latent geometry stabilizes by N=10,000
- Cross-scale ratios within 1-sigma of within-scale noise floor
- Embedding stability: knn_overlap_k10 ratios = 1.02, 1.07, 0.93
- Cluster stability: ARI ratios approx 1.0

### Geometric Quantities (N=10,000)

| Quantity | Stopping Time r | Best Target | Embedding max|r|
|----------|----------------|-------------|-------------|
| Neighborhood-overlap descriptor | -0.206 | ancestor_count (0.43) | 0.024 |
| Branching Entropy | -0.125 | branching_factor (0.64) | 0.039 |
| Ancestor Density (k=2) | +0.088 | ancestor_count (-0.70) | 0.028 |
| Local Potential | -0.192 | distance_from_root (0.997) | 0.031 |
| Flow Energy | -0.107 | ancestor_count (0.88) | 0.030 |

### Most Informative: Local Potential
- Near-perfect proxy for graph distance (r=0.997 with distance_from_root)
- 2nd strongest correlation with stopping time
- Defined by summing `log(ancestor_count + 1)` along the finite forward path
- Computed with memoized path resolution on the bounded functional graph

### Best Embedding Alignment
- Branching entropy <-> dim_105 (r=0.039)
- All geometric quantities weakly aligned with embeddings (max |r| < 0.04)
- Node2Vec optimizes random-walk co-occurrence, not intrinsic geometry

## Reproduction
```bash
python run_all.py
```

### Interpretation

The reported relationships are empirical observations on finite,
domain-truncated graphs. They do not constitute proofs about the infinite
Collatz graph. The neighborhood-overlap descriptor is not Ollivier-Ricci
curvature: it does not compute Wasserstein distance between neighborhood
measures.

## Figures Generated (SVG + PDF)
1. fig01_embedding_spectrum - Eigenvalue spectrum & cumulative variance
2. fig02_dimension_variance - Per-dimension variance
3. fig03_geo_distributions - Histograms of 5 geometric quantities
4. fig04_geo_vs_stopping_time - Scatter vs stopping time
5. fig05_geo_vs_distance - Scatter vs distance from root
6. fig06_geo_vs_ancestors - Scatter vs ancestor count
7. fig07_geo_correlation_heatmap - Geo vs targets correlation matrix
8. fig08_geo_vs_embedding_heatmap - Geo vs 128 embedding dims
9. fig09_key_relationships - Perfect correlations (potential/distance, energy/ancestors)
10. fig10_geo_pairwise - Pairwise geo quantities
11. fig11_top_embedding_dims - Top 5 dims per graph property
12. fig12_convergence_summary - Cross-scale convergence ratios
13. fig13_graph_scaling - Graph statistics across N=10k,50k,100k
14. fig14_prediction_scaling - Predictive R^2 scaling

## Tables Generated (CSV + LaTeX)
1. table01_geo_statistics - Descriptive stats for 5 geo quantities
2. table02_geo_correlations - Geo vs targets correlation matrix
3. table03_geo_embedding_correlations - Top 10 embedding dims per geo quantity
4. table04_convergence_summary - Cross-scale convergence ratios
5. table05_graph_scaling - Graph statistics across scales
6. table06_prediction_scaling - Predictive R^2 across scales
7. table07_embedding_geometry - Embedding geometry summary

## Data in Results/
- features_with_geometry.csv - Node features + 5 geometric quantities
- comparison/ - All scaling study comparison tables
- supervised_N*/ - Supervised prediction results per scale
- clustering_N*_seed*/ - Clustering results per scale/seed

## Data in Embeddings/
- node2vec_N{10000,50000,100000}_seed{20260722,20260723,20260724}/ - Embeddings + metadata

## Data in Graphs/
- statistics_N{10000,50000,100000}/ - Graph statistics per scale

## Requirements
- Python 3.10+
- numpy, pandas, networkx, matplotlib, seaborn, scikit-learn, torch, umap-learn
- Install: pip install -r requirements.txt -r requirements-ml.txt
