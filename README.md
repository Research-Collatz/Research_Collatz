# Inverse Collatz Graph

A reproducible Python research project for exploratory computation on the inverse Collatz graph.

## Mathematical convention

The forward Collatz map is

\[
T(n) = \begin{cases} n/2, & n \equiv 0 \pmod 2,\\ 3n+1, & n \equiv 1 \pmod 2.\end{cases}
\]

The directed graph stores forward edges `u -> T(u)`. Inverse exploration starts from a target node and enumerates its valid predecessors:

- `2n` is always a predecessor because `T(2n) = n`.
- `(n - 1) / 3` is a predecessor only when it is a positive odd integer.

All arithmetic is integer-exact. Graph expansion is bounded by an explicit node budget; the budget is a computational truncation, not a mathematical claim about the infinite graph.

## Layout

- `collatz_graph/`: reusable typed package code.
- `outputs/`: generated node, edge, and metadata artifacts (ignored by Git).
- `notebooks/`: Colab-compatible experiment notebooks.
- `tests/`: focused correctness and regression tests.
- `requirements.txt`: runtime and test dependencies.
- `pyproject.toml`: package metadata and pytest configuration.

## Google Colab

From a fresh Colab runtime, clone the repository and install dependencies:

```python
!git clone <repository-url>
%cd Research_Colatz
!pip install -r requirements.txt
```

Then open `notebooks/01_inverse_graph_exploration.ipynb`. The notebook inserts the repository root into `sys.path`, so package imports are explicit and reusable.

## Build and export the finite graph on `1..N`

The bounded-domain API stores forward-oriented edges `u -> T(u)` only when
both endpoints are in `1..N`. It creates nodes from a `range` iterator and
evaluates one successor per source, so construction takes `O(N)` time. If
`E_N` edges remain inside the domain, NetworkX storage is `O(N + E_N)` and
`E_N <= N`. Export uses NetworkX node and edge views directly, avoiding
duplicate edge or node lists in memory.

```python
from collatz_graph import build_and_save_inverse_graph

result = build_and_save_inverse_graph(100_000, "outputs", show_progress=True)
print(result.runtime_seconds)
```

This writes `nodes_100000.csv`, `edges_100000.csv`, and
`metadata_100000.json`. Metadata records the domain, edge orientation, node
and edge counts, measured runtime, Python version, and NetworkX version.

The finite graph is an induced subgraph of the positive-integer Collatz
graph. An odd source can map above `N`; that edge is intentionally omitted,
not clipped. This distinction matters when interpreting boundary degree
statistics.

## Node features

For a node `n`, the feature extractor in `collatz_graph.features` returns a
Pandas DataFrame with one row per node:

- `stopping_time`: the least `k >= 1` such that `T^k(n) < n`; it is `0` for
	the root `1`. This is the usual stopping time, not necessarily the time to
	reach `1`.
- `total_stopping_time`: the least `k >= 0` such that `T^k(n) = 1`, when the
	trajectory reaches `1`; the implementation reports the computed trajectory
	length for finite experiments.
- `maximum_excursion`: `max(T^j(n) : 0 <= j <= total_stopping_time)`, including
	the starting value.
- `binary_length`: `floor(log_2(n)) + 1`, the number of bits in the ordinary
	binary representation of `n`.
- `parity_vector_length`: the number of parity decisions in the trajectory to
	`1`; with a parity vector containing one bit per transition, this equals
	`total_stopping_time`.
- `in_degree`: the number of retained forward predecessors in the finite
	graph.
- `out_degree`: zero or one, indicating whether `T(n)` remains in `1..N`.
- `distance_from_root`: shortest directed distance from `n` to root `1` in the
	finite graph; `-1` indicates that truncation removes a required edge or the
	node cannot reach the root.
- `ancestor_count`: the number of nodes whose finite forward orbit reaches
	the node, including the node itself. Nodes in the same directed cycle share
	the component's total count.
- `branching_factor`: inverse branching, defined here as the number of
	retained predecessors; therefore it equals `in_degree` under forward edge
	orientation.

Compute and save the table with:

```python
from collatz_graph import compute_node_features, save_node_features

features = compute_node_features(graph, show_progress=True)
save_node_features(features, "outputs/node_features_100000.csv")
```

The structural passes are linear in `N + E_N`; arithmetic work is linear in
the number of trajectory transitions evaluated. Fixed-width NumPy arrays,
memoization for values inside `1..N`, and streamed progress/export handling
keep avoidable memory overhead low for million-node experiments. NetworkX and
the final Pandas DataFrame still require memory proportional to the graph and
table, respectively.

## Descriptive graph statistics

`collatz_graph.statistics` computes paper-ready tables for degree structure,
centrality, components, tree depth, and spectral diagnostics. For the
forward-oriented graph, in-degree is inverse branching: it counts retained
predecessors of a node. Out-degree is at most one because the Collatz map is a
function. The degree-distribution table reports joint in/out-degree counts and
proportions.

The centrality table contains normalized in-degree and out-degree centrality,
PageRank, closeness, and betweenness. PageRank is a damped random-walk score;
high values identify nodes receiving probability mass from many predecessor
paths. Closeness is NetworkX's directed inbound-distance convention, so it
measures how easily other nodes can reach a node. Betweenness measures the
fraction of shortest paths passing through a node. For graphs larger than
2,000 nodes, betweenness defaults to a reproducible sample of 512 source
nodes; the method and seed are recorded in the summary table.

Weak components ignore edge direction and describe undirected connectivity;
strong components require directed reachability in both directions. A finite
domain has many boundary components because edges leaving `1..N` are omitted.
Tree depth is shortest distance from root `1` in the reversed graph, meaning
inverse-generation depth. The reported diameter is the exact undirected
diameter of the largest weak component when that component is below the
configured feasibility limit. Directed diameter is generally infinite for
this graph and is not substituted silently.

The spectral table reports an edge-wise power-iteration estimate of the
adjacency spectral radius, not an exact dense eigendecomposition, together
with the exact adjacency Frobenius norm and maximum out-degree. This avoids
forming an `N x N` matrix and is suitable for sparse million-node graphs.

Generate all tables and figures:

```python
from collatz_graph import (
	build_inverse_graph_up_to,
	compute_graph_statistics,
	save_publication_figures,
	save_statistics_tables,
)

graph = build_inverse_graph_up_to(100_000, show_progress=True).graph
statistics = compute_graph_statistics(graph, show_progress=True)
save_statistics_tables(statistics, "outputs/statistics_100000", stem="graph_100000")
save_publication_figures(graph, statistics, "outputs/statistics_100000")
```

## Local verification

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

## Node2Vec embeddings on the inverse graph

The package trains skip-gram Node2Vec embeddings on the bounded graph.  The
stored graph orientation is forward (`u -> T(u)`), while Node2Vec walks are
**inverse by default**, moving from a number to its retained predecessors.
This is the appropriate direction for representing the inverse Collatz tree.

Run a reproducible 128-dimensional experiment (CUDA is selected automatically
when PyTorch is installed and a GPU is available):

```powershell
python -m collatz_graph.node2vec_cli --max-node 10000 --seed 20260722 --backend auto
```

Each run is stored at `outputs/node2vec_N10000_seed20260722/` with:

- `node2vec_embeddings.npy`: the returned `float32` matrix `(n_nodes, 128)`.
- `node2vec_nodes.csv`: row-to-node mapping; row `i` in the matrix represents
  this file's `i`th node.
- `node2vec_training.csv` and `node2vec_training_loss.png`: per-epoch loss and
  a publication-resolution convergence plot.
- `node2vec_metadata.json`: exact hyperparameters, graph and walk orientation,
  backend/device, versions, and runtime for later experiments.

Programmatic use returns the matrix directly:

```python
from collatz_graph import build_inverse_graph_up_to, train_node2vec

graph = build_inverse_graph_up_to(10_000, show_progress=True).graph
result = train_node2vec(graph, show_progress=True)
embedding_matrix = result.embeddings  # shape: (10_000, 128)
```

### Hyperparameters

`dimensions=128` is the number of learned coordinates per node. `seed=20260722`
initializes walk sampling, pair shuffling, and model weights reproducibly.
`walk_length=40` caps the nodes visited in one walk, and `walks_per_node=2`
controls the sampling coverage of every node. `context_size=5` is the
skip-gram window radius; it creates positive center/context pairs from nodes
that occur within five positions in a walk.

`p=1.0` is the return parameter: lower values make an immediate backtrack more
likely, while higher values discourage it. `q=1.0` is the exploration
parameter: lower values bias toward nodes farther from the previous node
(outward/DFS-like behavior), while higher values favor local/BFS-like walks.
Both defaults make the second-order walk unbiased apart from graph structure.

`negative_samples=5` draws five noise contexts for every observed pair. Noise
nodes are sampled in proportion to total walk-graph degree raised to `0.75`,
the standard word2vec/Node2Vec smoothing rule. `epochs=3` is the number of SGD
passes over generated pairs, `learning_rate=0.025` is the SGD step size, and
`batch_size=512` bounds the in-memory positive-pair batch and trades memory for
optimization throughput; walks and pairs are streamed rather than retained for
the complete training run. `backend="auto"` uses PyTorch/CUDA when available
and otherwise uses deterministic NumPy CPU training. `follow_reverse=True` selects inverse walks; the CLI's
`--forward-walks` explicitly changes this. 

### UMAP embedding figures

Create reproducible two-dimensional publication figures from a saved run:

```powershell
python -m collatz_graph.embedding_plots_cli outputs/node2vec_N10000_seed20260722
```

One UMAP layout is shared by all five figures (`n_neighbors=30`,
`min_dist=0.15`, cosine metric, and seed `20260722`), so locations are
comparable between colours. `level_set` means the inverse-tree breadth-first
distance from node 1. Nodes outside the finite root component receive `-1` and
are rendered in grey. The command saves 450-DPI PNGs, vector PDFs, UMAP
coordinates, and their aligned feature table under the run's `umap/` folder.

## Reproducibility and experimental limitations

Graph results depend on the selected root set and node budget. A bounded inverse graph can omit predecessors beyond the budget, so degree and depth statistics are censored. Layout coordinates are visualization artifacts; the plotting helper uses a fixed NetworkX seed for stable figures but those coordinates have no mathematical meaning. Report configuration values, package version, Python version, and dependency versions with published results.

## Development quality checks

Install development dependencies and run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
mypy collatz_graph
```

Validation on 2026-09-07: editable installation with the `dev` extra succeeds,
and the full test suite passes (`25 passed`, with two existing sklearn future
warnings). Ruff passes for the files changed by the current PR. Repository-wide
Ruff still reports pre-existing violations elsewhere. Mypy reports the same 68
pre-existing errors as `origin/main`.

### Bounded inverse-graph truncation

The node-bounded builder treats `max_nodes` as a hard admission limit. Once the limit is reached, undiscovered predecessors are skipped and traversal continues with already-admitted nodes. This avoids prematurely terminating the entire traversal while keeping the result within the requested node budget.

The result remains a finite computational approximation. Degree, depth, connectivity, and embedding results can be affected by omitted predecessors or boundary nodes and must not be interpreted as properties of the infinite Collatz graph.

### Reproducibility

Graph and Node2Vec artifact metadata record the Git commit SHA, Python and
package versions, operating system, machine and processor details, CPU count,
and optional PyTorch/CUDA/GPU information. Node2Vec artifacts also snapshot
the complete training configuration. A fixed random seed improves
reproducibility but does not guarantee bit-for-bit equality across different
hardware, operating systems, CUDA/PyTorch versions, or numerical libraries.
