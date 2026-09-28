# Method notes

The geometry and evaluation components use NumPy and SciPy.

## Representations and PCA

For each persona, `center_activations` averages the difference between that
persona's response activations and the neutral response activations for the same
probes. The caller extracts the residual stream at the last generated response
token of a selected layer; prompt-only activations are a different measurement.

`fit` performs centered PCA using SVD. Components are orthonormal, and
`to_activation` restores both the retained projection and the empirical mean.
Discarded directions cannot be recovered. Steering endpoints therefore equal
the PCA-reconstructed inputs, not necessarily the original unprojected inputs.

## Local metrics and edges

For point `i`, use offsets of its `k` nearest neighbors from `i` (not offsets
from the neighbors' mean) to form `C_i = offsets.T @ offsets / k`. Retain the
largest `tangent_dim` eigenpairs and compute

```text
g_i = sum_a v_a v_a^T / (lambda_a + ridge)
w_ij = (sqrt(delta^T g_i delta) + sqrt(delta^T g_j delta)) / 2
```

The graph contains an undirected edge if either endpoint selects the other.
Shortest paths use these weights. All-pairs distances retain infinity between
components; interpolating between disconnected endpoints raises an error.
Duplicate projected points and zero-weight edges are rejected explicitly.

The tangent dimension is user-specified. Metric distances have a different scale
from ambient Euclidean distance: do not interpret their raw ratio as a pure
curvature estimate without controlling for metric anisotropy.

## Steering

Shortest-path vertices are sampled at equal cumulative **graph-weight** intervals.
Within an edge the code uses linear interpolation. The resulting polyline
segments can leave the underlying continuous manifold.

Inverse PCA gives neutral-relative activation vectors. Each direction is
normalized, then scaled by a linear interpolation of the reconstructed endpoint
norms and the requested strength. This is an explicit activation-space convention
for centered PCA. A zero direction raises an error rather than producing NaNs.

The package returns vectors. The caller must choose a compatible residual-stream
hook and confirm layer, token positions, device, dtype, and generation behavior.


## Curvature

`edge_curvature` computes Ollivier–Ricci curvature using an exact transport linear
program. The ground cost is graph shortest-path distance. Neighbor measures are
uniform on the **symmetrized graph**; optional idleness places probability mass at
the node itself. The default idleness is zero. Different measure conventions can
change curvature, so report the convention with results.

This helper solves one edge at a time, recomputing graph distances. Its runtime
depends on the graph size and the number of neighbors at each endpoint.

## BST-style evaluation

Behavioral distance combines mean forced-choice **disagreement** (smaller means
more similar) and mean per-question cosine distance between open-ended responses.
The default choice weight is one third. For choice-only/embedding-only distances,
set beta to 1/0 respectively (both input arrays are still required).

`triplet_accuracy` measures the fraction of strict inequalities
`d(anchor, positive) < d(anchor, negative)`. Ties count as incorrect. Invalid
indices, repeated IDs within a triplet, and nonfinite evaluated distances are
errors. Labels must come from independent behavioral observations; using a
graph's own nearest neighbors as its evaluation labels is circular.

The synthetic example uses known arc coordinates as independent labels.

## Scale

PCA, pairwise distances, and local metric tensors are dense. Storage includes
O(N²) distances and O(N d²) metrics in the retained dimension. Start with a small
subset; the compact implementation prioritizes readability over large-scale
memory and compute efficiency.
