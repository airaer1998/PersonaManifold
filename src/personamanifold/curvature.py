"""Exact edge Ollivier--Ricci curvature for small demonstration graphs."""

import numpy as np
from scipy.optimize import linprog

from .geometry import _integer


def edge_curvature(manifold, source, target, idleness=0.0):
    """Compute 1 - W1(mu_source, mu_target) / graph_distance(source, target).

    mu is uniform over union-kNN graph neighbors, with optional mass at the node.
    Uses shortest-path ground costs and an exact transport LP, not Euclidean
    costs. This dense illustrative implementation is expensive for large graphs.
    """
    manifold._check_fitted()
    n = len(manifold.points_)
    source = _integer(source, "source", 0, n - 1)
    target = _integer(target, "target", 0, n - 1)
    if source == target or manifold.graph_[source, target] <= 0:
        raise ValueError("source and target must form a graph edge")
    if not np.isfinite(idleness) or not 0 <= idleness < 1:
        raise ValueError("idleness must be in [0, 1)")

    def measure(node):
        neighbors = manifold.graph_.getrow(node).indices
        support = np.r_[node, neighbors]
        mass = np.r_[idleness, np.full(len(neighbors), (1 - idleness) / len(neighbors))]
        return support, mass

    left, a = measure(source)
    right, b = measure(target)
    distances = manifold.distances()
    costs = distances[np.ix_(left, right)]
    rows, cols = costs.shape
    constraints = np.vstack((np.kron(np.eye(rows), np.ones((1, cols))), np.kron(np.ones((1, rows)), np.eye(cols))))
    result = linprog(costs.ravel(), A_eq=constraints, b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"transport solver failed: {result.message}")
    return float(1 - result.fun / distances[source, target])
