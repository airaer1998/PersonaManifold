"""PCA, local metrics, graph geodesics, and steering vectors for small datasets."""

import operator

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist


def _integer(value, name, minimum, maximum):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _finite_array(values, ndim, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != ndim or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty, finite {ndim}D array")
    return values


def center_activations(persona, neutral):
    """Mean neutral-subtracted response activations: (N, M, D), (M, D) -> (N, D).

    Each probe must occupy the same position in both arrays. Extract the residual
    stream at the last generated response token before calling this function.
    """
    persona = _finite_array(persona, 3, "persona")
    neutral = _finite_array(neutral, 2, "neutral")
    if persona.shape[1:] != neutral.shape:
        raise ValueError("neutral must match the probe and hidden dimensions")
    return (persona - neutral[None, :, :]).mean(axis=1)


class PersonaManifold:
    """Fit a local-metric kNN graph to neutral-relative persona vectors.

    This reference deliberately uses dense PCA/pairwise distances, a user-chosen
    tangent dimension, and polyline paths. It is intended for small datasets,
    not paper-scale experiment reproduction. Disconnected components stay
    disconnected rather than being joined with artificial shortcut edges.
    """

    def __init__(self, n_neighbors=8, variance=0.99, tangent_dim=2, ridge=1e-3):
        if not np.isfinite(variance) or not 0 < variance <= 1:
            raise ValueError("variance must be in (0, 1]")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        self.n_neighbors = n_neighbors
        self.variance = variance
        self.tangent_dim = tangent_dim
        self.ridge = ridge

    def fit(self, vectors):
        vectors = _finite_array(vectors, 2, "vectors")
        n = len(vectors)
        k = _integer(self.n_neighbors, "n_neighbors", 1, n - 1)
        rank = _integer(self.tangent_dim, "tangent_dim", 1, vectors.shape[1])
        mean = vectors.mean(axis=0)
        centered = vectors - mean
        _, singular, components = np.linalg.svd(centered, full_matrices=False)
        energy = singular**2
        if energy.sum() <= np.finfo(float).tiny:
            raise ValueError("vectors must contain variation")
        # Exclude numerically null PCA axes, including for variance=1.
        effective_rank = int(np.sum(singular > singular[0] * max(centered.shape) * np.finfo(float).eps))
        kept = min(np.searchsorted(np.cumsum(energy) / energy.sum(), self.variance) + 1, effective_rank)
        if rank > kept:
            raise ValueError(f"tangent_dim={rank} exceeds retained PCA dimension {kept}")
        basis = components[:kept]
        points = centered @ basis.T
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        if np.any(distances <= np.finfo(float).eps * max(1.0, np.linalg.norm(points))):
            raise ValueError("duplicate points after PCA; deduplicate or retain more variance")
        neighbors = np.argsort(distances, axis=1, kind="stable")[:, :k]
        metrics = []
        for i, ids in enumerate(neighbors):
            offsets = points[ids] - points[i]
            covariance = offsets.T @ offsets / k
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            tangent = eigenvectors[:, -rank:]
            metrics.append((tangent / (eigenvalues[-rank:] + self.ridge)) @ tangent.T)
        metrics = np.asarray(metrics)
        # Union symmetrization: one undirected edge if either node selects the other.
        edges = sorted({tuple(sorted((i, int(j)))) for i, ids in enumerate(neighbors) for j in ids})
        rows, cols, weights = [], [], []
        for i, j in edges:
            delta = points[j] - points[i]
            a = np.sqrt(max(0.0, float(delta @ metrics[i] @ delta)))
            b = np.sqrt(max(0.0, float(delta @ metrics[j] @ delta)))
            weight = (a + b) / 2
            if weight <= np.finfo(float).eps:
                raise ValueError("degenerate metric edge; increase tangent_dim or adjust neighbors")
            rows.extend((i, j))
            cols.extend((j, i))
            weights.extend((weight, weight))
        # Commit state only after a successful fit.
        self.mean_ = mean
        self.components_ = basis
        self.points_ = points
        self.metrics_ = metrics
        self.neighbors_ = neighbors
        self.graph_ = csr_matrix((weights, (rows, cols)), shape=(n, n))
        return self

    def _check_fitted(self):
        if not hasattr(self, "graph_"):
            raise ValueError("call fit before querying the manifold")

    def distances(self):
        """All-pairs graph distances; disconnected pairs have distance infinity."""
        self._check_fitted()
        return dijkstra(self.graph_, directed=False)

    def shortest_path(self, start, end):
        """Return node indices along a shortest path, including both endpoints."""
        self._check_fitted()
        n = len(self.points_)
        start = _integer(start, "start", 0, n - 1)
        end = _integer(end, "end", 0, n - 1)
        distances, parents = dijkstra(self.graph_, directed=False, indices=start, return_predecessors=True)
        if not np.isfinite(distances[end]):
            raise ValueError("endpoints are disconnected; inspect data or increase n_neighbors")
        path = [end]
        while path[-1] != start:
            path.append(int(parents[path[-1]]))
        return np.asarray(path[::-1], dtype=int)

    def interpolate(self, start, end, n_steps=11):
        """Sample a graph polyline at equal weighted arc length, in PCA space.

        This is the release's explicit substitute for the paper's local-tangent
        cubic spline. Each sample stays on a graph edge; no spline overshoot.
        """
        n_steps = _integer(n_steps, "n_steps", 2, 100000)
        path = self.shortest_path(start, end)
        vertices = self.points_[path]
        if len(path) == 1:
            return np.repeat(vertices, n_steps, axis=0)
        lengths = np.array([self.graph_[a, b] for a, b in zip(path[:-1], path[1:])])
        arc = np.r_[0.0, np.cumsum(lengths)]
        samples = np.linspace(0, arc[-1], n_steps)
        return np.column_stack([np.interp(samples, arc, vertices[:, j]) for j in range(vertices.shape[1])])

    def to_activation(self, points):
        """Invert centered PCA; restore the mean to obtain neutral-relative vectors."""
        self._check_fitted()
        points = _finite_array(points, 2, "points")
        if points.shape[1] != len(self.components_):
            raise ValueError("points must have the retained PCA dimension")
        return points @ self.components_ + self.mean_

    def steering_vectors(self, start, end, n_steps=11, strength=1.0):
        """Activation-space path directions with linearly interpolated endpoint norms.

        Add one returned vector to a chosen residual stream outside this package.
        Model hooks, generation, and layer/strength calibration are caller-owned.
        """
        if not np.isfinite(strength):
            raise ValueError("strength must be finite")
        points = self.to_activation(self.interpolate(start, end, n_steps))
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError("zero activation direction on path; choose different endpoints")
        scales = np.linspace(norms[0], norms[-1], n_steps)
        return strength * points / norms[:, None] * scales[:, None]
