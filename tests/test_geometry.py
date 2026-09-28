import numpy as np
import pytest
from scipy.spatial.distance import cdist

from personamanifold import PersonaManifold, center_activations
from personamanifold.curvature import edge_curvature


def arc():
    theta = np.linspace(0, 1.8 * np.pi, 80)
    return np.column_stack((np.cos(theta), np.sin(theta)))


def test_neutral_subtraction_uses_matching_probes():
    neutral = np.array([[3., 4.], [7., 2.]])
    offsets = np.array([[1., 2.], [-1., 3.]])
    responses = neutral[None] + offsets[:, None]
    np.testing.assert_allclose(center_activations(responses, neutral), offsets)
    with pytest.raises(ValueError, match="match"):
        center_activations(responses, neutral[:1])


def test_pca_roundtrip_and_metric_edge_formula():
    points = np.array([[0., 0.], [1., 0.], [0., 2.], [1., 2.], [2., 1.]]) + [20, -7]
    m = PersonaManifold(n_neighbors=3, tangent_dim=2, variance=1, ridge=.1).fit(points)
    np.testing.assert_allclose(m.to_activation(m.points_), points, atol=1e-12)
    delta = m.points_[1] - m.points_[0]
    norms = []
    for node in (0, 1):
        offsets = m.points_[m.neighbors_[node]] - m.points_[node]
        covariance = offsets.T @ offsets / 3
        inverse = np.linalg.inv(covariance + .1 * np.eye(2))
        np.testing.assert_allclose(m.metrics_[node], inverse, atol=1e-12)
        norms.append(np.sqrt(delta @ inverse @ delta))
    assert m.graph_[0, 1] == pytest.approx(sum(norms) / 2)
    np.testing.assert_allclose(m.graph_.toarray(), m.graph_.toarray().T)


def test_geodesic_follows_curve_and_preserves_endpoints():
    points = arc()
    m = PersonaManifold(n_neighbors=4, tangent_dim=1, variance=1).fit(points)
    samples = m.to_activation(m.interpolate(0, 79, n_steps=101))
    np.testing.assert_allclose(samples[[0, -1]], points[[0, -1]], atol=1e-12)
    assert len(m.shortest_path(0, 79)) > 20
    assert cdist(samples, points).min(axis=1).max() < .05
    assert np.linalg.norm(samples[50] - np.mean(points[[0, -1]], axis=0)) > 1.5
    distances = m.distances()
    np.testing.assert_allclose(distances, distances.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(distances), 0)
    vectors = m.steering_vectors(0, 79, strength=.5)
    np.testing.assert_allclose(vectors[[0, -1]], .5 * points[[0, -1]], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), .5, atol=1e-12)


def test_disconnected_graph_is_not_silently_bridged():
    points = np.array([[0.], [.1], [10.], [10.1]])
    m = PersonaManifold(n_neighbors=1, tangent_dim=1).fit(points)
    assert np.isinf(m.distances()[0, 3])
    with pytest.raises(ValueError, match="disconnected"):
        m.interpolate(0, 3)


def test_constant_path_and_validation():
    m = PersonaManifold(n_neighbors=4, tangent_dim=1).fit(arc())
    expected = np.repeat(m.points_[[5]], 3, axis=0)
    np.testing.assert_allclose(m.interpolate(5, 5, 3), expected)
    for invalid in (-1, 80, 1.5, True):
        with pytest.raises(ValueError):
            m.shortest_path(invalid, 2)
    with pytest.raises(ValueError, match="variation"):
        PersonaManifold(n_neighbors=2).fit(np.ones((5, 2)))
    with pytest.raises(ValueError, match="duplicate"):
        PersonaManifold(n_neighbors=2, tangent_dim=1).fit([[0.], [1.], [1.]])
    with pytest.raises(ValueError, match="fit"):
        PersonaManifold().distances()


def test_curvature_of_equilateral_triangle():
    points = np.array([[0., 0.], [1., 0.], [.5, np.sqrt(3) / 2]])
    m = PersonaManifold(n_neighbors=2, tangent_dim=2, variance=1).fit(points)
    # Two uniform neighbor measures share half their mass; moving the other
    # half across one edge costs d/2, so Ollivier curvature is 1/2.
    assert edge_curvature(m, 0, 1) == pytest.approx(.5)
    assert edge_curvature(m, 1, 0) == pytest.approx(.5)
