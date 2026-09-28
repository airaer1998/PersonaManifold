import numpy as np
import pytest

from personamanifold import behavioral_distances, triplet_accuracy


def test_behavioral_distance_has_correct_orientation_and_weight():
    choices = np.array([[0, 1], [0, 1], [1, 0]])
    embeddings = np.array([[[1., 0.]], [[1., 0.]], [[0., 1.]]])
    distance = behavioral_distances(choices, embeddings)
    np.testing.assert_allclose(distance, [[0, 0, 1], [0, 0, 1], [1, 1, 0]])
    embeddings[2, 0] = [-1, 0]
    assert behavioral_distances(choices, embeddings)[0, 2] == pytest.approx(5 / 3)


def test_triplets_count_ties_as_incorrect():
    distance = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    assert triplet_accuracy(distance, [[0, 1, 2], [1, 0, 2]]) == .5
    assert triplet_accuracy(distance, [[0, 2, 1]]) == 0
    for bad in ([], [[0, 0, 1]], [[0, 1, 3]], [[0., 1., 2.]]):
        with pytest.raises(ValueError):
            triplet_accuracy(distance, bad)
    distance[0, 2] = np.inf
    with pytest.raises(ValueError, match="finite"):
        triplet_accuracy(distance, [[0, 1, 2]])


def test_reject_missing_behavioral_data():
    with pytest.raises(ValueError, match="nonzero"):
        behavioral_distances([[0], [1]], np.zeros((2, 1, 3)))
    with pytest.raises(ValueError, match="same personas"):
        behavioral_distances([[0], [1]], np.ones((3, 1, 3)))
