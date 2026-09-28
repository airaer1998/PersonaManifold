"""BST-style behavioral distances and strict triplet consistency."""

import numpy as np

from .geometry import _finite_array


def behavioral_distances(choices, embeddings, beta=1 / 3):
    """Combine forced-choice disagreement and mean per-question cosine distance.

    choices: integer option IDs (N, Q_choice), aligned by question.
    embeddings: response embeddings (N, Q_open, D), aligned by question.
    Missing choices/zero embeddings are rejected. Questions and responses must
    be provided by the caller; this function does not generate the BST corpus.
    """
    choices = np.asarray(choices)
    embeddings = _finite_array(embeddings, 3, "embeddings")
    if choices.ndim != 2 or 0 in choices.shape or not np.issubdtype(choices.dtype, np.integer) or np.any(choices < 0):
        raise ValueError("choices must be a nonempty matrix of nonnegative integer option IDs")
    if choices.shape[0] != embeddings.shape[0]:
        raise ValueError("choices and embeddings must describe the same personas")
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("response embeddings must have nonzero norms")
    unit = embeddings / norms
    choice_distance = np.mean(choices[:, None, :] != choices[None, :, :], axis=-1)
    cosine = np.einsum("iqd,jqd->ij", unit, unit) / embeddings.shape[1]
    result = beta * choice_distance + (1 - beta) * np.clip(1 - cosine, 0, 2)
    np.fill_diagonal(result, 0)
    return result


def triplet_accuracy(distances, triplets):
    """Fraction with d(anchor, positive) < d(anchor, negative); ties score zero."""
    distances = np.asarray(distances, dtype=float)
    triplets = np.asarray(triplets)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1] or not len(distances):
        raise ValueError("distances must be a nonempty square matrix")
    if triplets.ndim != 2 or triplets.shape[1] != 3 or len(triplets) == 0 or not np.issubdtype(triplets.dtype, np.integer):
        raise ValueError("triplets must be a nonempty integer array with shape (T, 3)")
    if np.any(triplets < 0) or np.any(triplets >= len(distances)):
        raise ValueError("triplet index outside distance matrix")
    a, p, n = triplets.T
    if np.any((a == p) | (a == n) | (p == n)):
        raise ValueError("each triplet must contain three distinct personas")
    positive, negative = distances[a, p], distances[a, n]
    if not np.isfinite(np.r_[positive, negative]).all() or np.any(np.r_[positive, negative] < 0):
        raise ValueError("evaluated distances must be finite and nonnegative")
    return float(np.mean(positive < negative))
