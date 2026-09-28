"""Small, CPU-only building blocks for persona geometry."""

from .geometry import PersonaManifold, center_activations
from .evaluation import behavioral_distances, triplet_accuracy

__all__ = [
    "PersonaManifold",
    "center_activations",
    "behavioral_distances",
    "triplet_accuracy",
]
