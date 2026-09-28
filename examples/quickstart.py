"""Run a synthetic curved-manifold demo, with no LLM, dataset, or API key."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from personamanifold import PersonaManifold, triplet_accuracy
from personamanifold.curvature import edge_curvature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, help="optional image output (requires matplotlib)")
    args = parser.parse_args()
    # A nearly complete circular arc. Endpoints are close in ambient space but
    # far apart along the sampled curve. These are synthetic points, not personas.
    angles = np.linspace(0, 1.8 * np.pi, 100)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    manifold = PersonaManifold(n_neighbors=4, tangent_dim=1, variance=1).fit(points)
    curved = manifold.to_activation(manifold.interpolate(0, 99, n_steps=101))
    straight = np.linspace(points[0], points[-1], 101)
    rng = np.random.default_rng(7)
    triplets = []
    for _ in range(1000):
        anchor, b, c = rng.choice(len(points), size=3, replace=False)
        # Labels come from the known generating angle, independently of the fit.
        if abs(angles[b] - angles[anchor]) == abs(angles[c] - angles[anchor]):
            continue
        positive, negative = sorted((b, c), key=lambda x: abs(angles[x] - angles[anchor]))
        triplets.append((anchor, positive, negative))
    report = {
        "data": "synthetic circular arc; these are NOT paper benchmark results",
        "points": len(points),
        "triplets": len(triplets),
        "euclidean_tcr": triplet_accuracy(cdist(points, points), triplets),
        "graph_tcr": triplet_accuracy(manifold.distances(), triplets),
        "polyline_mean_distance_to_samples": float(cdist(curved, points).min(axis=1).mean()),
        "straight_mean_distance_to_samples": float(cdist(straight, points).min(axis=1).mean()),
        "example_edge_curvature": edge_curvature(manifold, 50, 51),
        "steering_vectors_shape": list(manifold.steering_vectors(0, 99).shape),
    }
    print(json.dumps(report, indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(*points.T, s=12, color="#8594a8", label="Synthetic samples")
        ax.plot(*curved.T, color="#247cce", linewidth=2, label="Graph path (polyline)")
        ax.plot(*straight.T, "--", color="#e98643", linewidth=2, label="Straight interpolation")
        ax.scatter(*points[[0, -1]].T, s=65, color="#20344a", zorder=5)
        ax.set(aspect="equal", title="Synthetic illustration: following the sampled curve")
        ax.legend(loc="center", frameon=False, fontsize=9)
        ax.set_axis_off()
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
