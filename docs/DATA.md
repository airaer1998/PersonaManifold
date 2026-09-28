# Data interface

The quickstart generates a mathematical curve in memory. For persona
experiments, supply the activation and behavioral arrays described below.

## Activations

Supply NumPy arrays with finite floating-point values:

| File / array | Shape | Meaning |
| --- | --- | --- |
| `responses.npy` | `(N, M, D)` | Last response-token residual activations, one layer, for N personas and M probes |
| `neutral.npy` | `(M, D)` | Matching neutral-prompt responses for the same probes and layer |
| `vectors.npy` (alternative) | `(N, D)` | Already aggregated, neutral-relative persona vectors |

Use the same model, tokenizer, layer, extraction position, and probe order in
both response arrays. Persona indices must be stable across activation and
behavioral data. The library cannot detect misaligned IDs from numeric arrays.

If you already have aggregated vectors:

```python
import numpy as np
from personamanifold import PersonaManifold

vectors = np.load("vectors.npy", allow_pickle=False)
m = PersonaManifold(n_neighbors=8, tangent_dim=2).fit(vectors)
np.save("geodesic_distances.npy", m.distances())
np.save("steering.npy", m.steering_vectors(0, 1, n_steps=11))
```

## Behavior and triplets

`behavioral_distances(choices, embeddings)` accepts:

- `choices`: integer array `(N, Q_choice)`, nonnegative option IDs, with option
  meanings consistent across personas for each question. Missing choices are
  not supported; filter incomplete data before evaluation.
- `embeddings`: float array `(N, Q_open, E)` of response embeddings, one aligned
  embedding per persona and question. Zero-norm responses are invalid.

The distance output has shape `(N, N)`. Lower values mean more similar behavior.
Question counts need not match between forced-choice and open-ended inputs.

Triplets are integer rows `[anchor, positive, negative]` with shape `(T, 3)`.
Each row has three distinct IDs in `[0, N)`. Construct and validate these labels
from held-out behavioral data. Keep layer/parameter selection separate from
the test triplets.

## Output

- `shortest_path(a, b)`: ordered input-row indices, including endpoints.
- `interpolate(a, b, n_steps)`: `(n_steps, retained_dimension)` PCA coordinates.
- `steering_vectors(a, b, n_steps)`: `(n_steps, D)` scaled residual additions.
- `triplet_accuracy(...)`: fraction in `[0, 1]`, not a percentage.

Source-checkout examples and images are repository assets; built wheels contain
the importable `personamanifold` package, not datasets or example input files.
