"""Two moons data, the plotting grid and the figure folder shared by the examples."""

from pathlib import Path

import numpy as np

# README figures, found from the repo root so it works from a script and from the notebook
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def generate_two_moons(
    n_samples: int = 100, noise: float = 0.1, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half circles with Gaussian noise. Returns X of shape N by 2 and labels 0 or 1."""
    rng = np.random.default_rng(random_state)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    theta_out = rng.uniform(0, np.pi, n_samples_out)
    x_out = np.cos(theta_out)
    y_out = np.sin(theta_out)

    theta_in = rng.uniform(0, np.pi, n_samples_in)
    x_in = 1.0 - np.cos(theta_in)
    y_in = 0.5 - np.sin(theta_in)

    X = np.vstack([np.column_stack((x_out, y_out)), np.column_stack((x_in, y_in))])
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    if noise > 0:
        X += rng.normal(scale=noise, size=X.shape)

    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def grid(
    X: np.ndarray, pad: float = 1.5, n: int = 150
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh around the data, padded on every side. Returns xx, yy and the flat points."""
    lo, hi = X.min(axis=0) - pad, X.max(axis=0) + pad
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])
