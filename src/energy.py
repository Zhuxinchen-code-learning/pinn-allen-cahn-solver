import numpy as np


def compute_energy(u: np.ndarray, epsilon: float, dx: float) -> float:
    """Compute the discrete 1D Allen-Cahn energy with periodic differences."""
    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
    energy_density = 0.5 * epsilon**2 * ux**2 + 0.25 * (u**2 - 1.0) ** 2
    return float(np.sum(energy_density) * dx)


def compute_energy_series(snapshots: np.ndarray, epsilon: float, dx: float) -> np.ndarray:
    """Compute energy for each saved solution snapshot."""
    return np.array([compute_energy(u, epsilon, dx) for u in snapshots])
