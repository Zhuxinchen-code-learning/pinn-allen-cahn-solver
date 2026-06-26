from dataclasses import dataclass

import numpy as np

from src.energy import compute_energy


@dataclass(frozen=True)
class FDMSolution:
    x: np.ndarray
    times: np.ndarray
    snapshots: np.ndarray
    energies: np.ndarray
    epsilon: float
    dx: float
    dt: float


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Initial condition u(x, 0) = 0.5 sin(pi x)."""
    return 0.5 * np.sin(np.pi * x)


def compute_laplacian_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central difference for u_xx with periodic boundaries."""
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2


def solve_allen_cahn_fdm(
    epsilon: float = 0.05,
    x_min: float = -1.0,
    x_max: float = 1.0,
    nx: int = 256,
    T: float = 1.0,
    dt: float = 1e-4,
    num_snapshots: int = 201,
) -> FDMSolution:
    """
    Explicit finite difference solver for the 1D Allen-Cahn equation:
        u_t = epsilon^2 u_xx + u - u^3

    Periodic boundary conditions are implemented by np.roll.
    """
    x = np.linspace(x_min, x_max, nx, endpoint=False)
    dx = (x_max - x_min) / nx
    nt = int(round(T / dt))
    save_every = max(nt // max(num_snapshots - 1, 1), 1)

    u = initial_condition(x)
    snapshots = []
    times = []
    energies = []

    for n in range(nt + 1):
        if n % save_every == 0 or n == nt:
            snapshots.append(u.copy())
            times.append(n * dt)
            energies.append(compute_energy(u, epsilon, dx))

        if n == nt:
            break

        lap = compute_laplacian_periodic(u, dx)
        reaction = u - u**3
        u = u + dt * (epsilon**2 * lap + reaction)

    return FDMSolution(
        x=x,
        times=np.array(times),
        snapshots=np.array(snapshots),
        energies=np.array(energies),
        epsilon=epsilon,
        dx=dx,
        dt=dt,
    )


def save_solution(solution: FDMSolution, output_path: str) -> None:
    """Save a reference solution as a compressed NumPy archive."""
    np.savez_compressed(
        output_path,
        x=solution.x,
        times=solution.times,
        snapshots=solution.snapshots,
        energies=solution.energies,
        epsilon=solution.epsilon,
        dx=solution.dx,
        dt=solution.dt,
    )
