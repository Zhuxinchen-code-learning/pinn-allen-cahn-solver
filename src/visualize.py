from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_solution_heatmap(
    x: np.ndarray,
    times: np.ndarray,
    snapshots: np.ndarray,
    output_path: str | Path,
    title: str = "FDM Solution of 1D Allen-Cahn Equation",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.imshow(
        snapshots,
        extent=[x[0], x[-1], times[-1], times[0]],
        aspect="auto",
        cmap="coolwarm",
    )
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_solution_profiles(
    x: np.ndarray,
    times: np.ndarray,
    snapshots: np.ndarray,
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(8, 5))
    selected_indices = sorted(
        set([0, len(times) // 4, len(times) // 2, 3 * len(times) // 4, len(times) - 1])
    )

    for idx in selected_indices:
        plt.plot(x, snapshots[idx], label=f"t = {times[idx]:.2f}")

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Solution Profiles at Different Times")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_energy(times: np.ndarray, energies: np.ndarray, output_path: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(times, energies, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("Energy")
    plt.title("Discrete Energy Decay")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
