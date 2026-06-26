from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fdm_solver import save_solution, solve_allen_cahn_fdm
from src.utils import ensure_dir, load_config
from src.visualize import plot_energy, plot_solution_heatmap, plot_solution_profiles


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    log("Loading configuration...")
    config = load_config(PROJECT_ROOT / "configs" / "config_1d.yaml")
    equation = config["equation"]
    fdm = config["fdm"]
    paths = config["paths"]

    data_dir = ensure_dir(PROJECT_ROOT / paths["data_dir"])
    figures_dir = ensure_dir(PROJECT_ROOT / paths["figures_dir"])

    log("Solving Allen-Cahn equation with FDM...")
    solution = solve_allen_cahn_fdm(
        epsilon=equation["epsilon"],
        x_min=equation["x_min"],
        x_max=equation["x_max"],
        nx=fdm["nx"],
        T=equation["T"],
        dt=fdm["dt"],
        num_snapshots=fdm["num_snapshots"],
    )

    log("Saving reference solution...")
    save_solution(solution, data_dir / "reference_solution.npz")

    log("Saving FDM heatmap...")
    plot_solution_heatmap(
        solution.x,
        solution.times,
        solution.snapshots,
        figures_dir / "fdm_solution.png",
    )

    log("Saving solution profiles...")
    plot_solution_profiles(
        solution.x,
        solution.times,
        solution.snapshots,
        figures_dir / "fdm_profiles.png",
    )

    log("Saving energy curve...")
    plot_energy(solution.times, solution.energies, figures_dir / "energy_decay.png")

    energy_change = solution.energies[-1] - solution.energies[0]
    monotone_ratio = (solution.energies[1:] <= solution.energies[:-1] + 1e-12).mean()

    log("FDM baseline complete")
    log(f"snapshots: {solution.snapshots.shape}")
    log(f"initial energy: {solution.energies[0]:.8f}")
    log(f"final energy: {solution.energies[-1]:.8f}")
    log(f"energy change: {energy_change:.8f}")
    log(f"monotone energy steps: {monotone_ratio:.2%}")
    log(f"data saved to: {data_dir / 'reference_solution.npz'}")
    log(f"figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
