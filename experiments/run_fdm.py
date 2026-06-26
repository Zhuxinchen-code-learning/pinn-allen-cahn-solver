from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fdm_solver import save_solution, solve_allen_cahn_fdm
from src.utils import ensure_dir, load_config
from src.visualize import plot_energy, plot_solution_heatmap, plot_solution_profiles


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config_1d.yaml")
    equation = config["equation"]
    fdm = config["fdm"]
    paths = config["paths"]

    data_dir = ensure_dir(PROJECT_ROOT / paths["data_dir"])
    figures_dir = ensure_dir(PROJECT_ROOT / paths["figures_dir"])

    solution = solve_allen_cahn_fdm(
        epsilon=equation["epsilon"],
        x_min=equation["x_min"],
        x_max=equation["x_max"],
        nx=fdm["nx"],
        T=equation["T"],
        dt=fdm["dt"],
        num_snapshots=fdm["num_snapshots"],
    )

    save_solution(solution, data_dir / "reference_solution.npz")
    plot_solution_heatmap(
        solution.x,
        solution.times,
        solution.snapshots,
        figures_dir / "fdm_solution.png",
    )
    plot_solution_profiles(
        solution.x,
        solution.times,
        solution.snapshots,
        figures_dir / "fdm_profiles.png",
    )
    plot_energy(solution.times, solution.energies, figures_dir / "energy_decay.png")

    energy_change = solution.energies[-1] - solution.energies[0]
    monotone_ratio = (solution.energies[1:] <= solution.energies[:-1] + 1e-12).mean()

    print("FDM baseline complete")
    print(f"snapshots: {solution.snapshots.shape}")
    print(f"initial energy: {solution.energies[0]:.8f}")
    print(f"final energy: {solution.energies[-1]:.8f}")
    print(f"energy change: {energy_change:.8f}")
    print(f"monotone energy steps: {monotone_ratio:.2%}")
    print(f"data saved to: {data_dir / 'reference_solution.npz'}")
    print(f"figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
