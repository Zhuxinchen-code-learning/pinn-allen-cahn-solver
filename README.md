# Physics-Informed Neural Network for Allen-Cahn Equation

This project studies the 1D Allen-Cahn equation with a finite difference baseline and a planned physics-informed neural network (PINN) solver.

## Research Question

For the 1D Allen-Cahn equation, how do a traditional finite difference method and a PINN compare in accuracy, energy dissipation, compute cost, and generalization across different `epsilon` values?

## Equation

The equation is `u_t = epsilon^2 u_xx + u - u^3`.

Domain: `x in [-1, 1]`, `t in [0, T]`.

Periodic boundary conditions: `u(-1, t) = u(1, t)` and `u_x(-1, t) = u_x(1, t)`.

Initial condition: `u(x, 0) = 0.5 sin(pi x)`.

## Current Stage

Stage 1 implements the finite difference method baseline:

- spatial grid generation
- periodic boundary handling
- explicit time stepping
- discrete Allen-Cahn energy calculation
- heatmap, profile, and energy visualizations

## Project Structure

```text
pinn-allen-cahn-solver/
|-- README.md
|-- requirements.txt
|-- configs/
|   `-- config_1d.yaml
|-- data/
|   `-- reference_solution.npz
|-- src/
|   |-- fdm_solver.py
|   |-- energy.py
|   |-- visualize.py
|   |-- utils.py
|   |-- pinn_model.py
|   |-- train_pinn.py
|   `-- evaluate.py
|-- experiments/
|   |-- run_fdm.py
|   |-- run_pinn.py
|   `-- compare_results.py
|-- figures/
|   |-- fdm_solution.png
|   |-- fdm_profiles.png
|   `-- energy_decay.png
`-- report/
    `-- project_report.md
```

The `data/` and `figures/` folders are generated when the FDM experiment is run.

## Setup

```bash
pip install -r requirements.txt
```

On this Windows machine, use `py -3` if `python` does not print anything:

```powershell
py -3 -m pip install -r requirements.txt
```

## Run the FDM Baseline

```bash
python experiments/run_fdm.py
```

or on this Windows machine:

```powershell
py -3 -u experiments/run_fdm.py
```

This creates:

- `data/reference_solution.npz`
- `figures/fdm_solution.png`
- `figures/fdm_profiles.png`
- `figures/energy_decay.png`

## Baseline Method

The solver uses a second-order central difference for the spatial Laplacian: `u_xx ~= (u[i+1] - 2u[i] + u[i-1]) / dx^2`.

It uses explicit Euler time stepping: `u^{n+1} = u^n + dt * (epsilon^2 u_xx^n + u^n - (u^n)^3)`.

Periodic boundaries are implemented with cyclic array shifts.

## Energy

The discrete energy approximates `E[u] = integral [epsilon^2 / 2 * |u_x|^2 + 1/4 * (u^2 - 1)^2] dx`.

The energy curve should generally decrease over time, which confirms that the baseline is capturing the dissipative behavior of the Allen-Cahn equation.

## Results

The first FDM baseline run produced a stable solution with monotonically decreasing energy:

```text
snapshots: (201, 256)
initial energy: 0.38980238
final energy: 0.20684054
energy change: -0.18296184
monotone energy steps: 100.00%
```

### FDM Solution Heatmap

![FDM solution heatmap](figures/fdm_solution.png)

### Solution Profiles

![FDM solution profiles](figures/fdm_profiles.png)

### Energy Decay

![Energy decay curve](figures/energy_decay.png)

## Next Steps

- implement the PyTorch PINN model
- train with PDE residual, initial condition, and periodic boundary losses
- compare PINN output against the FDM reference
- report L2 error, max error, energy behavior, training time, and inference time
