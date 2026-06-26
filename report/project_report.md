# Physics-Informed Neural Network for Allen-Cahn Equation

## 1. Introduction

This project studies numerical and neural-network-based solvers for the 1D Allen-Cahn equation. The first stage builds a finite difference baseline that will later serve as the reference solution for PINN evaluation.

## 2. Allen-Cahn Equation

We solve `u_t = epsilon^2 u_xx + u - u^3` on `x in [-1, 1]` and `t in [0, T]`, with periodic boundary conditions and initial condition `u(x, 0) = 0.5 sin(pi x)`.

## 3. Finite Difference Baseline

The baseline uses a second-order central finite difference for `u_xx` and explicit Euler time stepping. Periodic boundary conditions are implemented through array shifts.

## 4. Energy Stability

The Allen-Cahn energy is `E[u] = integral [epsilon^2 / 2 * |u_x|^2 + 1/4 * (u^2 - 1)^2] dx`.

For a stable dissipative simulation, this energy should generally decrease over time.

## 5. Next Step

The next stage will implement a PyTorch PINN that learns `u(x, t)` directly from PDE residual, initial condition, and periodic boundary losses.
