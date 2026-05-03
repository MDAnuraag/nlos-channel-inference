"""Huber-loss reconstruction under simplex constraints.

This solver targets the case where the latent variable x is interpreted as a
non-negative distribution over positions with unit sum:

    x >= 0
    sum(x) = 1

That is appropriate for a single-object prior or a probability-like position map.

Model
-----
We solve

    minimise_x   sum_i rho_delta((A x - y)_i)
    subject to   x >= 0,  sum(x) = 1

where rho_delta is the Huber penalty.

Important note
--------------
If x is constrained to the simplex, then ||x||_1 = 1 identically, so an added
lambda * ||x||_1 term would be constant and therefore redundant. This module
makes that explicit instead of pretending lambda still tunes sparsity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from nlos_cs.inverse.base import (
    InverseProblem,
    InverseSolver,
    ReconstructionResult,
    compute_residual_norm,
    compute_solution_norm,
)
from nlos_cs.inverse.tikhonov import TikhonovSolver

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = FloatArray | ComplexArray


def simplex_project(x: FloatArray, *, eps: float = 1e-15) -> FloatArray:
    """Project a vector onto the probability simplex.

    This implementation is intentionally simple and robust for warm starts:
    1. clip negatives to zero
    2. renormalise to unit sum

    If all mass vanishes after clipping, return the uniform vector.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x_arr.shape}")

    x_nn = np.maximum(x_arr, 0.0)
    total = float(np.sum(x_nn))
    if total <= eps:
        return np.ones_like(x_nn) / x_nn.shape[0]
    return x_nn / total


def huber_penalty(residual: FloatArray, delta: float) -> FloatArray:
    """Elementwise Huber penalty."""
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")

    r = np.asarray(residual, dtype=np.float64)
    abs_r = np.abs(r)
    return np.where(
        abs_r <= delta,
        0.5 * r**2,
        delta * (abs_r - 0.5 * delta),
    )


def huber_gradient_wrt_residual(residual: FloatArray, delta: float) -> FloatArray:
    """Gradient of Huber penalty with respect to the residual."""
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")

    r = np.asarray(residual, dtype=np.float64)
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, r, delta * np.sign(r))


def _real_system_for_real_x(
    A: OperatorArray,
    y: FloatArray | ComplexArray,
) -> tuple[FloatArray, FloatArray]:
    """Convert a possibly complex system into a real residual system for real x.

    We keep x real-valued and solve against the stacked real residual:

        [Re(A)] x - [Re(y)]
        [Im(A)]     [Im(y)]

    If A and y are already real, this is a no-op.
    """
    A_arr = np.asarray(A)
    y_arr = np.asarray(y)

    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A_arr.shape}")
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
    if A_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Row mismatch: A has {A_arr.shape[0]} rows but y has length {y_arr.shape[0]}"
        )

    if np.iscomplexobj(A_arr) or np.iscomplexobj(y_arr):
        A_real = np.vstack([A_arr.real, A_arr.imag]).astype(np.float64, copy=False)
        y_real = np.concatenate([y_arr.real, y_arr.imag], axis=0).astype(
            np.float64, copy=False
        )
        return A_real, y_real

    return (
        A_arr.astype(np.float64, copy=False),
        y_arr.astype(np.float64, copy=False),
    )


def solve_huber_simplex(
    A: OperatorArray,
    y: FloatArray | ComplexArray,
    *,
    delta: float,
    warm_start: FloatArray | None = None,
    maxiter: int = 500,
    ftol: float = 1e-9,
) -> FloatArray:
    """Solve the simplex-constrained Huber problem.

    Parameters
    ----------
    A, y:
        Linear system.
    delta:
        Huber transition point in residual units.
    warm_start:
        Optional initial vector in the latent space. It will be projected onto
        the simplex before optimisation.
    maxiter, ftol:
        SLSQP settings.

    Returns
    -------
    x_hat:
        Real-valued simplex-constrained solution.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if maxiter <= 0:
        raise ValueError(f"maxiter must be positive, got {maxiter}")
    if ftol <= 0.0:
        raise ValueError(f"ftol must be positive, got {ftol}")

    A_real, y_real = _real_system_for_real_x(A, y)
    n = A_real.shape[1]

    if warm_start is None:
        x0 = np.ones(n, dtype=np.float64) / n
    else:
        warm = np.asarray(warm_start, dtype=np.float64)
        if warm.shape != (n,):
            raise ValueError(f"warm_start must have shape ({n},), got {warm.shape}")
        x0 = simplex_project(warm)

    def objective(x: FloatArray) -> float:
        r = A_real @ x - y_real
        return float(np.sum(huber_penalty(r, delta)))

    def gradient(x: FloatArray) -> FloatArray:
        r = A_real @ x - y_real
        g_r = huber_gradient_wrt_residual(r, delta)
        return (A_real.T @ g_r).astype(np.float64, copy=False)

    constraints = {
        "type": "eq",
        "fun": lambda x: float(np.sum(x) - 1.0),
        "jac": lambda x: np.ones(n, dtype=np.float64),
    }
    bounds = [(0.0, None)] * n

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol},
    )

    if not result.success:
        raise RuntimeError(
            f"Huber-simplex optimisation failed: {result.message}"
        )

    return simplex_project(np.asarray(result.x, dtype=np.float64))


@dataclass(frozen=True)
class HuberSimplexConfig:
    """Configuration for the Huber-simplex solver."""

    delta: float
    maxiter: int = 500
    ftol: float = 1e-9
    use_tikhonov_warm_start: bool = True
    tikhonov_lambda_for_warm_start: float = 1.0


class HuberSimplexSolver(InverseSolver):
    """Huber-loss solver under simplex constraints."""

    name = "huber_simplex"

    def __init__(self, config: HuberSimplexConfig) -> None:
        if config.delta <= 0.0:
            raise ValueError(f"delta must be positive, got {config.delta}")
        self.config = config

    def _warm_start(self, problem: InverseProblem) -> FloatArray | None:
        if not self.config.use_tikhonov_warm_start:
            return None

        tik = TikhonovSolver(
            lambda_value=self.config.tikhonov_lambda_for_warm_start,
            use_svd=True,
            precompute_svd=False,
        )
        x0 = tik.solve(problem).x_hat
        x0_real = np.abs(np.asarray(x0)).astype(np.float64, copy=False)
        return simplex_project(x0_real)

    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        problem.validate()

        warm_start = self._warm_start(problem)

        x_hat = solve_huber_simplex(
            problem.A,
            problem.y,
            delta=self.config.delta,
            warm_start=warm_start,
            maxiter=self.config.maxiter,
            ftol=self.config.ftol,
        )

        residual_norm = compute_residual_norm(problem.A, x_hat, problem.y)
        solution_norm = compute_solution_norm(x_hat)
        objective_value = float(
            np.sum(
                huber_penalty(
                    _real_system_for_real_x(problem.A, problem.y)[0] @ x_hat
                    - _real_system_for_real_x(problem.A, problem.y)[1],
                    self.config.delta,
                )
            )
        )

        return ReconstructionResult(
            x_hat=x_hat,
            residual_norm=residual_norm,
            solution_norm=solution_norm,
            solver_name=self.name,
            objective_value=objective_value,
            lambda_value=None,
            metadata={
                "delta": self.config.delta,
                "maxiter": self.config.maxiter,
                "ftol": self.config.ftol,
                "use_tikhonov_warm_start": self.config.use_tikhonov_warm_start,
                "tikhonov_lambda_for_warm_start": self.config.tikhonov_lambda_for_warm_start,
                "simplex_constraint": True,
                "l1_term_redundant_under_simplex": True,
            },
        )