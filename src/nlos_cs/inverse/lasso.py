"""L1-regularised least-squares solvers (LASSO / sparse recovery).

Model
-----
Solve

    minimise_x   0.5 ||A x - y||_2^2 + alpha ||x||_1

This is a core sparse-recovery baseline for compressed sensing.

Notes
-----
- This implementation assumes x is real-valued.
- If A and/or y are complex, the system is converted to an equivalent
  stacked real system while keeping x real.
- The solver uses ISTA or FISTA with soft-thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from nlos_cs.inverse.base import (
    InverseProblem,
    InverseSolver,
    ReconstructionResult,
    compute_residual_norm,
    compute_solution_norm,
)

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = FloatArray | ComplexArray


def _real_system_for_real_x(
    A: OperatorArray,
    y: FloatArray | ComplexArray,
) -> tuple[FloatArray, FloatArray]:
    """Convert a possibly complex system into a real system for real x."""
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


def soft_threshold(x: FloatArray, threshold: float) -> FloatArray:
    """Apply elementwise soft-thresholding."""
    if threshold < 0.0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    x_arr = np.asarray(x, dtype=np.float64)
    return np.sign(x_arr) * np.maximum(np.abs(x_arr) - threshold, 0.0)


def lasso_objective(
    A: FloatArray,
    y: FloatArray,
    x: FloatArray,
    alpha: float,
) -> float:
    """Evaluate 0.5 ||Ax-y||_2^2 + alpha ||x||_1."""
    r = A @ x - y
    return float(0.5 * np.dot(r, r) + alpha * np.sum(np.abs(x)))


def _estimate_lipschitz_constant(A: FloatArray) -> float:
    """Estimate L = ||A^T A||_2 = sigma_max(A)^2."""
    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    if s.size == 0:
        raise ValueError("Cannot estimate Lipschitz constant for empty matrix")
    return float(s[0] ** 2)


def solve_lasso_ista(
    A: OperatorArray,
    y: FloatArray | ComplexArray,
    *,
    alpha: float,
    maxiter: int = 1000,
    tol: float = 1e-8,
    step_size: float | None = None,
    x0: FloatArray | None = None,
    use_fista: bool = True,
) -> tuple[FloatArray, dict[str, float | int | bool]]:
    """Solve LASSO using ISTA or FISTA.

    Parameters
    ----------
    A, y:
        Linear system.
    alpha:
        L1 regularisation strength.
    maxiter:
        Maximum number of iterations.
    tol:
        Stopping tolerance based on relative iterate change.
    step_size:
        Optional fixed step size. If omitted, uses 1 / L where
        L = ||A^T A||_2.
    x0:
        Optional initial vector.
    use_fista:
        If True, use FISTA acceleration. Else use plain ISTA.

    Returns
    -------
    x_hat, metadata
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if maxiter <= 0:
        raise ValueError(f"maxiter must be positive, got {maxiter}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")

    A_real, y_real = _real_system_for_real_x(A, y)
    n = A_real.shape[1]

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64)
        if x.shape != (n,):
            raise ValueError(f"x0 must have shape ({n},), got {x.shape}")
        x = x.copy()

    if step_size is None:
        L = _estimate_lipschitz_constant(A_real)
        if L <= 0.0:
            raise ValueError(f"Estimated Lipschitz constant must be positive, got {L}")
        step = 1.0 / L
    else:
        step = float(step_size)
        if step <= 0.0:
            raise ValueError(f"step_size must be positive, got {step}")

    if use_fista:
        z = x.copy()
        t = 1.0
    else:
        z = x

    converged = False
    iterations = 0

    for k in range(1, maxiter + 1):
        x_old = x.copy()

        grad = A_real.T @ (A_real @ z - y_real)
        x = soft_threshold(z - step * grad, step * alpha)

        if use_fista:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = x + ((t - 1.0) / t_new) * (x - x_old)
            t = t_new
        else:
            z = x

        denom = max(np.linalg.norm(x_old), 1e-12)
        rel_change = float(np.linalg.norm(x - x_old) / denom)

        iterations = k
        if rel_change < tol:
            converged = True
            break

    metadata: dict[str, float | int | bool] = {
        "alpha": alpha,
        "maxiter": maxiter,
        "tol": tol,
        "step_size": step,
        "use_fista": use_fista,
        "iterations": iterations,
        "converged": converged,
    }
    return x.astype(np.float64, copy=False), metadata


@dataclass(frozen=True)
class LassoSolverConfig:
    """Configuration for the LASSO solver."""

    alpha: float
    maxiter: int = 1000
    tol: float = 1e-8
    step_size: float | None = None
    use_fista: bool = True
    normalise_by_sum: bool = False
    normalise_by_peak: bool = False


class LassoSolver(InverseSolver):
    """L1-regularised least-squares solver via ISTA/FISTA."""

    name = "lasso"

    def __init__(self, config: LassoSolverConfig) -> None:
        if config.alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {config.alpha}")
        if config.maxiter <= 0:
            raise ValueError(f"maxiter must be positive, got {config.maxiter}")
        if config.tol <= 0.0:
            raise ValueError(f"tol must be positive, got {config.tol}")
        if config.step_size is not None and config.step_size <= 0.0:
            raise ValueError(f"step_size must be positive, got {config.step_size}")
        if config.normalise_by_sum and config.normalise_by_peak:
            raise ValueError(
                "normalise_by_sum and normalise_by_peak cannot both be True"
            )
        self.config = config

    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        problem.validate()

        x_hat, meta = solve_lasso_ista(
            problem.A,
            problem.y,
            alpha=self.config.alpha,
            maxiter=self.config.maxiter,
            tol=self.config.tol,
            step_size=self.config.step_size,
            x0=None,
            use_fista=self.config.use_fista,
        )

        if self.config.normalise_by_sum:
            total = float(np.sum(np.abs(x_hat)))
            if total > 1e-15:
                x_hat = x_hat / total

        if self.config.normalise_by_peak:
            peak = float(np.max(np.abs(x_hat)))
            if peak > 1e-15:
                x_hat = x_hat / peak

        residual_norm = compute_residual_norm(problem.A, x_hat, problem.y)
        solution_norm = compute_solution_norm(x_hat)
        objective_value = lasso_objective(
            _real_system_for_real_x(problem.A, problem.y)[0],
            _real_system_for_real_x(problem.A, problem.y)[1],
            x_hat,
            self.config.alpha,
        )

        return ReconstructionResult(
            x_hat=x_hat,
            residual_norm=residual_norm,
            solution_norm=solution_norm,
            solver_name=self.name,
            objective_value=objective_value,
            lambda_value=self.config.alpha,
            metadata={
                **meta,
                "normalise_by_sum": self.config.normalise_by_sum,
                "normalise_by_peak": self.config.normalise_by_peak,
                "real_x_assumption": True,
            },
        )