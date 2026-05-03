"""Non-negative least-squares solvers.

Model
-----
Solve

    minimise_x   ||A x - y||_2^2
    subject to   x >= 0

This is a useful baseline when negative latent weights are physically
meaningless, but a simplex constraint is too strong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import nnls

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
    """Convert a possibly complex system into a real system for real x.

    For complex A and y, solve against the stacked real residual:

        [Re(A)] x ≈ [Re(y)]
        [Im(A)]     [Im(y)]

    so x remains real and non-negative.
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


def solve_nnls(
    A: OperatorArray,
    y: FloatArray | ComplexArray,
) -> FloatArray:
    """Solve non-negative least squares.

    Parameters
    ----------
    A, y:
        Linear system.

    Returns
    -------
    x_hat:
        Real-valued non-negative least-squares solution.
    """
    A_real, y_real = _real_system_for_real_x(A, y)
    x_hat, _ = nnls(A_real, y_real)
    return x_hat.astype(np.float64, copy=False)


@dataclass(frozen=True)
class NNLSSolverConfig:
    """Configuration for the NNLS solver."""

    normalise_by_sum: bool = False
    normalise_by_peak: bool = False


class NNLSSolver(InverseSolver):
    """Non-negative least-squares solver."""

    name = "nnls"

    def __init__(self, config: NNLSSolverConfig | None = None) -> None:
        self.config = NNLSSolverConfig() if config is None else config

        if self.config.normalise_by_sum and self.config.normalise_by_peak:
            raise ValueError(
                "normalise_by_sum and normalise_by_peak cannot both be True"
            )

    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        problem.validate()

        x_hat = solve_nnls(problem.A, problem.y)

        if self.config.normalise_by_sum:
            total = float(np.sum(x_hat))
            if total > 1e-15:
                x_hat = x_hat / total

        if self.config.normalise_by_peak:
            peak = float(np.max(np.abs(x_hat)))
            if peak > 1e-15:
                x_hat = x_hat / peak

        residual_norm = compute_residual_norm(problem.A, x_hat, problem.y)
        solution_norm = compute_solution_norm(x_hat)
        objective_value = float(residual_norm**2)

        return ReconstructionResult(
            x_hat=x_hat,
            residual_norm=residual_norm,
            solution_norm=solution_norm,
            solver_name=self.name,
            objective_value=objective_value,
            lambda_value=None,
            metadata={
                "nonnegative_constraint": True,
                "sum_constraint": False,
                "normalise_by_sum": self.config.normalise_by_sum,
                "normalise_by_peak": self.config.normalise_by_peak,
            },
        )