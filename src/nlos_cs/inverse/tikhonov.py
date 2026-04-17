"""Tikhonov inverse solvers and L-curve lambda selection."""

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


@dataclass(frozen=True)
class TikhonovSweepResult:
    """Full lambda sweep output for L-curve analysis."""

    lambdas: FloatArray
    residual_norms: FloatArray
    solution_norms: FloatArray
    x_hats: OperatorArray
    curvature: FloatArray
    lambda_opt: float
    idx_opt: int


def _validate_lambda_value(lambda_value: float) -> float:
    lam = float(lambda_value)
    if lam < 0.0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    return lam


def compute_svd(
    A: OperatorArray,
) -> tuple[OperatorArray, FloatArray, OperatorArray]:
    """Compute compact SVD of A."""
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    return U, s.astype(np.float64, copy=False), Vh


def tikhonov_svd_solve(
    U: OperatorArray,
    s: FloatArray,
    Vh: OperatorArray,
    y: FloatArray | ComplexArray,
    lambda_value: float,
) -> FloatArray | ComplexArray:
    """Solve Tikhonov regularisation in SVD filter form.

    Minimises

        ||A x - y||_2^2 + lambda ||x||_2^2

    using A = U diag(s) Vh, giving

        x_hat = V diag(s / (s^2 + lambda)) U^H y
    """
    lam = _validate_lambda_value(lambda_value)

    if U.ndim != 2 or Vh.ndim != 2 or s.ndim != 1:
        raise ValueError("U, s, Vh must have shapes (m,n), (n,), (n,n)")

    if U.shape[1] != s.shape[0] or Vh.shape[0] != s.shape[0]:
        raise ValueError(
            f"Incompatible SVD shapes: U{U.shape}, s{s.shape}, Vh{Vh.shape}"
        )

    Uy = U.conj().T @ y
    filters = s / (s**2 + lam)
    x_hat = Vh.conj().T @ (filters * Uy)
    return x_hat


def tikhonov_direct_solve(
    A: OperatorArray,
    y: FloatArray | ComplexArray,
    lambda_value: float,
) -> FloatArray | ComplexArray:
    """Solve Tikhonov regularisation by direct linear solve.

    Solves

        (A^H A + lambda I) x = A^H y
    """
    lam = _validate_lambda_value(lambda_value)

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if A.shape[0] != y.shape[0]:
        raise ValueError(
            f"Row mismatch: A has {A.shape[0]} rows but y has length {y.shape[0]}"
        )

    n = A.shape[1]
    AhA = A.conj().T @ A
    Ahy = A.conj().T @ y
    M = AhA + lam * np.eye(n, dtype=AhA.dtype)
    return np.linalg.solve(M, Ahy)


def make_lambda_grid(
    lambda_min_exp: float = 0.0,
    lambda_max_exp: float = 6.0,
    n_lambda: int = 200,
) -> FloatArray:
    """Construct a log-spaced lambda grid."""
    if n_lambda < 3:
        raise ValueError("n_lambda must be at least 3 for L-curve curvature analysis")
    if lambda_max_exp <= lambda_min_exp:
        raise ValueError("lambda_max_exp must be greater than lambda_min_exp")

    return np.logspace(lambda_min_exp, lambda_max_exp, n_lambda).astype(np.float64)


def compute_lcurve_sweep(
    A: OperatorArray,
    y: FloatArray | ComplexArray,
    lambdas: FloatArray,
    *,
    precomputed_svd: tuple[OperatorArray, FloatArray, OperatorArray] | None = None,
) -> TikhonovSweepResult:
    """Compute residual and solution norms across a lambda sweep."""
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if A.shape[0] != y.shape[0]:
        raise ValueError(
            f"Row mismatch: A has {A.shape[0]} rows but y has length {y.shape[0]}"
        )

    lambdas = np.asarray(lambdas, dtype=np.float64)
    if lambdas.ndim != 1:
        raise ValueError("lambdas must be 1D")
    if np.any(lambdas < 0.0):
        raise ValueError("lambdas must be non-negative")

    U, s, Vh = precomputed_svd if precomputed_svd is not None else compute_svd(A)

    n_lam = lambdas.shape[0]
    n_unknowns = A.shape[1]
    x_dtype = np.complex128 if np.iscomplexobj(A) or np.iscomplexobj(y) else np.float64

    x_hats = np.zeros((n_lam, n_unknowns), dtype=x_dtype)
    residual_norms = np.zeros(n_lam, dtype=np.float64)
    solution_norms = np.zeros(n_lam, dtype=np.float64)

    for i, lam in enumerate(lambdas):
        x_hat = tikhonov_svd_solve(U, s, Vh, y, float(lam))
        x_hats[i] = x_hat
        residual_norms[i] = compute_residual_norm(A, x_hat, y)
        solution_norms[i] = compute_solution_norm(x_hat)

    lambda_opt, idx_opt, curvature = find_lcurve_corner(
        residual_norms=residual_norms,
        solution_norms=solution_norms,
        lambdas=lambdas,
    )

    return TikhonovSweepResult(
        lambdas=lambdas,
        residual_norms=residual_norms,
        solution_norms=solution_norms,
        x_hats=x_hats,
        curvature=curvature,
        lambda_opt=lambda_opt,
        idx_opt=idx_opt,
    )


def find_lcurve_corner(
    residual_norms: FloatArray,
    solution_norms: FloatArray,
    lambdas: FloatArray,
    *,
    eps: float = 1e-15,
) -> tuple[float, int, FloatArray]:
    """Locate the maximum-curvature point of the L-curve in log-log space."""
    residual_norms = np.asarray(residual_norms, dtype=np.float64)
    solution_norms = np.asarray(solution_norms, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)

    if not (
        residual_norms.ndim == solution_norms.ndim == lambdas.ndim == 1
        and residual_norms.size == solution_norms.size == lambdas.size
    ):
        raise ValueError("residual_norms, solution_norms, and lambdas must be 1D and equal-length")

    if lambdas.size < 3:
        raise ValueError("At least 3 lambda values are required for curvature estimation")

    log_res = np.log(np.maximum(residual_norms, eps))
    log_sol = np.log(np.maximum(solution_norms, eps))
    log_lam = np.log(np.maximum(lambdas, eps))

    d_res = np.gradient(log_res, log_lam)
    d_sol = np.gradient(log_sol, log_lam)
    dd_res = np.gradient(d_res, log_lam)
    dd_sol = np.gradient(d_sol, log_lam)

    numerator = np.abs(d_res * dd_sol - d_sol * dd_res)
    denominator = np.maximum((d_res**2 + d_sol**2) ** 1.5, eps)
    curvature = numerator / denominator

    idx_opt = int(np.argmax(curvature))
    lambda_opt = float(lambdas[idx_opt])

    return lambda_opt, idx_opt, curvature.astype(np.float64, copy=False)


class TikhonovSolver(InverseSolver):
    """Fixed-lambda Tikhonov solver."""

    name = "tikhonov"

    def __init__(
        self,
        lambda_value: float,
        *,
        use_svd: bool = True,
        precompute_svd: bool = True,
    ) -> None:
        self.lambda_value = _validate_lambda_value(lambda_value)
        self.use_svd = bool(use_svd)
        self.precompute_svd = bool(precompute_svd)
        self._cached_shape: tuple[int, int] | None = None
        self._cached_svd: tuple[OperatorArray, FloatArray, OperatorArray] | None = None

    def _get_svd(
        self,
        A: OperatorArray,
    ) -> tuple[OperatorArray, FloatArray, OperatorArray]:
        if (
            self.precompute_svd
            and self._cached_svd is not None
            and self._cached_shape == A.shape
        ):
            return self._cached_svd

        svd = compute_svd(A)
        if self.precompute_svd:
            self._cached_svd = svd
            self._cached_shape = A.shape
        return svd

    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        problem.validate()

        if self.use_svd:
            U, s, Vh = self._get_svd(problem.A)
            x_hat = tikhonov_svd_solve(U, s, Vh, problem.y, self.lambda_value)
        else:
            x_hat = tikhonov_direct_solve(problem.A, problem.y, self.lambda_value)

        residual_norm = compute_residual_norm(problem.A, x_hat, problem.y)
        solution_norm = compute_solution_norm(x_hat)
        objective = residual_norm**2 + self.lambda_value * solution_norm**2

        return ReconstructionResult(
            x_hat=x_hat,
            residual_norm=residual_norm,
            solution_norm=solution_norm,
            solver_name=self.name,
            objective_value=float(objective),
            lambda_value=self.lambda_value,
            metadata={
                "use_svd": self.use_svd,
                "precompute_svd": self.precompute_svd,
            },
        )


class LCurveTikhonovSolver(InverseSolver):
    """Tikhonov solver with lambda selected by L-curve maximum curvature."""

    name = "tikhonov_lcurve"

    def __init__(
        self,
        lambdas: FloatArray | None = None,
        *,
        lambda_min_exp: float = 0.0,
        lambda_max_exp: float = 6.0,
        n_lambda: int = 200,
    ) -> None:
        self.lambdas = (
            np.asarray(lambdas, dtype=np.float64)
            if lambdas is not None
            else make_lambda_grid(
                lambda_min_exp=lambda_min_exp,
                lambda_max_exp=lambda_max_exp,
                n_lambda=n_lambda,
            )
        )

    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        problem.validate()

        svd = compute_svd(problem.A)
        sweep = compute_lcurve_sweep(
            A=problem.A,
            y=problem.y,
            lambdas=self.lambdas,
            precomputed_svd=svd,
        )

        x_hat = sweep.x_hats[sweep.idx_opt]
        residual_norm = sweep.residual_norms[sweep.idx_opt]
        solution_norm = sweep.solution_norms[sweep.idx_opt]
        objective = residual_norm**2 + sweep.lambda_opt * solution_norm**2

        return ReconstructionResult(
            x_hat=x_hat,
            residual_norm=float(residual_norm),
            solution_norm=float(solution_norm),
            solver_name=self.name,
            objective_value=float(objective),
            lambda_value=float(sweep.lambda_opt),
            metadata={
                "lambdas": sweep.lambdas,
                "residual_norms": sweep.residual_norms,
                "solution_norms": sweep.solution_norms,
                "curvature": sweep.curvature,
                "idx_opt": sweep.idx_opt,
            },
        )