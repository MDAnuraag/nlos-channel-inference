"""Base interfaces and result containers for inverse solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = FloatArray | ComplexArray


@dataclass(frozen=True)
class InverseProblem:
    """One linear inverse problem instance.

    Model
    -----
        y = A x + eps
    """

    A: OperatorArray
    y: FloatArray | ComplexArray
    positions_mm: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_measurements(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_unknowns(self) -> int:
        return int(self.A.shape[1])

    @property
    def is_complex(self) -> bool:
        return np.iscomplexobj(self.A) or np.iscomplexobj(self.y)

    def validate(self) -> None:
        """Raise if shapes are inconsistent."""
        if self.A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {self.A.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {self.y.shape}")
        if self.A.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"Row mismatch: A has {self.A.shape[0]} rows but y has length {self.y.shape[0]}"
            )
        if self.positions_mm is not None:
            if self.positions_mm.ndim != 1:
                raise ValueError("positions_mm must be 1D")
            if self.positions_mm.shape[0] != self.A.shape[1]:
                raise ValueError(
                    f"Column mismatch: A has {self.A.shape[1]} columns but "
                    f"positions_mm has length {self.positions_mm.shape[0]}"
                )


@dataclass(frozen=True)
class ReconstructionResult:
    """Standard solver output."""

    x_hat: FloatArray | ComplexArray
    residual_norm: float
    solution_norm: float
    solver_name: str
    objective_value: float | None = None
    lambda_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def peak_index(self) -> int:
        return int(np.argmax(np.abs(self.x_hat)))

    @property
    def peak_value(self) -> float:
        return float(np.max(np.abs(self.x_hat)))

    def peak_position_mm(self, positions_mm: FloatArray | None) -> float | None:
        """Return the physical position of the dominant component if available."""
        if positions_mm is None:
            return None
        return float(positions_mm[self.peak_index])


class InverseSolver(ABC):
    """Abstract base class for all inverse solvers."""

    name: str = "inverse_solver"

    @abstractmethod
    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        """Solve the inverse problem and return a reconstruction."""
        raise NotImplementedError


def compute_residual(
    A: OperatorArray,
    x_hat: FloatArray | ComplexArray,
    y: FloatArray | ComplexArray,
) -> FloatArray | ComplexArray:
    """Compute residual vector r = A x_hat - y."""
    return A @ x_hat - y


def compute_residual_norm(
    A: OperatorArray,
    x_hat: FloatArray | ComplexArray,
    y: FloatArray | ComplexArray,
) -> float:
    """Compute ||A x_hat - y||_2."""
    r = compute_residual(A, x_hat, y)
    return float(np.linalg.norm(r))


def compute_solution_norm(x_hat: FloatArray | ComplexArray) -> float:
    """Compute ||x_hat||_2."""
    return float(np.linalg.norm(x_hat))


def normalise_by_peak(x_hat: FloatArray | ComplexArray) -> FloatArray | ComplexArray:
    """Scale x_hat so that max(|x_hat|) = 1, if possible."""
    peak = np.max(np.abs(x_hat))
    if peak <= 1e-15:
        return x_hat.copy()
    return x_hat / peak


def to_real_problem(problem: InverseProblem) -> InverseProblem:
    """Convert a complex problem into an equivalent real-stacked problem.

    If A and y are real, the problem is returned unchanged.

    For complex A and y, construct:

        [Re(y)]   [Re(A) -Im(A)] [Re(x)]
        [Im(y)] = [Im(A)  Re(A)] [Im(x)]

    This is useful for solvers implemented only for real-valued variables.
    """
    problem.validate()

    if not problem.is_complex:
        return problem

    A = np.asarray(problem.A)
    y = np.asarray(problem.y)

    A_real = np.block([
        [A.real, -A.imag],
        [A.imag,  A.real],
    ])
    y_real = np.concatenate([y.real, y.imag], axis=0)

    if problem.positions_mm is None:
        positions_mm = None
    else:
        positions_mm = np.concatenate([problem.positions_mm, problem.positions_mm], axis=0)

    meta = dict(problem.metadata)
    meta["derived_from_complex"] = True

    return InverseProblem(
        A=A_real.astype(np.float64, copy=False),
        y=y_real.astype(np.float64, copy=False),
        positions_mm=positions_mm.astype(np.float64, copy=False) if positions_mm is not None else None,
        metadata=meta,
    )