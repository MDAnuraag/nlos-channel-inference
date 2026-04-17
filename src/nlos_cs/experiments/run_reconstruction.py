"""Experiment runner for one reconstruction solve."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from nlos_cs.inverse.base import (
    InverseProblem,
    InverseSolver,
    ReconstructionResult,
    compute_residual,
)
from nlos_cs.io.artifacts import init_run_dir, save_json, save_reconstruction_artifact

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = FloatArray | ComplexArray

MeasurementMode = Literal["provided", "operator_column"]


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration for one reconstruction experiment."""

    run_name: str
    output_root: str = "outputs"

    measurement_mode: MeasurementMode = "operator_column"
    true_index: int | None = None

    noise_fraction_of_rms: float = 0.0
    random_seed: int = 42

    save_measurement: bool = True
    save_residual: bool = True


@dataclass(frozen=True)
class ReconstructionExperimentResult:
    """Outputs from one reconstruction experiment."""

    reconstruction: ReconstructionResult
    y: OperatorArray
    y_true: OperatorArray | None
    residual: OperatorArray
    run_dir: Path
    report: dict[str, Any]


def _extract_operator_data(
    *,
    operator: Any | None,
    A: OperatorArray | None,
    positions_mm: FloatArray | None,
) -> tuple[OperatorArray, FloatArray | None, dict[str, Any]]:
    """Extract A and positions from an operator object or explicit arrays."""
    if operator is not None:
        if not hasattr(operator, "A"):
            raise ValueError("operator must have attribute 'A'")
        A_out = operator.A
        positions_out = getattr(operator, "positions_mm", None)

        meta = {
            "operator_type": type(operator).__name__,
        }
        if hasattr(operator, "state_id"):
            meta["state_id"] = operator.state_id
        if hasattr(operator, "state_ids"):
            meta["state_ids"] = list(operator.state_ids)
        if hasattr(operator, "measurement_kind"):
            meta["measurement_kind"] = operator.measurement_kind

        return A_out, positions_out, meta

    if A is None:
        raise ValueError("Provide either operator=... or A=...")

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    if positions_mm is not None:
        positions_arr = np.asarray(positions_mm, dtype=np.float64)
        if positions_arr.ndim != 1:
            raise ValueError("positions_mm must be 1D")
        if positions_arr.shape[0] != A.shape[1]:
            raise ValueError(
                f"positions_mm length {positions_arr.shape[0]} does not match A columns {A.shape[1]}"
            )
    else:
        positions_arr = None

    return A, positions_arr, {"operator_type": "explicit_matrix"}


def make_operator_column_measurement(
    A: OperatorArray,
    *,
    true_index: int,
    noise_fraction_of_rms: float = 0.0,
    random_seed: int = 42,
) -> tuple[OperatorArray, OperatorArray]:
    """Construct a synthetic measurement from one operator column.

    Returns
    -------
    y, y_true
        y_true is the clean column A[:, true_index]
        y is the optionally noise-corrupted version
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    n_cols = A.shape[1]
    if true_index < 0 or true_index >= n_cols:
        raise ValueError(f"true_index {true_index} out of range for {n_cols} columns")

    if noise_fraction_of_rms < 0.0:
        raise ValueError("noise_fraction_of_rms must be non-negative")

    y_true = A[:, true_index].copy()
    y = y_true.copy()

    if noise_fraction_of_rms > 0.0:
        rng = np.random.default_rng(random_seed)
        rms = float(np.sqrt(np.mean(np.abs(y_true) ** 2)))

        if np.iscomplexobj(y_true):
            noise = (
                rng.standard_normal(size=y_true.shape) + 1j * rng.standard_normal(size=y_true.shape)
            ) * (noise_fraction_of_rms * rms / np.sqrt(2.0))
        else:
            noise = rng.standard_normal(size=y_true.shape) * (noise_fraction_of_rms * rms)

        y = y + noise.astype(y.dtype, copy=False)

    return y, y_true


def run_reconstruction_experiment(
    *,
    solver: InverseSolver,
    config: ReconstructionConfig,
    operator: Any | None = None,
    A: OperatorArray | None = None,
    positions_mm: FloatArray | None = None,
    y: OperatorArray | None = None,
    measurement_metadata: dict[str, Any] | None = None,
) -> ReconstructionExperimentResult:
    """Run one reconstruction experiment.

    Modes
    -----
    provided:
        Use the supplied y directly.

    operator_column:
        Build y from A[:, true_index], optionally with additive noise.
    """
    A_use, positions_use, operator_meta = _extract_operator_data(
        operator=operator,
        A=A,
        positions_mm=positions_mm,
    )

    if config.measurement_mode == "provided":
        if y is None:
            raise ValueError("measurement_mode='provided' requires y=...")
        y_use = np.asarray(y)
        y_true = None

    elif config.measurement_mode == "operator_column":
        if config.true_index is None:
            raise ValueError("measurement_mode='operator_column' requires true_index")
        y_use, y_true = make_operator_column_measurement(
            A_use,
            true_index=config.true_index,
            noise_fraction_of_rms=config.noise_fraction_of_rms,
            random_seed=config.random_seed,
        )

    else:
        raise ValueError(f"Unknown measurement_mode: {config.measurement_mode}")

    problem = InverseProblem(
        A=A_use,
        y=y_use,
        positions_mm=positions_use,
        metadata={
            **operator_meta,
            **({} if measurement_metadata is None else dict(measurement_metadata)),
        },
    )
    problem.validate()

    recon = solver.solve(problem)
    residual = compute_residual(problem.A, recon.x_hat, problem.y)

    run_dir = init_run_dir(config.output_root, config.run_name)

    extra_arrays: dict[str, OperatorArray] = {}
    if y_true is not None:
        extra_arrays["y_true"] = y_true

    save_reconstruction_artifact(
        run_dir,
        x_hat=recon.x_hat,
        y=y_use if config.save_measurement else None,
        residual=residual if config.save_residual else None,
        extra_arrays=extra_arrays if extra_arrays else None,
        metadata={
            "solver_name": recon.solver_name,
            "lambda_value": recon.lambda_value,
            "objective_value": recon.objective_value,
            "peak_index": recon.peak_index,
            "peak_position_mm": recon.peak_position_mm(positions_use),
            "residual_norm": recon.residual_norm,
            "solution_norm": recon.solution_norm,
            "measurement_mode": config.measurement_mode,
            "true_index": config.true_index,
            "noise_fraction_of_rms": config.noise_fraction_of_rms,
            "random_seed": config.random_seed,
            "operator_meta": operator_meta,
            "solver_metadata": recon.metadata,
        },
    )

    report = {
        "experiment_type": "run_reconstruction",
        "solver_name": recon.solver_name,
        "lambda_value": recon.lambda_value,
        "objective_value": recon.objective_value,
        "residual_norm": recon.residual_norm,
        "solution_norm": recon.solution_norm,
        "peak_index": recon.peak_index,
        "peak_position_mm": recon.peak_position_mm(positions_use),
        "measurement_mode": config.measurement_mode,
        "true_index": config.true_index,
        "true_position_mm": (
            None
            if positions_use is None or config.true_index is None
            else float(positions_use[config.true_index])
        ),
        "noise_fraction_of_rms": config.noise_fraction_of_rms,
        "random_seed": config.random_seed,
        "operator": operator_meta,
        "measurement_metadata": {} if measurement_metadata is None else measurement_metadata,
    }
    save_json(report, run_dir / "reports" / "reconstruction_report.json")

    return ReconstructionExperimentResult(
        reconstruction=recon,
        y=y_use,
        y_true=y_true,
        residual=residual,
        run_dir=run_dir,
        report=report,
    )