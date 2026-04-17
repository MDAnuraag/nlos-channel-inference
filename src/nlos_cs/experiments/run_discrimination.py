"""Experiment runner for discrimination / leakage analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from nlos_cs.inverse.base import InverseSolver
from nlos_cs.io.artifacts import init_run_dir, save_json, save_named_arrays, write_manifest
from nlos_cs.metrics.discrimination import (
    DiscriminationResult,
    compute_discrimination_from_measurements,
    compute_discrimination_from_operator,
    group_leakage_summary,
    hardest_pairs,
)

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = FloatArray | ComplexArray


@dataclass(frozen=True)
class DiscriminationConfig:
    """Configuration for one discrimination experiment."""

    run_name: str
    output_root: str = "outputs"

    mode: str = "matched"  # "matched" | "measurements"
    top_k_pairs: int = 10

    save_x_hats: bool = True
    save_measurements: bool = False


@dataclass(frozen=True)
class DiscriminationExperimentResult:
    """Outputs from one discrimination experiment."""

    result: DiscriminationResult
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


def _result_summary(
    result: DiscriminationResult,
    *,
    top_k_pairs: int,
    groups: dict[str, set[int]] | None = None,
) -> dict[str, Any]:
    """Build a compact JSON-safe report summary."""
    worst_i, worst_j, worst_leak = result.worst_pair()

    summary: dict[str, Any] = {
        "n_positions": result.n_positions,
        "mean_off_diagonal_leakage": result.mean_off_diagonal_leakage(),
        "mean_off_diagonal_discrimination": result.mean_off_diagonal_discrimination(),
        "worst_pair": {
            "i": worst_i,
            "j": worst_j,
            "leakage": worst_leak,
        },
        "hardest_pairs": [
            {"i": i, "j": j, "leakage": val}
            for i, j, val in hardest_pairs(result, top_k=top_k_pairs)
        ],
        "peak_indices": result.peak_indices.tolist(),
    }

    if result.positions_mm is not None:
        summary["positions_mm"] = result.positions_mm.tolist()
        summary["worst_pair"]["i_position_mm"] = float(result.positions_mm[worst_i])
        summary["worst_pair"]["j_position_mm"] = float(result.positions_mm[worst_j])

        for item in summary["hardest_pairs"]:
            i = item["i"]
            j = item["j"]
            item["i_position_mm"] = float(result.positions_mm[i])
            item["j_position_mm"] = float(result.positions_mm[j])

    if groups is not None:
        summary["group_leakage_summary"] = group_leakage_summary(result, groups)

    return summary


def run_discrimination_experiment(
    *,
    solver: InverseSolver,
    config: DiscriminationConfig,
    operator: Any | None = None,
    A: OperatorArray | None = None,
    positions_mm: FloatArray | None = None,
    Y: OperatorArray | None = None,
    true_indices: npt.NDArray[np.int64] | None = None,
    groups: dict[str, set[int]] | None = None,
    measurement_metadata: dict[str, Any] | None = None,
) -> DiscriminationExperimentResult:
    """Run a discrimination / leakage experiment.

    Modes
    -----
    matched:
        Reconstruct each operator column A[:, i] and compute square discrimination.

    measurements:
        Reconstruct each row of Y against A and compute square discrimination.
        This is the route for perturbed or mismatched measurements.
    """
    A_use, positions_use, operator_meta = _extract_operator_data(
        operator=operator,
        A=A,
        positions_mm=positions_mm,
    )

    if config.mode == "matched":
        result = compute_discrimination_from_operator(
            A=A_use,
            solver=solver,
            positions_mm=positions_use,
            metadata=measurement_metadata,
        )
        measurements_used = None

    elif config.mode == "measurements":
        if Y is None:
            raise ValueError("mode='measurements' requires Y=...")

        Y_use = np.asarray(Y)
        result = compute_discrimination_from_measurements(
            A=A_use,
            Y=Y_use,
            solver=solver,
            true_indices=true_indices,
            positions_mm=positions_use,
            metadata_rows=None,
        )
        measurements_used = Y_use

    else:
        raise ValueError(f"Unknown discrimination mode: {config.mode}")

    run_dir = init_run_dir(config.output_root, config.run_name)

    arrays_to_save: dict[str, OperatorArray] = {
        "discrimination": result.discrimination,
        "leakage": result.leakage,
        "reference_values": result.reference_values,
        "peak_indices": result.peak_indices,
    }

    if config.save_x_hats:
        arrays_to_save["x_hats"] = result.x_hats
    if config.mode == "measurements" and config.save_measurements and measurements_used is not None:
        arrays_to_save["Y"] = measurements_used
    if positions_use is not None:
        arrays_to_save["positions_mm"] = positions_use

    save_named_arrays(arrays_to_save, run_dir / "arrays")

    report = {
        "experiment_type": "run_discrimination",
        "mode": config.mode,
        "solver_name": solver.name,
        "operator": operator_meta,
        "measurement_metadata": {} if measurement_metadata is None else measurement_metadata,
        "summary": _result_summary(
            result,
            top_k_pairs=config.top_k_pairs,
            groups=groups,
        ),
    }
    save_json(report, run_dir / "reports" / "discrimination_report.json")

    manifest = {
        "artifact_type": "discrimination",
        "arrays": {name: f"arrays/{name}.npy" for name in arrays_to_save},
        "metadata": {
            "mode": config.mode,
            "solver_name": solver.name,
            "operator": operator_meta,
        },
    }
    write_manifest(run_dir, manifest=manifest)

    return DiscriminationExperimentResult(
        result=result,
        run_dir=run_dir,
        report=report,
    )