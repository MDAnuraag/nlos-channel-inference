"""Experiment runner for AWGN robustness sweeps with a multi-state operator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from nlos_cs.inverse.base import InverseProblem, InverseSolver
from nlos_cs.io.artifacts import init_run_dir, save_json, save_named_arrays, write_manifest
from nlos_cs.perturb.awgn import add_awgn, snr_db_from_noise_fraction

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = FloatArray | ComplexArray


@dataclass(frozen=True)
class MultiStateRobustnessConfig:
    """Configuration for one multi-state AWGN robustness sweep."""

    run_name: str
    output_root: str = "outputs"

    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.10)
    n_trials: int = 50
    random_seed: int = 42

    save_mean_x_hats: bool = True
    save_peak_count_cube: bool = True


@dataclass(frozen=True)
class MultiStateRobustnessExperimentResult:
    """Outputs from one multi-state robustness experiment."""

    run_dir: Path
    report: dict[str, Any]

    success_rate_matrix: FloatArray
    mean_peak_margin_matrix: FloatArray
    std_peak_margin_matrix: FloatArray

    mean_x_hats: npt.NDArray[np.float64] | npt.NDArray[np.complex128] | None
    peak_count_cube: npt.NDArray[np.int64] | None


def _extract_multi_state_data(
    *,
    operator: Any | None,
    A: OperatorArray | None,
    positions_mm: FloatArray | None,
) -> tuple[OperatorArray, FloatArray | None, dict[str, Any], dict[str, slice]]:
    """Extract A, positions, metadata, and optional row slices."""
    if operator is not None:
        if not hasattr(operator, "A"):
            raise ValueError("operator must have attribute 'A'")

        A_out = operator.A
        positions_out = getattr(operator, "positions_mm", None)
        row_slices = getattr(operator, "row_slices", {})

        meta = {
            "operator_type": type(operator).__name__,
        }
        if hasattr(operator, "state_ids"):
            meta["state_ids"] = list(operator.state_ids)
        if hasattr(operator, "measurement_kind"):
            meta["measurement_kind"] = operator.measurement_kind
        if hasattr(operator, "n_states"):
            meta["n_states"] = operator.n_states

        return A_out, positions_out, meta, dict(row_slices)

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

    return A, positions_arr, {"operator_type": "explicit_matrix"}, {}


def _peak_margin(x_hat: OperatorArray) -> float:
    """Return max(|x|) - second_max(|x|)."""
    vals = np.sort(np.abs(x_hat).astype(np.float64, copy=False))[::-1]
    if vals.size == 0:
        raise ValueError("x_hat must not be empty")
    if vals.size == 1:
        return float(vals[0])
    return float(vals[0] - vals[1])


def _per_noise_summary(
    *,
    success_rate_matrix: FloatArray,
    positions_mm: FloatArray | None,
    noise_levels: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Create per-noise summary rows."""
    rows: list[dict[str, Any]] = []
    for k, nl in enumerate(noise_levels):
        row = {
            "noise_fraction_of_rms": float(nl),
            "snr_db_nominal": snr_db_from_noise_fraction(float(nl)),
            "mean_success_rate": float(np.mean(success_rate_matrix[:, k])),
            "min_success_rate": float(np.min(success_rate_matrix[:, k])),
            "max_success_rate": float(np.max(success_rate_matrix[:, k])),
        }
        if positions_mm is not None:
            worst_idx = int(np.argmin(success_rate_matrix[:, k]))
            best_idx = int(np.argmax(success_rate_matrix[:, k]))
            row["worst_position_index"] = worst_idx
            row["worst_position_mm"] = float(positions_mm[worst_idx])
            row["best_position_index"] = best_idx
            row["best_position_mm"] = float(positions_mm[best_idx])
        rows.append(row)
    return rows


def run_multi_state_robustness_experiment(
    *,
    solver: InverseSolver,
    config: MultiStateRobustnessConfig,
    operator: Any | None = None,
    A: OperatorArray | None = None,
    positions_mm: FloatArray | None = None,
    measurement_metadata: dict[str, Any] | None = None,
) -> MultiStateRobustnessExperimentResult:
    """Run an AWGN robustness sweep using a multi-state operator."""
    A_use, positions_use, operator_meta, row_slices = _extract_multi_state_data(
        operator=operator,
        A=A,
        positions_mm=positions_mm,
    )

    if A_use.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A_use.shape}")
    if config.n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if len(config.noise_levels) == 0:
        raise ValueError("noise_levels must not be empty")
    if any(nl < 0.0 for nl in config.noise_levels):
        raise ValueError("noise_levels must all be non-negative")

    _, n_positions = A_use.shape
    noise_levels = tuple(float(v) for v in config.noise_levels)
    n_noise = len(noise_levels)

    success_rate_matrix = np.zeros((n_positions, n_noise), dtype=np.float64)
    mean_peak_margin_matrix = np.zeros((n_positions, n_noise), dtype=np.float64)
    std_peak_margin_matrix = np.zeros((n_positions, n_noise), dtype=np.float64)

    x_dtype = np.complex128 if np.iscomplexobj(A_use) else np.float64
    mean_x_hats = (
        np.zeros((n_positions, n_noise, n_positions), dtype=x_dtype)
        if config.save_mean_x_hats
        else None
    )
    peak_count_cube = (
        np.zeros((n_positions, n_noise, n_positions), dtype=np.int64)
        if config.save_peak_count_cube
        else None
    )

    for i in range(n_positions):
        y_true = A_use[:, i]

        for k, noise_level in enumerate(noise_levels):
            successes = 0
            margins: list[float] = []
            x_hats_this_cell = np.zeros((config.n_trials, n_positions), dtype=x_dtype)

            for trial in range(config.n_trials):
                trial_seed = int(config.random_seed + 10_000 * i + 100 * k + trial)

                awgn = add_awgn(
                    y_true,
                    noise_fraction_of_rms=noise_level,
                    random_seed=trial_seed,
                )
                y_noisy = awgn.y_noisy

                problem = InverseProblem(
                    A=A_use,
                    y=y_noisy,
                    positions_mm=positions_use,
                    metadata={
                        **operator_meta,
                        **({} if measurement_metadata is None else dict(measurement_metadata)),
                        "true_index": i,
                        "noise_fraction_of_rms": noise_level,
                        "trial_index": trial,
                        "trial_seed": trial_seed,
                    },
                )
                problem.validate()

                recon = solver.solve(problem)
                x_hat = recon.x_hat
                x_hats_this_cell[trial] = x_hat

                pred_idx = int(np.argmax(np.abs(x_hat)))
                if pred_idx == i:
                    successes += 1

                margins.append(_peak_margin(x_hat))

                if peak_count_cube is not None:
                    peak_count_cube[i, k, pred_idx] += 1

            success_rate_matrix[i, k] = successes / config.n_trials
            mean_peak_margin_matrix[i, k] = float(np.mean(margins))
            std_peak_margin_matrix[i, k] = float(np.std(margins))

            if mean_x_hats is not None:
                mean_x_hats[i, k] = np.mean(x_hats_this_cell, axis=0)

    run_dir = init_run_dir(config.output_root, config.run_name)

    arrays_to_save: dict[str, np.ndarray] = {
        "success_rate_matrix": success_rate_matrix,
        "mean_peak_margin_matrix": mean_peak_margin_matrix,
        "std_peak_margin_matrix": std_peak_margin_matrix,
        "noise_levels": np.asarray(noise_levels, dtype=np.float64),
    }
    if positions_use is not None:
        arrays_to_save["positions_mm"] = positions_use
    if mean_x_hats is not None:
        arrays_to_save["mean_x_hats"] = mean_x_hats
    if peak_count_cube is not None:
        arrays_to_save["peak_count_cube"] = peak_count_cube

    save_named_arrays(arrays_to_save, run_dir / "arrays")

    report = {
        "experiment_type": "run_multi_state_robustness",
        "solver_name": solver.name,
        "operator": operator_meta,
        "row_slices": {
            state_id: [sl.start, sl.stop] for state_id, sl in row_slices.items()
        },
        "config": {
            "noise_levels": list(noise_levels),
            "n_trials": config.n_trials,
            "random_seed": config.random_seed,
            "save_mean_x_hats": config.save_mean_x_hats,
            "save_peak_count_cube": config.save_peak_count_cube,
        },
        "measurement_metadata": {} if measurement_metadata is None else measurement_metadata,
        "summary": {
            "n_positions": n_positions,
            "overall_mean_success_rate": float(np.mean(success_rate_matrix)),
            "overall_min_success_rate": float(np.min(success_rate_matrix)),
            "overall_max_success_rate": float(np.max(success_rate_matrix)),
            "per_noise_level": _per_noise_summary(
                success_rate_matrix=success_rate_matrix,
                positions_mm=positions_use,
                noise_levels=noise_levels,
            ),
        },
    }
    save_json(report, run_dir / "reports" / "multi_state_robustness_report.json")

    manifest = {
        "artifact_type": "multi_state_robustness",
        "arrays": {name: f"arrays/{name}.npy" for name in arrays_to_save},
        "metadata": {
            "solver_name": solver.name,
            "operator": operator_meta,
            "row_slices": {
                state_id: [sl.start, sl.stop] for state_id, sl in row_slices.items()
            },
            "n_trials": config.n_trials,
            "noise_levels": list(noise_levels),
        },
    }
    write_manifest(run_dir, manifest=manifest)

    return MultiStateRobustnessExperimentResult(
        run_dir=run_dir,
        report=report,
        success_rate_matrix=success_rate_matrix,
        mean_peak_margin_matrix=mean_peak_margin_matrix,
        std_peak_margin_matrix=std_peak_margin_matrix,
        mean_x_hats=mean_x_hats,
        peak_count_cube=peak_count_cube,
    )