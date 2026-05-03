"""Experiment runner for robustness sweeps under composed perturbations.

This extends the AWGN-only robustness runner by allowing a configurable
perturbation pipeline, including:
- multipath leakage
- correlated additive noise
- AWGN
- dropout

The perturbation order is delegated to `apply_combined_perturbations(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from nlos_cs.inverse.base import InverseProblem, InverseSolver
from nlos_cs.io.artifacts import init_run_dir, save_json, save_named_arrays, write_manifest
from nlos_cs.perturb.combined import (
    CombinedPerturbationConfig,
    apply_combined_perturbations,
)

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = FloatArray | ComplexArray


@dataclass(frozen=True)
class CombinedRobustnessConfig:
    """Configuration for one combined-perturbation robustness sweep."""

    run_name: str
    perturb_config: CombinedPerturbationConfig

    output_root: str = "outputs"
    n_trials: int = 50
    random_seed: int = 42

    save_mean_x_hats: bool = True
    save_peak_count_cube: bool = True


@dataclass(frozen=True)
class CombinedRobustnessExperimentResult:
    """Outputs from one combined-perturbation robustness experiment."""

    run_dir: Path
    report: dict[str, Any]

    success_rate: FloatArray
    mean_peak_margin: FloatArray
    std_peak_margin: FloatArray

    mean_x_hats: npt.NDArray[np.float64] | npt.NDArray[np.complex128] | None
    peak_count_cube: npt.NDArray[np.int64] | None


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


def _peak_margin(x_hat: OperatorArray) -> float:
    """Return max(|x|) - second_max(|x|)."""
    vals = np.sort(np.abs(x_hat).astype(np.float64, copy=False))[::-1]
    if vals.size == 0:
        raise ValueError("x_hat must not be empty")
    if vals.size == 1:
        return float(vals[0])
    return float(vals[0] - vals[1])


def _config_summary(cfg: CombinedPerturbationConfig) -> dict[str, Any]:
    """Return JSON-safe summary of perturbation config."""
    return {
        "apply_awgn": cfg.apply_awgn,
        "awgn_fraction_of_rms": cfg.awgn_fraction_of_rms,
        "apply_correlated": cfg.apply_correlated,
        "correlated_fraction_of_rms": cfg.correlated_fraction_of_rms,
        "corr_length": cfg.corr_length,
        "apply_dropout": cfg.apply_dropout,
        "dropout_fraction": cfg.dropout_fraction,
        "apply_multipath": cfg.apply_multipath,
        "multipath_fraction_of_rms": cfg.multipath_fraction_of_rms,
        "n_leak": cfg.n_leak,
        "exclude_index": cfg.exclude_index,
    }


def run_combined_robustness_experiment(
    *,
    solver: InverseSolver,
    config: CombinedRobustnessConfig,
    operator: Any | None = None,
    A: OperatorArray | None = None,
    positions_mm: FloatArray | None = None,
    measurement_metadata: dict[str, Any] | None = None,
) -> CombinedRobustnessExperimentResult:
    """Run a robustness sweep under composed perturbations.

    For each true position i:
        - start from y_true = A[:, i]
        - run n_trials independently
        - apply configured perturbation pipeline
        - reconstruct x_hat
        - record:
            success rate
            peak margin statistics
            mean reconstruction
            predicted-peak histogram
    """
    A_use, positions_use, operator_meta = _extract_operator_data(
        operator=operator,
        A=A,
        positions_mm=positions_mm,
    )

    if A_use.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A_use.shape}")
    if config.n_trials <= 0:
        raise ValueError("n_trials must be positive")

    n_measurements, n_positions = A_use.shape

    success_rate = np.zeros(n_positions, dtype=np.float64)
    mean_peak_margin = np.zeros(n_positions, dtype=np.float64)
    std_peak_margin = np.zeros(n_positions, dtype=np.float64)

    x_dtype = np.complex128 if np.iscomplexobj(A_use) else np.float64
    mean_x_hats = (
        np.zeros((n_positions, n_positions), dtype=x_dtype)
        if config.save_mean_x_hats
        else None
    )
    peak_count_cube = (
        np.zeros((n_positions, n_positions), dtype=np.int64)
        if config.save_peak_count_cube
        else None
    )

    stage_usage: dict[str, int] = {
        "multipath": 0,
        "correlated": 0,
        "awgn": 0,
        "dropout": 0,
    }

    for i in range(n_positions):
        y_true = A_use[:, i]
        margins: list[float] = []
        successes = 0
        x_hats_this_position = np.zeros((config.n_trials, n_positions), dtype=x_dtype)

        # Per-position perturbation config: fill exclude_index for multipath automatically
        per_position_cfg = CombinedPerturbationConfig(
            apply_awgn=config.perturb_config.apply_awgn,
            awgn_fraction_of_rms=config.perturb_config.awgn_fraction_of_rms,
            apply_correlated=config.perturb_config.apply_correlated,
            correlated_fraction_of_rms=config.perturb_config.correlated_fraction_of_rms,
            corr_length=config.perturb_config.corr_length,
            apply_dropout=config.perturb_config.apply_dropout,
            dropout_fraction=config.perturb_config.dropout_fraction,
            apply_multipath=config.perturb_config.apply_multipath,
            multipath_fraction_of_rms=config.perturb_config.multipath_fraction_of_rms,
            n_leak=config.perturb_config.n_leak,
            exclude_index=i if config.perturb_config.exclude_index is None else config.perturb_config.exclude_index,
        )

        for trial in range(config.n_trials):
            trial_seed = int(config.random_seed + 10_000 * i + trial)

            perturbed = apply_combined_perturbations(
                y_true,
                config=per_position_cfg,
                A=A_use,
                random_seed=trial_seed,
            )

            for stage in perturbed.applied_stages:
                stage_usage[stage] += 1

            problem = InverseProblem(
                A=A_use,
                y=perturbed.y_perturbed,
                positions_mm=positions_use,
                metadata={
                    **operator_meta,
                    **({} if measurement_metadata is None else dict(measurement_metadata)),
                    "true_index": i,
                    "trial_index": trial,
                    "trial_seed": trial_seed,
                    "applied_stages": list(perturbed.applied_stages),
                },
            )
            problem.validate()

            recon = solver.solve(problem)
            x_hat = recon.x_hat
            x_hats_this_position[trial] = x_hat

            pred_idx = int(np.argmax(np.abs(x_hat)))
            if pred_idx == i:
                successes += 1

            margins.append(_peak_margin(x_hat))

            if peak_count_cube is not None:
                peak_count_cube[i, pred_idx] += 1

        success_rate[i] = successes / config.n_trials
        mean_peak_margin[i] = float(np.mean(margins))
        std_peak_margin[i] = float(np.std(margins))

        if mean_x_hats is not None:
            mean_x_hats[i] = np.mean(x_hats_this_position, axis=0)

    run_dir = init_run_dir(config.output_root, config.run_name)

    arrays_to_save: dict[str, np.ndarray] = {
        "success_rate": success_rate,
        "mean_peak_margin": mean_peak_margin,
        "std_peak_margin": std_peak_margin,
    }
    if positions_use is not None:
        arrays_to_save["positions_mm"] = positions_use
    if mean_x_hats is not None:
        arrays_to_save["mean_x_hats"] = mean_x_hats
    if peak_count_cube is not None:
        arrays_to_save["peak_count_cube"] = peak_count_cube

    save_named_arrays(arrays_to_save, run_dir / "arrays")

    report = {
        "experiment_type": "run_combined_robustness",
        "solver_name": solver.name,
        "operator": operator_meta,
        "config": {
            "n_trials": config.n_trials,
            "random_seed": config.random_seed,
            "perturb_config": _config_summary(config.perturb_config),
            "save_mean_x_hats": config.save_mean_x_hats,
            "save_peak_count_cube": config.save_peak_count_cube,
        },
        "measurement_metadata": {} if measurement_metadata is None else measurement_metadata,
        "summary": {
            "n_measurements": n_measurements,
            "n_positions": n_positions,
            "overall_mean_success_rate": float(np.mean(success_rate)),
            "overall_min_success_rate": float(np.min(success_rate)),
            "overall_max_success_rate": float(np.max(success_rate)),
            "overall_mean_peak_margin": float(np.mean(mean_peak_margin)),
            "stage_usage_counts": stage_usage,
        },
    }
    if positions_use is not None:
        worst_idx = int(np.argmin(success_rate))
        best_idx = int(np.argmax(success_rate))
        report["summary"]["worst_position_index"] = worst_idx
        report["summary"]["worst_position_mm"] = float(positions_use[worst_idx])
        report["summary"]["best_position_index"] = best_idx
        report["summary"]["best_position_mm"] = float(positions_use[best_idx])

    save_json(report, run_dir / "reports" / "combined_robustness_report.json")

    manifest = {
        "artifact_type": "combined_robustness",
        "arrays": {name: f"arrays/{name}.npy" for name in arrays_to_save},
        "metadata": {
            "solver_name": solver.name,
            "operator": operator_meta,
            "n_trials": config.n_trials,
            "perturb_config": _config_summary(config.perturb_config),
        },
    }
    write_manifest(run_dir, manifest=manifest)

    return CombinedRobustnessExperimentResult(
        run_dir=run_dir,
        report=report,
        success_rate=success_rate,
        mean_peak_margin=mean_peak_margin,
        std_peak_margin=std_peak_margin,
        mean_x_hats=mean_x_hats,
        peak_count_cube=peak_count_cube,
    )