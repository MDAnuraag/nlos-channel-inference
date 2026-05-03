"""Config-driven experiment dispatch for the CLI.

This module is the first bridge between:
- JSON config files
- experiment runner functions

It intentionally supports only a small set of experiment types at first.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nlos_cs.experiments.build_operator import (
    BuildOperatorConfig,
    PlaneSpec,
    build_single_state_operator_experiment,
)
from nlos_cs.experiments.run_discrimination import (
    DiscriminationConfig,
    run_discrimination_experiment,
)
from nlos_cs.experiments.run_reconstruction import (
    ReconstructionConfig,
    run_reconstruction_experiment,
)
from nlos_cs.experiments.run_robustness import (
    RobustnessConfig,
    run_awgn_robustness_experiment,
)
from nlos_cs.inverse.huber_simplex import HuberSimplexConfig, HuberSimplexSolver
from nlos_cs.inverse.lasso import LassoSolver, LassoSolverConfig
from nlos_cs.inverse.nnls import NNLSSolver, NNLSSolverConfig
from nlos_cs.inverse.tikhonov import LCurveTikhonovSolver, TikhonovSolver
from nlos_cs.io.artifacts import load_operator_artifact


def _require(config: dict[str, Any], *fields: str) -> None:
    missing = [field for field in fields if field not in config]
    if missing:
        raise ValueError(f"Missing required config field(s): {missing}")


def _as_path_str(value: Any) -> str:
    return str(value)


def _build_solver(solver_cfg: dict[str, Any]):
    """Construct a solver from a small config block."""
    _require(solver_cfg, "type")
    solver_type = solver_cfg["type"]

    if solver_type == "tikhonov":
        _require(solver_cfg, "lambda_value")
        return TikhonovSolver(
            lambda_value=float(solver_cfg["lambda_value"]),
            use_svd=bool(solver_cfg.get("use_svd", True)),
            precompute_svd=bool(solver_cfg.get("precompute_svd", True)),
        )

    if solver_type == "tikhonov_lcurve":
        lambdas = solver_cfg.get("lambdas", None)
        return LCurveTikhonovSolver(
            lambdas=None if lambdas is None else np.asarray(lambdas, dtype=np.float64),
            lambda_min_exp=float(solver_cfg.get("lambda_min_exp", 0.0)),
            lambda_max_exp=float(solver_cfg.get("lambda_max_exp", 6.0)),
            n_lambda=int(solver_cfg.get("n_lambda", 200)),
        )

    if solver_type == "nnls":
        return NNLSSolver(
            NNLSSolverConfig(
                normalise_by_sum=bool(solver_cfg.get("normalise_by_sum", False)),
                normalise_by_peak=bool(solver_cfg.get("normalise_by_peak", False)),
            )
        )

    if solver_type == "lasso":
        _require(solver_cfg, "alpha")
        step_size = solver_cfg.get("step_size", None)
        return LassoSolver(
            LassoSolverConfig(
                alpha=float(solver_cfg["alpha"]),
                maxiter=int(solver_cfg.get("maxiter", 1000)),
                tol=float(solver_cfg.get("tol", 1e-8)),
                step_size=None if step_size is None else float(step_size),
                use_fista=bool(solver_cfg.get("use_fista", True)),
                normalise_by_sum=bool(solver_cfg.get("normalise_by_sum", False)),
                normalise_by_peak=bool(solver_cfg.get("normalise_by_peak", False)),
            )
        )

    if solver_type == "huber_simplex":
        _require(solver_cfg, "delta")
        return HuberSimplexSolver(
            HuberSimplexConfig(
                delta=float(solver_cfg["delta"]),
                maxiter=int(solver_cfg.get("maxiter", 500)),
                ftol=float(solver_cfg.get("ftol", 1e-9)),
                use_tikhonov_warm_start=bool(
                    solver_cfg.get("use_tikhonov_warm_start", True)
                ),
                tikhonov_lambda_for_warm_start=float(
                    solver_cfg.get("tikhonov_lambda_for_warm_start", 1.0)
                ),
            )
        )

    raise ValueError(f"Unknown solver type: {solver_type}")


def _load_operator_from_artifact(operator_cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Load an operator bundle from an artifact directory."""
    _require(operator_cfg, "artifact_dir")
    artifact = load_operator_artifact(operator_cfg["artifact_dir"])

    class LoadedOperator:
        def __init__(self, A, positions_mm, metadata):
            self.A = A
            self.positions_mm = positions_mm
            self.metadata = metadata

    operator = LoadedOperator(
        A=artifact["A"],
        positions_mm=artifact["positions_mm"],
        metadata=artifact.get("metadata", {}),
    )
    return operator, artifact


def run_experiment_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a config dictionary to a supported experiment runner.

    Returns a small summary dict suitable for CLI printing.
    """
    _require(config, "experiment_type")
    exp_type = config["experiment_type"]

    if exp_type == "build_single_state_operator":
        _require(
            config,
            "state_id",
            "measurement_kind",
            "plane",
            "position_to_file",
            "run_name",
        )
        plane_cfg = config["plane"]
        _require(plane_cfg, "axis", "value_mm", "tol_mm")

        result = build_single_state_operator_experiment(
            BuildOperatorConfig(
                state_id=str(config["state_id"]),
                measurement_kind=str(config["measurement_kind"]),
                plane=PlaneSpec(
                    axis=str(plane_cfg["axis"]),
                    value_mm=float(plane_cfg["value_mm"]),
                    tol_mm=float(plane_cfg["tol_mm"]),
                ),
                position_to_file={
                    float(k): _as_path_str(v)
                    for k, v in config["position_to_file"].items()
                },
                run_name=str(config["run_name"]),
                output_root=str(config.get("output_root", "outputs")),
                cst_skiprows=int(config.get("cst_skiprows", 2)),
                save_probe_vectors=bool(config.get("save_probe_vectors", False)),
            )
        )

        return {
            "experiment_type": exp_type,
            "run_dir": str(result.run_dir),
            "n_measurements": result.operator.n_measurements,
            "n_positions": result.operator.n_positions,
            "state_id": result.operator.state_id,
        }

    if exp_type == "run_reconstruction":
        _require(config, "operator", "solver", "run_name")
        operator, _ = _load_operator_from_artifact(config["operator"])
        solver = _build_solver(config["solver"])

        recon_cfg = ReconstructionConfig(
            run_name=str(config["run_name"]),
            output_root=str(config.get("output_root", "outputs")),
            measurement_mode=str(config.get("measurement_mode", "operator_column")),
            true_index=config.get("true_index", None),
            noise_fraction_of_rms=float(config.get("noise_fraction_of_rms", 0.0)),
            random_seed=int(config.get("random_seed", 42)),
            save_measurement=bool(config.get("save_measurement", True)),
            save_residual=bool(config.get("save_residual", True)),
        )

        y = config.get("y", None)
        result = run_reconstruction_experiment(
            solver=solver,
            config=recon_cfg,
            operator=operator,
            y=None if y is None else np.asarray(y),
            measurement_metadata=config.get("measurement_metadata", None),
        )

        return {
            "experiment_type": exp_type,
            "run_dir": str(result.run_dir),
            "solver_name": result.reconstruction.solver_name,
            "peak_index": result.reconstruction.peak_index,
            "peak_position_mm": result.reconstruction.peak_position_mm(
                getattr(operator, "positions_mm", None)
            ),
        }

    if exp_type == "run_discrimination":
        _require(config, "operator", "solver", "run_name")
        operator, _ = _load_operator_from_artifact(config["operator"])
        solver = _build_solver(config["solver"])

        disc_cfg = DiscriminationConfig(
            run_name=str(config["run_name"]),
            output_root=str(config.get("output_root", "outputs")),
            mode=str(config.get("mode", "matched")),
            top_k_pairs=int(config.get("top_k_pairs", 10)),
            save_x_hats=bool(config.get("save_x_hats", True)),
            save_measurements=bool(config.get("save_measurements", False)),
        )

        Y = config.get("Y", None)
        true_indices = config.get("true_indices", None)
        groups = config.get("groups", None)

        result = run_discrimination_experiment(
            solver=solver,
            config=disc_cfg,
            operator=operator,
            Y=None if Y is None else np.asarray(Y),
            true_indices=None if true_indices is None else np.asarray(true_indices, dtype=np.int64),
            groups=groups,
            measurement_metadata=config.get("measurement_metadata", None),
        )

        worst_i, worst_j, worst_leak = result.result.worst_pair()
        return {
            "experiment_type": exp_type,
            "run_dir": str(result.run_dir),
            "solver_name": solver.name,
            "worst_pair": [worst_i, worst_j],
            "worst_leakage": worst_leak,
        }

    if exp_type == "run_awgn_robustness":
        _require(config, "operator", "solver", "run_name")
        operator, _ = _load_operator_from_artifact(config["operator"])
        solver = _build_solver(config["solver"])

        robust_cfg = RobustnessConfig(
            run_name=str(config["run_name"]),
            output_root=str(config.get("output_root", "outputs")),
            noise_levels=tuple(
                float(v)
                for v in config.get("noise_levels", (0.0, 0.01, 0.03, 0.05, 0.10))
            ),
            n_trials=int(config.get("n_trials", 50)),
            random_seed=int(config.get("random_seed", 42)),
            save_mean_x_hats=bool(config.get("save_mean_x_hats", True)),
            save_peak_count_cubes=bool(config.get("save_peak_count_cubes", True)),
        )

        result = run_awgn_robustness_experiment(
            solver=solver,
            config=robust_cfg,
            operator=operator,
            measurement_metadata=config.get("measurement_metadata", None),
        )

        return {
            "experiment_type": exp_type,
            "run_dir": str(result.run_dir),
            "solver_name": solver.name,
            "overall_mean_success_rate": float(np.mean(result.success_rate_matrix)),
        }

    raise ValueError(f"Unsupported experiment_type: {exp_type}")