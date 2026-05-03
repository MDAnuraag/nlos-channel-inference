from pathlib import Path

import numpy as np

from nlos_cs.experiments.run_robustness_combined import (
    CombinedRobustnessConfig,
    run_combined_robustness_experiment,
)
from nlos_cs.inverse.base import InverseProblem, InverseSolver, ReconstructionResult
from nlos_cs.operators.single_state import SingleStateOperator
from nlos_cs.perturb.combined import CombinedPerturbationConfig


class IdentitySolver(InverseSolver):
    name = "identity"

    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        problem.validate()
        x_hat = problem.y.copy()
        return ReconstructionResult(
            x_hat=x_hat,
            residual_norm=0.0,
            solution_norm=float(np.linalg.norm(x_hat)),
            solver_name=self.name,
        )


class ArgmaxOneHotSolver(InverseSolver):
    name = "argmax_onehot"

    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        problem.validate()
        idx = int(np.argmax(np.abs(problem.y)))
        x_hat = np.zeros(problem.A.shape[1], dtype=float)
        x_hat[idx] = 1.0
        return ReconstructionResult(
            x_hat=x_hat,
            residual_norm=0.0,
            solution_norm=1.0,
            solver_name=self.name,
        )


def _make_test_operator() -> SingleStateOperator:
    A = np.array(
        [
            [4.0, 0.2, 0.1],
            [0.1, 5.0, 0.2],
            [0.2, 0.3, 6.0],
        ],
        dtype=float,
    )
    positions_mm = np.array([65.0, 70.0, 75.0], dtype=float)
    coords_in_plane_mm = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    return SingleStateOperator(
        state_id="flat",
        positions_mm=positions_mm,
        coords_in_plane_mm=coords_in_plane_mm,
        in_plane_axes=("y", "z"),
        measurement_kind="e_mag",
        A=A,
    )


def test_run_combined_robustness_no_perturbations_basic(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    cfg = CombinedRobustnessConfig(
        run_name="combined_basic",
        perturb_config=CombinedPerturbationConfig(),
        output_root=str(tmp_path / "outputs"),
        n_trials=5,
        random_seed=123,
        save_mean_x_hats=True,
        save_peak_count_cube=True,
    )

    result = run_combined_robustness_experiment(
        solver=solver,
        config=cfg,
        operator=op,
    )

    assert result.run_dir.exists()
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "reports" / "combined_robustness_report.json").exists()

    assert (result.run_dir / "arrays" / "success_rate.npy").exists()
    assert (result.run_dir / "arrays" / "mean_peak_margin.npy").exists()
    assert (result.run_dir / "arrays" / "std_peak_margin.npy").exists()
    assert (result.run_dir / "arrays" / "positions_mm.npy").exists()
    assert (result.run_dir / "arrays" / "mean_x_hats.npy").exists()
    assert (result.run_dir / "arrays" / "peak_count_cube.npy").exists()

    assert result.success_rate.shape == (3,)
    assert result.mean_peak_margin.shape == (3,)
    assert result.std_peak_margin.shape == (3,)
    assert result.mean_x_hats is not None
    assert result.mean_x_hats.shape == (3, 3)
    assert result.peak_count_cube is not None
    assert result.peak_count_cube.shape == (3, 3)

    assert result.report["experiment_type"] == "run_combined_robustness"
    assert result.report["solver_name"] == "identity"
    assert result.report["operator"]["state_id"] == "flat"
    assert result.report["summary"]["stage_usage_counts"] == {
        "multipath": 0,
        "correlated": 0,
        "awgn": 0,
        "dropout": 0,
    }

    assert np.allclose(result.success_rate, 1.0)


def test_run_combined_robustness_with_explicit_matrix(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    cfg = CombinedRobustnessConfig(
        run_name="combined_explicit",
        perturb_config=CombinedPerturbationConfig(),
        output_root=str(tmp_path / "outputs"),
        n_trials=3,
        random_seed=1,
    )

    result = run_combined_robustness_experiment(
        solver=solver,
        config=cfg,
        A=op.A,
        positions_mm=op.positions_mm,
    )

    assert result.success_rate.shape == (3,)
    assert result.report["operator"]["operator_type"] == "explicit_matrix"


def test_run_combined_robustness_respects_save_flags(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    cfg = CombinedRobustnessConfig(
        run_name="combined_flags",
        perturb_config=CombinedPerturbationConfig(),
        output_root=str(tmp_path / "outputs"),
        n_trials=4,
        random_seed=5,
        save_mean_x_hats=False,
        save_peak_count_cube=False,
    )

    result = run_combined_robustness_experiment(
        solver=solver,
        config=cfg,
        operator=op,
    )

    assert (result.run_dir / "arrays" / "success_rate.npy").exists()
    assert (result.run_dir / "arrays" / "mean_peak_margin.npy").exists()
    assert (result.run_dir / "arrays" / "std_peak_margin.npy").exists()
    assert not (result.run_dir / "arrays" / "mean_x_hats.npy").exists()
    assert not (result.run_dir / "arrays" / "peak_count_cube.npy").exists()

    assert result.mean_x_hats is None
    assert result.peak_count_cube is None


def test_run_combined_robustness_peak_count_sums_to_trials(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    cfg = CombinedRobustnessConfig(
        run_name="combined_peak_counts",
        perturb_config=CombinedPerturbationConfig(),
        output_root=str(tmp_path / "outputs"),
        n_trials=7,
        random_seed=11,
        save_peak_count_cube=True,
    )

    result = run_combined_robustness_experiment(
        solver=solver,
        config=cfg,
        operator=op,
    )

    assert result.peak_count_cube is not None
    sums = np.sum(result.peak_count_cube, axis=1)
    assert np.array_equal(sums, np.full(3, 7))


def test_run_combined_robustness_with_dropout_stage(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    cfg = CombinedRobustnessConfig(
        run_name="combined_dropout",
        perturb_config=CombinedPerturbationConfig(
            apply_dropout=True,
            dropout_fraction=0.5,
        ),
        output_root=str(tmp_path / "outputs"),
        n_trials=5,
        random_seed=21,
    )

    result = run_combined_robustness_experiment(
        solver=solver,
        config=cfg,
        operator=op,
    )

    assert result.report["summary"]["stage_usage_counts"]["dropout"] == 3 * 5
    assert result.report["summary"]["stage_usage_counts"]["awgn"] == 0
    assert result.report["summary"]["stage_usage_counts"]["correlated"] == 0
    assert result.report["summary"]["stage_usage_counts"]["multipath"] == 0


def test_run_combined_robustness_with_all_stages(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    perturb_cfg = CombinedPerturbationConfig(
        apply_multipath=True,
        multipath_fraction_of_rms=0.05,
        n_leak=2,
        apply_correlated=True,
        correlated_fraction_of_rms=0.05,
        corr_length=2,
        apply_awgn=True,
        awgn_fraction_of_rms=0.05,
        apply_dropout=True,
        dropout_fraction=0.25,
    )
    cfg = CombinedRobustnessConfig(
        run_name="combined_all_stages",
        perturb_config=perturb_cfg,
        output_root=str(tmp_path / "outputs"),
        n_trials=4,
        random_seed=33,
    )

    result = run_combined_robustness_experiment(
        solver=solver,
        config=cfg,
        operator=op,
    )

    expected_count = 3 * 4
    summary = result.report["summary"]["stage_usage_counts"]
    assert summary["multipath"] == expected_count
    assert summary["correlated"] == expected_count
    assert summary["awgn"] == expected_count
    assert summary["dropout"] == expected_count


def test_run_combined_robustness_zero_perturbation_with_argmax_solver_is_perfect(tmp_path: Path):
    op = _make_test_operator()
    solver = ArgmaxOneHotSolver()

    cfg = CombinedRobustnessConfig(
        run_name="combined_argmax",
        perturb_config=CombinedPerturbationConfig(),
        output_root=str(tmp_path / "outputs"),
        n_trials=5,
        random_seed=77,
    )

    result = run_combined_robustness_experiment(
        solver=solver,
        config=cfg,
        operator=op,
    )

    assert np.allclose(result.success_rate, 1.0)
    assert np.all(result.mean_peak_margin == 1.0)
    assert np.all(result.std_peak_margin == 0.0)


def test_run_combined_robustness_reproducible(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    perturb_cfg = CombinedPerturbationConfig(
        apply_awgn=True,
        awgn_fraction_of_rms=0.05,
        apply_dropout=True,
        dropout_fraction=0.25,
    )

    cfg1 = CombinedRobustnessConfig(
        run_name="combined_repro_1",
        perturb_config=perturb_cfg,
        output_root=str(tmp_path / "outputs1"),
        n_trials=6,
        random_seed=999,
    )
    cfg2 = CombinedRobustnessConfig(
        run_name="combined_repro_2",
        perturb_config=perturb_cfg,
        output_root=str(tmp_path / "outputs2"),
        n_trials=6,
        random_seed=999,
    )

    result1 = run_combined_robustness_experiment(
        solver=solver,
        config=cfg1,
        operator=op,
    )
    result2 = run_combined_robustness_experiment(
        solver=solver,
        config=cfg2,
        operator=op,
    )

    assert np.allclose(result1.success_rate, result2.success_rate)
    assert np.allclose(result1.mean_peak_margin, result2.mean_peak_margin)
    assert np.allclose(result1.std_peak_margin, result2.std_peak_margin)

    if result1.mean_x_hats is not None and result2.mean_x_hats is not None:
        assert np.allclose(result1.mean_x_hats, result2.mean_x_hats)

    if result1.peak_count_cube is not None and result2.peak_count_cube is not None:
        assert np.array_equal(result1.peak_count_cube, result2.peak_count_cube)


def test_run_combined_robustness_rejects_nonpositive_trials(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    cfg = CombinedRobustnessConfig(
        run_name="combined_bad_trials",
        perturb_config=CombinedPerturbationConfig(),
        output_root=str(tmp_path / "outputs"),
        n_trials=0,
        random_seed=1,
    )

    try:
        run_combined_robustness_experiment(
            solver=solver,
            config=cfg,
            operator=op,
        )
        assert False, "Expected ValueError for non-positive n_trials"
    except ValueError as exc:
        assert "n_trials must be positive" in str(exc)