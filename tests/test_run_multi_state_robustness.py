from pathlib import Path

import numpy as np

from nlos_cs.experiments.run_multi_state_robustness import (
    MultiStateRobustnessConfig,
    run_multi_state_robustness_experiment,
)
from nlos_cs.inverse.base import InverseProblem, InverseSolver, ReconstructionResult
from nlos_cs.operators.multi_state import build_multi_state_operator
from nlos_cs.operators.single_state import SingleStateOperator


class ArgmaxOneHotSolver(InverseSolver):
    name = "argmax_onehot"

    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        problem.validate()
        idx = int(np.argmax(np.abs(problem.A.T @ problem.y)))
        x_hat = np.zeros(problem.A.shape[1], dtype=float)
        x_hat[idx] = 1.0
        return ReconstructionResult(
            x_hat=x_hat,
            residual_norm=0.0,
            solution_norm=1.0,
            solver_name=self.name,
        )


def _make_single_state_operator(
    state_id: str,
    scale: float,
) -> SingleStateOperator:
    A = scale * np.array(
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
        state_id=state_id,
        positions_mm=positions_mm,
        coords_in_plane_mm=coords_in_plane_mm,
        in_plane_axes=("y", "z"),
        measurement_kind="e_mag",
        A=A,
    )


def _make_multi_state_operator():
    op1 = _make_single_state_operator("flat", 1.0)
    op2 = _make_single_state_operator("tilted", 2.0)
    return build_multi_state_operator(op1, op2)


def test_run_multi_state_robustness_basic(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = ArgmaxOneHotSolver()

    cfg = MultiStateRobustnessConfig(
        run_name="multi_robust_basic",
        output_root=str(tmp_path / "outputs"),
        noise_levels=(0.0, 0.01),
        n_trials=5,
        random_seed=123,
        save_mean_x_hats=True,
        save_peak_count_cube=True,
    )

    result = run_multi_state_robustness_experiment(
        solver=solver,
        config=cfg,
        operator=op,
    )

    assert result.run_dir.exists()
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "reports" / "multi_state_robustness_report.json").exists()

    assert (result.run_dir / "arrays" / "success_rate_matrix.npy").exists()
    assert (result.run_dir / "arrays" / "mean_peak_margin_matrix.npy").exists()
    assert (result.run_dir / "arrays" / "std_peak_margin_matrix.npy").exists()
    assert (result.run_dir / "arrays" / "noise_levels.npy").exists()
    assert (result.run_dir / "arrays" / "positions_mm.npy").exists()
    assert (result.run_dir / "arrays" / "mean_x_hats.npy").exists()
    assert (result.run_dir / "arrays" / "peak_count_cube.npy").exists()

    assert result.success_rate_matrix.shape == (3, 2)
    assert result.mean_peak_margin_matrix.shape == (3, 2)
    assert result.std_peak_margin_matrix.shape == (3, 2)
    assert result.mean_x_hats is not None
    assert result.mean_x_hats.shape == (3, 2, 3)
    assert result.peak_count_cube is not None
    assert result.peak_count_cube.shape == (3, 2, 3)

    assert result.report["experiment_type"] == "run_multi_state_robustness"
    assert result.report["solver_name"] == "argmax_onehot"
    assert result.report["operator"]["state_ids"] == ["flat", "tilted"]
    assert "flat" in result.report["row_slices"]
    assert "tilted" in result.report["row_slices"]

    assert np.allclose(result.success_rate_matrix[:, 0], 1.0)


def test_run_multi_state_robustness_with_explicit_matrix(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = ArgmaxOneHotSolver()

    cfg = MultiStateRobustnessConfig(
        run_name="multi_robust_explicit",
        output_root=str(tmp_path / "outputs"),
        noise_levels=(0.0,),
        n_trials=3,
        random_seed=1,
    )

    result = run_multi_state_robustness_experiment(
        solver=solver,
        config=cfg,
        A=op.A,
        positions_mm=op.positions_mm,
    )

    assert result.success_rate_matrix.shape == (3, 1)
    assert result.report["operator"]["operator_type"] == "explicit_matrix"


def test_run_multi_state_robustness_respects_save_flags(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = ArgmaxOneHotSolver()

    cfg = MultiStateRobustnessConfig(
        run_name="multi_robust_flags",
        output_root=str(tmp_path / "outputs"),
        noise_levels=(0.0, 0.05),
        n_trials=4,
        random_seed=5,
        save_mean_x_hats=False,
        save_peak_count_cube=False,
    )

    result = run_multi_state_robustness_experiment(
        solver=solver,
        config=cfg,
        operator=op,
    )

    assert (result.run_dir / "arrays" / "success_rate_matrix.npy").exists()
    assert (result.run_dir / "arrays" / "mean_peak_margin_matrix.npy").exists()
    assert (result.run_dir / "arrays" / "std_peak_margin_matrix.npy").exists()
    assert not (result.run_dir / "arrays" / "mean_x_hats.npy").exists()
    assert not (result.run_dir / "arrays" / "peak_count_cube.npy").exists()

    assert result.mean_x_hats is None
    assert result.peak_count_cube is None


def test_run_multi_state_robustness_peak_count_sums_to_trials(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = ArgmaxOneHotSolver()

    cfg = MultiStateRobustnessConfig(
        run_name="multi_robust_peak_counts",
        output_root=str(tmp_path / "outputs"),
        noise_levels=(0.0, 0.01, 0.05),
        n_trials=7,
        random_seed=11,
        save_peak_count_cube=True,
    )

    result = run_multi_state_robustness_experiment(
        solver=solver,
        config=cfg,
        operator=op,
    )

    assert result.peak_count_cube is not None
    sums = np.sum(result.peak_count_cube, axis=2)
    assert np.array_equal(sums, np.full((3, 3), 7))


def test_run_multi_state_robustness_reproducible(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = ArgmaxOneHotSolver()

    cfg1 = MultiStateRobustnessConfig(
        run_name="multi_robust_repro_1",
        output_root=str(tmp_path / "outputs1"),
        noise_levels=(0.01, 0.05),
        n_trials=6,
        random_seed=999,
    )
    cfg2 = MultiStateRobustnessConfig(
        run_name="multi_robust_repro_2",
        output_root=str(tmp_path / "outputs2"),
        noise_levels=(0.01, 0.05),
        n_trials=6,
        random_seed=999,
    )

    result1 = run_multi_state_robustness_experiment(
        solver=solver,
        config=cfg1,
        operator=op,
    )
    result2 = run_multi_state_robustness_experiment(
        solver=solver,
        config=cfg2,
        operator=op,
    )

    assert np.allclose(result1.success_rate_matrix, result2.success_rate_matrix)
    assert np.allclose(result1.mean_peak_margin_matrix, result2.mean_peak_margin_matrix)
    assert np.allclose(result1.std_peak_margin_matrix, result2.std_peak_margin_matrix)

    if result1.mean_x_hats is not None and result2.mean_x_hats is not None:
        assert np.allclose(result1.mean_x_hats, result2.mean_x_hats)

    if result1.peak_count_cube is not None and result2.peak_count_cube is not None:
        assert np.array_equal(result1.peak_count_cube, result2.peak_count_cube)


def test_run_multi_state_robustness_rejects_empty_noise_levels(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = ArgmaxOneHotSolver()

    cfg = MultiStateRobustnessConfig(
        run_name="multi_robust_empty_noise",
        output_root=str(tmp_path / "outputs"),
        noise_levels=(),
        n_trials=5,
        random_seed=1,
    )

    try:
        run_multi_state_robustness_experiment(
            solver=solver,
            config=cfg,
            operator=op,
        )
        assert False, "Expected ValueError for empty noise_levels"
    except ValueError as exc:
        assert "noise_levels must not be empty" in str(exc)


def test_run_multi_state_robustness_rejects_negative_noise_level(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = ArgmaxOneHotSolver()

    cfg = MultiStateRobustnessConfig(
        run_name="multi_robust_negative_noise",
        output_root=str(tmp_path / "outputs"),
        noise_levels=(0.0, -0.01),
        n_trials=5,
        random_seed=1,
    )

    try:
        run_multi_state_robustness_experiment(
            solver=solver,
            config=cfg,
            operator=op,
        )
        assert False, "Expected ValueError for negative noise level"
    except ValueError as exc:
        assert "noise_levels must all be non-negative" in str(exc)


def test_run_multi_state_robustness_rejects_nonpositive_trials(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = ArgmaxOneHotSolver()

    cfg = MultiStateRobustnessConfig(
        run_name="multi_robust_bad_trials",
        output_root=str(tmp_path / "outputs"),
        noise_levels=(0.0, 0.01),
        n_trials=0,
        random_seed=1,
    )

    try:
        run_multi_state_robustness_experiment(
            solver=solver,
            config=cfg,
            operator=op,
        )
        assert False, "Expected ValueError for non-positive n_trials"
    except ValueError as exc:
        assert "n_trials must be positive" in str(exc)