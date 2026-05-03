from pathlib import Path

import numpy as np

from nlos_cs.experiments.run_multi_state_discrimination import (
    MultiStateDiscriminationConfig,
    run_multi_state_discrimination_experiment,
)
from nlos_cs.inverse.base import InverseProblem, InverseSolver, ReconstructionResult
from nlos_cs.operators.multi_state import build_multi_state_operator
from nlos_cs.operators.single_state import SingleStateOperator


class BackProjectionSolver(InverseSolver):
    name = "backprojection"

    def solve(self, problem: InverseProblem) -> ReconstructionResult:
        problem.validate()
        x_hat = problem.A.T @ problem.y
        return ReconstructionResult(
            x_hat=x_hat,
            residual_norm=0.0,
            solution_norm=float(np.linalg.norm(x_hat)),
            solver_name=self.name,
        )


def _make_single_state_operator(
    state_id: str,
    scale: float,
) -> SingleStateOperator:
    A = scale * np.array(
        [
            [1.0, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [0.1, 0.5, 4.0],
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


def test_run_multi_state_discrimination_matched_mode(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = BackProjectionSolver()

    config = MultiStateDiscriminationConfig(
        run_name="multi_disc_matched",
        output_root=str(tmp_path / "outputs"),
        mode="matched",
        top_k_pairs=5,
        save_x_hats=True,
        save_measurements=False,
    )

    result = run_multi_state_discrimination_experiment(
        solver=solver,
        config=config,
        operator=op,
    )

    assert result.run_dir.exists()
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "reports" / "multi_state_discrimination_report.json").exists()

    assert (result.run_dir / "arrays" / "discrimination.npy").exists()
    assert (result.run_dir / "arrays" / "leakage.npy").exists()
    assert (result.run_dir / "arrays" / "reference_values.npy").exists()
    assert (result.run_dir / "arrays" / "peak_indices.npy").exists()
    assert (result.run_dir / "arrays" / "x_hats.npy").exists()
    assert (result.run_dir / "arrays" / "positions_mm.npy").exists()

    assert result.report["experiment_type"] == "run_multi_state_discrimination"
    assert result.report["mode"] == "matched"
    assert result.report["solver_name"] == "backprojection"
    assert result.report["operator"]["state_ids"] == ["flat", "tilted"]
    assert "flat" in result.report["row_slices"]
    assert "tilted" in result.report["row_slices"]


def test_run_multi_state_discrimination_measurements_mode(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = BackProjectionSolver()

    Y = op.A.T.copy()

    config = MultiStateDiscriminationConfig(
        run_name="multi_disc_measurements",
        output_root=str(tmp_path / "outputs"),
        mode="measurements",
        top_k_pairs=3,
        save_x_hats=True,
        save_measurements=True,
    )

    result = run_multi_state_discrimination_experiment(
        solver=solver,
        config=config,
        operator=op,
        Y=Y,
    )

    assert result.report["mode"] == "measurements"
    assert (result.run_dir / "arrays" / "Y.npy").exists()
    assert (result.run_dir / "arrays" / "x_hats.npy").exists()


def test_run_multi_state_discrimination_group_summary(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = BackProjectionSolver()

    config = MultiStateDiscriminationConfig(
        run_name="multi_disc_groups",
        output_root=str(tmp_path / "outputs"),
        mode="matched",
        top_k_pairs=4,
    )

    groups = {
        "A": {0, 1},
        "B": {2},
    }

    result = run_multi_state_discrimination_experiment(
        solver=solver,
        config=config,
        operator=op,
        groups=groups,
    )

    summary = result.report["summary"]
    assert "group_leakage_summary" in summary
    assert set(summary["group_leakage_summary"].keys()) == {"A->A", "A->B", "B->A", "B->B"}


def test_run_multi_state_discrimination_with_explicit_matrix(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = BackProjectionSolver()

    config = MultiStateDiscriminationConfig(
        run_name="multi_disc_explicit",
        output_root=str(tmp_path / "outputs"),
        mode="matched",
    )

    result = run_multi_state_discrimination_experiment(
        solver=solver,
        config=config,
        A=op.A,
        positions_mm=op.positions_mm,
    )

    assert result.report["operator"]["operator_type"] == "explicit_matrix"
    assert (result.run_dir / "arrays" / "positions_mm.npy").exists()


def test_run_multi_state_discrimination_respects_save_flags(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = BackProjectionSolver()

    config = MultiStateDiscriminationConfig(
        run_name="multi_disc_flags",
        output_root=str(tmp_path / "outputs"),
        mode="matched",
        save_x_hats=False,
    )

    result = run_multi_state_discrimination_experiment(
        solver=solver,
        config=config,
        operator=op,
    )

    assert (result.run_dir / "arrays" / "discrimination.npy").exists()
    assert (result.run_dir / "arrays" / "leakage.npy").exists()
    assert not (result.run_dir / "arrays" / "x_hats.npy").exists()


def test_run_multi_state_discrimination_measurements_mode_requires_Y(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = BackProjectionSolver()

    config = MultiStateDiscriminationConfig(
        run_name="multi_disc_missing_Y",
        output_root=str(tmp_path / "outputs"),
        mode="measurements",
    )

    try:
        run_multi_state_discrimination_experiment(
            solver=solver,
            config=config,
            operator=op,
        )
        assert False, "Expected ValueError when Y is missing"
    except ValueError as exc:
        assert "requires Y" in str(exc)


def test_run_multi_state_discrimination_invalid_mode(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = BackProjectionSolver()

    config = MultiStateDiscriminationConfig(
        run_name="multi_disc_bad_mode",
        output_root=str(tmp_path / "outputs"),
        mode="not_a_mode",
    )

    try:
        run_multi_state_discrimination_experiment(
            solver=solver,
            config=config,
            operator=op,
        )
        assert False, "Expected ValueError for invalid mode"
    except ValueError as exc:
        assert "Unknown discrimination mode" in str(exc)