from pathlib import Path

import numpy as np

from nlos_cs.experiments.run_discrimination import (
    DiscriminationConfig,
    run_discrimination_experiment,
)
from nlos_cs.inverse.base import InverseProblem, InverseSolver, ReconstructionResult
from nlos_cs.operators.single_state import SingleStateOperator


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


def _make_test_operator() -> SingleStateOperator:
    A = np.array(
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
        state_id="flat",
        positions_mm=positions_mm,
        coords_in_plane_mm=coords_in_plane_mm,
        in_plane_axes=("y", "z"),
        measurement_kind="e_mag",
        A=A,
    )


def test_run_discrimination_experiment_matched_mode(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    config = DiscriminationConfig(
        run_name="disc_matched",
        output_root=str(tmp_path / "outputs"),
        mode="matched",
        top_k_pairs=5,
        save_x_hats=True,
        save_measurements=False,
    )

    result = run_discrimination_experiment(
        solver=solver,
        config=config,
        operator=op,
    )

    assert result.run_dir.exists()
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "reports" / "discrimination_report.json").exists()

    assert (result.run_dir / "arrays" / "discrimination.npy").exists()
    assert (result.run_dir / "arrays" / "leakage.npy").exists()
    assert (result.run_dir / "arrays" / "reference_values.npy").exists()
    assert (result.run_dir / "arrays" / "peak_indices.npy").exists()
    assert (result.run_dir / "arrays" / "x_hats.npy").exists()
    assert (result.run_dir / "arrays" / "positions_mm.npy").exists()

    assert result.report["experiment_type"] == "run_discrimination"
    assert result.report["mode"] == "matched"
    assert result.report["solver_name"] == "identity"
    assert result.report["operator"]["state_id"] == "flat"

    summary = result.report["summary"]
    assert summary["n_positions"] == 3
    assert "worst_pair" in summary
    assert "hardest_pairs" in summary
    assert len(summary["hardest_pairs"]) <= 5


def test_run_discrimination_experiment_measurements_mode(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    # Row i is the reconstruction we want IdentitySolver to return.
    Y = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [0.1, 0.5, 4.0],
        ],
        dtype=float,
    )

    config = DiscriminationConfig(
        run_name="disc_measurements",
        output_root=str(tmp_path / "outputs"),
        mode="measurements",
        top_k_pairs=3,
        save_x_hats=True,
        save_measurements=True,
    )

    result = run_discrimination_experiment(
        solver=solver,
        config=config,
        operator=op,
        Y=Y,
    )

    assert result.report["mode"] == "measurements"
    assert (result.run_dir / "arrays" / "Y.npy").exists()
    assert (result.run_dir / "arrays" / "x_hats.npy").exists()

    leakage = result.result.leakage
    expected = np.array(
        [
            [0.0, 0.2 / 1.0, 0.1 / 1.0],
            [0.3 / 2.0, 0.0, 0.4 / 2.0],
            [0.1 / 4.0, 0.5 / 4.0, 0.0],
        ]
    )
    assert np.allclose(leakage, expected)


def test_run_discrimination_experiment_group_summary(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    config = DiscriminationConfig(
        run_name="disc_groups",
        output_root=str(tmp_path / "outputs"),
        mode="matched",
        top_k_pairs=4,
    )

    groups = {
        "A": {0, 1},
        "B": {2},
    }

    result = run_discrimination_experiment(
        solver=solver,
        config=config,
        operator=op,
        groups=groups,
    )

    summary = result.report["summary"]
    assert "group_leakage_summary" in summary
    assert set(summary["group_leakage_summary"].keys()) == {"A->A", "A->B", "B->A", "B->B"}


def test_run_discrimination_experiment_with_explicit_matrix(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    config = DiscriminationConfig(
        run_name="disc_explicit",
        output_root=str(tmp_path / "outputs"),
        mode="matched",
    )

    result = run_discrimination_experiment(
        solver=solver,
        config=config,
        A=op.A,
        positions_mm=op.positions_mm,
    )

    assert result.report["operator"]["operator_type"] == "explicit_matrix"
    assert (result.run_dir / "arrays" / "positions_mm.npy").exists()


def test_run_discrimination_experiment_respects_save_flags(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    config = DiscriminationConfig(
        run_name="disc_flags",
        output_root=str(tmp_path / "outputs"),
        mode="matched",
        save_x_hats=False,
    )

    result = run_discrimination_experiment(
        solver=solver,
        config=config,
        operator=op,
    )

    assert (result.run_dir / "arrays" / "discrimination.npy").exists()
    assert (result.run_dir / "arrays" / "leakage.npy").exists()
    assert not (result.run_dir / "arrays" / "x_hats.npy").exists()


def test_run_discrimination_experiment_measurements_mode_requires_Y(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    config = DiscriminationConfig(
        run_name="disc_missing_Y",
        output_root=str(tmp_path / "outputs"),
        mode="measurements",
    )

    try:
        run_discrimination_experiment(
            solver=solver,
            config=config,
            operator=op,
        )
        assert False, "Expected ValueError when Y is missing in measurements mode"
    except ValueError as exc:
        assert "requires Y" in str(exc)


def test_run_discrimination_experiment_invalid_mode(tmp_path: Path):
    op = _make_test_operator()
    solver = IdentitySolver()

    config = DiscriminationConfig(
        run_name="disc_bad_mode",
        output_root=str(tmp_path / "outputs"),
        mode="not_a_mode",
    )

    try:
        run_discrimination_experiment(
            solver=solver,
            config=config,
            operator=op,
        )
        assert False, "Expected ValueError for invalid discrimination mode"
    except ValueError as exc:
        assert "Unknown discrimination mode" in str(exc)