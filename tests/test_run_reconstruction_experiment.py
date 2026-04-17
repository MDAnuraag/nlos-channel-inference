from pathlib import Path

import numpy as np

from nlos_cs.experiments.run_reconstruction import (
    ReconstructionConfig,
    make_operator_column_measurement,
    run_reconstruction_experiment,
)
from nlos_cs.inverse.tikhonov import TikhonovSolver
from nlos_cs.operators.single_state import SingleStateOperator


def _make_test_operator() -> SingleStateOperator:
    A = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [0.1, 0.5, 4.0],
            [0.2, 0.1, 0.3],
        ],
        dtype=float,
    )
    positions_mm = np.array([65.0, 70.0, 75.0], dtype=float)
    coords_in_plane_mm = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
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


def test_make_operator_column_measurement_no_noise():
    op = _make_test_operator()

    y, y_true = make_operator_column_measurement(
        op.A,
        true_index=1,
        noise_fraction_of_rms=0.0,
        random_seed=42,
    )

    assert np.array_equal(y_true, op.A[:, 1])
    assert np.array_equal(y, op.A[:, 1])


def test_make_operator_column_measurement_with_noise_is_reproducible():
    op = _make_test_operator()

    y1, y_true1 = make_operator_column_measurement(
        op.A,
        true_index=2,
        noise_fraction_of_rms=0.05,
        random_seed=123,
    )
    y2, y_true2 = make_operator_column_measurement(
        op.A,
        true_index=2,
        noise_fraction_of_rms=0.05,
        random_seed=123,
    )

    assert np.array_equal(y_true1, op.A[:, 2])
    assert np.array_equal(y_true1, y_true2)
    assert np.allclose(y1, y2)
    assert not np.allclose(y1, y_true1)


def test_run_reconstruction_experiment_operator_column_mode(tmp_path: Path):
    op = _make_test_operator()
    solver = TikhonovSolver(lambda_value=0.5)

    config = ReconstructionConfig(
        run_name="recon_operator_column",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="operator_column",
        true_index=2,
        noise_fraction_of_rms=0.0,
        random_seed=42,
        save_measurement=True,
        save_residual=True,
    )

    result = run_reconstruction_experiment(
        solver=solver,
        config=config,
        operator=op,
    )

    assert result.run_dir.exists()
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "reports" / "reconstruction_report.json").exists()
    assert (result.run_dir / "arrays" / "x_hat.npy").exists()
    assert (result.run_dir / "arrays" / "y.npy").exists()
    assert (result.run_dir / "arrays" / "residual.npy").exists()
    assert (result.run_dir / "arrays" / "y_true.npy").exists()

    assert np.array_equal(result.y, op.A[:, 2])
    assert np.array_equal(result.y_true, op.A[:, 2])
    assert result.reconstruction.solver_name == "tikhonov"

    assert result.report["measurement_mode"] == "operator_column"
    assert result.report["true_index"] == 2
    assert result.report["true_position_mm"] == 75.0
    assert result.report["operator"]["state_id"] == "flat"


def test_run_reconstruction_experiment_provided_mode(tmp_path: Path):
    op = _make_test_operator()
    solver = TikhonovSolver(lambda_value=1.0)

    y = np.array([1.2, 0.8, 0.5, 0.1], dtype=float)

    config = ReconstructionConfig(
        run_name="recon_provided",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="provided",
        true_index=None,
        noise_fraction_of_rms=0.0,
        random_seed=42,
        save_measurement=True,
        save_residual=True,
    )

    result = run_reconstruction_experiment(
        solver=solver,
        config=config,
        operator=op,
        y=y,
        measurement_metadata={"source": "unit_test"},
    )

    assert np.array_equal(result.y, y)
    assert result.y_true is None
    assert result.report["measurement_mode"] == "provided"
    assert result.report["true_index"] is None
    assert result.report["measurement_metadata"]["source"] == "unit_test"

    assert (result.run_dir / "arrays" / "x_hat.npy").exists()
    assert (result.run_dir / "arrays" / "y.npy").exists()
    assert (result.run_dir / "arrays" / "residual.npy").exists()
    assert not (result.run_dir / "arrays" / "y_true.npy").exists()


def test_run_reconstruction_experiment_with_explicit_matrix(tmp_path: Path):
    op = _make_test_operator()
    solver = TikhonovSolver(lambda_value=0.25)

    config = ReconstructionConfig(
        run_name="recon_explicit_matrix",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="operator_column",
        true_index=0,
        noise_fraction_of_rms=0.0,
        random_seed=42,
    )

    result = run_reconstruction_experiment(
        solver=solver,
        config=config,
        A=op.A,
        positions_mm=op.positions_mm,
    )

    assert np.array_equal(result.y, op.A[:, 0])
    assert result.report["operator"]["operator_type"] == "explicit_matrix"
    assert result.report["true_position_mm"] == 65.0


def test_run_reconstruction_experiment_respects_save_flags(tmp_path: Path):
    op = _make_test_operator()
    solver = TikhonovSolver(lambda_value=0.5)

    config = ReconstructionConfig(
        run_name="recon_save_flags",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="operator_column",
        true_index=1,
        noise_fraction_of_rms=0.0,
        random_seed=42,
        save_measurement=False,
        save_residual=False,
    )

    result = run_reconstruction_experiment(
        solver=solver,
        config=config,
        operator=op,
    )

    assert (result.run_dir / "arrays" / "x_hat.npy").exists()
    assert not (result.run_dir / "arrays" / "y.npy").exists()
    assert not (result.run_dir / "arrays" / "residual.npy").exists()
    assert (result.run_dir / "arrays" / "y_true.npy").exists()


def test_run_reconstruction_experiment_requires_y_in_provided_mode(tmp_path: Path):
    op = _make_test_operator()
    solver = TikhonovSolver(lambda_value=1.0)

    config = ReconstructionConfig(
        run_name="recon_missing_y",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="provided",
    )

    try:
        run_reconstruction_experiment(
            solver=solver,
            config=config,
            operator=op,
        )
        assert False, "Expected ValueError when y is missing in provided mode"
    except ValueError as exc:
        assert "requires y" in str(exc)


def test_run_reconstruction_experiment_requires_true_index_in_operator_column_mode(tmp_path: Path):
    op = _make_test_operator()
    solver = TikhonovSolver(lambda_value=1.0)

    config = ReconstructionConfig(
        run_name="recon_missing_true_index",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="operator_column",
        true_index=None,
    )

    try:
        run_reconstruction_experiment(
            solver=solver,
            config=config,
            operator=op,
        )
        assert False, "Expected ValueError when true_index is missing"
    except ValueError as exc:
        assert "requires true_index" in str(exc)