from pathlib import Path

import numpy as np

from nlos_cs.experiments.run_multi_state_reconstruction import (
    MultiStateReconstructionConfig,
    run_multi_state_reconstruction_experiment,
)
from nlos_cs.inverse.tikhonov import TikhonovSolver
from nlos_cs.operators.multi_state import build_multi_state_operator
from nlos_cs.operators.single_state import SingleStateOperator


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


def test_run_multi_state_reconstruction_operator_column_mode(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = TikhonovSolver(lambda_value=0.5)

    config = MultiStateReconstructionConfig(
        run_name="multi_recon_operator_column",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="operator_column",
        true_index=2,
        noise_fraction_of_rms=0.0,
        random_seed=42,
        save_measurement=True,
        save_residual=True,
        save_per_state_measurement_blocks=True,
    )

    result = run_multi_state_reconstruction_experiment(
        solver=solver,
        config=config,
        operator=op,
    )

    assert result.run_dir.exists()
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "reports" / "multi_state_reconstruction_report.json").exists()
    assert (result.run_dir / "arrays" / "x_hat.npy").exists()
    assert (result.run_dir / "arrays" / "y.npy").exists()
    assert (result.run_dir / "arrays" / "residual.npy").exists()
    assert (result.run_dir / "arrays" / "y_true.npy").exists()

    assert (result.run_dir / "arrays" / "y_block_flat.npy").exists()
    assert (result.run_dir / "arrays" / "y_block_tilted.npy").exists()
    assert (result.run_dir / "arrays" / "y_true_block_flat.npy").exists()
    assert (result.run_dir / "arrays" / "y_true_block_tilted.npy").exists()

    assert np.array_equal(result.y, op.A[:, 2])
    assert np.array_equal(result.y_true, op.A[:, 2])
    assert result.report["experiment_type"] == "run_multi_state_reconstruction"
    assert result.report["measurement_mode"] == "operator_column"
    assert result.report["true_position_mm"] == 75.0
    assert result.report["operator"]["state_ids"] == ["flat", "tilted"]


def test_run_multi_state_reconstruction_provided_mode(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = TikhonovSolver(lambda_value=1.0)
    y = np.arange(op.A.shape[0], dtype=float)

    config = MultiStateReconstructionConfig(
        run_name="multi_recon_provided",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="provided",
        true_index=None,
        save_measurement=True,
        save_residual=True,
    )

    result = run_multi_state_reconstruction_experiment(
        solver=solver,
        config=config,
        operator=op,
        y=y,
        measurement_metadata={"source": "unit_test"},
    )

    assert np.array_equal(result.y, y)
    assert result.y_true is None
    assert result.report["measurement_mode"] == "provided"
    assert result.report["measurement_metadata"]["source"] == "unit_test"
    assert not (result.run_dir / "arrays" / "y_true.npy").exists()


def test_run_multi_state_reconstruction_explicit_matrix(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = TikhonovSolver(lambda_value=0.25)

    config = MultiStateReconstructionConfig(
        run_name="multi_recon_explicit",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="operator_column",
        true_index=0,
        noise_fraction_of_rms=0.0,
        random_seed=42,
    )

    result = run_multi_state_reconstruction_experiment(
        solver=solver,
        config=config,
        A=op.A,
        positions_mm=op.positions_mm,
    )

    assert np.array_equal(result.y, op.A[:, 0])
    assert result.report["operator"]["operator_type"] == "explicit_matrix"


def test_run_multi_state_reconstruction_respects_save_flags(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = TikhonovSolver(lambda_value=0.5)

    config = MultiStateReconstructionConfig(
        run_name="multi_recon_flags",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="operator_column",
        true_index=1,
        noise_fraction_of_rms=0.0,
        random_seed=42,
        save_measurement=False,
        save_residual=False,
        save_per_state_measurement_blocks=False,
    )

    result = run_multi_state_reconstruction_experiment(
        solver=solver,
        config=config,
        operator=op,
    )

    assert (result.run_dir / "arrays" / "x_hat.npy").exists()
    assert not (result.run_dir / "arrays" / "y.npy").exists()
    assert not (result.run_dir / "arrays" / "residual.npy").exists()
    assert (result.run_dir / "arrays" / "y_true.npy").exists()
    assert not (result.run_dir / "arrays" / "y_block_flat.npy").exists()
    assert not (result.run_dir / "arrays" / "y_block_tilted.npy").exists()


def test_run_multi_state_reconstruction_requires_y_in_provided_mode(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = TikhonovSolver(lambda_value=1.0)

    config = MultiStateReconstructionConfig(
        run_name="multi_recon_missing_y",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="provided",
    )

    try:
        run_multi_state_reconstruction_experiment(
            solver=solver,
            config=config,
            operator=op,
        )
        assert False, "Expected ValueError when y is missing"
    except ValueError as exc:
        assert "requires y" in str(exc)


def test_run_multi_state_reconstruction_requires_true_index_in_operator_column_mode(tmp_path: Path):
    op = _make_multi_state_operator()
    solver = TikhonovSolver(lambda_value=1.0)

    config = MultiStateReconstructionConfig(
        run_name="multi_recon_missing_true_index",
        output_root=str(tmp_path / "outputs"),
        measurement_mode="operator_column",
        true_index=None,
    )

    try:
        run_multi_state_reconstruction_experiment(
            solver=solver,
            config=config,
            operator=op,
        )
        assert False, "Expected ValueError when true_index is missing"
    except ValueError as exc:
        assert "requires true_index" in str(exc)