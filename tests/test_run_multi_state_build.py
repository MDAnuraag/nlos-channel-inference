from pathlib import Path

import numpy as np

from nlos_cs.experiments.run_multi_state_build import (
    MultiStateBuildConfig,
    run_multi_state_build_experiment,
)
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


def test_run_multi_state_build_experiment_basic(tmp_path: Path):
    op1 = _make_single_state_operator("flat", 1.0)
    op2 = _make_single_state_operator("tilted", 2.0)
    op3 = _make_single_state_operator("stepped", 3.0)

    cfg = MultiStateBuildConfig(
        run_name="multi_state_basic",
        output_root=str(tmp_path / "outputs"),
        save_component_blocks=True,
    )

    result = run_multi_state_build_experiment(
        operators=[op1, op2, op3],
        config=cfg,
    )

    assert result.run_dir.exists()
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "reports" / "multi_state_build_report.json").exists()

    assert (result.run_dir / "arrays" / "A.npy").exists()
    assert (result.run_dir / "arrays" / "positions_mm.npy").exists()
    assert (result.run_dir / "arrays" / "singular_values.npy").exists()
    assert (result.run_dir / "arrays" / "gram_matrix.npy").exists()

    assert (result.run_dir / "arrays" / "A_block_flat.npy").exists()
    assert (result.run_dir / "arrays" / "A_block_tilted.npy").exists()
    assert (result.run_dir / "arrays" / "A_block_stepped.npy").exists()

    assert result.operator.n_states == 3
    assert result.operator.shape == (9, 3)
    assert result.report["experiment_type"] == "build_multi_state_operator"
    assert result.report["operator_summary"]["n_states"] == 3
    assert result.report["operator_summary"]["state_ids"] == ["flat", "tilted", "stepped"]


def test_run_multi_state_build_experiment_without_component_blocks(tmp_path: Path):
    op1 = _make_single_state_operator("flat", 1.0)
    op2 = _make_single_state_operator("tilted", 2.0)

    cfg = MultiStateBuildConfig(
        run_name="multi_state_no_blocks",
        output_root=str(tmp_path / "outputs"),
        save_component_blocks=False,
    )

    result = run_multi_state_build_experiment(
        operators=[op1, op2],
        config=cfg,
    )

    assert (result.run_dir / "arrays" / "A.npy").exists()
    assert not (result.run_dir / "arrays" / "A_block_flat.npy").exists()
    assert not (result.run_dir / "arrays" / "A_block_tilted.npy").exists()


def test_run_multi_state_build_experiment_rejects_empty_operator_list(tmp_path: Path):
    cfg = MultiStateBuildConfig(
        run_name="multi_state_empty",
        output_root=str(tmp_path / "outputs"),
    )

    try:
        run_multi_state_build_experiment(
            operators=[],
            config=cfg,
        )
        assert False, "Expected ValueError for empty operator list"
    except ValueError as exc:
        assert "operators must not be empty" in str(exc)