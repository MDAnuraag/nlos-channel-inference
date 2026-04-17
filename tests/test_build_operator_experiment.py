from pathlib import Path

import numpy as np

from nlos_cs.experiments.build_operator import (
    BuildOperatorConfig,
    PlaneSpec,
    build_single_state_operator_experiment,
)


def _write_mock_cst_field_file(
    path: Path,
    *,
    probe_x: float,
    yz_points: list[tuple[float, float]],
    amplitude_scale: float,
) -> None:
    """Write a minimal CST-style ASCII field export.

    Format:
        x  y  z  Re(Ex) Im(Ex) Re(Ey) Im(Ey) Re(Ez) Im(Ez)

    We keep Ex only, with purely real values, so |E| = |Ex|.
    """
    lines = [
        "Header line 1",
        "Header line 2",
    ]

    for y, z in yz_points:
        ex = amplitude_scale * (1.0 + 0.1 * y + 0.01 * z)
        row = [probe_x, y, z, ex, 0.0, 0.0, 0.0, 0.0, 0.0]
        lines.append(" ".join(f"{v:.6f}" for v in row))

    path.write_text("\n".join(lines), encoding="utf-8")


def test_build_single_state_operator_experiment_basic(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    yz_points = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]

    f1 = data_dir / "pos65.txt"
    f2 = data_dir / "pos70.txt"
    f3 = data_dir / "pos75.txt"

    _write_mock_cst_field_file(f1, probe_x=105.0, yz_points=yz_points, amplitude_scale=1.0)
    _write_mock_cst_field_file(f2, probe_x=105.0, yz_points=yz_points, amplitude_scale=2.0)
    _write_mock_cst_field_file(f3, probe_x=105.0, yz_points=yz_points, amplitude_scale=3.0)

    config = BuildOperatorConfig(
        state_id="flat",
        measurement_kind="e_mag",
        plane=PlaneSpec(axis="x", value_mm=105.0, tol_mm=1e-6),
        position_to_file={
            65.0: str(f1),
            70.0: str(f2),
            75.0: str(f3),
        },
        run_name="unit_test_run",
        output_root=str(tmp_path / "outputs"),
        cst_skiprows=2,
        save_probe_vectors=False,
    )

    result = build_single_state_operator_experiment(config)

    op = result.operator
    assert op.state_id == "flat"
    assert op.measurement_kind == "e_mag"
    assert op.shape == (4, 3)
    assert np.allclose(op.positions_mm, np.array([65.0, 70.0, 75.0]))

    # Because amplitude scales are 1, 2, 3 and |E| = |Ex|,
    # columns should scale accordingly.
    col0 = op.A[:, 0]
    col1 = op.A[:, 1]
    col2 = op.A[:, 2]

    assert np.allclose(col1, 2.0 * col0)
    assert np.allclose(col2, 3.0 * col0)

    assert result.run_dir.exists()
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "reports" / "build_operator_report.json").exists()
    assert (result.run_dir / "arrays" / "A.npy").exists()
    assert (result.run_dir / "arrays" / "positions_mm.npy").exists()
    assert (result.run_dir / "arrays" / "coords_in_plane_mm.npy").exists()
    assert (result.run_dir / "arrays" / "singular_values.npy").exists()
    assert (result.run_dir / "arrays" / "gram_matrix.npy").exists()

    assert "summary" in result.diagnostics
    assert "verdict" in result.diagnostics
    assert result.manifest["experiment_type"] == "build_single_state_operator"


def test_build_single_state_operator_experiment_saves_probe_vectors(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    yz_points = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]

    f1 = data_dir / "pos65.txt"
    f2 = data_dir / "pos70.txt"

    _write_mock_cst_field_file(f1, probe_x=105.0, yz_points=yz_points, amplitude_scale=1.0)
    _write_mock_cst_field_file(f2, probe_x=105.0, yz_points=yz_points, amplitude_scale=2.0)

    config = BuildOperatorConfig(
        state_id="tilted",
        measurement_kind="e_mag",
        plane=PlaneSpec(axis="x", value_mm=105.0, tol_mm=1e-6),
        position_to_file={
            65.0: str(f1),
            70.0: str(f2),
        },
        run_name="unit_test_probe_vectors",
        output_root=str(tmp_path / "outputs"),
        cst_skiprows=2,
        save_probe_vectors=True,
    )

    result = build_single_state_operator_experiment(config)

    assert (result.run_dir / "arrays" / "probe_65mm.npy").exists()
    assert (result.run_dir / "arrays" / "probe_70mm.npy").exists()


def test_build_single_state_operator_experiment_rejects_bad_plane(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    yz_points = [
        (0.0, 0.0),
        (0.0, 1.0),
    ]

    f1 = data_dir / "pos65.txt"
    _write_mock_cst_field_file(f1, probe_x=105.0, yz_points=yz_points, amplitude_scale=1.0)

    config = BuildOperatorConfig(
        state_id="flat",
        measurement_kind="e_mag",
        plane=PlaneSpec(axis="x", value_mm=999.0, tol_mm=1e-6),
        position_to_file={65.0: str(f1)},
        run_name="bad_plane_run",
        output_root=str(tmp_path / "outputs"),
        cst_skiprows=2,
        save_probe_vectors=False,
    )

    try:
        build_single_state_operator_experiment(config)
        assert False, "Expected ValueError for missing probe plane"
    except ValueError as exc:
        assert "No points found within" in str(exc)


def test_build_single_state_operator_experiment_rejects_empty_mapping(tmp_path: Path):
    config = BuildOperatorConfig(
        state_id="flat",
        measurement_kind="e_mag",
        plane=PlaneSpec(axis="x", value_mm=105.0, tol_mm=1e-6),
        position_to_file={},
        run_name="empty_run",
        output_root=str(tmp_path / "outputs"),
        cst_skiprows=2,
        save_probe_vectors=False,
    )

    try:
        build_single_state_operator_experiment(config)
        assert False, "Expected ValueError for empty position_to_file"
    except ValueError as exc:
        assert "position_to_file must not be empty" in str(exc)