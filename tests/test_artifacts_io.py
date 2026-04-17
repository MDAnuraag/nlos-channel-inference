from pathlib import Path

import numpy as np

from nlos_cs.io.artifacts import (
    ensure_dir,
    init_run_dir,
    load_array,
    load_json,
    load_operator_artifact,
    load_reconstruction_artifact,
    load_named_arrays,
    read_manifest,
    save_array,
    save_json,
    save_named_arrays,
    save_operator_artifact,
    save_reconstruction_artifact,
    write_manifest,
)


def test_ensure_dir_creates_directory(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    out = ensure_dir(target)

    assert out == target
    assert target.exists()
    assert target.is_dir()


def test_save_and_load_json(tmp_path: Path):
    data = {
        "name": "test_run",
        "value": 3.14,
        "count": 5,
        "flags": [True, False],
    }
    path = tmp_path / "meta.json"

    save_json(data, path)
    loaded = load_json(path)

    assert loaded == data


def test_save_and_load_array(tmp_path: Path):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = tmp_path / "arr.npy"

    save_array(arr, path)
    loaded = load_array(path)

    assert np.array_equal(loaded, arr)


def test_save_and_load_named_arrays(tmp_path: Path):
    arrays = {
        "A": np.eye(3),
        "positions_mm": np.array([65.0, 70.0, 75.0]),
    }

    save_named_arrays(arrays, tmp_path)
    loaded = load_named_arrays(["A", "positions_mm"], tmp_path)

    assert np.array_equal(loaded["A"], arrays["A"])
    assert np.array_equal(loaded["positions_mm"], arrays["positions_mm"])


def test_init_run_dir_creates_standard_subdirs(tmp_path: Path):
    run_dir = init_run_dir(tmp_path, "run_001")

    assert run_dir == tmp_path / "run_001"
    assert run_dir.exists()
    assert (run_dir / "arrays").exists()
    assert (run_dir / "figures").exists()
    assert (run_dir / "reports").exists()


def test_write_and_read_manifest(tmp_path: Path):
    run_dir = init_run_dir(tmp_path, "run_manifest")
    manifest = {
        "artifact_type": "operator",
        "metadata": {"state_id": "flat"},
        "arrays": {"A": "arrays/A.npy"},
    }

    write_manifest(run_dir, manifest=manifest)
    loaded = read_manifest(run_dir)

    assert loaded == manifest


def test_save_and_load_operator_artifact(tmp_path: Path):
    run_dir = init_run_dir(tmp_path, "operator_run")

    A = np.array([[1.0, 0.2], [0.3, 2.0]])
    positions_mm = np.array([65.0, 70.0])
    coords_in_plane_mm = np.array([[0.0, 0.0], [1.0, 0.0]])
    extra = {"singular_values": np.array([2.1, 0.9])}
    metadata = {
        "state_id": "flat",
        "measurement_kind": "e_mag",
    }

    save_operator_artifact(
        run_dir,
        A=A,
        positions_mm=positions_mm,
        coords_in_plane_mm=coords_in_plane_mm,
        extra_arrays=extra,
        metadata=metadata,
    )

    loaded = load_operator_artifact(run_dir)

    assert np.array_equal(loaded["A"], A)
    assert np.array_equal(loaded["positions_mm"], positions_mm)
    assert np.array_equal(loaded["coords_in_plane_mm"], coords_in_plane_mm)
    assert np.array_equal(loaded["singular_values"], extra["singular_values"])
    assert loaded["metadata"] == metadata

    manifest = loaded["manifest"]
    assert manifest["artifact_type"] == "operator"
    assert "A" in manifest["arrays"]
    assert "positions_mm" in manifest["arrays"]


def test_save_and_load_reconstruction_artifact(tmp_path: Path):
    run_dir = init_run_dir(tmp_path, "recon_run")

    x_hat = np.array([0.1, 1.0, 0.2])
    y = np.array([2.0, 3.0, 4.0])
    residual = np.array([0.01, -0.02, 0.03])
    extra = {"lambdas": np.array([0.1, 1.0, 10.0])}
    metadata = {
        "solver_name": "tikhonov_lcurve",
        "lambda_value": 1.0,
    }

    save_reconstruction_artifact(
        run_dir,
        x_hat=x_hat,
        y=y,
        residual=residual,
        extra_arrays=extra,
        metadata=metadata,
    )

    loaded = load_reconstruction_artifact(run_dir)

    assert np.array_equal(loaded["x_hat"], x_hat)
    assert np.array_equal(loaded["y"], y)
    assert np.array_equal(loaded["residual"], residual)
    assert np.array_equal(loaded["lambdas"], extra["lambdas"])
    assert loaded["metadata"] == metadata

    manifest = loaded["manifest"]
    assert manifest["artifact_type"] == "reconstruction"
    assert "x_hat" in manifest["arrays"]


def test_save_json_handles_numpy_types(tmp_path: Path):
    data = {
        "float": np.float64(1.25),
        "int": np.int64(7),
        "bool": np.bool_(True),
        "array": np.array([1, 2, 3]),
    }
    path = tmp_path / "numpy_meta.json"

    save_json(data, path)
    loaded = load_json(path)

    assert loaded["float"] == 1.25
    assert loaded["int"] == 7
    assert loaded["bool"] is True
    assert loaded["array"] == [1, 2, 3]


def test_manifest_uses_relative_paths(tmp_path: Path):
    run_dir = init_run_dir(tmp_path, "relative_paths_run")

    A = np.eye(2)
    positions_mm = np.array([65.0, 70.0])

    save_operator_artifact(
        run_dir,
        A=A,
        positions_mm=positions_mm,
        metadata={"state_id": "tilted"},
    )

    manifest = read_manifest(run_dir)

    assert manifest["arrays"]["A"] == "arrays/A.npy"
    assert manifest["arrays"]["positions_mm"] == "arrays/positions_mm.npy"