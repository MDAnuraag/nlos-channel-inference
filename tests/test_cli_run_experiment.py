from pathlib import Path

import numpy as np

from nlos_cs.cli.run_experiment import run_experiment_from_config
from nlos_cs.io.artifacts import init_run_dir, save_operator_artifact


def _make_operator_artifact(tmp_path: Path) -> Path:
    run_dir = init_run_dir(tmp_path, "operator_artifact")
    A = np.array(
        [
            [4.0, 0.2, 0.1],
            [0.1, 5.0, 0.2],
            [0.2, 0.3, 6.0],
        ],
        dtype=float,
    )
    positions_mm = np.array([65.0, 70.0, 75.0], dtype=float)

    save_operator_artifact(
        run_dir,
        A=A,
        positions_mm=positions_mm,
        metadata={"state_id": "flat", "measurement_kind": "e_mag"},
    )
    return run_dir


def test_run_experiment_from_config_build_single_state_operator(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    field_path = data_dir / "pos65.txt"
    field_path.write_text(
        "\n".join(
            [
                "Header line 1",
                "Header line 2",
                "105.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000",
                "105.000000 0.000000 1.000000 1.100000 0.000000 0.000000 0.000000 0.000000 0.000000",
                "105.000000 1.000000 0.000000 1.200000 0.000000 0.000000 0.000000 0.000000 0.000000",
                "105.000000 1.000000 1.000000 1.300000 0.000000 0.000000 0.000000 0.000000 0.000000",
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "experiment_type": "build_single_state_operator",
        "state_id": "flat",
        "measurement_kind": "e_mag",
        "plane": {
            "axis": "x",
            "value_mm": 105.0,
            "tol_mm": 1e-6,
        },
        "position_to_file": {
            "65.0": str(field_path),
        },
        "run_name": "cli_build_test",
        "output_root": str(tmp_path / "outputs"),
        "cst_skiprows": 2,
    }

    result = run_experiment_from_config(config)

    assert result["experiment_type"] == "build_single_state_operator"
    assert result["state_id"] == "flat"
    assert result["n_positions"] == 1
    assert Path(result["run_dir"]).exists()


def test_run_experiment_from_config_reconstruction(tmp_path: Path):
    artifact_dir = _make_operator_artifact(tmp_path)

    config = {
        "experiment_type": "run_reconstruction",
        "operator": {"artifact_dir": str(artifact_dir)},
        "solver": {
            "type": "tikhonov",
            "lambda_value": 0.5,
        },
        "run_name": "cli_recon_test",
        "output_root": str(tmp_path / "outputs"),
        "measurement_mode": "operator_column",
        "true_index": 2,
        "noise_fraction_of_rms": 0.0,
    }

    result = run_experiment_from_config(config)

    assert result["experiment_type"] == "run_reconstruction"
    assert result["solver_name"] == "tikhonov"
    assert result["peak_index"] in (0, 1, 2)
    assert Path(result["run_dir"]).exists()


def test_run_experiment_from_config_discrimination(tmp_path: Path):
    artifact_dir = _make_operator_artifact(tmp_path)

    config = {
        "experiment_type": "run_discrimination",
        "operator": {"artifact_dir": str(artifact_dir)},
        "solver": {
            "type": "tikhonov",
            "lambda_value": 0.5,
        },
        "run_name": "cli_disc_test",
        "output_root": str(tmp_path / "outputs"),
        "mode": "matched",
        "top_k_pairs": 5,
    }

    result = run_experiment_from_config(config)

    assert result["experiment_type"] == "run_discrimination"
    assert result["solver_name"] == "tikhonov"
    assert len(result["worst_pair"]) == 2
    assert Path(result["run_dir"]).exists()


def test_run_experiment_from_config_awgn_robustness(tmp_path: Path):
    artifact_dir = _make_operator_artifact(tmp_path)

    config = {
        "experiment_type": "run_awgn_robustness",
        "operator": {"artifact_dir": str(artifact_dir)},
        "solver": {
            "type": "tikhonov",
            "lambda_value": 0.5,
        },
        "run_name": "cli_robust_test",
        "output_root": str(tmp_path / "outputs"),
        "noise_levels": [0.0, 0.01],
        "n_trials": 3,
        "random_seed": 42,
    }

    result = run_experiment_from_config(config)

    assert result["experiment_type"] == "run_awgn_robustness"
    assert result["solver_name"] == "tikhonov"
    assert 0.0 <= result["overall_mean_success_rate"] <= 1.0
    assert Path(result["run_dir"]).exists()


def test_run_experiment_from_config_rejects_unknown_experiment(tmp_path: Path):
    config = {
        "experiment_type": "not_real",
    }

    try:
        run_experiment_from_config(config)
        assert False, "Expected ValueError for unsupported experiment_type"
    except ValueError as exc:
        assert "Unsupported experiment_type" in str(exc)


def test_run_experiment_from_config_rejects_unknown_solver(tmp_path: Path):
    artifact_dir = _make_operator_artifact(tmp_path)

    config = {
        "experiment_type": "run_reconstruction",
        "operator": {"artifact_dir": str(artifact_dir)},
        "solver": {
            "type": "not_real",
        },
        "run_name": "bad_solver",
    }

    try:
        run_experiment_from_config(config)
        assert False, "Expected ValueError for unknown solver"
    except ValueError as exc:
        assert "Unknown solver type" in str(exc)