import json
from pathlib import Path

from nlos_cs.cli.main import app
from nlos_cs.io.artifacts import init_run_dir, save_operator_artifact


def _make_operator_artifact(tmp_path: Path) -> Path:
    run_dir = init_run_dir(tmp_path, "operator_artifact")

    import numpy as np

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


def test_app_run_command_executes_config(capsys, tmp_path: Path):
    artifact_dir = _make_operator_artifact(tmp_path)

    config = {
        "experiment_type": "run_reconstruction",
        "operator": {"artifact_dir": str(artifact_dir)},
        "solver": {
            "type": "tikhonov",
            "lambda_value": 0.5,
        },
        "run_name": "cli_main_run_test",
        "output_root": str(tmp_path / "outputs"),
        "measurement_mode": "operator_column",
        "true_index": 2,
        "noise_fraction_of_rms": 0.0,
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    code = app(["run", str(config_path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "experiment_type: run_reconstruction" in captured.out
    assert "solver_name: tikhonov" in captured.out
    assert "run_dir:" in captured.out


def test_app_run_command_missing_config_raises(tmp_path: Path):
    missing = tmp_path / "missing.json"

    try:
        app(["run", str(missing)])
        assert False, "Expected FileNotFoundError for missing config"
    except FileNotFoundError as exc:
        assert "Config file not found" in str(exc)


def test_app_info_lists_run_command(capsys):
    code = app(["info"])
    captured = capsys.readouterr()

    assert code == 0
    assert "Available commands: info, run, build-operator, reconstruct, discrim, robustness" in captured.out