from pathlib import Path
import json

from nlos_cs.io.config import (
    LoadedConfig,
    load_and_prepare_config,
    load_json_config,
    require_field_types,
    require_fields,
    resolve_path,
    resolve_path_list,
    resolve_path_mapping,
    resolve_selected_fields,
    save_json_config,
)


def test_load_json_config_basic(tmp_path: Path):
    config_path = tmp_path / "configs" / "test.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = {"run_name": "demo", "n_trials": 5}

    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    loaded = load_json_config(config_path)

    assert isinstance(loaded, LoadedConfig)
    assert loaded.data == config_data
    assert loaded.source_path == config_path.resolve()
    assert loaded.base_dir == config_path.resolve().parent


def test_load_json_config_rejects_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.json"

    try:
        load_json_config(missing)
        assert False, "Expected FileNotFoundError for missing config"
    except FileNotFoundError as exc:
        assert "Config file not found" in str(exc)


def test_load_json_config_rejects_non_object_top_level(tmp_path: Path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    try:
        load_json_config(config_path)
        assert False, "Expected ValueError for non-object top level"
    except ValueError as exc:
        assert "Top-level config object must be a JSON object" in str(exc)


def test_save_json_config_basic(tmp_path: Path):
    config = {"run_name": "demo", "alpha": 0.1}
    path = tmp_path / "configs" / "saved.json"

    out = save_json_config(config, path)

    assert out == path
    assert path.exists()

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == config


def test_resolve_path_relative(tmp_path: Path):
    base_dir = tmp_path / "configs"
    base_dir.mkdir()

    resolved = resolve_path("data/file.txt", base_dir=base_dir)

    assert resolved == (base_dir / "data" / "file.txt").resolve()


def test_resolve_path_absolute(tmp_path: Path):
    base_dir = tmp_path / "configs"
    base_dir.mkdir()
    absolute = (tmp_path / "already_absolute.txt").resolve()

    resolved = resolve_path(absolute, base_dir=base_dir)

    assert resolved == absolute


def test_resolve_path_list(tmp_path: Path):
    base_dir = tmp_path / "configs"
    base_dir.mkdir()

    resolved = resolve_path_list(["a.txt", "b.txt"], base_dir=base_dir)

    assert resolved == [
        (base_dir / "a.txt").resolve(),
        (base_dir / "b.txt").resolve(),
    ]


def test_resolve_path_mapping(tmp_path: Path):
    base_dir = tmp_path / "configs"
    base_dir.mkdir()

    resolved = resolve_path_mapping(
        {65.0: "flat/65.txt", 70.0: "flat/70.txt"},
        base_dir=base_dir,
    )

    assert resolved[65.0] == (base_dir / "flat" / "65.txt").resolve()
    assert resolved[70.0] == (base_dir / "flat" / "70.txt").resolve()


def test_resolve_selected_fields(tmp_path: Path):
    base_dir = tmp_path / "configs"
    base_dir.mkdir()

    config = {
        "output_root": "outputs",
        "input_files": ["a.txt", "b.txt"],
        "position_to_file": {65.0: "flat/65.txt"},
        "run_name": "demo",
    }

    resolved = resolve_selected_fields(
        config,
        base_dir=base_dir,
        scalar_fields=("output_root",),
        list_fields=("input_files",),
        mapping_fields=("position_to_file",),
    )

    assert resolved["output_root"] == (base_dir / "outputs").resolve()
    assert resolved["input_files"] == [
        (base_dir / "a.txt").resolve(),
        (base_dir / "b.txt").resolve(),
    ]
    assert resolved["position_to_file"][65.0] == (base_dir / "flat" / "65.txt").resolve()
    assert resolved["run_name"] == "demo"


def test_resolve_selected_fields_rejects_bad_list_field(tmp_path: Path):
    config = {"input_files": "not_a_list"}

    try:
        resolve_selected_fields(
            config,
            base_dir=tmp_path,
            list_fields=("input_files",),
        )
        assert False, "Expected ValueError for bad list field"
    except ValueError as exc:
        assert "must be a list" in str(exc)


def test_resolve_selected_fields_rejects_bad_mapping_field(tmp_path: Path):
    config = {"position_to_file": ["not", "a", "mapping"]}

    try:
        resolve_selected_fields(
            config,
            base_dir=tmp_path,
            mapping_fields=("position_to_file",),
        )
        assert False, "Expected ValueError for bad mapping field"
    except ValueError as exc:
        assert "must be a mapping" in str(exc)


def test_require_fields_basic():
    config = {"run_name": "demo", "n_trials": 5}
    require_fields(config, ("run_name", "n_trials"))


def test_require_fields_rejects_missing():
    config = {"run_name": "demo"}

    try:
        require_fields(config, ("run_name", "n_trials"))
        assert False, "Expected ValueError for missing required fields"
    except ValueError as exc:
        assert "Missing required config field" in str(exc)


def test_require_field_types_basic():
    config = {"run_name": "demo", "n_trials": 5, "noise_levels": [0.0, 0.1]}
    require_field_types(
        config,
        {
            "run_name": str,
            "n_trials": int,
            "noise_levels": list,
        },
    )


def test_require_field_types_rejects_wrong_type():
    config = {"n_trials": "five"}

    try:
        require_field_types(config, {"n_trials": int})
        assert False, "Expected ValueError for wrong field type"
    except ValueError as exc:
        assert "Field 'n_trials' has type str" in str(exc)


def test_load_and_prepare_config(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    config_path = config_dir / "experiment.json"
    config_data = {
        "run_name": "demo",
        "position_to_file": {
            "65.0": "flat/65.txt",
            "70.0": "flat/70.txt",
        },
        "output_root": "outputs",
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    loaded = load_and_prepare_config(
        config_path,
        required_fields=("run_name", "position_to_file"),
        type_map={
            "run_name": str,
            "position_to_file": dict,
            "output_root": str,
        },
        scalar_path_fields=("output_root",),
        mapping_path_fields=("position_to_file",),
    )

    assert loaded.data["run_name"] == "demo"
    assert loaded.data["output_root"] == (config_dir / "outputs").resolve()
    assert loaded.data["position_to_file"]["65.0"] == (config_dir / "flat" / "65.txt").resolve()
    assert loaded.data["position_to_file"]["70.0"] == (config_dir / "flat" / "70.txt").resolve()