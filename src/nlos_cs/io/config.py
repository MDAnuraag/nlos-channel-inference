"""Config file loading and path-resolution helpers.

Current scope
-------------
- JSON config loading/saving
- resolving relative paths against the config file location
- lightweight validation helpers for experiment configs

This stays deliberately small. It is not a schema system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class LoadedConfig:
    """Loaded config plus source-path context."""

    data: dict[str, Any]
    source_path: Path

    @property
    def base_dir(self) -> Path:
        """Directory containing the config file."""
        return self.source_path.parent


def load_json_config(filepath: str | Path) -> LoadedConfig:
    """Load a JSON config file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Top-level config object must be a JSON object")

    return LoadedConfig(data=data, source_path=path.resolve())


def save_json_config(
    data: dict[str, Any],
    filepath: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Save a JSON config file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=True)

    return path


def resolve_path(value: str | Path, *, base_dir: str | Path) -> Path:
    """Resolve a path relative to a config base directory.

    Absolute paths are preserved.
    Relative paths are resolved against `base_dir`.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def resolve_path_list(
    values: list[str] | list[Path],
    *,
    base_dir: str | Path,
) -> list[Path]:
    """Resolve a list of paths relative to a config base directory."""
    return [resolve_path(v, base_dir=base_dir) for v in values]


def resolve_path_mapping(
    mapping: dict[str, str] | dict[float, str] | dict[int, str],
    *,
    base_dir: str | Path,
) -> dict[Any, Path]:
    """Resolve a mapping whose values are paths relative to a config base directory."""
    return {k: resolve_path(v, base_dir=base_dir) for k, v in mapping.items()}


def resolve_selected_fields(
    config: dict[str, Any],
    *,
    base_dir: str | Path,
    scalar_fields: tuple[str, ...] = (),
    list_fields: tuple[str, ...] = (),
    mapping_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resolve selected config fields that contain paths."""
    out = dict(config)

    for field in scalar_fields:
        if field in out and out[field] is not None:
            out[field] = resolve_path(out[field], base_dir=base_dir)

    for field in list_fields:
        if field in out and out[field] is not None:
            if not isinstance(out[field], list):
                raise ValueError(f"Field '{field}' must be a list")
            out[field] = resolve_path_list(out[field], base_dir=base_dir)

    for field in mapping_fields:
        if field in out and out[field] is not None:
            if not isinstance(out[field], dict):
                raise ValueError(f"Field '{field}' must be a mapping")
            out[field] = resolve_path_mapping(out[field], base_dir=base_dir)

    return out


def require_fields(
    config: dict[str, Any],
    required_fields: tuple[str, ...],
) -> None:
    """Raise if any required fields are missing."""
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required config field(s): {missing}")


def require_field_types(
    config: dict[str, Any],
    type_map: dict[str, type | tuple[type, ...]],
) -> None:
    """Raise if present fields do not have the expected types."""
    for field, expected in type_map.items():
        if field not in config:
            continue
        if not isinstance(config[field], expected):
            raise ValueError(
                f"Field '{field}' has type {type(config[field]).__name__}, expected {expected}"
            )


def load_and_prepare_config(
    filepath: str | Path,
    *,
    required_fields: tuple[str, ...] = (),
    type_map: dict[str, type | tuple[type, ...]] | None = None,
    scalar_path_fields: tuple[str, ...] = (),
    list_path_fields: tuple[str, ...] = (),
    mapping_path_fields: tuple[str, ...] = (),
) -> LoadedConfig:
    """Load a config, validate basic shape, and resolve selected path fields."""
    loaded = load_json_config(filepath)

    require_fields(loaded.data, required_fields)
    if type_map is not None:
        require_field_types(loaded.data, type_map)

    resolved = resolve_selected_fields(
        loaded.data,
        base_dir=loaded.base_dir,
        scalar_fields=scalar_path_fields,
        list_fields=list_path_fields,
        mapping_fields=mapping_path_fields,
    )

    return LoadedConfig(
        data=resolved,
        source_path=loaded.source_path,
    )