"""Artifact I/O helpers for saving and loading experiment outputs.

Design goals
------------
- one manifest per run
- predictable folder layout
- no hard-coded thesis-era filenames
- small metadata in JSON
- numeric arrays in .npy
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
ArrayLike = npt.NDArray[Any]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_jsonable(obj: Any) -> Any:
    """Convert common Python / NumPy / dataclass objects into JSON-safe values."""
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(data: dict[str, Any], filepath: str | Path, *, indent: int = 2) -> Path:
    """Save a JSON metadata file."""
    path = Path(filepath)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=indent, sort_keys=True)
    return path


def load_json(filepath: str | Path) -> dict[str, Any]:
    """Load a JSON metadata file."""
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_array(array: ArrayLike, filepath: str | Path) -> Path:
    """Save one NumPy array as .npy."""
    path = Path(filepath)
    ensure_dir(path.parent)
    np.save(path, array)
    return path


def load_array(filepath: str | Path) -> ArrayLike:
    """Load one NumPy array from .npy."""
    path = Path(filepath)
    return np.load(path)


def save_named_arrays(arrays: dict[str, ArrayLike], directory: str | Path) -> dict[str, Path]:
    """Save multiple arrays into one directory as <name>.npy."""
    out_dir = ensure_dir(directory)
    saved: dict[str, Path] = {}
    for name, arr in arrays.items():
        saved[name] = save_array(arr, out_dir / f"{name}.npy")
    return saved


def load_named_arrays(names: list[str], directory: str | Path) -> dict[str, ArrayLike]:
    """Load multiple arrays from one directory using <name>.npy convention."""
    base = Path(directory)
    return {name: load_array(base / f"{name}.npy") for name in names}


def init_run_dir(
    root: str | Path,
    run_name: str,
    *,
    subdirs: tuple[str, ...] = ("arrays", "figures", "reports"),
) -> Path:
    """Create a run directory with standard subfolders.

    Example result:
        root/run_name/
            manifest.json
            arrays/
            figures/
            reports/
    """
    run_dir = ensure_dir(Path(root) / run_name)
    for subdir in subdirs:
        ensure_dir(run_dir / subdir)
    return run_dir


def write_manifest(
    run_dir: str | Path,
    *,
    manifest: dict[str, Any],
    filename: str = "manifest.json",
) -> Path:
    """Write the main run manifest."""
    return save_json(manifest, Path(run_dir) / filename)


def read_manifest(
    run_dir: str | Path,
    *,
    filename: str = "manifest.json",
) -> dict[str, Any]:
    """Read the main run manifest."""
    return load_json(Path(run_dir) / filename)


def save_operator_artifact(
    run_dir: str | Path,
    *,
    A: ArrayLike,
    positions_mm: ArrayLike,
    coords_in_plane_mm: ArrayLike | None = None,
    extra_arrays: dict[str, ArrayLike] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save a standard operator artifact bundle.

    Saved arrays:
    - arrays/A.npy
    - arrays/positions_mm.npy
    - arrays/coords_in_plane_mm.npy (optional)
    - any extra arrays in arrays/

    Also updates / writes manifest.json with lightweight metadata.
    """
    base = Path(run_dir)
    arrays_dir = ensure_dir(base / "arrays")

    saved: dict[str, Path] = {}
    saved["A"] = save_array(A, arrays_dir / "A.npy")
    saved["positions_mm"] = save_array(positions_mm, arrays_dir / "positions_mm.npy")

    if coords_in_plane_mm is not None:
        saved["coords_in_plane_mm"] = save_array(
            coords_in_plane_mm, arrays_dir / "coords_in_plane_mm.npy"
        )

    if extra_arrays:
        for name, arr in extra_arrays.items():
            saved[name] = save_array(arr, arrays_dir / f"{name}.npy")

    manifest = {
        "artifact_type": "operator",
        "arrays": {key: path.relative_to(base).as_posix() for key, path in saved.items()},
        "metadata": {} if metadata is None else _to_jsonable(metadata),
    }
    write_manifest(base, manifest=manifest)

    return saved


def load_operator_artifact(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Load a standard operator artifact bundle."""
    base = Path(run_dir)
    manifest = read_manifest(base)

    arrays_info = manifest.get("arrays", {})
    out = {
        "manifest": manifest,
        "metadata": manifest.get("metadata", {}),
    }

    for key, rel_path in arrays_info.items():
        out[key] = load_array(base / rel_path)

    return out


def save_reconstruction_artifact(
    run_dir: str | Path,
    *,
    x_hat: ArrayLike,
    y: ArrayLike | None = None,
    residual: ArrayLike | None = None,
    extra_arrays: dict[str, ArrayLike] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save a standard reconstruction artifact bundle."""
    base = Path(run_dir)
    arrays_dir = ensure_dir(base / "arrays")

    saved: dict[str, Path] = {}
    saved["x_hat"] = save_array(x_hat, arrays_dir / "x_hat.npy")

    if y is not None:
        saved["y"] = save_array(y, arrays_dir / "y.npy")
    if residual is not None:
        saved["residual"] = save_array(residual, arrays_dir / "residual.npy")

    if extra_arrays:
        for name, arr in extra_arrays.items():
            saved[name] = save_array(arr, arrays_dir / f"{name}.npy")

    manifest = {
        "artifact_type": "reconstruction",
        "arrays": {key: path.relative_to(base).as_posix() for key, path in saved.items()},
        "metadata": {} if metadata is None else _to_jsonable(metadata),
    }
    write_manifest(base, manifest=manifest)

    return saved


def load_reconstruction_artifact(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Load a standard reconstruction artifact bundle."""
    base = Path(run_dir)
    manifest = read_manifest(base)

    arrays_info = manifest.get("arrays", {})
    out = {
        "manifest": manifest,
        "metadata": manifest.get("metadata", {}),
    }

    for key, rel_path in arrays_info.items():
        out[key] = load_array(base / rel_path)

    return out