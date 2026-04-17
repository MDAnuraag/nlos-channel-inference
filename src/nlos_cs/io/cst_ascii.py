"""Utilities for parsing CST E-field ASCII exports.

Expected CST export format after the header:

    x(mm)  y(mm)  z(mm)  Re(Ex) Im(Ex)  Re(Ey) Im(Ey)  Re(Ez) Im(Ez)

This module is intentionally narrow:
- parse one CST ASCII field export
- expose coordinates, complex vector field, and |E|
- avoid any probe-plane extraction or experiment logic
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


@dataclass(frozen=True)
class CSTFieldExport:
    """Parsed CST field export.

    Attributes
    ----------
    source_path:
        Original file path.
    coords_mm:
        Array of shape (N, 3) with columns [x_mm, y_mm, z_mm].
    e_complex:
        Array of shape (N, 3) with complex phasor components [Ex, Ey, Ez].
    e_mag:
        Array of shape (N,) with total field magnitude:
            sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)
    """

    source_path: Path
    coords_mm: FloatArray
    e_complex: ComplexArray
    e_mag: FloatArray

    @property
    def n_points(self) -> int:
        """Number of sampled field points."""
        return int(self.coords_mm.shape[0])


def load_cst_efield_ascii(filepath: str | Path, skiprows: int = 2) -> CSTFieldExport:
    """Load a CST ASCII E-field export.

    Parameters
    ----------
    filepath:
        Path to the CST ASCII export.
    skiprows:
        Number of initial header rows to skip. Defaults to 2, matching the
        legacy thesis exports.

    Returns
    -------
    CSTFieldExport
        Parsed field export with coordinates, complex field, and |E|.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed as a 9-column CST export.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CST export not found: {path}")

    raw = np.loadtxt(path, skiprows=skiprows)

    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError(
            f"Expected a 9-column CST field export after skipping {skiprows} rows, "
            f"got shape {raw.shape} from {path}"
        )

    coords_mm = raw[:, 0:3].astype(np.float64, copy=False)

    ex = raw[:, 3] + 1j * raw[:, 4]
    ey = raw[:, 5] + 1j * raw[:, 6]
    ez = raw[:, 7] + 1j * raw[:, 8]

    e_complex = np.stack([ex, ey, ez], axis=1).astype(np.complex128, copy=False)
    e_mag = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2).astype(
        np.float64, copy=False
    )

    return CSTFieldExport(
        source_path=path,
        coords_mm=coords_mm,
        e_complex=e_complex,
        e_mag=e_mag,
    )


def infer_axis_values(coords_mm: FloatArray, decimals: int = 6) -> dict[str, FloatArray]:
    """Return sorted unique axis values for quick diagnostics.

    Parameters
    ----------
    coords_mm:
        Coordinate array of shape (N, 3).
    decimals:
        Decimal rounding used before uniqueness, to suppress tiny export noise.

    Returns
    -------
    dict[str, ndarray]
        Keys: ``"x"``, ``"y"``, ``"z"``.
    """
    if coords_mm.ndim != 2 or coords_mm.shape[1] != 3:
        raise ValueError(f"coords_mm must have shape (N, 3), got {coords_mm.shape}")

    rounded = np.round(coords_mm, decimals=decimals)
    return {
        "x": np.unique(rounded[:, 0]),
        "y": np.unique(rounded[:, 1]),
        "z": np.unique(rounded[:, 2]),
    }


def validate_finite_values(field: CSTFieldExport) -> None:
    """Raise if coordinates or fields contain NaN/Inf."""
    if not np.all(np.isfinite(field.coords_mm)):
        raise ValueError(f"Non-finite coordinate values found in {field.source_path}")
    if not np.all(np.isfinite(field.e_mag)):
        raise ValueError(f"Non-finite |E| values found in {field.source_path}")
    if not np.all(np.isfinite(field.e_complex.real)) or not np.all(
        np.isfinite(field.e_complex.imag)
    ):
        raise ValueError(f"Non-finite complex field values found in {field.source_path}")


def summarise_field_export(field: CSTFieldExport) -> dict[str, float | int]:
    """Return lightweight numeric summary for logs and tests."""
    return {
        "n_points": field.n_points,
        "e_mag_min": float(np.min(field.e_mag)),
        "e_mag_max": float(np.max(field.e_mag)),
        "e_mag_mean": float(np.mean(field.e_mag)),
        "e_mag_rms": float(np.sqrt(np.mean(field.e_mag**2))),
    }