"""Probe-plane extraction utilities.

This module converts a full 3D CST field export into a single 2D measurement
plane suitable for operator construction.

Design rules:
- extraction is geometric only
- no filename logic
- no experiment-specific assumptions
- deterministic row ordering across files
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from nlos_cs.io.cst_ascii import CSTFieldExport, infer_axis_values


FloatArray = npt.NDArray[np.float64]
AxisName = Literal["x", "y", "z"]


@dataclass(frozen=True)
class ProbePlane:
    """A 2D measurement plane extracted from a 3D field export.

    Attributes
    ----------
    axis:
        The axis held fixed, e.g. "x" for an x = const plane.
    value_mm:
        The target plane value along that axis.
    tol_mm:
        Absolute tolerance used for plane selection.
    coords_in_plane_mm:
        Array of shape (M, 2), giving the two in-plane coordinates after sorting.
    field_indices:
        Indices into the original 3D export rows.
    e_mag:
        |E| values on the selected plane, ordered consistently with coords_in_plane_mm.
    e_complex:
        Complex field vectors on the selected plane, ordered consistently.
    in_plane_axes:
        Names of the two axes spanning the plane, e.g. ("y", "z") for x = const.
    """

    axis: AxisName
    value_mm: float
    tol_mm: float
    coords_in_plane_mm: FloatArray
    field_indices: npt.NDArray[np.int64]
    e_mag: FloatArray
    e_complex: npt.NDArray[np.complex128]
    in_plane_axes: tuple[AxisName, AxisName]

    @property
    def n_points(self) -> int:
        """Number of points in the extracted plane."""
        return int(self.e_mag.shape[0])

    @property
    def unique_axis_1(self) -> FloatArray:
        """Unique values along the first in-plane axis."""
        return np.unique(self.coords_in_plane_mm[:, 0])

    @property
    def unique_axis_2(self) -> FloatArray:
        """Unique values along the second in-plane axis."""
        return np.unique(self.coords_in_plane_mm[:, 1])


_AXIS_TO_INDEX: dict[AxisName, int] = {"x": 0, "y": 1, "z": 2}


def _in_plane_axes(axis: AxisName) -> tuple[AxisName, AxisName]:
    if axis == "x":
        return ("y", "z")
    if axis == "y":
        return ("x", "z")
    return ("x", "y")


def extract_probe_plane(
    field: CSTFieldExport,
    axis: AxisName,
    value_mm: float,
    tol_mm: float,
) -> ProbePlane:
    """Extract a plane defined by one fixed coordinate.

    Parameters
    ----------
    field:
        Parsed CST field export.
    axis:
        Axis to hold fixed: "x", "y", or "z".
    value_mm:
        Target coordinate value in mm for the plane.
    tol_mm:
        Absolute tolerance for selecting points on the plane.

    Returns
    -------
    ProbePlane
        Extracted and deterministically sorted plane.

    Raises
    ------
    ValueError
        If no points are found on the requested plane.
    """
    axis_idx = _AXIS_TO_INDEX[axis]
    mask = np.abs(field.coords_mm[:, axis_idx] - value_mm) < tol_mm

    if not np.any(mask):
        axes = infer_axis_values(field.coords_mm, decimals=6)
        present = axes[axis]
        raise ValueError(
            f"No points found within {tol_mm} mm of {axis}={value_mm} mm. "
            f"Available {axis}-values include: {present}"
        )

    field_indices = np.where(mask)[0].astype(np.int64)
    coords_sel = field.coords_mm[field_indices]
    e_mag_sel = field.e_mag[field_indices]
    e_complex_sel = field.e_complex[field_indices]

    axis1, axis2 = _in_plane_axes(axis)
    axis1_idx = _AXIS_TO_INDEX[axis1]
    axis2_idx = _AXIS_TO_INDEX[axis2]

    coords_2d = coords_sel[:, [axis1_idx, axis2_idx]]

    # Deterministic ordering:
    # sort by first in-plane axis primary, second in-plane axis secondary
    sort_idx = np.lexsort((coords_2d[:, 1], coords_2d[:, 0]))

    coords_2d = coords_2d[sort_idx].astype(np.float64, copy=False)
    field_indices = field_indices[sort_idx]
    e_mag_sel = e_mag_sel[sort_idx].astype(np.float64, copy=False)
    e_complex_sel = e_complex_sel[sort_idx].astype(np.complex128, copy=False)

    return ProbePlane(
        axis=axis,
        value_mm=float(value_mm),
        tol_mm=float(tol_mm),
        coords_in_plane_mm=coords_2d,
        field_indices=field_indices,
        e_mag=e_mag_sel,
        e_complex=e_complex_sel,
        in_plane_axes=(axis1, axis2),
    )


def validate_rectangular_grid(plane: ProbePlane, decimals: int = 6) -> None:
    """Check whether the extracted plane forms a complete rectangular grid.

    This is important because operator construction usually assumes that each
    measurement plane has identical row count and identical coordinate layout
    across all sensing states / object positions.
    """
    coords = np.round(plane.coords_in_plane_mm, decimals=decimals)
    axis1_vals = np.unique(coords[:, 0])
    axis2_vals = np.unique(coords[:, 1])

    expected = len(axis1_vals) * len(axis2_vals)
    actual = plane.n_points

    if actual != expected:
        raise ValueError(
            f"Probe plane is not a complete rectangular grid: "
            f"expected {expected} points from "
            f"{len(axis1_vals)} x {len(axis2_vals)}, got {actual}"
        )


def compare_plane_coordinates(
    reference: ProbePlane,
    candidate: ProbePlane,
    atol: float = 1e-9,
) -> None:
    """Raise if two probe planes do not share identical ordered coordinates.

    This is the strictest check needed before stacking columns into a sensing matrix.
    """
    if reference.in_plane_axes != candidate.in_plane_axes:
        raise ValueError(
            f"In-plane axes differ: {reference.in_plane_axes} vs {candidate.in_plane_axes}"
        )

    if reference.coords_in_plane_mm.shape != candidate.coords_in_plane_mm.shape:
        raise ValueError(
            f"Plane shapes differ: "
            f"{reference.coords_in_plane_mm.shape} vs {candidate.coords_in_plane_mm.shape}"
        )

    if not np.allclose(
        reference.coords_in_plane_mm,
        candidate.coords_in_plane_mm,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("Probe-plane coordinates differ between exports")


def plane_to_image_grid(
    plane: ProbePlane,
    values: FloatArray | None = None,
    decimals: int = 6,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Reshape plane data into a 2D grid for plotting.

    Parameters
    ----------
    plane:
        Extracted probe plane.
    values:
        Optional values of shape (M,) to reshape. If omitted, plane.e_mag is used.
    decimals:
        Rounding applied before unique-axis detection.

    Returns
    -------
    axis1_vals, axis2_vals, grid
        axis1_vals has length N1, axis2_vals has length N2, and grid has shape (N1, N2).

    Notes
    -----
    Because the plane is sorted lexicographically by (axis1, axis2), a simple reshape
    is valid once the grid completeness has been checked.
    """
    validate_rectangular_grid(plane, decimals=decimals)

    vals = plane.e_mag if values is None else values
    if vals.shape[0] != plane.n_points:
        raise ValueError(
            f"values length {vals.shape[0]} does not match plane size {plane.n_points}"
        )

    coords = np.round(plane.coords_in_plane_mm, decimals=decimals)
    axis1_vals = np.unique(coords[:, 0])
    axis2_vals = np.unique(coords[:, 1])

    grid = vals.reshape(len(axis1_vals), len(axis2_vals))
    return axis1_vals, axis2_vals, grid