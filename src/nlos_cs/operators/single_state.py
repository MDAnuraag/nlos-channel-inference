"""Single-state sensing operator construction.

A single-state operator corresponds to one metasurface / boundary condition
or one fixed sensing configuration. Each column of A is built from one
probe-plane response at one object position.

Typical forward model:

    y = A x + eps

where:
- A is the sensing operator for one state
- x is the latent scene / object-position vector
- y is the probe-plane measurement vector
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from nlos_cs.preprocessing.probe_plane import ProbePlane
from nlos_cs.preprocessing.validation import (
    ValidationReport,
    summarise_matrix_statistics,
    validate_measurement_matrix,
    validate_plane_collection_compatibility,
)

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = npt.NDArray[np.float64] | npt.NDArray[np.complex128]

MeasurementKind = Literal["e_mag", "ex", "ey", "ez"]


@dataclass(frozen=True)
class SingleStateOperator:
    """A sensing operator built from one sensing state.

    Attributes
    ----------
    state_id:
        Identifier for the sensing state, e.g. "flat", "tilted", "stepped".
    positions_mm:
        1D array of object positions corresponding to the columns of A.
    coords_in_plane_mm:
        Ordered 2D measurement coordinates shared by all columns.
    in_plane_axes:
        Names of the two axes spanning the probe plane.
    measurement_kind:
        Which field quantity was used to build the columns:
        "e_mag", "ex", "ey", or "ez".
    A:
        Operator matrix of shape (M, N), where M is the number of measurement
        points and N is the number of object positions.
    """

    state_id: str
    positions_mm: FloatArray
    coords_in_plane_mm: FloatArray
    in_plane_axes: tuple[str, str]
    measurement_kind: MeasurementKind
    A: OperatorArray

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape (M, N)."""
        return self.A.shape

    @property
    def n_measurements(self) -> int:
        """Number of measurement rows M."""
        return int(self.A.shape[0])

    @property
    def n_positions(self) -> int:
        """Number of position columns N."""
        return int(self.A.shape[1])

    @property
    def is_complex(self) -> bool:
        """True if A is complex-valued."""
        return np.iscomplexobj(self.A)

    def summary(self) -> dict[str, float | int | str]:
        """Compact operator summary."""
        stats = summarise_matrix_statistics(np.abs(self.A) if self.is_complex else self.A)
        return {
            "state_id": self.state_id,
            "measurement_kind": self.measurement_kind,
            "n_measurements": self.n_measurements,
            "n_positions": self.n_positions,
            "is_complex": int(self.is_complex),
            **stats,
        }


def _extract_measurement_vector(
    plane: ProbePlane,
    measurement_kind: MeasurementKind,
) -> FloatArray | ComplexArray:
    """Extract one column vector from a probe plane."""
    if measurement_kind == "e_mag":
        return plane.e_mag.astype(np.float64, copy=False)

    component_map = {"ex": 0, "ey": 1, "ez": 2}
    idx = component_map[measurement_kind]
    return plane.e_complex[:, idx].astype(np.complex128, copy=False)


def _validate_positions(positions_mm: Sequence[float | int]) -> FloatArray:
    """Validate and normalise the position list."""
    positions = np.asarray(positions_mm, dtype=np.float64)

    if positions.ndim != 1:
        raise ValueError(f"positions_mm must be 1D, got shape {positions.shape}")
    if positions.size == 0:
        raise ValueError("positions_mm must contain at least one position")

    unique = np.unique(positions)
    if unique.size != positions.size:
        raise ValueError("positions_mm contains duplicate values")

    return positions


def build_single_state_operator(
    *,
    state_id: str,
    positions_mm: Sequence[float | int],
    planes: Sequence[ProbePlane],
    measurement_kind: MeasurementKind = "e_mag",
    validate: bool = True,
) -> SingleStateOperator:
    """Build a single-state sensing operator from a sequence of probe planes.

    Parameters
    ----------
    state_id:
        Identifier for this sensing state.
    positions_mm:
        Object positions corresponding to the supplied planes. These define
        the operator column order.
    planes:
        Probe planes, one per object position, already extracted and sorted.
    measurement_kind:
        Which quantity to use as the column vector. Defaults to "e_mag".
        - "e_mag" yields a real-valued operator.
        - "ex", "ey", "ez" yield complex-valued operators.
    validate:
        If True, run compatibility and matrix sanity checks before returning.

    Returns
    -------
    SingleStateOperator
        Constructed sensing operator.

    Raises
    ------
    ValueError
        If the input planes are incompatible or the operator is invalid.
    """
    positions = _validate_positions(positions_mm)

    if len(planes) != len(positions):
        raise ValueError(
            f"Number of planes ({len(planes)}) must match number of positions ({len(positions)})"
        )

    if validate:
        plane_report = validate_plane_collection_compatibility(planes)
        plane_report.raise_if_errors()

    first_column = _extract_measurement_vector(planes[0], measurement_kind)
    n_rows = first_column.shape[0]
    n_cols = len(planes)

    if np.iscomplexobj(first_column):
        A = np.zeros((n_rows, n_cols), dtype=np.complex128)
    else:
        A = np.zeros((n_rows, n_cols), dtype=np.float64)

    A[:, 0] = first_column
    for j, plane in enumerate(planes[1:], start=1):
        column = _extract_measurement_vector(plane, measurement_kind)
        if column.shape[0] != n_rows:
            raise ValueError(
                f"Plane {j} has column length {column.shape[0]}, expected {n_rows}"
            )
        A[:, j] = column

    if validate:
        matrix_report = validate_measurement_matrix(np.abs(A) if np.iscomplexobj(A) else A)
        matrix_report.raise_if_errors()

    return SingleStateOperator(
        state_id=state_id,
        positions_mm=positions,
        coords_in_plane_mm=planes[0].coords_in_plane_mm.copy(),
        in_plane_axes=planes[0].in_plane_axes,
        measurement_kind=measurement_kind,
        A=A,
    )


def build_single_state_operator_from_pairs(
    *,
    state_id: str,
    position_plane_pairs: Sequence[tuple[float | int, ProbePlane]],
    measurement_kind: MeasurementKind = "e_mag",
    validate: bool = True,
) -> SingleStateOperator:
    """Build an operator from unordered (position, plane) pairs.

    This is a convenience wrapper that sorts by ascending position before
    constructing the operator.
    """
    if len(position_plane_pairs) == 0:
        raise ValueError("position_plane_pairs must not be empty")

    sorted_pairs = sorted(position_plane_pairs, key=lambda item: float(item[0]))
    positions = [pair[0] for pair in sorted_pairs]
    planes = [pair[1] for pair in sorted_pairs]

    return build_single_state_operator(
        state_id=state_id,
        positions_mm=positions,
        planes=planes,
        measurement_kind=measurement_kind,
        validate=validate,
    )


def operator_column_norms(operator: SingleStateOperator) -> FloatArray:
    """Return Euclidean norm of each operator column."""
    return np.linalg.norm(operator.A, axis=0).astype(np.float64, copy=False)


def operator_column_means(operator: SingleStateOperator) -> FloatArray:
    """Return mean value of each operator column.

    For complex operators, this returns the mean complex phasor.
    """
    return np.mean(operator.A, axis=0)


def validate_single_state_operator(operator: SingleStateOperator) -> ValidationReport:
    """Validate an already constructed operator object."""
    report = ValidationReport()

    if operator.positions_mm.ndim != 1:
        report.add_error("bad_positions_shape", "positions_mm must be 1D")

    if operator.coords_in_plane_mm.ndim != 2 or operator.coords_in_plane_mm.shape[1] != 2:
        report.add_error(
            "bad_coordinate_shape",
            f"coords_in_plane_mm must have shape (M, 2), got {operator.coords_in_plane_mm.shape}",
        )

    if operator.A.ndim != 2:
        report.add_error("bad_matrix_shape", f"A must be 2D, got shape {operator.A.shape}")
        return report

    if operator.A.shape[0] != operator.coords_in_plane_mm.shape[0]:
        report.add_error(
            "row_coordinate_mismatch",
            "Number of rows in A does not match number of probe-plane coordinates",
        )

    if operator.A.shape[1] != operator.positions_mm.shape[0]:
        report.add_error(
            "column_position_mismatch",
            "Number of columns in A does not match number of positions",
        )

    matrix_report = validate_measurement_matrix(
        np.abs(operator.A) if np.iscomplexobj(operator.A) else operator.A
    )
    report.issues.extend(matrix_report.issues)

    return report