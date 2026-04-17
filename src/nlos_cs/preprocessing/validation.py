"""Validation utilities for parsed CST fields, probe planes, and operator inputs.

This module is intentionally generic. It does not know about:
- thesis-era "flat / tilted / stepped" labels
- specific benchmark positions like 75 mm or 100 mm
- experiment plots or report wording

It only answers:
- is the data finite?
- is the probe plane structurally valid?
- are multiple planes compatible for stacking?
- are measurement vectors non-trivial and sufficiently distinct?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from nlos_cs.io.cst_ascii import CSTFieldExport, validate_finite_values
from nlos_cs.preprocessing.probe_plane import (
    ProbePlane,
    compare_plane_coordinates,
    validate_rectangular_grid,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding."""

    level: str  # "error" | "warning"
    code: str
    message: str


@dataclass
class ValidationReport:
    """Accumulated validation findings."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(level="error", code=code, message=message))

    def add_warning(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(level="warning", code=code, message=message))

    @property
    def ok(self) -> bool:
        """True if no errors were recorded."""
        return not any(issue.level == "error" for issue in self.issues)

    def raise_if_errors(self) -> None:
        """Raise ValueError if any errors are present."""
        errors = [i for i in self.issues if i.level == "error"]
        if not errors:
            return

        formatted = "\n".join(f"[{e.code}] {e.message}" for e in errors)
        raise ValueError(f"Validation failed with {len(errors)} error(s):\n{formatted}")


def validate_field_export_basic(field: CSTFieldExport) -> ValidationReport:
    """Validate one parsed CST field export.

    Checks:
    - finite coordinates and fields
    - non-empty sample count
    - non-zero field magnitude
    """
    report = ValidationReport()

    try:
        validate_finite_values(field)
    except ValueError as exc:
        report.add_error("non_finite_field", str(exc))

    if field.n_points == 0:
        report.add_error("empty_field", "Parsed field export contains zero sample points")

    if field.n_points > 0 and float(np.max(np.abs(field.e_mag))) == 0.0:
        report.add_error(
            "zero_field",
            f"Field export {field.source_path} contains only zero-magnitude samples",
        )

    return report


def validate_probe_plane_basic(plane: ProbePlane) -> ValidationReport:
    """Validate one extracted probe plane.

    Checks:
    - finite plane values
    - non-empty plane
    - non-zero |E|
    - rectangular grid completeness
    """
    report = ValidationReport()

    if plane.n_points == 0:
        report.add_error("empty_plane", "Probe plane contains zero points")
        return report

    if not np.all(np.isfinite(plane.coords_in_plane_mm)):
        report.add_error("non_finite_plane_coords", "Probe plane coordinates contain NaN/Inf")

    if not np.all(np.isfinite(plane.e_mag)):
        report.add_error("non_finite_plane_mag", "Probe plane |E| contains NaN/Inf")

    if not np.all(np.isfinite(plane.e_complex.real)) or not np.all(
        np.isfinite(plane.e_complex.imag)
    ):
        report.add_error("non_finite_plane_complex", "Probe plane complex field contains NaN/Inf")

    if float(np.max(np.abs(plane.e_mag))) == 0.0:
        report.add_error("zero_plane", "Probe plane contains only zero-magnitude samples")

    try:
        validate_rectangular_grid(plane)
    except ValueError as exc:
        report.add_error("non_rectangular_grid", str(exc))

    return report


def validate_plane_collection_compatibility(
    planes: Sequence[ProbePlane],
    atol: float = 1e-9,
) -> ValidationReport:
    """Validate that a set of probe planes can be stacked into one operator.

    Checks:
    - collection is non-empty
    - every plane passes basic validation
    - all planes share identical ordered coordinates
    """
    report = ValidationReport()

    if len(planes) == 0:
        report.add_error("empty_collection", "No probe planes were provided")
        return report

    for idx, plane in enumerate(planes):
        sub = validate_probe_plane_basic(plane)
        for issue in sub.issues:
            report.issues.append(
                ValidationIssue(
                    level=issue.level,
                    code=f"plane_{idx}:{issue.code}",
                    message=issue.message,
                )
            )

    if not report.ok:
        return report

    ref = planes[0]
    for idx, plane in enumerate(planes[1:], start=1):
        try:
            compare_plane_coordinates(ref, plane, atol=atol)
        except ValueError as exc:
            report.add_error(
                f"plane_{idx}:coordinate_mismatch",
                f"Plane {idx} is incompatible with reference plane: {exc}",
            )

    return report


def validate_measurement_matrix(
    A: FloatArray,
    *,
    min_relative_spread: float = 1e-6,
    duplicate_tol: float = 1e-12,
) -> ValidationReport:
    """Validate a candidate sensing matrix.

    Parameters
    ----------
    A:
        Measurement matrix of shape (M, N), usually one column per object position.
    min_relative_spread:
        Minimum acceptable relative spread in column means. If the spread is below
        this level, the matrix may be effectively insensitive to position.
    duplicate_tol:
        Tolerance for detecting near-identical columns via normalised correlation.

    Checks
    ------
    - finite values
    - non-empty shape
    - no all-zero columns
    - non-trivial variation across columns
    - warning on near-duplicate columns
    """
    report = ValidationReport()

    if A.ndim != 2:
        report.add_error("bad_ndim", f"Measurement matrix must be 2D, got shape {A.shape}")
        return report

    m, n = A.shape
    if m == 0 or n == 0:
        report.add_error("empty_matrix", f"Measurement matrix has invalid shape {A.shape}")
        return report

    if not np.all(np.isfinite(A)):
        report.add_error("non_finite_matrix", "Measurement matrix contains NaN/Inf")
        return report

    col_max = np.max(np.abs(A), axis=0)
    zero_cols = np.where(col_max == 0.0)[0]
    if len(zero_cols) > 0:
        report.add_error(
            "zero_columns",
            f"Measurement matrix has {len(zero_cols)} all-zero column(s): {zero_cols.tolist()}",
        )

    col_means = np.mean(A, axis=0)
    mean_scale = max(float(np.max(np.abs(col_means))), 1e-12)
    relative_spread = float((np.max(col_means) - np.min(col_means)) / mean_scale)
    if relative_spread < min_relative_spread:
        report.add_warning(
            "low_column_spread",
            "Column means vary only weakly; position sensitivity may be poor",
        )

    # Detect near-identical columns using column-normalised Gram matrix
    norms = np.linalg.norm(A, axis=0)
    good = norms > 0
    if np.sum(good) >= 2:
        A_norm = A[:, good] / norms[good]
        G = A_norm.T @ A_norm
        np.fill_diagonal(G, 0.0)
        max_corr = float(np.max(np.abs(G)))
        if max_corr >= 1.0 - duplicate_tol:
            report.add_warning(
                "near_duplicate_columns",
                f"At least one column pair is nearly identical (max |corr| = {max_corr:.12f})",
            )

    return report


def summarise_matrix_statistics(A: FloatArray) -> dict[str, float | int]:
    """Return compact numeric diagnostics for logs, tests, or manifests."""
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    col_norms = np.linalg.norm(A, axis=0)
    col_means = np.mean(A, axis=0)

    return {
        "n_rows": int(A.shape[0]),
        "n_cols": int(A.shape[1]),
        "global_min": float(np.min(A)),
        "global_max": float(np.max(A)),
        "global_mean": float(np.mean(A)),
        "global_rms": float(np.sqrt(np.mean(A**2))),
        "col_norm_min": float(np.min(col_norms)),
        "col_norm_max": float(np.max(col_norms)),
        "col_mean_min": float(np.min(col_means)),
        "col_mean_max": float(np.max(col_means)),
    }


def compute_column_correlation_matrix(A: FloatArray) -> FloatArray:
    """Compute the column-normalised Gram matrix.

    Returns
    -------
    G : ndarray of shape (N, N)
        G[i, j] = <a_i, a_j> / (||a_i|| ||a_j||)

    Notes
    -----
    For magnitude-only operators this is a useful first diagnostic for
    distinguishability. High off-diagonal values imply poor separability.
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    norms = np.linalg.norm(A, axis=0)
    if np.any(norms == 0.0):
        raise ValueError("Cannot compute column correlation matrix with zero-norm columns")

    A_norm = A / norms[None, :]
    return A_norm.T @ A_norm


def validate_detectability_order(
    reference_values: Sequence[float],
    *,
    should_increase: bool | None = None,
    should_decrease: bool | None = None,
    atol: float = 0.0,
) -> ValidationReport:
    """Optional monotonicity validator for experiment-specific checks.

    This intentionally does not know what the values mean. It only checks a trend.
    Use this for controlled experiments where a monotonic detectability or response
    order is physically expected.

    Exactly one of `should_increase` or `should_decrease` should be True.
    """
    report = ValidationReport()

    vals = np.asarray(reference_values, dtype=np.float64)
    if vals.ndim != 1 or vals.size < 2:
        report.add_error(
            "bad_reference_values",
            "reference_values must be a 1D sequence with at least two elements",
        )
        return report

    if bool(should_increase) == bool(should_decrease):
        report.add_error(
            "bad_monotonicity_spec",
            "Specify exactly one of should_increase or should_decrease",
        )
        return report

    diffs = np.diff(vals)
    if should_increase:
        bad = np.where(diffs < -atol)[0]
        if len(bad) > 0:
            report.add_warning(
                "monotonicity_violation",
                f"Sequence is not non-decreasing at indices {bad.tolist()}",
            )
    else:
        bad = np.where(diffs > atol)[0]
        if len(bad) > 0:
            report.add_warning(
                "monotonicity_violation",
                f"Sequence is not non-increasing at indices {bad.tolist()}",
            )

    return report


def merge_reports(reports: Iterable[ValidationReport]) -> ValidationReport:
    """Merge multiple validation reports into one."""
    out = ValidationReport()
    for report in reports:
        out.issues.extend(report.issues)
    return out