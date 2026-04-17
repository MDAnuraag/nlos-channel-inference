"""Experiment runner for building and saving a single-state sensing operator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nlos_cs.io.artifacts import init_run_dir, save_operator_artifact, save_json
from nlos_cs.io.cst_ascii import load_cst_efield_ascii, summarise_field_export
from nlos_cs.operators.diagnostics import (
    analyse_single_state_operator,
    summarise_operator_quality,
    quality_verdict,
)
from nlos_cs.operators.single_state import (
    MeasurementKind,
    SingleStateOperator,
    build_single_state_operator_from_pairs,
)
from nlos_cs.preprocessing.probe_plane import ProbePlane, extract_probe_plane
from nlos_cs.preprocessing.validation import (
    ValidationReport,
    merge_reports,
    validate_field_export_basic,
    validate_probe_plane_basic,
)


@dataclass(frozen=True)
class PlaneSpec:
    """Definition of the measurement plane to extract."""

    axis: str
    value_mm: float
    tol_mm: float


@dataclass(frozen=True)
class BuildOperatorConfig:
    """Configuration for one single-state operator build."""

    state_id: str
    measurement_kind: MeasurementKind
    plane: PlaneSpec
    position_to_file: dict[float, str]
    run_name: str
    output_root: str = "outputs"
    cst_skiprows: int = 2
    save_probe_vectors: bool = False


@dataclass(frozen=True)
class BuildOperatorResult:
    """Outputs from one operator-build experiment."""

    operator: SingleStateOperator
    diagnostics: dict[str, Any]
    run_dir: Path
    manifest: dict[str, Any]


def _validate_config(config: BuildOperatorConfig) -> None:
    if len(config.position_to_file) == 0:
        raise ValueError("position_to_file must not be empty")
    if config.cst_skiprows < 0:
        raise ValueError("cst_skiprows must be non-negative")
    if config.plane.tol_mm <= 0:
        raise ValueError("plane.tol_mm must be positive")


def _load_and_extract_one(
    *,
    position_mm: float,
    filepath: str,
    plane_spec: PlaneSpec,
    cst_skiprows: int,
) -> tuple[ProbePlane, dict[str, Any], ValidationReport]:
    """Load one CST export, extract the requested plane, and validate it."""
    field = load_cst_efield_ascii(filepath, skiprows=cst_skiprows)
    field_report = validate_field_export_basic(field)

    plane = extract_probe_plane(
        field=field,
        axis=plane_spec.axis,  # type: ignore[arg-type]
        value_mm=plane_spec.value_mm,
        tol_mm=plane_spec.tol_mm,
    )
    plane_report = validate_probe_plane_basic(plane)

    report = merge_reports([field_report, plane_report])

    metadata = {
        "position_mm": float(position_mm),
        "source_file": str(field.source_path),
        "field_summary": summarise_field_export(field),
        "probe_plane": {
            "axis": plane.axis,
            "value_mm": plane.value_mm,
            "tol_mm": plane.tol_mm,
            "n_points": plane.n_points,
            "in_plane_axes": list(plane.in_plane_axes),
        },
        "validation": [
            {"level": issue.level, "code": issue.code, "message": issue.message}
            for issue in report.issues
        ],
    }

    return plane, metadata, report


def build_single_state_operator_experiment(
    config: BuildOperatorConfig,
) -> BuildOperatorResult:
    """Run the single-state operator build pipeline.

    Steps
    -----
    1. Load CST ASCII exports for all requested positions.
    2. Extract the requested probe plane from each file.
    3. Validate all field exports and planes.
    4. Build the operator matrix A.
    5. Run operator diagnostics.
    6. Save arrays and metadata into a run directory.
    """
    _validate_config(config)

    position_plane_pairs: list[tuple[float, ProbePlane]] = []
    per_position_metadata: list[dict[str, Any]] = []
    reports: list[ValidationReport] = []

    for position_mm, filepath in sorted(config.position_to_file.items(), key=lambda kv: float(kv[0])):
        plane, meta, report = _load_and_extract_one(
            position_mm=float(position_mm),
            filepath=filepath,
            plane_spec=config.plane,
            cst_skiprows=config.cst_skiprows,
        )
        position_plane_pairs.append((float(position_mm), plane))
        per_position_metadata.append(meta)
        reports.append(report)

    merged_report = merge_reports(reports)
    merged_report.raise_if_errors()

    operator = build_single_state_operator_from_pairs(
        state_id=config.state_id,
        position_plane_pairs=position_plane_pairs,
        measurement_kind=config.measurement_kind,
        validate=True,
    )

    diag = analyse_single_state_operator(operator)
    diag_summary = summarise_operator_quality(diag)
    verdict = quality_verdict(diag)

    run_dir = init_run_dir(config.output_root, config.run_name)

    extra_arrays: dict[str, np.ndarray] = {
        "singular_values": diag.svd.singular_values,
        "gram_matrix": diag.coherence.gram_matrix,
    }

    if config.save_probe_vectors:
        for pos, plane in position_plane_pairs:
            extra_arrays[f"probe_{int(pos)}mm"] = (
                plane.e_mag if config.measurement_kind == "e_mag"
                else operator.A[:, list(operator.positions_mm).index(pos)]
            )

    save_operator_artifact(
        run_dir,
        A=operator.A,
        positions_mm=operator.positions_mm,
        coords_in_plane_mm=operator.coords_in_plane_mm,
        extra_arrays=extra_arrays,
        metadata={
            "state_id": operator.state_id,
            "measurement_kind": operator.measurement_kind,
            "in_plane_axes": list(operator.in_plane_axes),
            "diagnostics": diag_summary,
            "quality_verdict": verdict,
        },
    )

    manifest = {
        "experiment_type": "build_single_state_operator",
        "config": {
            "state_id": config.state_id,
            "measurement_kind": config.measurement_kind,
            "plane": {
                "axis": config.plane.axis,
                "value_mm": config.plane.value_mm,
                "tol_mm": config.plane.tol_mm,
            },
            "position_to_file": {str(k): str(v) for k, v in config.position_to_file.items()},
            "cst_skiprows": config.cst_skiprows,
            "save_probe_vectors": config.save_probe_vectors,
        },
        "per_position": per_position_metadata,
        "operator_summary": operator.summary(),
        "diagnostics": diag_summary,
        "quality_verdict": verdict,
    }
    save_json(manifest, run_dir / "reports" / "build_operator_report.json")

    return BuildOperatorResult(
        operator=operator,
        diagnostics={
            "summary": diag_summary,
            "verdict": verdict,
            "singular_values": diag.svd.singular_values,
            "gram_matrix": diag.coherence.gram_matrix,
        },
        run_dir=run_dir,
        manifest=manifest,
    )