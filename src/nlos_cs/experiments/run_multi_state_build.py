"""Experiment runner for building and saving a multi-state sensing operator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nlos_cs.io.artifacts import init_run_dir, save_operator_artifact, save_json
from nlos_cs.operators.diagnostics import (
    compute_coherence_report,
    compute_svd_report,
)
from nlos_cs.operators.multi_state import (
    MultiStateOperator,
    build_multi_state_operator,
    per_state_row_counts,
    validate_multi_state_operator,
)


@dataclass(frozen=True)
class MultiStateBuildConfig:
    """Configuration for one multi-state operator build."""

    run_name: str
    output_root: str = "outputs"
    save_component_blocks: bool = True


@dataclass(frozen=True)
class MultiStateBuildResult:
    """Outputs from one multi-state build experiment."""

    operator: MultiStateOperator
    diagnostics: dict[str, Any]
    run_dir: Path
    report: dict[str, Any]


def _operator_summary(operator: MultiStateOperator) -> dict[str, Any]:
    """Compact summary for a multi-state operator."""
    return {
        "n_states": operator.n_states,
        "state_ids": list(operator.state_ids),
        "measurement_kind": operator.measurement_kind,
        "n_measurements": operator.n_measurements,
        "n_positions": operator.n_positions,
        "is_complex": bool(operator.is_complex),
        "per_state_row_counts": per_state_row_counts(operator),
    }


def run_multi_state_build_experiment(
    *,
    operators: list[Any],
    config: MultiStateBuildConfig,
) -> MultiStateBuildResult:
    """Build, validate, diagnose, and save a multi-state operator.

    Parameters
    ----------
    operators:
        List of compatible SingleStateOperator objects.
    config:
        Output and saving configuration.

    Returns
    -------
    MultiStateBuildResult
    """
    if len(operators) == 0:
        raise ValueError("operators must not be empty")

    multi = build_multi_state_operator(*operators)
    validate_multi_state_operator(multi)

    svd = compute_svd_report(multi.A)
    coh = compute_coherence_report(multi.A)

    diag_summary = {
        "condition_number": svd.condition_number,
        "effective_rank": svd.effective_rank,
        "frobenius_norm": svd.frobenius_norm,
        "spectral_norm": svd.spectral_norm,
        "smallest_singular_value": svd.smallest_singular_value,
        "leading_energy_fraction": svd.leading_energy_fraction,
        "mutual_coherence": coh.mutual_coherence,
        "mean_off_diagonal_correlation": coh.mean_off_diagonal_correlation,
        "max_corr_pair": list(coh.max_corr_pair),
    }

    run_dir = init_run_dir(config.output_root, config.run_name)

    extra_arrays: dict[str, np.ndarray] = {
        "singular_values": svd.singular_values,
        "gram_matrix": coh.gram_matrix,
    }

    if config.save_component_blocks:
        for state_id in multi.state_ids:
            extra_arrays[f"A_block_{state_id}"] = multi.block(state_id)
            extra_arrays[f"coords_{state_id}"] = multi.coords_by_state[state_id]

    save_operator_artifact(
        run_dir,
        A=multi.A,
        positions_mm=multi.positions_mm,
        coords_in_plane_mm=None,
        extra_arrays=extra_arrays,
        metadata={
            "state_ids": list(multi.state_ids),
            "measurement_kind": multi.measurement_kind,
            "per_state_row_counts": per_state_row_counts(multi),
            "diagnostics": diag_summary,
        },
    )

    report = {
        "experiment_type": "build_multi_state_operator",
        "config": {
            "save_component_blocks": config.save_component_blocks,
        },
        "operator_summary": _operator_summary(multi),
        "diagnostics": diag_summary,
        "states": [
            {
                "state_id": state_id,
                "row_slice": [multi.row_slices[state_id].start, multi.row_slices[state_id].stop],
                "in_plane_axes": list(multi.in_plane_axes_by_state[state_id]),
                "n_rows": int(multi.block(state_id).shape[0]),
            }
            for state_id in multi.state_ids
        ],
    }
    save_json(report, run_dir / "reports" / "multi_state_build_report.json")

    return MultiStateBuildResult(
        operator=multi,
        diagnostics={
            "svd": svd,
            "coherence": coh,
            "summary": diag_summary,
        },
        run_dir=run_dir,
        report=report,
    )