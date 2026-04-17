"""Diagnostics for sensing operators.

This module contains operator-level diagnostics that answer whether a sensing
matrix is numerically useful before any inverse solver is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from nlos_cs.operators.single_state import SingleStateOperator

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SVDReport:
    """Singular-value decomposition summary for an operator."""

    singular_values: FloatArray
    condition_number: float
    effective_rank: int
    frobenius_norm: float
    spectral_norm: float
    smallest_singular_value: float
    leading_energy_fraction: float

    @property
    def n_singular_values(self) -> int:
        return int(self.singular_values.shape[0])


@dataclass(frozen=True)
class CoherenceReport:
    """Column-correlation / coherence summary for an operator."""

    gram_matrix: FloatArray
    mutual_coherence: float
    mean_off_diagonal_correlation: float
    max_corr_pair: tuple[int, int]


@dataclass(frozen=True)
class OperatorDiagnostics:
    """Combined diagnostic summary for one operator."""

    state_id: str
    measurement_kind: str
    n_measurements: int
    n_positions: int
    svd: SVDReport
    coherence: CoherenceReport


def _as_real_matrix(A: npt.NDArray[np.float64] | npt.NDArray[np.complex128]) -> FloatArray:
    """Convert operator to a real matrix for real-valued diagnostics.

    For complex operators, diagnostics are based on magnitudes by default.
    """
    if np.iscomplexobj(A):
        return np.abs(A).astype(np.float64, copy=False)
    return A.astype(np.float64, copy=False)


def compute_svd_report(
    A: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    *,
    effective_rank_rtol: float = 1e-10,
) -> SVDReport:
    """Compute SVD-based diagnostics for an operator matrix."""
    A_real = _as_real_matrix(A)
    _, s, _ = np.linalg.svd(A_real, full_matrices=False)

    if s.size == 0:
        raise ValueError("Cannot compute SVD diagnostics for an empty matrix")

    s_max = float(s[0])
    s_min = float(s[-1])

    if s_min == 0.0:
        kappa = float("inf")
    else:
        kappa = s_max / s_min

    eff_rank = int(np.sum(s > s_max * effective_rank_rtol))
    frob = float(np.linalg.norm(A_real, ord="fro"))
    spec = s_max
    energy = s**2
    lead_frac = float(energy[0] / np.sum(energy)) if np.sum(energy) > 0 else 0.0

    return SVDReport(
        singular_values=s.astype(np.float64, copy=False),
        condition_number=float(kappa),
        effective_rank=eff_rank,
        frobenius_norm=frob,
        spectral_norm=spec,
        smallest_singular_value=s_min,
        leading_energy_fraction=lead_frac,
    )


def compute_column_gram_matrix(
    A: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> FloatArray:
    """Compute the column-normalised Gram matrix.

    G[i, j] = <a_i, a_j> / (||a_i|| ||a_j||)

    Returned as a real matrix.
    """
    A_real = _as_real_matrix(A)
    norms = np.linalg.norm(A_real, axis=0)

    if np.any(norms == 0.0):
        raise ValueError("Cannot compute Gram matrix with zero-norm columns")

    A_norm = A_real / norms[None, :]
    G = A_norm.T @ A_norm
    return G.astype(np.float64, copy=False)


def compute_coherence_report(
    A: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> CoherenceReport:
    """Compute coherence-style diagnostics from the column Gram matrix."""
    G = compute_column_gram_matrix(A)
    n = G.shape[0]

    if n < 2:
        return CoherenceReport(
            gram_matrix=G,
            mutual_coherence=0.0,
            mean_off_diagonal_correlation=0.0,
            max_corr_pair=(0, 0),
        )

    G_off = np.abs(G.copy())
    np.fill_diagonal(G_off, 0.0)

    max_idx = np.unravel_index(np.argmax(G_off), G_off.shape)
    mutual_coherence = float(G_off[max_idx])

    mask = ~np.eye(n, dtype=bool)
    mean_off = float(np.mean(G_off[mask])) if np.any(mask) else 0.0

    return CoherenceReport(
        gram_matrix=G,
        mutual_coherence=mutual_coherence,
        mean_off_diagonal_correlation=mean_off,
        max_corr_pair=(int(max_idx[0]), int(max_idx[1])),
    )


def analyse_single_state_operator(
    operator: SingleStateOperator,
    *,
    effective_rank_rtol: float = 1e-10,
) -> OperatorDiagnostics:
    """Compute the full diagnostic bundle for one single-state operator."""
    svd = compute_svd_report(operator.A, effective_rank_rtol=effective_rank_rtol)
    coh = compute_coherence_report(operator.A)

    return OperatorDiagnostics(
        state_id=operator.state_id,
        measurement_kind=operator.measurement_kind,
        n_measurements=operator.n_measurements,
        n_positions=operator.n_positions,
        svd=svd,
        coherence=coh,
    )


def compare_condition_numbers(
    *operators: SingleStateOperator,
    effective_rank_rtol: float = 1e-10,
) -> dict[str, float]:
    """Return condition numbers keyed by state_id."""
    if len(operators) == 0:
        raise ValueError("At least one operator must be provided")

    out: dict[str, float] = {}
    for op in operators:
        svd = compute_svd_report(op.A, effective_rank_rtol=effective_rank_rtol)
        out[op.state_id] = svd.condition_number
    return out


def compare_mutual_coherence(*operators: SingleStateOperator) -> dict[str, float]:
    """Return mutual coherence keyed by state_id."""
    if len(operators) == 0:
        raise ValueError("At least one operator must be provided")

    out: dict[str, float] = {}
    for op in operators:
        coh = compute_coherence_report(op.A)
        out[op.state_id] = coh.mutual_coherence
    return out


def compare_smallest_singular_values(
    *operators: SingleStateOperator,
    effective_rank_rtol: float = 1e-10,
) -> dict[str, float]:
    """Return smallest singular value keyed by state_id."""
    if len(operators) == 0:
        raise ValueError("At least one operator must be provided")

    out: dict[str, float] = {}
    for op in operators:
        svd = compute_svd_report(op.A, effective_rank_rtol=effective_rank_rtol)
        out[op.state_id] = svd.smallest_singular_value
    return out


def summarise_operator_quality(diag: OperatorDiagnostics) -> dict[str, float | int | str]:
    """Return a compact dictionary summary suitable for logs or manifests."""
    return {
        "state_id": diag.state_id,
        "measurement_kind": diag.measurement_kind,
        "n_measurements": diag.n_measurements,
        "n_positions": diag.n_positions,
        "condition_number": diag.svd.condition_number,
        "effective_rank": diag.svd.effective_rank,
        "frobenius_norm": diag.svd.frobenius_norm,
        "spectral_norm": diag.svd.spectral_norm,
        "smallest_singular_value": diag.svd.smallest_singular_value,
        "leading_energy_fraction": diag.svd.leading_energy_fraction,
        "mutual_coherence": diag.coherence.mutual_coherence,
        "mean_off_diagonal_correlation": diag.coherence.mean_off_diagonal_correlation,
        "max_corr_i": diag.coherence.max_corr_pair[0],
        "max_corr_j": diag.coherence.max_corr_pair[1],
    }


def quality_verdict(
    diag: OperatorDiagnostics,
    *,
    kappa_good: float = 100.0,
    kappa_moderate: float = 1000.0,
    coherence_good: float = 0.5,
    coherence_moderate: float = 0.9,
) -> str:
    """Return a coarse textual verdict for early-stage screening."""
    kappa = diag.svd.condition_number
    mu = diag.coherence.mutual_coherence

    if kappa < kappa_good and mu < coherence_good:
        return "good"
    if kappa < kappa_moderate and mu < coherence_moderate:
        return "moderate"
    return "poor"