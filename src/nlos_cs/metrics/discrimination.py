"""Discrimination and leakage metrics for reconstructed position responses.

Core definitions
----------------
Given a reconstruction x_hat produced when the true position is i:

    leakage[i, j] = |x_hat[j]| / |x_hat[i]|
    discrim[i, j] = 1 - leakage[i, j]

Interpretation
--------------
- discrim[i, j] close to 1:
    position j is strongly suppressed when i is true
- discrim[i, j] close to 0:
    position j is almost as plausible as i
- leakage[i, j] large:
    strong ambiguity from i -> j
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt

from nlos_cs.inverse.base import InverseProblem, InverseSolver

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = FloatArray | ComplexArray


@dataclass(frozen=True)
class DiscriminationResult:
    """Discrimination analysis for a set of reconstructions.

    Attributes
    ----------
    discrimination:
        Matrix D of shape (N, N), where D[i, j] = 1 - leakage[i, j].
    leakage:
        Matrix L of shape (N, N), where L[i, j] = |x_hat[j]| / |x_hat[i]|.
    x_hats:
        Reconstruction matrix of shape (N, N). Row i is the reconstruction
        produced when the true position is i.
    reference_values:
        Reference denominator per row, usually |x_hat[i]| for the true position i.
    peak_indices:
        Dominant reconstruction index for each row.
    positions_mm:
        Optional physical position grid.
    """

    discrimination: FloatArray
    leakage: FloatArray
    x_hats: OperatorArray
    reference_values: FloatArray
    peak_indices: npt.NDArray[np.int64]
    positions_mm: FloatArray | None = None

    @property
    def n_positions(self) -> int:
        return int(self.discrimination.shape[0])

    def worst_pair(self) -> tuple[int, int, float]:
        """Return (i, j, leakage[i, j]) for the hardest-to-distinguish pair."""
        leakage = self.leakage.copy()
        np.fill_diagonal(leakage, -np.inf)
        idx = np.unravel_index(np.argmax(leakage), leakage.shape)
        return int(idx[0]), int(idx[1]), float(leakage[idx])

    def mean_off_diagonal_leakage(self) -> float:
        """Mean leakage excluding the diagonal."""
        mask = ~np.eye(self.n_positions, dtype=bool)
        return float(np.mean(self.leakage[mask]))

    def mean_off_diagonal_discrimination(self) -> float:
        """Mean discrimination excluding the diagonal."""
        mask = ~np.eye(self.n_positions, dtype=bool)
        return float(np.mean(self.discrimination[mask]))


def _validate_square_reconstruction_matrix(x_hats: OperatorArray) -> None:
    """Validate that x_hats is square with one row per true position."""
    if x_hats.ndim != 2:
        raise ValueError(f"x_hats must be 2D, got shape {x_hats.shape}")
    if x_hats.shape[0] != x_hats.shape[1]:
        raise ValueError(
            f"x_hats must be square for position-to-position discrimination, got {x_hats.shape}"
        )


def compute_discrimination_from_xhats(
    x_hats: OperatorArray,
    *,
    use_abs: bool = True,
    true_indices: npt.NDArray[np.int64] | None = None,
    positions_mm: FloatArray | None = None,
) -> DiscriminationResult:
    """Compute leakage and discrimination matrices from a reconstruction matrix.

    Parameters
    ----------
    x_hats:
        Array of shape (N, N). Row i is the reconstruction produced when the
        true position is i.
    use_abs:
        If True, use absolute amplitudes before forming ratios.
    true_indices:
        Optional array of length N giving the reference index in each row.
        Defaults to [0, 1, ..., N-1].
    positions_mm:
        Optional physical position grid of length N.

    Returns
    -------
    DiscriminationResult
    """
    _validate_square_reconstruction_matrix(x_hats)

    n = x_hats.shape[0]
    if true_indices is None:
        true_indices_arr = np.arange(n, dtype=np.int64)
    else:
        true_indices_arr = np.asarray(true_indices, dtype=np.int64)
        if true_indices_arr.shape != (n,):
            raise ValueError(
                f"true_indices must have shape ({n},), got {true_indices_arr.shape}"
            )
        if np.any(true_indices_arr < 0) or np.any(true_indices_arr >= n):
            raise ValueError("true_indices contains out-of-range values")

    if positions_mm is not None:
        positions_mm = np.asarray(positions_mm, dtype=np.float64)
        if positions_mm.shape != (n,):
            raise ValueError(
                f"positions_mm must have shape ({n},), got {positions_mm.shape}"
            )

    x_eval = np.abs(x_hats) if use_abs else np.asarray(x_hats)
    x_eval = x_eval.astype(np.float64, copy=False)

    ref = x_eval[np.arange(n), true_indices_arr]
    if np.any(ref <= 1e-15):
        bad = np.where(ref <= 1e-15)[0].tolist()
        raise ValueError(
            f"Reference value is zero or near-zero for row(s) {bad}; cannot form leakage ratios"
        )

    leakage = x_eval / ref[:, None]
    discrimination = 1.0 - leakage

    # By definition, the true index in each row should have zero leakage and unit discrimination
    leakage[np.arange(n), true_indices_arr] = 0.0
    discrimination[np.arange(n), true_indices_arr] = 1.0

    peak_indices = np.argmax(x_eval, axis=1).astype(np.int64)

    return DiscriminationResult(
        discrimination=discrimination.astype(np.float64, copy=False),
        leakage=leakage.astype(np.float64, copy=False),
        x_hats=x_hats,
        reference_values=ref.astype(np.float64, copy=False),
        peak_indices=peak_indices,
        positions_mm=positions_mm,
    )


def compute_discrimination_from_operator(
    A: OperatorArray,
    solver: InverseSolver,
    *,
    positions_mm: FloatArray | None = None,
    metadata: dict | None = None,
) -> DiscriminationResult:
    """Compute discrimination by reconstructing each operator column in turn.

    For each column i:
        y_i = A[:, i]
        x_hat_i = solver.solve(InverseProblem(A=A, y=y_i)).x_hat

    The resulting row stack is analysed with the default true index mapping.
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    n = A.shape[1]
    x_dtype = np.complex128 if np.iscomplexobj(A) else np.float64
    x_hats = np.zeros((n, n), dtype=x_dtype)

    for i in range(n):
        problem = InverseProblem(
            A=A,
            y=A[:, i],
            positions_mm=positions_mm,
            metadata={} if metadata is None else dict(metadata),
        )
        result = solver.solve(problem)
        if result.x_hat.shape != (n,):
            raise ValueError(
                f"Solver returned x_hat with shape {result.x_hat.shape}, expected ({n},)"
            )
        x_hats[i] = result.x_hat

    return compute_discrimination_from_xhats(
        x_hats=x_hats,
        use_abs=True,
        true_indices=np.arange(n, dtype=np.int64),
        positions_mm=positions_mm,
    )


def compute_discrimination_from_measurements(
    A: OperatorArray,
    Y: OperatorArray,
    solver: InverseSolver,
    *,
    true_indices: npt.NDArray[np.int64] | None = None,
    positions_mm: FloatArray | None = None,
    metadata_rows: Iterable[dict] | None = None,
) -> DiscriminationResult:
    """Compute discrimination from an arbitrary set of measurement vectors.

    Parameters
    ----------
    A:
        Operator matrix of shape (M, N).
    Y:
        Measurement matrix of shape (K, M), where each row is one measurement vector.
    solver:
        Inverse solver used to reconstruct each row of Y.
    true_indices:
        Reference column index per row, shape (K,). If omitted, requires K == N and
        defaults to [0, 1, ..., N-1].
    positions_mm:
        Optional physical position grid of length N.
    metadata_rows:
        Optional iterable of metadata dicts, one per row of Y.

    Notes
    -----
    This function is more general than the square position-to-position case.
    It is useful for mismatched or perturbed measurements. If K != N, this does
    not return a square DiscriminationResult and therefore should not be used here.
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y.shape}")
    if Y.shape[1] != A.shape[0]:
        raise ValueError(
            f"Measurement row length mismatch: Y has {Y.shape[1]} columns but A has {A.shape[0]} rows"
        )
    if Y.shape[0] != A.shape[1]:
        raise ValueError(
            "This helper is restricted to square position-to-position analysis; "
            f"got Y with {Y.shape[0]} rows and A with {A.shape[1]} columns"
        )

    n = A.shape[1]
    x_dtype = np.complex128 if np.iscomplexobj(A) or np.iscomplexobj(Y) else np.float64
    x_hats = np.zeros((n, n), dtype=x_dtype)

    row_meta_list = (
        [None] * n if metadata_rows is None else list(metadata_rows)
    )
    if len(row_meta_list) != n:
        raise ValueError(f"metadata_rows must have length {n}, got {len(row_meta_list)}")

    for i in range(n):
        meta = {} if row_meta_list[i] is None else dict(row_meta_list[i])
        problem = InverseProblem(A=A, y=Y[i], positions_mm=positions_mm, metadata=meta)
        result = solver.solve(problem)
        if result.x_hat.shape != (n,):
            raise ValueError(
                f"Solver returned x_hat with shape {result.x_hat.shape}, expected ({n},)"
            )
        x_hats[i] = result.x_hat

    return compute_discrimination_from_xhats(
        x_hats=x_hats,
        use_abs=True,
        true_indices=true_indices,
        positions_mm=positions_mm,
    )


def hardest_pairs(
    result: DiscriminationResult,
    *,
    top_k: int = 10,
) -> list[tuple[int, int, float]]:
    """Return the top-k hardest pairs ranked by leakage descending."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    leakage = result.leakage.copy()
    np.fill_diagonal(leakage, -np.inf)

    flat_indices = np.argsort(leakage.ravel())[::-1]
    pairs: list[tuple[int, int, float]] = []

    for flat_idx in flat_indices:
        i, j = np.unravel_index(flat_idx, leakage.shape)
        value = float(leakage[i, j])
        if not np.isfinite(value):
            continue
        pairs.append((int(i), int(j), value))
        if len(pairs) >= top_k:
            break

    return pairs


def group_leakage_summary(
    result: DiscriminationResult,
    groups: dict[str, set[int]],
) -> dict[str, float]:
    """Compute mean leakage for within-group and cross-group pairings.

    Parameters
    ----------
    result:
        Discrimination analysis result.
    groups:
        Mapping such as {"A": {0, 2, 5}, "B": {1, 3, 4}} using column indices.

    Returns
    -------
    dict[str, float]
        Keys look like:
        - "A->A"
        - "A->B"
        - "B->A"
        - "B->B"
    """
    n = result.n_positions
    all_indices = set(range(n))
    covered = set().union(*groups.values()) if groups else set()

    if covered != all_indices:
        missing = sorted(all_indices - covered)
        extra = sorted(covered - all_indices)
        raise ValueError(
            f"Group coverage must match 0..{n-1}. Missing={missing}, Extra={extra}"
        )

    out: dict[str, float] = {}
    for g1, idxs1 in groups.items():
        for g2, idxs2 in groups.items():
            vals = []
            for i in idxs1:
                for j in idxs2:
                    if i == j:
                        continue
                    vals.append(result.leakage[i, j])
            key = f"{g1}->{g2}"
            out[key] = float(np.mean(vals)) if vals else float("nan")

    return out