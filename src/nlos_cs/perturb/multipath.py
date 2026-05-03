"""Multipath / column-leakage perturbations.

Model
-----
Start from a clean measurement y and add contamination from one or more
other operator columns:

    y_perturbed = y + c

where c is formed from other columns of A and scaled to a requested RMS
fraction of the clean signal.

This is useful for modelling:
- multipath leakage
- structured interference from alternative object hypotheses
- encoding-state cross-talk
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from nlos_cs.perturb.awgn import signal_rms

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
SignalArray = FloatArray | ComplexArray
IndexArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class MultipathResult:
    """Result of applying multipath / column-leakage contamination."""

    y_perturbed: SignalArray
    y_clean: SignalArray
    contamination: SignalArray
    chosen_indices: IndexArray
    contamination_fraction_of_rms: float
    rms_signal: float
    rms_contamination: float
    random_seed: int | None


def _validate_operator_and_signal(
    A: SignalArray,
    y: SignalArray,
) -> tuple[np.ndarray, np.ndarray]:
    A_arr = np.asarray(A)
    y_arr = np.asarray(y)

    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A_arr.shape}")
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
    if A_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Row mismatch: A has {A_arr.shape[0]} rows but y has length {y_arr.shape[0]}"
        )

    return A_arr, y_arr


def choose_leakage_indices(
    n_columns: int,
    *,
    exclude_index: int | None = None,
    n_leak: int = 2,
    random_seed: int | None = None,
) -> IndexArray:
    """Choose contamination column indices."""
    if n_columns <= 0:
        raise ValueError(f"n_columns must be positive, got {n_columns}")
    if n_leak <= 0:
        raise ValueError(f"n_leak must be positive, got {n_leak}")

    candidates = np.arange(n_columns, dtype=np.int64)
    if exclude_index is not None:
        if exclude_index < 0 or exclude_index >= n_columns:
            raise ValueError(
                f"exclude_index {exclude_index} out of range for {n_columns} columns"
            )
        candidates = candidates[candidates != exclude_index]

    if candidates.size == 0:
        raise ValueError("No candidate columns remain after exclusion")

    k = min(int(n_leak), int(candidates.size))
    rng = np.random.default_rng(random_seed)
    chosen = rng.choice(candidates, size=k, replace=False)
    return np.asarray(chosen, dtype=np.int64)


def build_leakage_contamination(
    A: SignalArray,
    *,
    chosen_indices: IndexArray,
    weights: FloatArray | None = None,
) -> SignalArray:
    """Build an unscaled contamination vector from selected columns of A."""
    A_arr = np.asarray(A)
    idx = np.asarray(chosen_indices, dtype=np.int64)

    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A_arr.shape}")
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("chosen_indices must be a non-empty 1D array")
    if np.any(idx < 0) or np.any(idx >= A_arr.shape[1]):
        raise ValueError("chosen_indices contains out-of-range column indices")

    if weights is None:
        w = np.ones(idx.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != idx.shape:
            raise ValueError(
                f"weights must have shape {idx.shape}, got {w.shape}"
            )

    contamination = np.zeros(A_arr.shape[0], dtype=A_arr.dtype)
    for col_idx, weight in zip(idx, w):
        contamination = contamination + weight * A_arr[:, int(col_idx)]

    return contamination


def add_multipath_leakage(
    y: SignalArray,
    A: SignalArray,
    *,
    contamination_fraction_of_rms: float,
    exclude_index: int | None = None,
    chosen_indices: IndexArray | None = None,
    n_leak: int = 2,
    random_seed: int | None = None,
    weights: FloatArray | None = None,
) -> MultipathResult:
    """Add structured leakage contamination from other columns of A.

    Parameters
    ----------
    y:
        Clean measurement vector.
    A:
        Operator matrix whose columns define contamination candidates.
    contamination_fraction_of_rms:
        Target contamination RMS as a fraction of clean signal RMS.
    exclude_index:
        Optional column index to exclude from contamination selection. Typically
        this is the true column index.
    chosen_indices:
        Optional explicit contamination indices. If provided, random selection
        is skipped.
    n_leak:
        Number of contamination columns to choose if `chosen_indices` is not provided.
    random_seed:
        RNG seed for random column selection.
    weights:
        Optional weights for the selected contamination columns before RMS scaling.

    Returns
    -------
    MultipathResult
    """
    A_arr, y_arr = _validate_operator_and_signal(A, y)

    frac = float(contamination_fraction_of_rms)
    if frac < 0.0:
        raise ValueError(
            f"contamination_fraction_of_rms must be non-negative, got {frac}"
        )

    rms_signal = signal_rms(y_arr)

    if chosen_indices is None:
        idx = choose_leakage_indices(
            A_arr.shape[1],
            exclude_index=exclude_index,
            n_leak=n_leak,
            random_seed=random_seed,
        )
    else:
        idx = np.asarray(chosen_indices, dtype=np.int64)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("chosen_indices must be a non-empty 1D array")
        if np.any(idx < 0) or np.any(idx >= A_arr.shape[1]):
            raise ValueError("chosen_indices contains out-of-range values")
        if exclude_index is not None and np.any(idx == exclude_index):
            raise ValueError("chosen_indices includes exclude_index")

    raw_contamination = build_leakage_contamination(
        A_arr,
        chosen_indices=idx,
        weights=weights,
    )

    if frac == 0.0 or rms_signal == 0.0:
        contamination = np.zeros_like(y_arr)
    else:
        raw_rms = signal_rms(raw_contamination)
        if raw_rms <= 1e-15:
            contamination = np.zeros_like(y_arr)
        else:
            contamination = raw_contamination * ((frac * rms_signal) / raw_rms)

    y_perturbed = y_arr + contamination
    rms_contamination = signal_rms(contamination)

    return MultipathResult(
        y_perturbed=y_perturbed,
        y_clean=y_arr.copy(),
        contamination=contamination,
        chosen_indices=idx.copy(),
        contamination_fraction_of_rms=frac,
        rms_signal=rms_signal,
        rms_contamination=rms_contamination,
        random_seed=random_seed,
    )


def add_multipath_leakage_rows(
    Y: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    A: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    *,
    contamination_fraction_of_rms: float,
    exclude_indices: IndexArray | None = None,
    n_leak: int = 2,
    random_seed: int | None = None,
) -> tuple[npt.NDArray[np.float64] | npt.NDArray[np.complex128], list[MultipathResult]]:
    """Apply multipath leakage row-by-row to a matrix of measurements."""
    Y_arr = np.asarray(Y)
    A_arr = np.asarray(A)

    if Y_arr.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y_arr.shape}")
    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A_arr.shape}")
    if Y_arr.shape[1] != A_arr.shape[0]:
        raise ValueError(
            f"Row-length mismatch: Y rows have length {Y_arr.shape[1]} but A has {A_arr.shape[0]} rows"
        )

    if exclude_indices is not None:
        exc = np.asarray(exclude_indices, dtype=np.int64)
        if exc.shape != (Y_arr.shape[0],):
            raise ValueError(
                f"exclude_indices must have shape ({Y_arr.shape[0]},), got {exc.shape}"
            )
    else:
        exc = None

    Y_out = np.zeros_like(Y_arr)
    results: list[MultipathResult] = []

    for i in range(Y_arr.shape[0]):
        seed_i = None if random_seed is None else int(random_seed + i)
        result = add_multipath_leakage(
            Y_arr[i],
            A_arr,
            contamination_fraction_of_rms=contamination_fraction_of_rms,
            exclude_index=None if exc is None else int(exc[i]),
            chosen_indices=None,
            n_leak=n_leak,
            random_seed=seed_i,
            weights=None,
        )
        Y_out[i] = result.y_perturbed
        results.append(result)

    return Y_out, results