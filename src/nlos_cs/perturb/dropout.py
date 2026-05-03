"""Measurement dropout perturbations.

Model
-----
A fraction of measurement channels is dropped, typically by setting them to zero.

This is useful for modelling:
- blocked probe rays
- dead sensors / missing samples
- partial occlusion
- communication / acquisition loss
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
BoolArray = npt.NDArray[np.bool_]
SignalArray = FloatArray | ComplexArray


@dataclass(frozen=True)
class DropoutResult:
    """Result of applying measurement dropout."""

    y_dropped: SignalArray
    y_clean: SignalArray
    mask: BoolArray
    dropout_fraction: float
    kept_fraction: float
    random_seed: int | None


def make_dropout_mask(
    n: int,
    *,
    dropout_fraction: float,
    random_seed: int | None = None,
) -> BoolArray:
    """Create a boolean keep-mask for measurement dropout.

    The mask drops an exact number of channels:
        n_drop = round(dropout_fraction * n)

    so the realised dropout fraction is deterministic given n and the seed.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    frac = float(dropout_fraction)
    if frac < 0.0 or frac > 1.0:
        raise ValueError(f"dropout_fraction must be in [0, 1], got {frac}")

    if frac == 0.0:
        return np.ones(n, dtype=bool)
    if frac == 1.0:
        return np.zeros(n, dtype=bool)

    n_drop = int(round(frac * n))
    n_drop = max(0, min(n_drop, n))

    if n_drop == 0:
        return np.ones(n, dtype=bool)
    if n_drop == n:
        return np.zeros(n, dtype=bool)

    rng = np.random.default_rng(random_seed)
    drop_idx = rng.choice(n, size=n_drop, replace=False)

    mask = np.ones(n, dtype=bool)
    mask[drop_idx] = False
    return mask


def apply_dropout_mask(
    y: SignalArray,
    mask: BoolArray,
) -> DropoutResult:
    """Apply a boolean keep-mask to a signal.

    Parameters
    ----------
    y:
        Input signal of shape (M,).
    mask:
        Boolean keep-mask of shape (M,), where True means keep.

    Returns
    -------
    DropoutResult
    """
    y_arr = np.asarray(y)
    mask_arr = np.asarray(mask, dtype=bool)

    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
    if mask_arr.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape {mask_arr.shape}")
    if y_arr.shape[0] != mask_arr.shape[0]:
        raise ValueError(
            f"Length mismatch: y has length {y_arr.shape[0]} but mask has length {mask_arr.shape[0]}"
        )

    y_dropped = y_arr.copy()
    y_dropped[~mask_arr] = 0

    dropout_fraction = float(np.mean(~mask_arr))
    kept_fraction = float(np.mean(mask_arr))

    return DropoutResult(
        y_dropped=y_dropped,
        y_clean=y_arr.copy(),
        mask=mask_arr.copy(),
        dropout_fraction=dropout_fraction,
        kept_fraction=kept_fraction,
        random_seed=None,
    )


def add_dropout(
    y: SignalArray,
    *,
    dropout_fraction: float,
    random_seed: int | None = None,
) -> DropoutResult:
    """Apply random measurement dropout to a signal."""
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}")

    mask = make_dropout_mask(
        y_arr.shape[0],
        dropout_fraction=dropout_fraction,
        random_seed=random_seed,
    )
    result = apply_dropout_mask(y_arr, mask)
    return DropoutResult(
        y_dropped=result.y_dropped,
        y_clean=result.y_clean,
        mask=result.mask,
        dropout_fraction=result.dropout_fraction,
        kept_fraction=result.kept_fraction,
        random_seed=random_seed,
    )


def apply_dropout_mask_rows(
    Y: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    mask: BoolArray,
) -> tuple[npt.NDArray[np.float64] | npt.NDArray[np.complex128], list[DropoutResult]]:
    """Apply the same dropout mask to every row of a signal matrix."""
    Y_arr = np.asarray(Y)
    mask_arr = np.asarray(mask, dtype=bool)

    if Y_arr.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y_arr.shape}")
    if mask_arr.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape {mask_arr.shape}")
    if Y_arr.shape[1] != mask_arr.shape[0]:
        raise ValueError(
            f"Column mismatch: Y has {Y_arr.shape[1]} columns but mask has length {mask_arr.shape[0]}"
        )

    Y_out = np.zeros_like(Y_arr)
    results: list[DropoutResult] = []

    for i in range(Y_arr.shape[0]):
        res = apply_dropout_mask(Y_arr[i], mask_arr)
        Y_out[i] = res.y_dropped
        results.append(res)

    return Y_out, results


def add_dropout_rows(
    Y: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    *,
    dropout_fraction: float,
    random_seed: int | None = None,
    shared_mask: bool = False,
) -> tuple[npt.NDArray[np.float64] | npt.NDArray[np.complex128], list[DropoutResult]]:
    """Apply random dropout row-by-row to a matrix of signals.

    Parameters
    ----------
    Y:
        Array of shape (K, M), where each row is one signal.
    dropout_fraction:
        Fraction of channels to drop.
    random_seed:
        Base seed.
    shared_mask:
        If True, one mask is generated and applied to all rows.
        If False, each row gets its own mask.
    """
    Y_arr = np.asarray(Y)
    if Y_arr.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y_arr.shape}")

    Y_out = np.zeros_like(Y_arr)
    results: list[DropoutResult] = []

    if shared_mask:
        mask = make_dropout_mask(
            Y_arr.shape[1],
            dropout_fraction=dropout_fraction,
            random_seed=random_seed,
        )
        for i in range(Y_arr.shape[0]):
            res = apply_dropout_mask(Y_arr[i], mask)
            Y_out[i] = res.y_dropped
            results.append(
                DropoutResult(
                    y_dropped=res.y_dropped,
                    y_clean=res.y_clean,
                    mask=res.mask,
                    dropout_fraction=res.dropout_fraction,
                    kept_fraction=res.kept_fraction,
                    random_seed=random_seed,
                )
            )
        return Y_out, results

    for i in range(Y_arr.shape[0]):
        seed_i = None if random_seed is None else int(random_seed + i)
        res = add_dropout(
            Y_arr[i],
            dropout_fraction=dropout_fraction,
            random_seed=seed_i,
        )
        Y_out[i] = res.y_dropped
        results.append(res)

    return Y_out, results