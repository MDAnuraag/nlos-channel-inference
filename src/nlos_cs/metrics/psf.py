"""Point-spread-function style metrics for reconstructed position responses.

Given a 1D reconstruction x_hat over discrete positions, this module measures:

- peak value and peak index
- strongest sidelobe and sidelobe level
- peak margin
- half-maximum width on the discrete position grid

These metrics are useful when x_hat is interpreted as a position response
rather than a full image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
SignalArray = FloatArray | ComplexArray


@dataclass(frozen=True)
class PSFMetrics:
    """Discrete PSF-style metrics for one reconstruction vector."""

    peak_index: int
    peak_position_mm: float | None
    peak_value: float

    sidelobe_index: int | None
    sidelobe_position_mm: float | None
    sidelobe_value: float

    sidelobe_db: float
    peak_margin: float

    half_max_value: float
    half_max_width_samples: int
    half_max_width_mm: float | None

    x_abs: FloatArray
    x_norm: FloatArray


def _validate_inputs(
    x_hat: SignalArray,
    positions_mm: FloatArray | None,
) -> tuple[FloatArray, FloatArray | None]:
    x = np.asarray(x_hat)
    if x.ndim != 1:
        raise ValueError(f"x_hat must be 1D, got shape {x.shape}")

    x_abs = np.abs(x).astype(np.float64, copy=False)

    pos = None
    if positions_mm is not None:
        pos = np.asarray(positions_mm, dtype=np.float64)
        if pos.ndim != 1:
            raise ValueError(f"positions_mm must be 1D, got shape {pos.shape}")
        if pos.shape[0] != x_abs.shape[0]:
            raise ValueError(
                f"positions_mm length {pos.shape[0]} does not match x_hat length {x_abs.shape[0]}"
            )

    return x_abs, pos


def compute_psf_metrics(
    x_hat: SignalArray,
    *,
    positions_mm: FloatArray | None = None,
    true_index: int | None = None,
) -> PSFMetrics:
    """Compute PSF-style metrics for a reconstruction vector.

    Parameters
    ----------
    x_hat:
        Reconstruction vector over discrete positions.
    positions_mm:
        Optional physical position grid.
    true_index:
        Optional reference index to force the peak location used for metrics.
        If omitted, the dominant index of |x_hat| is used.

    Returns
    -------
    PSFMetrics
    """
    x_abs, pos = _validate_inputs(x_hat, positions_mm)
    n = x_abs.shape[0]

    if n == 0:
        raise ValueError("x_hat must not be empty")

    peak_index = int(np.argmax(x_abs)) if true_index is None else int(true_index)
    if peak_index < 0 or peak_index >= n:
        raise ValueError(f"true_index out of range: {peak_index}")

    peak_value = float(x_abs[peak_index])
    peak_position_mm = None if pos is None else float(pos[peak_index])

    if peak_value <= 1e-15:
        x_norm = x_abs.copy()
        return PSFMetrics(
            peak_index=peak_index,
            peak_position_mm=peak_position_mm,
            peak_value=peak_value,
            sidelobe_index=None,
            sidelobe_position_mm=None,
            sidelobe_value=0.0,
            sidelobe_db=float("-inf"),
            peak_margin=0.0,
            half_max_value=0.0,
            half_max_width_samples=0,
            half_max_width_mm=0.0 if pos is not None else None,
            x_abs=x_abs,
            x_norm=x_norm,
        )

    sidelobe_mask = np.ones(n, dtype=bool)
    sidelobe_mask[peak_index] = False

    if np.any(sidelobe_mask):
        sidelobe_candidates = x_abs.copy()
        sidelobe_candidates[peak_index] = -np.inf
        sidelobe_index = int(np.argmax(sidelobe_candidates))
        sidelobe_value = float(x_abs[sidelobe_index])
        sidelobe_position_mm = None if pos is None else float(pos[sidelobe_index])
    else:
        sidelobe_index = None
        sidelobe_value = 0.0
        sidelobe_position_mm = None

    peak_margin = float(peak_value - sidelobe_value)
    sidelobe_db = (
        float(20.0 * np.log10(sidelobe_value / peak_value))
        if sidelobe_value > 1e-15
        else float("-inf")
    )

    half_max_value = 0.5 * peak_value
    above_half = np.where(x_abs >= half_max_value)[0]

    if above_half.size == 0:
        half_max_width_samples = 0
        half_max_width_mm = 0.0 if pos is not None else None
    else:
        half_max_width_samples = int(above_half[-1] - above_half[0])
        if pos is None:
            half_max_width_mm = None
        else:
            half_max_width_mm = float(pos[above_half[-1]] - pos[above_half[0]])

    x_norm = (x_abs / peak_value).astype(np.float64, copy=False)

    return PSFMetrics(
        peak_index=peak_index,
        peak_position_mm=peak_position_mm,
        peak_value=peak_value,
        sidelobe_index=sidelobe_index,
        sidelobe_position_mm=sidelobe_position_mm,
        sidelobe_value=sidelobe_value,
        sidelobe_db=sidelobe_db,
        peak_margin=peak_margin,
        half_max_value=float(half_max_value),
        half_max_width_samples=half_max_width_samples,
        half_max_width_mm=half_max_width_mm,
        x_abs=x_abs,
        x_norm=x_norm,
    )


def batch_compute_psf_metrics(
    x_hats: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    *,
    positions_mm: FloatArray | None = None,
    true_indices: npt.NDArray[np.int64] | None = None,
) -> list[PSFMetrics]:
    """Compute PSF metrics row-by-row for a stack of reconstructions.

    Parameters
    ----------
    x_hats:
        Array of shape (K, N), where each row is one reconstruction.
    positions_mm:
        Optional physical position grid of length N.
    true_indices:
        Optional array of shape (K,) specifying the reference index for each row.
        If omitted, each row uses its own argmax.
    """
    X = np.asarray(x_hats)
    if X.ndim != 2:
        raise ValueError(f"x_hats must be 2D, got shape {X.shape}")

    k, n = X.shape

    if positions_mm is not None:
        pos = np.asarray(positions_mm, dtype=np.float64)
        if pos.shape != (n,):
            raise ValueError(f"positions_mm must have shape ({n},), got {pos.shape}")
    else:
        pos = None

    if true_indices is not None:
        idxs = np.asarray(true_indices, dtype=np.int64)
        if idxs.shape != (k,):
            raise ValueError(f"true_indices must have shape ({k},), got {idxs.shape}")
    else:
        idxs = None

    out: list[PSFMetrics] = []
    for i in range(k):
        out.append(
            compute_psf_metrics(
                X[i],
                positions_mm=pos,
                true_index=None if idxs is None else int(idxs[i]),
            )
        )
    return out


def summarise_psf_metrics(metrics: PSFMetrics) -> dict[str, float | int | None]:
    """Return compact numeric summary for logs or manifests."""
    return {
        "peak_index": metrics.peak_index,
        "peak_position_mm": metrics.peak_position_mm,
        "peak_value": metrics.peak_value,
        "sidelobe_index": metrics.sidelobe_index,
        "sidelobe_position_mm": metrics.sidelobe_position_mm,
        "sidelobe_value": metrics.sidelobe_value,
        "sidelobe_db": metrics.sidelobe_db,
        "peak_margin": metrics.peak_margin,
        "half_max_value": metrics.half_max_value,
        "half_max_width_samples": metrics.half_max_width_samples,
        "half_max_width_mm": metrics.half_max_width_mm,
    }


def mean_peak_margin(metrics_list: list[PSFMetrics]) -> float:
    """Mean peak margin across a batch of PSF metrics."""
    if len(metrics_list) == 0:
        raise ValueError("metrics_list must not be empty")
    return float(np.mean([m.peak_margin for m in metrics_list]))


def mean_sidelobe_db(metrics_list: list[PSFMetrics]) -> float:
    """Mean sidelobe level in dB across finite entries only."""
    if len(metrics_list) == 0:
        raise ValueError("metrics_list must not be empty")
    vals = np.array([m.sidelobe_db for m in metrics_list], dtype=np.float64)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return float("-inf")
    return float(np.mean(vals[finite]))