"""Spatially correlated additive noise perturbations.

Model
-----
Generate white Gaussian noise, smooth it along the measurement axis,
renormalise it to the requested RMS, then add it to the clean signal.

This is useful for modelling structured measurement corruption where nearby
channels are not independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d

from nlos_cs.perturb.awgn import signal_rms

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
SignalArray = FloatArray | ComplexArray


@dataclass(frozen=True)
class CorrelatedNoiseResult:
    """Result of applying correlated additive noise."""

    y_noisy: SignalArray
    y_clean: SignalArray
    noise: SignalArray
    rms_signal: float
    rms_noise: float
    noise_fraction_of_rms: float
    corr_length: int
    random_seed: int | None


def _smooth_real_noise(
    raw: FloatArray,
    *,
    corr_length: int,
) -> FloatArray:
    """Smooth 1D real noise using a uniform filter."""
    if corr_length <= 0:
        raise ValueError(f"corr_length must be positive, got {corr_length}")
    return uniform_filter1d(raw, size=corr_length, mode="wrap").astype(np.float64, copy=False)


def add_correlated_noise(
    y: SignalArray,
    *,
    noise_fraction_of_rms: float,
    corr_length: int,
    random_seed: int | None = None,
) -> CorrelatedNoiseResult:
    """Add spatially correlated noise to a signal.

    Parameters
    ----------
    y:
        Clean input signal, shape (M,).
    noise_fraction_of_rms:
        Target noise RMS as a fraction of the clean signal RMS.
    corr_length:
        Smoothing width along the measurement axis. Larger values produce
        more slowly varying noise.
    random_seed:
        Optional RNG seed.

    Returns
    -------
    CorrelatedNoiseResult
    """
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}")

    alpha = float(noise_fraction_of_rms)
    if alpha < 0.0:
        raise ValueError(f"noise_fraction_of_rms must be non-negative, got {alpha}")
    if corr_length <= 0:
        raise ValueError(f"corr_length must be positive, got {corr_length}")

    y_clean = y_arr.copy()
    rms_signal = signal_rms(y_clean)

    if alpha == 0.0 or rms_signal == 0.0:
        noise = np.zeros_like(y_clean)
        return CorrelatedNoiseResult(
            y_noisy=y_clean.copy(),
            y_clean=y_clean,
            noise=noise,
            rms_signal=rms_signal,
            rms_noise=0.0,
            noise_fraction_of_rms=alpha,
            corr_length=corr_length,
            random_seed=random_seed,
        )

    rng = np.random.default_rng(random_seed)
    target_rms = alpha * rms_signal

    if np.iscomplexobj(y_clean):
        raw_real = rng.standard_normal(size=y_clean.shape).astype(np.float64, copy=False)
        raw_imag = rng.standard_normal(size=y_clean.shape).astype(np.float64, copy=False)

        smooth_real = _smooth_real_noise(raw_real, corr_length=corr_length)
        smooth_imag = _smooth_real_noise(raw_imag, corr_length=corr_length)

        raw_noise = (smooth_real + 1j * smooth_imag).astype(np.complex128, copy=False)
    else:
        raw = rng.standard_normal(size=y_clean.shape).astype(np.float64, copy=False)
        raw_noise = _smooth_real_noise(raw, corr_length=corr_length)

    raw_rms = signal_rms(raw_noise)
    if raw_rms <= 1e-15:
        noise = np.zeros_like(y_clean)
    else:
        noise = raw_noise * (target_rms / raw_rms)

    y_noisy = y_clean + noise
    rms_noise = signal_rms(noise)

    return CorrelatedNoiseResult(
        y_noisy=y_noisy,
        y_clean=y_clean,
        noise=noise,
        rms_signal=rms_signal,
        rms_noise=rms_noise,
        noise_fraction_of_rms=alpha,
        corr_length=corr_length,
        random_seed=random_seed,
    )


def add_correlated_noise_rows(
    Y: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    *,
    noise_fraction_of_rms: float,
    corr_length: int,
    random_seed: int | None = None,
) -> tuple[npt.NDArray[np.float64] | npt.NDArray[np.complex128], list[CorrelatedNoiseResult]]:
    """Apply correlated noise row-by-row to a matrix of signals."""
    Y_arr = np.asarray(Y)
    if Y_arr.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y_arr.shape}")

    Y_noisy = np.zeros_like(Y_arr)
    results: list[CorrelatedNoiseResult] = []

    for i in range(Y_arr.shape[0]):
        seed_i = None if random_seed is None else int(random_seed + i)
        res = add_correlated_noise(
            Y_arr[i],
            noise_fraction_of_rms=noise_fraction_of_rms,
            corr_length=corr_length,
            random_seed=seed_i,
        )
        Y_noisy[i] = res.y_noisy
        results.append(res)

    return Y_noisy, results