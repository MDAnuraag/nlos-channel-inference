"""Additive white Gaussian noise perturbations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
SignalArray = FloatArray | ComplexArray


@dataclass(frozen=True)
class AWGNResult:
    """Result of applying AWGN to a signal."""

    y_noisy: SignalArray
    y_clean: SignalArray
    noise: SignalArray
    rms_signal: float
    rms_noise: float
    noise_fraction_of_rms: float
    random_seed: int | None


def signal_rms(y: SignalArray) -> float:
    """Return RMS magnitude of a real or complex signal."""
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
    return float(np.sqrt(np.mean(np.abs(y_arr) ** 2)))


def snr_db_from_noise_fraction(noise_fraction_of_rms: float) -> float:
    """Return nominal SNR in dB for RMS-scaled AWGN.

    If noise_fraction_of_rms = alpha, then

        SNR = 20 log10(1 / alpha)

    under the convention that noise RMS = alpha * signal RMS.
    """
    alpha = float(noise_fraction_of_rms)
    if alpha < 0.0:
        raise ValueError(f"noise_fraction_of_rms must be non-negative, got {alpha}")
    if alpha == 0.0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / alpha))


def add_awgn(
    y: SignalArray,
    *,
    noise_fraction_of_rms: float,
    random_seed: int | None = None,
) -> AWGNResult:
    """Add white Gaussian noise scaled to a fraction of the signal RMS.

    Parameters
    ----------
    y:
        Clean input signal, shape (M,).
    noise_fraction_of_rms:
        Noise RMS as a fraction of signal RMS.
        Example: 0.01 means 1% of signal RMS.
    random_seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    AWGNResult
    """
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}")

    alpha = float(noise_fraction_of_rms)
    if alpha < 0.0:
        raise ValueError(f"noise_fraction_of_rms must be non-negative, got {alpha}")

    y_clean = y_arr.copy()
    rms_signal = signal_rms(y_clean)

    if alpha == 0.0 or rms_signal == 0.0:
        noise = np.zeros_like(y_clean)
        y_noisy = y_clean.copy()
        rms_noise = 0.0
        return AWGNResult(
            y_noisy=y_noisy,
            y_clean=y_clean,
            noise=noise,
            rms_signal=rms_signal,
            rms_noise=rms_noise,
            noise_fraction_of_rms=alpha,
            random_seed=random_seed,
        )

    rng = np.random.default_rng(random_seed)

    target_rms = alpha * rms_signal

    if np.iscomplexobj(y_clean):
        raw_noise = (
            rng.standard_normal(size=y_clean.shape) + 1j * rng.standard_normal(size=y_clean.shape)
        ).astype(np.complex128, copy=False)
    else:
        raw_noise = rng.standard_normal(size=y_clean.shape).astype(np.float64, copy=False)

    raw_rms = signal_rms(raw_noise)
    if raw_rms <= 1e-15:
        noise = np.zeros_like(y_clean)
    else:
        noise = raw_noise * (target_rms / raw_rms)

    y_noisy = y_clean + noise
    rms_noise = signal_rms(noise)

    return AWGNResult(
        y_noisy=y_noisy,
        y_clean=y_clean,
        noise=noise,
        rms_signal=rms_signal,
        rms_noise=rms_noise,
        noise_fraction_of_rms=alpha,
        random_seed=random_seed,
    )


def add_awgn_rows(
    Y: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    *,
    noise_fraction_of_rms: float,
    random_seed: int | None = None,
) -> tuple[npt.NDArray[np.float64] | npt.NDArray[np.complex128], list[AWGNResult]]:
    """Apply AWGN row-by-row to a matrix of signals.

    Parameters
    ----------
    Y:
        Array of shape (K, M), where each row is one signal.
    noise_fraction_of_rms:
        Noise RMS as a fraction of each row's own RMS.
    random_seed:
        Base seed. Row i uses seed (random_seed + i) if provided.

    Returns
    -------
    Y_noisy, results
        Y_noisy has the same shape as Y.
        results contains one AWGNResult per row.
    """
    Y_arr = np.asarray(Y)
    if Y_arr.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y_arr.shape}")

    Y_noisy = np.zeros_like(Y_arr)
    results: list[AWGNResult] = []

    for i in range(Y_arr.shape[0]):
        seed_i = None if random_seed is None else int(random_seed + i)
        res = add_awgn(
            Y_arr[i],
            noise_fraction_of_rms=noise_fraction_of_rms,
            random_seed=seed_i,
        )
        Y_noisy[i] = res.y_noisy
        results.append(res)

    return Y_noisy, results