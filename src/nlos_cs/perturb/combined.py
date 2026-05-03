"""Composed perturbation pipelines.

This module lets you apply several perturbation types in sequence to the same
measurement vector. The order is explicit and reproducible.

Current supported stages
------------------------
- awgn
- correlated
- dropout
- multipath

The composition is intentionally simple and transparent. It does not try to
hide which perturbation happened first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from nlos_cs.perturb.awgn import AWGNResult, add_awgn
from nlos_cs.perturb.correlated import CorrelatedNoiseResult, add_correlated_noise
from nlos_cs.perturb.dropout import DropoutResult, add_dropout
from nlos_cs.perturb.multipath import MultipathResult, add_multipath_leakage

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
SignalArray = FloatArray | ComplexArray
OperatorArray = FloatArray | ComplexArray


@dataclass(frozen=True)
class CombinedPerturbationConfig:
    """Configuration for a composed perturbation pipeline."""

    apply_awgn: bool = False
    awgn_fraction_of_rms: float = 0.0

    apply_correlated: bool = False
    correlated_fraction_of_rms: float = 0.0
    corr_length: int = 3

    apply_dropout: bool = False
    dropout_fraction: float = 0.0

    apply_multipath: bool = False
    multipath_fraction_of_rms: float = 0.0
    n_leak: int = 2
    exclude_index: int | None = None


@dataclass(frozen=True)
class CombinedPerturbationResult:
    """Result of a composed perturbation pipeline."""

    y_perturbed: SignalArray
    y_clean: SignalArray

    awgn_result: AWGNResult | None
    correlated_result: CorrelatedNoiseResult | None
    dropout_result: DropoutResult | None
    multipath_result: MultipathResult | None

    applied_stages: tuple[str, ...]
    random_seed: int | None


def apply_combined_perturbations(
    y: SignalArray,
    *,
    config: CombinedPerturbationConfig,
    A: OperatorArray | None = None,
    random_seed: int | None = None,
) -> CombinedPerturbationResult:
    """Apply enabled perturbations in a fixed order.

    Order
    -----
    1. multipath
    2. correlated noise
    3. AWGN
    4. dropout

    Rationale
    ---------
    - multipath is structured signal contamination
    - correlated noise is structured additive corruption
    - AWGN is unstructured additive corruption
    - dropout zeroes channels after other corruptions are present
    """
    y_current = np.asarray(y).copy()
    if y_current.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_current.shape}")

    y_clean = y_current.copy()

    awgn_result: AWGNResult | None = None
    correlated_result: CorrelatedNoiseResult | None = None
    dropout_result: DropoutResult | None = None
    multipath_result: MultipathResult | None = None

    applied: list[str] = []

    base_seed = 0 if random_seed is None else int(random_seed)

    if config.apply_multipath:
        if A is None:
            raise ValueError("A must be provided when apply_multipath=True")
        multipath_result = add_multipath_leakage(
            y_current,
            A,
            contamination_fraction_of_rms=config.multipath_fraction_of_rms,
            exclude_index=config.exclude_index,
            chosen_indices=None,
            n_leak=config.n_leak,
            random_seed=base_seed + 0,
            weights=None,
        )
        y_current = multipath_result.y_perturbed
        applied.append("multipath")

    if config.apply_correlated:
        correlated_result = add_correlated_noise(
            y_current,
            noise_fraction_of_rms=config.correlated_fraction_of_rms,
            corr_length=config.corr_length,
            random_seed=base_seed + 1,
        )
        y_current = correlated_result.y_noisy
        applied.append("correlated")

    if config.apply_awgn:
        awgn_result = add_awgn(
            y_current,
            noise_fraction_of_rms=config.awgn_fraction_of_rms,
            random_seed=base_seed + 2,
        )
        y_current = awgn_result.y_noisy
        applied.append("awgn")

    if config.apply_dropout:
        dropout_result = add_dropout(
            y_current,
            dropout_fraction=config.dropout_fraction,
            random_seed=base_seed + 3,
        )
        y_current = dropout_result.y_dropped
        applied.append("dropout")

    return CombinedPerturbationResult(
        y_perturbed=y_current,
        y_clean=y_clean,
        awgn_result=awgn_result,
        correlated_result=correlated_result,
        dropout_result=dropout_result,
        multipath_result=multipath_result,
        applied_stages=tuple(applied),
        random_seed=random_seed,
    )