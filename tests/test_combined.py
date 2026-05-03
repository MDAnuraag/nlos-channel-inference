import numpy as np

from nlos_cs.perturb.combined import (
    CombinedPerturbationConfig,
    apply_combined_perturbations,
)


def _make_test_operator() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [0.1, 0.5, 4.0],
            [0.2, 0.1, 0.3],
        ],
        dtype=float,
    )


def test_combined_no_stages_returns_identity():
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    config = CombinedPerturbationConfig()

    result = apply_combined_perturbations(y, config=config, random_seed=42)

    assert np.array_equal(result.y_clean, y)
    assert np.array_equal(result.y_perturbed, y)
    assert result.awgn_result is None
    assert result.correlated_result is None
    assert result.dropout_result is None
    assert result.multipath_result is None
    assert result.applied_stages == ()


def test_combined_awgn_only():
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    config = CombinedPerturbationConfig(
        apply_awgn=True,
        awgn_fraction_of_rms=0.1,
    )

    result = apply_combined_perturbations(y, config=config, random_seed=42)

    assert result.awgn_result is not None
    assert result.correlated_result is None
    assert result.dropout_result is None
    assert result.multipath_result is None
    assert result.applied_stages == ("awgn",)
    assert not np.allclose(result.y_perturbed, y)


def test_combined_correlated_only():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    config = CombinedPerturbationConfig(
        apply_correlated=True,
        correlated_fraction_of_rms=0.1,
        corr_length=2,
    )

    result = apply_combined_perturbations(y, config=config, random_seed=42)

    assert result.awgn_result is None
    assert result.correlated_result is not None
    assert result.dropout_result is None
    assert result.multipath_result is None
    assert result.applied_stages == ("correlated",)
    assert not np.allclose(result.y_perturbed, y)


def test_combined_dropout_only():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    config = CombinedPerturbationConfig(
        apply_dropout=True,
        dropout_fraction=0.5,
    )

    result = apply_combined_perturbations(y, config=config, random_seed=42)

    assert result.dropout_result is not None
    assert result.applied_stages == ("dropout",)
    assert np.any(result.y_perturbed == 0.0)


def test_combined_multipath_requires_operator():
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    config = CombinedPerturbationConfig(
        apply_multipath=True,
        multipath_fraction_of_rms=0.1,
        n_leak=1,
    )

    try:
        apply_combined_perturbations(y, config=config, A=None, random_seed=42)
        assert False, "Expected ValueError when A is missing"
    except ValueError as exc:
        assert "A must be provided" in str(exc)


def test_combined_multipath_only():
    A = _make_test_operator()
    y = A[:, 0].copy()
    config = CombinedPerturbationConfig(
        apply_multipath=True,
        multipath_fraction_of_rms=0.1,
        n_leak=2,
        exclude_index=0,
    )

    result = apply_combined_perturbations(y, config=config, A=A, random_seed=42)

    assert result.multipath_result is not None
    assert result.applied_stages == ("multipath",)
    assert not np.allclose(result.y_perturbed, y)


def test_combined_all_stages_order_and_reproducibility():
    A = _make_test_operator()
    y = A[:, 1].copy()
    config = CombinedPerturbationConfig(
        apply_multipath=True,
        multipath_fraction_of_rms=0.05,
        n_leak=2,
        exclude_index=1,
        apply_correlated=True,
        correlated_fraction_of_rms=0.05,
        corr_length=2,
        apply_awgn=True,
        awgn_fraction_of_rms=0.05,
        apply_dropout=True,
        dropout_fraction=0.25,
    )

    r1 = apply_combined_perturbations(y, config=config, A=A, random_seed=99)
    r2 = apply_combined_perturbations(y, config=config, A=A, random_seed=99)

    assert r1.applied_stages == ("multipath", "correlated", "awgn", "dropout")
    assert np.allclose(r1.y_perturbed, r2.y_perturbed)
    assert np.array_equal(r1.dropout_result.mask, r2.dropout_result.mask)


def test_combined_rejects_non_1d_signal():
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    config = CombinedPerturbationConfig()

    try:
        apply_combined_perturbations(y, config=config, random_seed=42)
        assert False, "Expected ValueError for non-1D y"
    except ValueError as exc:
        assert "y must be 1D" in str(exc)