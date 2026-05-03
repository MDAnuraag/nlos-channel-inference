import numpy as np

from nlos_cs.perturb.correlated import (
    add_correlated_noise,
    add_correlated_noise_rows,
)


def test_add_correlated_noise_zero_noise_returns_identity():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    result = add_correlated_noise(
        y,
        noise_fraction_of_rms=0.0,
        corr_length=3,
        random_seed=42,
    )

    assert np.array_equal(result.y_clean, y)
    assert np.array_equal(result.y_noisy, y)
    assert np.array_equal(result.noise, np.zeros_like(y))
    assert result.rms_noise == 0.0


def test_add_correlated_noise_zero_signal_returns_identity():
    y = np.zeros(6, dtype=float)

    result = add_correlated_noise(
        y,
        noise_fraction_of_rms=0.1,
        corr_length=3,
        random_seed=42,
    )

    assert np.array_equal(result.y_noisy, y)
    assert np.array_equal(result.noise, np.zeros_like(y))
    assert result.rms_signal == 0.0
    assert result.rms_noise == 0.0


def test_add_correlated_noise_reproducible_real():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    r1 = add_correlated_noise(
        y,
        noise_fraction_of_rms=0.1,
        corr_length=3,
        random_seed=123,
    )
    r2 = add_correlated_noise(
        y,
        noise_fraction_of_rms=0.1,
        corr_length=3,
        random_seed=123,
    )

    assert np.allclose(r1.noise, r2.noise)
    assert np.allclose(r1.y_noisy, r2.y_noisy)


def test_add_correlated_noise_reproducible_complex():
    y = np.array([1.0 + 0.2j, 2.0 - 0.1j, 3.0 + 0.3j], dtype=complex)

    r1 = add_correlated_noise(
        y,
        noise_fraction_of_rms=0.1,
        corr_length=2,
        random_seed=999,
    )
    r2 = add_correlated_noise(
        y,
        noise_fraction_of_rms=0.1,
        corr_length=2,
        random_seed=999,
    )

    assert np.allclose(r1.noise, r2.noise)
    assert np.allclose(r1.y_noisy, r2.y_noisy)
    assert np.iscomplexobj(r1.noise)


def test_add_correlated_noise_hits_target_rms_reasonably():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    alpha = 0.1

    result = add_correlated_noise(
        y,
        noise_fraction_of_rms=alpha,
        corr_length=3,
        random_seed=7,
    )

    expected = alpha * result.rms_signal
    assert result.rms_noise > 0.0
    assert np.isclose(result.rms_noise, expected, rtol=1e-6, atol=1e-10)


def test_add_correlated_noise_rows_basic():
    Y = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
        dtype=float,
    )

    Y_noisy, results = add_correlated_noise_rows(
        Y,
        noise_fraction_of_rms=0.05,
        corr_length=2,
        random_seed=10,
    )

    assert Y_noisy.shape == Y.shape
    assert len(results) == 2
    assert results[0].random_seed == 10
    assert results[1].random_seed == 11


def test_add_correlated_noise_rows_rejects_non_2d():
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    try:
        add_correlated_noise_rows(
            y,
            noise_fraction_of_rms=0.1,
            corr_length=2,
            random_seed=1,
        )
        assert False, "Expected ValueError for non-2D input"
    except ValueError as exc:
        assert "Y must be 2D" in str(exc)


def test_add_correlated_noise_rejects_bad_inputs():
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    try:
        add_correlated_noise(
            y,
            noise_fraction_of_rms=-0.1,
            corr_length=2,
            random_seed=1,
        )
        assert False, "Expected ValueError for negative noise fraction"
    except ValueError as exc:
        assert "must be non-negative" in str(exc)

    try:
        add_correlated_noise(
            y,
            noise_fraction_of_rms=0.1,
            corr_length=0,
            random_seed=1,
        )
        assert False, "Expected ValueError for bad corr_length"
    except ValueError as exc:
        assert "corr_length must be positive" in str(exc)