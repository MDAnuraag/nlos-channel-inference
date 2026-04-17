import numpy as np

from nlos_cs.perturb.awgn import (
    add_awgn,
    add_awgn_rows,
    signal_rms,
    snr_db_from_noise_fraction,
)


def test_signal_rms_real():
    y = np.array([3.0, 4.0], dtype=float)
    expected = np.sqrt((3.0**2 + 4.0**2) / 2.0)

    assert np.isclose(signal_rms(y), expected)


def test_signal_rms_complex():
    y = np.array([3.0 + 4.0j, 0.0 + 0.0j], dtype=complex)
    expected = np.sqrt((5.0**2 + 0.0**2) / 2.0)

    assert np.isclose(signal_rms(y), expected)


def test_snr_db_from_noise_fraction():
    assert np.isclose(snr_db_from_noise_fraction(0.1), 20.0)
    assert np.isclose(snr_db_from_noise_fraction(0.01), 40.0)
    assert np.isinf(snr_db_from_noise_fraction(0.0))


def test_add_awgn_zero_noise_returns_identity():
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    result = add_awgn(y, noise_fraction_of_rms=0.0, random_seed=42)

    assert np.array_equal(result.y_clean, y)
    assert np.array_equal(result.y_noisy, y)
    assert np.array_equal(result.noise, np.zeros_like(y))
    assert result.rms_signal > 0.0
    assert result.rms_noise == 0.0
    assert result.noise_fraction_of_rms == 0.0
    assert result.random_seed == 42


def test_add_awgn_zero_signal_returns_identity():
    y = np.zeros(5, dtype=float)

    result = add_awgn(y, noise_fraction_of_rms=0.1, random_seed=42)

    assert np.array_equal(result.y_clean, y)
    assert np.array_equal(result.y_noisy, y)
    assert np.array_equal(result.noise, np.zeros_like(y))
    assert result.rms_signal == 0.0
    assert result.rms_noise == 0.0


def test_add_awgn_real_is_reproducible():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    r1 = add_awgn(y, noise_fraction_of_rms=0.05, random_seed=123)
    r2 = add_awgn(y, noise_fraction_of_rms=0.05, random_seed=123)

    assert np.allclose(r1.y_noisy, r2.y_noisy)
    assert np.allclose(r1.noise, r2.noise)
    assert np.array_equal(r1.y_clean, y)
    assert not np.allclose(r1.y_noisy, y)


def test_add_awgn_complex_is_reproducible():
    y = np.array([1.0 + 1.0j, 2.0 - 0.5j, -1.0 + 0.2j], dtype=complex)

    r1 = add_awgn(y, noise_fraction_of_rms=0.05, random_seed=999)
    r2 = add_awgn(y, noise_fraction_of_rms=0.05, random_seed=999)

    assert np.allclose(r1.y_noisy, r2.y_noisy)
    assert np.allclose(r1.noise, r2.noise)
    assert np.array_equal(r1.y_clean, y)
    assert np.iscomplexobj(r1.noise)
    assert np.iscomplexobj(r1.y_noisy)


def test_add_awgn_noise_level_is_reasonable_real():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    alpha = 0.1

    result = add_awgn(y, noise_fraction_of_rms=alpha, random_seed=7)

    expected_noise_rms = alpha * result.rms_signal
    assert result.rms_noise > 0.0
    assert np.isclose(result.rms_noise, expected_noise_rms, rtol=0.5)


def test_add_awgn_rows_shape_and_reproducibility():
    Y = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    Y1, results1 = add_awgn_rows(Y, noise_fraction_of_rms=0.05, random_seed=100)
    Y2, results2 = add_awgn_rows(Y, noise_fraction_of_rms=0.05, random_seed=100)

    assert Y1.shape == Y.shape
    assert Y2.shape == Y.shape
    assert len(results1) == Y.shape[0]
    assert len(results2) == Y.shape[0]

    assert np.allclose(Y1, Y2)
    for r1, r2 in zip(results1, results2):
        assert np.allclose(r1.y_noisy, r2.y_noisy)
        assert np.allclose(r1.noise, r2.noise)


def test_add_awgn_rows_uses_different_row_seeds():
    Y = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=float,
    )

    Y_noisy, results = add_awgn_rows(Y, noise_fraction_of_rms=0.05, random_seed=10)

    assert not np.allclose(Y_noisy[0], Y_noisy[1])
    assert results[0].random_seed == 10
    assert results[1].random_seed == 11


def test_add_awgn_rejects_negative_noise_fraction():
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    try:
        add_awgn(y, noise_fraction_of_rms=-0.1, random_seed=1)
        assert False, "Expected ValueError for negative noise fraction"
    except ValueError as exc:
        assert "must be non-negative" in str(exc)


def test_add_awgn_rows_rejects_non_2d_input():
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    try:
        add_awgn_rows(y, noise_fraction_of_rms=0.1, random_seed=1)
        assert False, "Expected ValueError for non-2D input"
    except ValueError as exc:
        assert "Y must be 2D" in str(exc)