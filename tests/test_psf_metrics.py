import numpy as np

from nlos_cs.metrics.psf import (
    batch_compute_psf_metrics,
    compute_psf_metrics,
    mean_peak_margin,
    mean_sidelobe_db,
    summarise_psf_metrics,
)


def test_compute_psf_metrics_basic_peak_and_sidelobe():
    x_hat = np.array([0.1, 0.4, 2.0, 0.5, 0.2], dtype=float)
    positions = np.array([65.0, 70.0, 75.0, 80.0, 85.0], dtype=float)

    metrics = compute_psf_metrics(x_hat, positions_mm=positions)

    assert metrics.peak_index == 2
    assert metrics.peak_position_mm == 75.0
    assert np.isclose(metrics.peak_value, 2.0)

    assert metrics.sidelobe_index == 3
    assert metrics.sidelobe_position_mm == 80.0
    assert np.isclose(metrics.sidelobe_value, 0.5)

    assert np.isclose(metrics.peak_margin, 1.5)
    assert np.isclose(metrics.sidelobe_db, 20.0 * np.log10(0.5 / 2.0))


def test_compute_psf_metrics_half_max_width_samples_and_mm():
    x_hat = np.array([0.2, 1.1, 2.0, 1.2, 0.3], dtype=float)
    positions = np.array([65.0, 70.0, 75.0, 80.0, 85.0], dtype=float)

    metrics = compute_psf_metrics(x_hat, positions_mm=positions)

    # half max = 1.0, so indices 1,2,3 are above half
    assert np.isclose(metrics.half_max_value, 1.0)
    assert metrics.half_max_width_samples == 2
    assert np.isclose(metrics.half_max_width_mm, 10.0)


def test_compute_psf_metrics_respects_true_index_override():
    x_hat = np.array([0.2, 2.5, 2.0, 0.4], dtype=float)
    positions = np.array([65.0, 70.0, 75.0, 80.0], dtype=float)

    metrics = compute_psf_metrics(x_hat, positions_mm=positions, true_index=2)

    assert metrics.peak_index == 2
    assert metrics.peak_position_mm == 75.0
    assert np.isclose(metrics.peak_value, 2.0)

    # strongest sidelobe should now be the actual max at index 1
    assert metrics.sidelobe_index == 1
    assert metrics.sidelobe_position_mm == 70.0
    assert np.isclose(metrics.sidelobe_value, 2.5)
    assert np.isclose(metrics.peak_margin, -0.5)


def test_compute_psf_metrics_uses_absolute_value_for_complex_input():
    x_hat = np.array([0.1 + 0.0j, -0.5j, -2.0 + 0.0j, 0.3 + 0.4j], dtype=complex)

    metrics = compute_psf_metrics(x_hat)

    assert metrics.peak_index == 2
    assert np.isclose(metrics.peak_value, 2.0)

    # |0.3 + 0.4j| = 0.5, tied with |-0.5j| = 0.5; argmax takes first
    assert metrics.sidelobe_index == 1
    assert np.isclose(metrics.sidelobe_value, 0.5)

    assert np.allclose(metrics.x_abs, np.array([0.1, 0.5, 2.0, 0.5]))
    assert np.isclose(metrics.x_norm[2], 1.0)


def test_compute_psf_metrics_handles_zero_vector():
    x_hat = np.zeros(4, dtype=float)
    positions = np.array([65.0, 70.0, 75.0, 80.0], dtype=float)

    metrics = compute_psf_metrics(x_hat, positions_mm=positions)

    assert metrics.peak_index == 0
    assert metrics.peak_position_mm == 65.0
    assert metrics.peak_value == 0.0
    assert metrics.sidelobe_index is None
    assert metrics.sidelobe_position_mm is None
    assert metrics.sidelobe_value == 0.0
    assert metrics.peak_margin == 0.0
    assert metrics.half_max_width_samples == 0
    assert metrics.half_max_width_mm == 0.0
    assert np.isneginf(metrics.sidelobe_db)


def test_batch_compute_psf_metrics():
    x_hats = np.array(
        [
            [0.1, 2.0, 0.3],
            [0.2, 0.4, 3.0],
        ],
        dtype=float,
    )
    positions = np.array([65.0, 70.0, 75.0], dtype=float)
    true_indices = np.array([1, 2], dtype=np.int64)

    metrics_list = batch_compute_psf_metrics(
        x_hats,
        positions_mm=positions,
        true_indices=true_indices,
    )

    assert len(metrics_list) == 2
    assert metrics_list[0].peak_index == 1
    assert metrics_list[0].peak_position_mm == 70.0
    assert metrics_list[1].peak_index == 2
    assert metrics_list[1].peak_position_mm == 75.0


def test_summarise_psf_metrics():
    x_hat = np.array([0.1, 0.4, 2.0, 0.5], dtype=float)
    positions = np.array([65.0, 70.0, 75.0, 80.0], dtype=float)

    metrics = compute_psf_metrics(x_hat, positions_mm=positions)
    summary = summarise_psf_metrics(metrics)

    expected_keys = {
        "peak_index",
        "peak_position_mm",
        "peak_value",
        "sidelobe_index",
        "sidelobe_position_mm",
        "sidelobe_value",
        "sidelobe_db",
        "peak_margin",
        "half_max_value",
        "half_max_width_samples",
        "half_max_width_mm",
    }
    assert set(summary.keys()) == expected_keys
    assert summary["peak_index"] == 2
    assert np.isclose(summary["peak_value"], 2.0)


def test_mean_peak_margin_and_mean_sidelobe_db():
    x_hats = [
        np.array([0.1, 2.0, 0.5], dtype=float),
        np.array([0.2, 0.4, 3.0], dtype=float),
    ]

    metrics_list = [compute_psf_metrics(x) for x in x_hats]

    expected_peak_margin = np.mean([2.0 - 0.5, 3.0 - 0.4])
    expected_sidelobe_db = np.mean([
        20.0 * np.log10(0.5 / 2.0),
        20.0 * np.log10(0.4 / 3.0),
    ])

    assert np.isclose(mean_peak_margin(metrics_list), expected_peak_margin)
    assert np.isclose(mean_sidelobe_db(metrics_list), expected_sidelobe_db)