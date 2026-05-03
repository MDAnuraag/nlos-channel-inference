import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlos_cs.metrics.discrimination import compute_discrimination_from_xhats
from nlos_cs.viz.reconstructions import (
    close_figure,
    plot_discrimination_heatmap,
    plot_leakage_heatmap,
    plot_psf_response,
    plot_reconstruction_bar,
)


def _make_test_x_hat() -> np.ndarray:
    return np.array([0.1, 0.4, 2.0, 0.5, 0.2], dtype=float)


def _make_test_positions() -> np.ndarray:
    return np.array([65.0, 70.0, 75.0, 80.0, 85.0], dtype=float)


def _make_test_discrimination_result():
    x_hats = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [0.1, 0.5, 4.0],
        ],
        dtype=float,
    )
    positions = np.array([65.0, 70.0, 75.0], dtype=float)
    return compute_discrimination_from_xhats(x_hats, positions_mm=positions)


def test_plot_reconstruction_bar_returns_figure():
    x_hat = _make_test_x_hat()
    positions = _make_test_positions()

    fig = plot_reconstruction_bar(
        x_hat,
        positions_mm=positions,
        true_index=2,
        title="Test Reconstruction",
    )

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_reconstruction_bar_saves_file(tmp_path: Path):
    x_hat = _make_test_x_hat()
    out = tmp_path / "figures" / "reconstruction_bar.png"

    fig = plot_reconstruction_bar(x_hat, output_path=out)

    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_reconstruction_bar_accepts_complex_input():
    x_hat = np.array([0.1 + 0.0j, -0.4j, -2.0 + 0.0j, 0.3 + 0.4j], dtype=complex)

    fig = plot_reconstruction_bar(x_hat, use_abs=True)

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_reconstruction_bar_rejects_bad_true_index():
    x_hat = _make_test_x_hat()

    try:
        fig = plot_reconstruction_bar(x_hat, true_index=99)
        close_figure(fig)
        assert False, "Expected ValueError for bad true_index"
    except ValueError as exc:
        assert "out of range" in str(exc)


def test_plot_psf_response_returns_figure():
    x_hat = _make_test_x_hat()
    positions = _make_test_positions()

    fig = plot_psf_response(
        x_hat,
        positions_mm=positions,
        true_index=2,
        title="Test PSF",
    )

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_psf_response_saves_file(tmp_path: Path):
    x_hat = _make_test_x_hat()
    out = tmp_path / "figures" / "psf_response.png"

    fig = plot_psf_response(x_hat, output_path=out)

    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_discrimination_heatmap_returns_figure():
    result = _make_test_discrimination_result()

    fig = plot_discrimination_heatmap(result, annotate=True)

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_discrimination_heatmap_saves_file(tmp_path: Path):
    result = _make_test_discrimination_result()
    out = tmp_path / "figures" / "discrimination.png"

    fig = plot_discrimination_heatmap(result, output_path=out)

    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_discrimination_heatmap_accepts_override_positions():
    result = _make_test_discrimination_result()
    positions = np.array([1.0, 2.0, 3.0], dtype=float)

    fig = plot_discrimination_heatmap(result, positions_mm=positions)

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_discrimination_heatmap_rejects_bad_positions_shape():
    result = _make_test_discrimination_result()
    bad_positions = np.array([65.0, 70.0], dtype=float)

    try:
        fig = plot_discrimination_heatmap(result, positions_mm=bad_positions)
        close_figure(fig)
        assert False, "Expected ValueError for bad positions shape"
    except ValueError as exc:
        assert "positions_mm must have shape" in str(exc)


def test_plot_leakage_heatmap_returns_figure_raw():
    result = _make_test_discrimination_result()

    fig = plot_leakage_heatmap(result, scale="raw", annotate=True)

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_leakage_heatmap_returns_figure_scaled():
    result = _make_test_discrimination_result()

    fig = plot_leakage_heatmap(result, scale="x1e3", annotate=True)

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_leakage_heatmap_saves_file(tmp_path: Path):
    result = _make_test_discrimination_result()
    out = tmp_path / "figures" / "leakage.png"

    fig = plot_leakage_heatmap(result, output_path=out)

    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_leakage_heatmap_rejects_bad_scale():
    result = _make_test_discrimination_result()

    try:
        fig = plot_leakage_heatmap(result, scale="bad_scale")
        close_figure(fig)
        assert False, "Expected ValueError for bad scale"
    except ValueError as exc:
        assert "Unknown scale" in str(exc)


def test_plot_leakage_heatmap_rejects_bad_positions_shape():
    result = _make_test_discrimination_result()
    bad_positions = np.array([65.0, 70.0], dtype=float)

    try:
        fig = plot_leakage_heatmap(result, positions_mm=bad_positions)
        close_figure(fig)
        assert False, "Expected ValueError for bad positions shape"
    except ValueError as exc:
        assert "positions_mm must have shape" in str(exc)


def test_close_figure_closes_without_error():
    fig = plot_reconstruction_bar(_make_test_x_hat())
    close_figure(fig)