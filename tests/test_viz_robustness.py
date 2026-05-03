import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlos_cs.viz.robustness import (
    close_figure,
    plot_mean_peak_margin_curves,
    plot_peak_margin_heatmap,
    plot_success_rate_curves,
    plot_success_rate_heatmap,
)


def _make_success_matrix() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.95, 0.80],
            [1.00, 0.90, 0.70],
            [1.00, 0.85, 0.60],
        ],
        dtype=float,
    )


def _make_margin_matrix() -> np.ndarray:
    return np.array(
        [
            [1.20, 1.00, 0.70],
            [1.10, 0.90, 0.60],
            [0.95, 0.75, 0.40],
        ],
        dtype=float,
    )


def _make_noise_levels() -> np.ndarray:
    return np.array([0.0, 0.01, 0.05], dtype=float)


def _make_positions() -> np.ndarray:
    return np.array([65.0, 70.0, 75.0], dtype=float)


def test_plot_success_rate_heatmap_returns_figure():
    fig = plot_success_rate_heatmap(
        _make_success_matrix(),
        noise_levels=_make_noise_levels(),
        positions_mm=_make_positions(),
        annotate=True,
    )
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_success_rate_heatmap_saves_file(tmp_path: Path):
    out = tmp_path / "figures" / "success_heatmap.png"
    fig = plot_success_rate_heatmap(
        _make_success_matrix(),
        noise_levels=_make_noise_levels(),
        output_path=out,
    )
    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_success_rate_heatmap_rejects_bad_noise_shape():
    try:
        fig = plot_success_rate_heatmap(
            _make_success_matrix(),
            noise_levels=np.array([0.0, 0.01]),
        )
        close_figure(fig)
        assert False, "Expected ValueError for bad noise_levels shape"
    except ValueError as exc:
        assert "noise_levels must have shape" in str(exc)


def test_plot_peak_margin_heatmap_returns_figure():
    fig = plot_peak_margin_heatmap(
        _make_margin_matrix(),
        noise_levels=_make_noise_levels(),
        positions_mm=_make_positions(),
        annotate=True,
    )
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_peak_margin_heatmap_saves_file(tmp_path: Path):
    out = tmp_path / "figures" / "margin_heatmap.png"
    fig = plot_peak_margin_heatmap(
        _make_margin_matrix(),
        noise_levels=_make_noise_levels(),
        output_path=out,
    )
    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_peak_margin_heatmap_rejects_bad_noise_shape():
    try:
        fig = plot_peak_margin_heatmap(
            _make_margin_matrix(),
            noise_levels=np.array([0.0, 0.01]),
        )
        close_figure(fig)
        assert False, "Expected ValueError for bad noise_levels shape"
    except ValueError as exc:
        assert "noise_levels must have shape" in str(exc)


def test_plot_success_rate_curves_returns_figure():
    fig = plot_success_rate_curves(
        _make_success_matrix(),
        noise_levels=_make_noise_levels(),
        positions_mm=_make_positions(),
    )
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_success_rate_curves_saves_file(tmp_path: Path):
    out = tmp_path / "figures" / "success_curves.png"
    fig = plot_success_rate_curves(
        _make_success_matrix(),
        noise_levels=_make_noise_levels(),
        output_path=out,
    )
    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_success_rate_curves_rejects_bad_noise_shape():
    try:
        fig = plot_success_rate_curves(
            _make_success_matrix(),
            noise_levels=np.array([0.0, 0.01]),
        )
        close_figure(fig)
        assert False, "Expected ValueError for bad noise_levels shape"
    except ValueError as exc:
        assert "noise_levels must have shape" in str(exc)


def test_plot_mean_peak_margin_curves_returns_figure():
    fig = plot_mean_peak_margin_curves(
        _make_margin_matrix(),
        noise_levels=_make_noise_levels(),
        positions_mm=_make_positions(),
    )
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_mean_peak_margin_curves_saves_file(tmp_path: Path):
    out = tmp_path / "figures" / "margin_curves.png"
    fig = plot_mean_peak_margin_curves(
        _make_margin_matrix(),
        noise_levels=_make_noise_levels(),
        output_path=out,
    )
    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_mean_peak_margin_curves_rejects_bad_noise_shape():
    try:
        fig = plot_mean_peak_margin_curves(
            _make_margin_matrix(),
            noise_levels=np.array([0.0, 0.01]),
        )
        close_figure(fig)
        assert False, "Expected ValueError for bad noise_levels shape"
    except ValueError as exc:
        assert "noise_levels must have shape" in str(exc)


def test_close_figure_closes_without_error():
    fig = plot_success_rate_heatmap(
        _make_success_matrix(),
        noise_levels=_make_noise_levels(),
    )
    close_figure(fig)