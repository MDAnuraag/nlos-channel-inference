from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlos_cs.operators.diagnostics import compute_coherence_report, compute_svd_report
from nlos_cs.viz.matrices import (
    close_figure,
    plot_gram_matrix,
    plot_operator_diagnostic_triptych,
    plot_operator_heatmap,
    plot_singular_value_spectrum,
)


def _make_test_matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [0.1, 0.5, 4.0],
            [0.2, 0.1, 0.3],
        ],
        dtype=float,
    )


def test_plot_operator_heatmap_returns_figure():
    A = _make_test_matrix()
    positions = np.array([65.0, 70.0, 75.0], dtype=float)

    fig = plot_operator_heatmap(A, positions_mm=positions, title="Test Heatmap")

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_operator_heatmap_saves_file(tmp_path: Path):
    A = _make_test_matrix()
    out = tmp_path / "figures" / "operator_heatmap.png"

    fig = plot_operator_heatmap(A, output_path=out)

    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_operator_heatmap_accepts_complex_matrix():
    A = _make_test_matrix().astype(complex)
    A = A + 1j * 0.1 * A

    fig = plot_operator_heatmap(A)

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_operator_heatmap_rejects_bad_positions_shape():
    A = _make_test_matrix()
    bad_positions = np.array([[65.0, 70.0, 75.0]])

    try:
        fig = plot_operator_heatmap(A, positions_mm=bad_positions)
        close_figure(fig)
        assert False, "Expected ValueError for bad positions_mm shape"
    except ValueError as exc:
        assert "positions_mm must have shape" in str(exc)


def test_plot_singular_value_spectrum_from_matrix():
    A = _make_test_matrix()

    fig = plot_singular_value_spectrum(A, title="SVD Plot")

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_singular_value_spectrum_from_report():
    A = _make_test_matrix()
    svd_report = compute_svd_report(A)

    fig = plot_singular_value_spectrum(svd_report=svd_report)

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_singular_value_spectrum_saves_file(tmp_path: Path):
    A = _make_test_matrix()
    out = tmp_path / "figures" / "svd.png"

    fig = plot_singular_value_spectrum(A, output_path=out)

    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_singular_value_spectrum_requires_input():
    try:
        fig = plot_singular_value_spectrum()
        close_figure(fig)
        assert False, "Expected ValueError when neither A nor svd_report is provided"
    except ValueError as exc:
        assert "Provide either A or svd_report" in str(exc)


def test_plot_gram_matrix_from_matrix():
    A = _make_test_matrix()
    positions = np.array([65.0, 70.0, 75.0], dtype=float)

    fig = plot_gram_matrix(A, positions_mm=positions, annotate=True)

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_gram_matrix_from_report():
    A = _make_test_matrix()
    coherence_report = compute_coherence_report(A)

    fig = plot_gram_matrix(coherence_report=coherence_report, annotate=False)

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_gram_matrix_saves_file(tmp_path: Path):
    A = _make_test_matrix()
    out = tmp_path / "figures" / "gram.png"

    fig = plot_gram_matrix(A, output_path=out)

    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_gram_matrix_rejects_bad_positions_shape():
    A = _make_test_matrix()
    bad_positions = np.array([65.0, 70.0])

    try:
        fig = plot_gram_matrix(A, positions_mm=bad_positions)
        close_figure(fig)
        assert False, "Expected ValueError for bad positions_mm shape"
    except ValueError as exc:
        assert "positions_mm must have shape" in str(exc)


def test_plot_gram_matrix_requires_input():
    try:
        fig = plot_gram_matrix()
        close_figure(fig)
        assert False, "Expected ValueError when neither A nor coherence_report is provided"
    except ValueError as exc:
        assert "Provide either A or coherence_report" in str(exc)


def test_plot_operator_diagnostic_triptych_returns_figure():
    A = _make_test_matrix()
    positions = np.array([65.0, 70.0, 75.0], dtype=float)

    fig = plot_operator_diagnostic_triptych(
        A,
        positions_mm=positions,
        title_prefix="Operator Diagnostics Test",
    )

    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_operator_diagnostic_triptych_saves_file(tmp_path: Path):
    A = _make_test_matrix()
    out = tmp_path / "figures" / "triptych.png"

    fig = plot_operator_diagnostic_triptych(A, output_path=out)

    assert out.exists()
    assert isinstance(fig, plt.Figure)
    close_figure(fig)


def test_plot_operator_diagnostic_triptych_rejects_bad_positions_shape():
    A = _make_test_matrix()
    bad_positions = np.array([65.0, 70.0])

    try:
        fig = plot_operator_diagnostic_triptych(A, positions_mm=bad_positions)
        close_figure(fig)
        assert False, "Expected ValueError for bad positions_mm shape"
    except ValueError as exc:
        assert "positions_mm must have shape" in str(exc)


def test_close_figure_closes_without_error():
    A = _make_test_matrix()
    fig = plot_operator_heatmap(A)

    close_figure(fig)