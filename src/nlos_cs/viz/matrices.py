"""Visualisation utilities for sensing matrices and operator diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from nlos_cs.operators.diagnostics import (
    CoherenceReport,
    SVDReport,
    compute_coherence_report,
    compute_svd_report,
)

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
OperatorArray = FloatArray | ComplexArray


def _ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directory for an output path if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _as_display_matrix(A: OperatorArray) -> FloatArray:
    """Convert an operator into a real matrix for plotting."""
    A_arr = np.asarray(A)
    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A_arr.shape}")
    if np.iscomplexobj(A_arr):
        return np.abs(A_arr).astype(np.float64, copy=False)
    return A_arr.astype(np.float64, copy=False)


def plot_operator_heatmap(
    A: OperatorArray,
    *,
    positions_mm: FloatArray | None = None,
    title: str = "Operator Heatmap",
    normalise_columns: bool = True,
    figsize: tuple[float, float] = (10.0, 5.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the sensing operator as a heatmap.

    Parameters
    ----------
    A:
        Operator matrix of shape (M, N).
    positions_mm:
        Optional position labels for the columns.
    title:
        Figure title.
    normalise_columns:
        If True, divide each column by its own maximum absolute value to show
        shape variation rather than raw amplitude dominance.
    figsize:
        Matplotlib figure size.
    output_path:
        If provided, save figure to this path.
    """
    A_plot = _as_display_matrix(A)

    if normalise_columns:
        col_max = np.max(np.abs(A_plot), axis=0, keepdims=True)
        col_max = np.where(col_max <= 1e-15, 1.0, col_max)
        A_plot = A_plot / col_max

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A_plot, aspect="auto", origin="upper")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalised amplitude" if normalise_columns else "Amplitude")

    ax.set_xlabel("Position index")
    ax.set_ylabel("Measurement row")
    ax.set_title(title)

    if positions_mm is not None:
        pos = np.asarray(positions_mm, dtype=np.float64)
        if pos.ndim != 1 or pos.shape[0] != A_plot.shape[1]:
            raise ValueError(
                f"positions_mm must have shape ({A_plot.shape[1]},), got {pos.shape}"
            )
        ax.set_xticks(np.arange(len(pos)))
        ax.set_xticklabels([f"{int(p)}" if float(p).is_integer() else f"{p:g}" for p in pos], rotation=45)

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_singular_value_spectrum(
    A: OperatorArray | None = None,
    *,
    svd_report: SVDReport | None = None,
    title: str = "Singular Value Spectrum",
    figsize: tuple[float, float] = (8.0, 5.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the singular values on a log-scale axis."""
    if svd_report is None:
        if A is None:
            raise ValueError("Provide either A or svd_report")
        svd_report = compute_svd_report(A)

    s = svd_report.singular_values
    idx = np.arange(1, len(s) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(idx, s, marker="o")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(
        f"{title}\n"
        f"$\\kappa$={svd_report.condition_number:.3g}, "
        f"rank={svd_report.effective_rank}/{len(s)}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(idx)

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_gram_matrix(
    A: OperatorArray | None = None,
    *,
    coherence_report: CoherenceReport | None = None,
    positions_mm: FloatArray | None = None,
    title: str = "Column Correlation Matrix",
    annotate: bool = True,
    figsize: tuple[float, float] = (8.0, 7.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    if coherence_report is None:
        if A is None:
            raise ValueError("Provide either A or coherence_report")
        coherence_report = compute_coherence_report(A)

    G = coherence_report.gram_matrix
    n = G.shape[0]

    labels = None
    if positions_mm is not None:
        pos = np.asarray(positions_mm, dtype=np.float64)
        if pos.ndim != 1 or pos.shape[0] != n:
            raise ValueError(f"positions_mm must have shape ({n},), got {pos.shape}")
        labels = [f"{int(p)}" if float(p).is_integer() else f"{p:g}" for p in pos]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(G, vmin=0.0, vmax=1.0, aspect="equal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalised column correlation")

    ax.set_xlabel("Column j")
    ax.set_ylabel("Column i")
    ax.set_title(
        f"{title}\n"
        f"mutual coherence={coherence_report.mutual_coherence:.4f}"
    )

    if labels is not None:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)

    if annotate:
        for i in range(n):
            for j in range(n):
                val = G[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val > 0.65 else "black",
                )

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_operator_diagnostic_triptych(
    A: OperatorArray,
    *,
    positions_mm: FloatArray | None = None,
    title_prefix: str = "Operator Diagnostics",
    figsize: tuple[float, float] = (18.0, 5.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a three-panel summary:
    1. operator heatmap
    2. singular value spectrum
    3. Gram matrix
    """
    A_plot = _as_display_matrix(A)
    svd_report = compute_svd_report(A)
    coherence_report = compute_coherence_report(A)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: operator heatmap
    col_max = np.max(np.abs(A_plot), axis=0, keepdims=True)
    col_max = np.where(col_max <= 1e-15, 1.0, col_max)
    A_norm = A_plot / col_max
    im0 = axes[0].imshow(A_norm, aspect="auto", origin="upper")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Normalised amplitude")
    axes[0].set_title("Operator")
    axes[0].set_xlabel("Position index")
    axes[0].set_ylabel("Measurement row")

    if positions_mm is not None:
        pos = np.asarray(positions_mm, dtype=np.float64)
        if pos.ndim != 1 or pos.shape[0] != A_plot.shape[1]:
            raise ValueError(
                f"positions_mm must have shape ({A_plot.shape[1]},), got {pos.shape}"
            )
        labels = [f"{int(p)}" if float(p).is_integer() else f"{p:g}" for p in pos]
        axes[0].set_xticks(np.arange(len(pos)))
        axes[0].set_xticklabels(labels, rotation=45)

    # Panel 2: singular values
    s = svd_report.singular_values
    idx = np.arange(1, len(s) + 1)
    axes[1].semilogy(idx, s, marker="o")
    axes[1].set_title(
        f"Singular values\n$\\kappa$={svd_report.condition_number:.3g}"
    )
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Singular value")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_xticks(idx)

    # Panel 3: Gram matrix
    G = coherence_report.gram_matrix
    im2 = axes[2].imshow(G, vmin=0.0, vmax=1.0, aspect="equal")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Correlation")
    axes[2].set_title(
        f"Gram matrix\n$\\mu$={coherence_report.mutual_coherence:.4f}"
    )
    axes[2].set_xlabel("Column j")
    axes[2].set_ylabel("Column i")

    if positions_mm is not None:
        axes[2].set_xticks(np.arange(len(pos)))
        axes[2].set_yticks(np.arange(len(pos)))
        axes[2].set_xticklabels(labels, rotation=45)
        axes[2].set_yticklabels(labels)

    fig.suptitle(title_prefix)
    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def close_figure(fig: plt.Figure) -> None:
    """Close a matplotlib figure explicitly."""
    plt.close(fig)