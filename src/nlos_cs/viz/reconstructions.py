"""Visualisation utilities for reconstructions, PSF-style responses, and
discrimination / leakage matrices.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from nlos_cs.metrics.discrimination import DiscriminationResult
from nlos_cs.metrics.psf import PSFMetrics, compute_psf_metrics

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
SignalArray = FloatArray | ComplexArray


def _ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directory for an output path if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _position_labels(positions_mm: FloatArray | None, n: int) -> list[str]:
    """Create x-axis labels from positions if available, else indices."""
    if positions_mm is None:
        return [str(i) for i in range(n)]

    pos = np.asarray(positions_mm, dtype=np.float64)
    if pos.ndim != 1 or pos.shape[0] != n:
        raise ValueError(f"positions_mm must have shape ({n},), got {pos.shape}")

    labels: list[str] = []
    for p in pos:
        labels.append(f"{int(p)}" if float(p).is_integer() else f"{p:g}")
    return labels


def plot_reconstruction_bar(
    x_hat: SignalArray,
    *,
    positions_mm: FloatArray | None = None,
    true_index: int | None = None,
    title: str = "Reconstruction $x_\\hat{}$",
    use_abs: bool = True,
    figsize: tuple[float, float] = (10.0, 5.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a single reconstruction vector as a bar chart."""
    x = np.asarray(x_hat)
    if x.ndim != 1:
        raise ValueError(f"x_hat must be 1D, got shape {x.shape}")

    vals = np.abs(x) if use_abs else x
    vals = np.asarray(vals, dtype=np.float64)
    n = vals.shape[0]

    if true_index is not None and (true_index < 0 or true_index >= n):
        raise ValueError(f"true_index {true_index} out of range for length {n}")

    peak_index = int(np.argmax(vals))
    labels = _position_labels(positions_mm, n)

    colours = ["steelblue"] * n
    colours[peak_index] = "tab:orange"
    if true_index is not None:
        colours[true_index] = "tab:red"

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(n), vals, color=colours, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Position")
    ax.set_ylabel("|x_hat|" if use_abs else "x_hat")
    ax.set_title(title)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(axis="y", alpha=0.3)

    if true_index is not None:
        ax.axvline(true_index, color="tab:red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(peak_index, color="tab:orange", linestyle=":", linewidth=1.5, alpha=0.7)

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_psf_response(
    x_hat: SignalArray,
    *,
    positions_mm: FloatArray | None = None,
    true_index: int | None = None,
    title: str = "PSF-style Response",
    figsize: tuple[float, float] = (10.0, 5.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a normalised PSF-style response from one reconstruction vector."""
    metrics = compute_psf_metrics(
        x_hat,
        positions_mm=positions_mm,
        true_index=true_index,
    )

    x = metrics.x_norm
    n = x.shape[0]
    labels = _position_labels(positions_mm, n)
    peak_index = metrics.peak_index

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(n), x, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Position")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title(
        f"{title}\n"
        f"peak={metrics.peak_value:.3g}, "
        f"sidelobe={metrics.sidelobe_db:.2f} dB, "
        f"margin={metrics.peak_margin:.3g}"
    )
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.3)

    ax.axhline(0.5, color="green", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(peak_index, color="tab:orange", linestyle=":", linewidth=1.5, alpha=0.8)

    if true_index is not None:
        ax.axvline(true_index, color="tab:red", linestyle="--", linewidth=1.5, alpha=0.7)

    if metrics.sidelobe_index is not None:
        ax.axvline(
            metrics.sidelobe_index,
            color="grey",
            linestyle=":",
            linewidth=1.2,
            alpha=0.6,
        )

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_discrimination_heatmap(
    result: DiscriminationResult,
    *,
    positions_mm: FloatArray | None = None,
    annotate: bool = True,
    title: str = "Discrimination Matrix",
    figsize: tuple[float, float] = (8.0, 7.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the discrimination matrix D[i, j] = 1 - leakage[i, j]."""
    D = np.asarray(result.discrimination, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"discrimination must be square, got shape {D.shape}")

    n = D.shape[0]
    pos = result.positions_mm if positions_mm is None else positions_mm
    labels = _position_labels(pos, n)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(D, vmin=0.0, vmax=1.0, aspect="equal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Discrimination")

    ax.set_xlabel("Alternative position j")
    ax.set_ylabel("True position i")
    ax.set_title(title)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    if annotate:
        for i in range(n):
            for j in range(n):
                val = D[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val < 0.4 else "black",
                )

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_leakage_heatmap(
    result: DiscriminationResult,
    *,
    positions_mm: FloatArray | None = None,
    annotate: bool = True,
    scale: str = "raw",  # "raw" | "x1e3"
    title: str = "Leakage Matrix",
    figsize: tuple[float, float] = (8.0, 7.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the leakage matrix L[i, j] = |x_hat[j]| / |x_hat[i]|."""
    L = np.asarray(result.leakage, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"leakage must be square, got shape {L.shape}")

    n = L.shape[0]
    pos = result.positions_mm if positions_mm is None else positions_mm
    labels = _position_labels(pos, n)

    if scale == "raw":
        L_plot = L
        cbar_label = "Leakage"
        fmt = "{:.2f}"
    elif scale == "x1e3":
        L_plot = L * 1e3
        cbar_label = "Leakage × 10^-3"
        fmt = "{:.2f}"
    else:
        raise ValueError(f"Unknown scale: {scale}")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(L_plot, aspect="equal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel("Alternative position j")
    ax.set_ylabel("True position i")
    ax.set_title(title)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    if annotate:
        for i in range(n):
            for j in range(n):
                val = L_plot[i, j]
                ax.text(
                    j,
                    i,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val > np.max(L_plot) * 0.6 else "black",
                )

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def close_figure(fig: plt.Figure) -> None:
    """Close a matplotlib figure explicitly."""
    plt.close(fig)