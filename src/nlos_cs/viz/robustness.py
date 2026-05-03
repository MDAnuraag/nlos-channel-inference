"""Visualisation utilities for robustness experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directory for an output path if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _position_labels(positions_mm: FloatArray | None, n: int) -> list[str]:
    """Create axis labels from positions if available, else indices."""
    if positions_mm is None:
        return [str(i) for i in range(n)]

    pos = np.asarray(positions_mm, dtype=np.float64)
    if pos.ndim != 1 or pos.shape[0] != n:
        raise ValueError(f"positions_mm must have shape ({n},), got {pos.shape}")

    labels: list[str] = []
    for p in pos:
        labels.append(f"{int(p)}" if float(p).is_integer() else f"{p:g}")
    return labels


def _noise_labels(noise_levels: FloatArray) -> list[str]:
    """Create readable labels for noise fractions."""
    noise = np.asarray(noise_levels, dtype=np.float64)
    if noise.ndim != 1:
        raise ValueError(f"noise_levels must be 1D, got shape {noise.shape}")
    return [f"{100.0 * x:.0f}%" for x in noise]


def plot_success_rate_heatmap(
    success_rate_matrix: FloatArray,
    *,
    noise_levels: FloatArray,
    positions_mm: FloatArray | None = None,
    annotate: bool = True,
    title: str = "Success Rate Heatmap",
    figsize: tuple[float, float] = (10.0, 7.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot success rate versus true position and noise level."""
    S = np.asarray(success_rate_matrix, dtype=np.float64)
    noise = np.asarray(noise_levels, dtype=np.float64)

    if S.ndim != 2:
        raise ValueError(f"success_rate_matrix must be 2D, got shape {S.shape}")
    if noise.ndim != 1 or noise.shape[0] != S.shape[1]:
        raise ValueError(
            f"noise_levels must have shape ({S.shape[1]},), got {noise.shape}"
        )

    n_positions, _ = S.shape
    pos_labels = _position_labels(positions_mm, n_positions)
    noise_labels = _noise_labels(noise)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(S, vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Success rate")

    ax.set_xlabel("Noise level")
    ax.set_ylabel("True position")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(noise_labels)))
    ax.set_xticklabels(noise_labels)
    ax.set_yticks(np.arange(n_positions))
    ax.set_yticklabels(pos_labels)

    if annotate:
        for i in range(n_positions):
            for j in range(len(noise_labels)):
                val = S[i, j]
                ax.text(
                    j,
                    i,
                    f"{100.0 * val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val < 0.5 else "black",
                )

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_peak_margin_heatmap(
    mean_peak_margin_matrix: FloatArray,
    *,
    noise_levels: FloatArray,
    positions_mm: FloatArray | None = None,
    annotate: bool = True,
    title: str = "Mean Peak Margin Heatmap",
    figsize: tuple[float, float] = (10.0, 7.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot mean peak margin versus true position and noise level."""
    M = np.asarray(mean_peak_margin_matrix, dtype=np.float64)
    noise = np.asarray(noise_levels, dtype=np.float64)

    if M.ndim != 2:
        raise ValueError(f"mean_peak_margin_matrix must be 2D, got shape {M.shape}")
    if noise.ndim != 1 or noise.shape[0] != M.shape[1]:
        raise ValueError(
            f"noise_levels must have shape ({M.shape[1]},), got {noise.shape}"
        )

    n_positions, _ = M.shape
    pos_labels = _position_labels(positions_mm, n_positions)
    noise_labels = _noise_labels(noise)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean peak margin")

    ax.set_xlabel("Noise level")
    ax.set_ylabel("True position")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(noise_labels)))
    ax.set_xticklabels(noise_labels)
    ax.set_yticks(np.arange(n_positions))
    ax.set_yticklabels(pos_labels)

    if annotate:
        vmax = float(np.max(np.abs(M))) if M.size > 0 else 0.0
        for i in range(n_positions):
            for j in range(len(noise_labels)):
                val = M[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if vmax > 0 and abs(val) > 0.6 * vmax else "black",
                )

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_success_rate_curves(
    success_rate_matrix: FloatArray,
    *,
    noise_levels: FloatArray,
    positions_mm: FloatArray | None = None,
    title: str = "Success Rate vs Noise Level",
    figsize: tuple[float, float] = (10.0, 6.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one success-rate curve per position across noise levels."""
    S = np.asarray(success_rate_matrix, dtype=np.float64)
    noise = np.asarray(noise_levels, dtype=np.float64)

    if S.ndim != 2:
        raise ValueError(f"success_rate_matrix must be 2D, got shape {S.shape}")
    if noise.ndim != 1 or noise.shape[0] != S.shape[1]:
        raise ValueError(
            f"noise_levels must have shape ({S.shape[1]},), got {noise.shape}"
        )

    n_positions, _ = S.shape
    pos_labels = _position_labels(positions_mm, n_positions)

    fig, ax = plt.subplots(figsize=figsize)
    x = 100.0 * noise

    for i in range(n_positions):
        ax.plot(
            x,
            100.0 * S[i],
            marker="o",
            linewidth=1.8,
            label=pos_labels[i],
        )

    ax.set_xlabel("Noise level (% of RMS)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(title)
    ax.set_ylim(0.0, 105.0)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Position", fontsize=8, title_fontsize=9, ncol=2)

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_mean_peak_margin_curves(
    mean_peak_margin_matrix: FloatArray,
    *,
    noise_levels: FloatArray,
    positions_mm: FloatArray | None = None,
    title: str = "Mean Peak Margin vs Noise Level",
    figsize: tuple[float, float] = (10.0, 6.0),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one mean peak-margin curve per position across noise levels."""
    M = np.asarray(mean_peak_margin_matrix, dtype=np.float64)
    noise = np.asarray(noise_levels, dtype=np.float64)

    if M.ndim != 2:
        raise ValueError(f"mean_peak_margin_matrix must be 2D, got shape {M.shape}")
    if noise.ndim != 1 or noise.shape[0] != M.shape[1]:
        raise ValueError(
            f"noise_levels must have shape ({M.shape[1]},), got {noise.shape}"
        )

    n_positions, _ = M.shape
    pos_labels = _position_labels(positions_mm, n_positions)

    fig, ax = plt.subplots(figsize=figsize)
    x = 100.0 * noise

    for i in range(n_positions):
        ax.plot(
            x,
            M[i],
            marker="o",
            linewidth=1.8,
            label=pos_labels[i],
        )

    ax.set_xlabel("Noise level (% of RMS)")
    ax.set_ylabel("Mean peak margin")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Position", fontsize=8, title_fontsize=9, ncol=2)

    fig.tight_layout()

    if output_path is not None:
        out = _ensure_parent_dir(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def close_figure(fig: plt.Figure) -> None:
    """Close a matplotlib figure explicitly."""
    plt.close(fig)