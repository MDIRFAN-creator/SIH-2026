"""
Baseline scan pattern visualization for SIH26055 (Phase 2).

Plots the open-loop sequential frequency sweep against time, clearly showing the
fixed sawtooth scan pattern and overlaying detection events (HIT, FALSE_ALARM, MISS).
"""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from environment.types import DetectionResult
from runners.episode_runner import EpisodeResult


def plot_baseline_scan_pattern(
    episode_result: EpisodeResult,
    time_range: Tuple[int, int] = (0, 200),
    save_path: Optional[str] = None,
    title: str = "Open-Loop Baseline: Fixed Sequential Frequency Sweep Pattern",
    show: bool = False,
) -> plt.Figure:
    """
    Plot the open-loop frequency sweep showing time vs scanned band.
    
    Args:
        episode_result: Recorded EpisodeResult from EpisodeRunner.
        time_range: (t_min, t_max) interval to display.
        save_path: Optional output file path.
        title: Plot title.
        show: If True, display interactive window.
        
    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    t_min, t_max = time_range
    num_bands = episode_result.environment_config.num_bands if episode_result.environment_config else 20

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Filter steps within time_range
    step_times = []
    step_bands = []
    hits_x, hits_y = [], []
    fa_x, fa_y = [], []
    miss_x, miss_y = [], []

    for rec in episode_result.step_records:
        if rec.end_time < t_min or rec.start_time >= t_max:
            continue

        d_start = max(rec.start_time, t_min)
        d_end = min(rec.end_time + 1, t_max)
        band = rec.action.frequency_band

        # Add points for sawtooth line
        step_times.extend([d_start, d_end])
        step_bands.extend([band, band])

        # Draw dwell interval line
        ax.hlines(
            y=band,
            xmin=d_start,
            xmax=d_end,
            colors="#38bdf8",
            linewidth=3.0,
            alpha=0.9,
            zorder=3,
        )

        mid_x = (d_start + d_end) / 2.0
        res = rec.observation.result
        if res == DetectionResult.HIT:
            hits_x.append(mid_x)
            hits_y.append(band)
        elif res == DetectionResult.FALSE_ALARM:
            fa_x.append(mid_x)
            fa_y.append(band)
        elif res == DetectionResult.MISS:
            miss_x.append(mid_x)
            miss_y.append(band)

    # Plot connecting sweep trendline (stepped/sawtooth)
    if step_times:
        ax.plot(
            step_times,
            step_bands,
            color="#0284c7",
            linestyle=":",
            linewidth=1.2,
            alpha=0.6,
            label="Sweep Trajectory",
            zorder=2,
        )

    # Plot detection markers
    if hits_x:
        ax.scatter(
            hits_x, hits_y,
            color="#22c55e", marker="*", s=160,
            edgecolors="white", linewidths=1.2,
            label="Receiver Result: HIT", zorder=5,
        )
    if fa_x:
        ax.scatter(
            fa_x, fa_y,
            color="#ef4444", marker="^", s=100,
            edgecolors="white", linewidths=1.0,
            label="Receiver Result: FALSE ALARM", zorder=5,
        )
    if miss_x:
        ax.scatter(
            miss_x, miss_y,
            color="#94a3b8", marker="x", s=50,
            linewidths=1.2, alpha=0.7,
            label="Receiver Result: MISS / Quiet", zorder=4,
        )

    # Formatting axes
    ax.set_yticks(range(num_bands))
    ax.set_yticklabels([f"B{b}" for b in range(num_bands)], color="#e2e8f0", fontsize=9)
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-0.5, num_bands - 0.5)

    ax.set_xlabel("Simulation Time Slot ($t$)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency Band", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_title(title, color="#f8fafc", fontsize=13, fontweight="bold", pad=12)

    ax.tick_params(colors="#cbd5e1", which="both")
    ax.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.legend(
        loc="upper right",
        facecolor="#0f172a",
        edgecolor="#475569",
        labelcolor="#f1f5f9",
        fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")

    if show:
        plt.show()

    return fig
