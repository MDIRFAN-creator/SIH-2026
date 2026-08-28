"""
Comparison visualization module for SIH26055 (Phase 3).

Plots Open-Loop Baseline vs XGBoost Adaptive Scheduler frequency scan trajectories
and feature importance distributions.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

from environment.types import DetectionResult
from runners.episode_runner import EpisodeResult


def plot_scheduler_comparison(
    open_loop_result: EpisodeResult,
    xgboost_result: EpisodeResult,
    time_range: Tuple[int, int] = (0, 200),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot stacked subplots comparing Open Loop and XGBoost scanning trajectories over time.
    
    Args:
        open_loop_result: EpisodeResult from OpenLoopScheduler.
        xgboost_result: EpisodeResult from XGBoostScheduler.
        time_range: (t_min, t_max) simulation time interval to display.
        save_path: Optional output image file path.
        show: If True, call plt.show().
        
    Returns:
        plt.Figure: The created figure.
    """
    t_min, t_max = time_range
    num_bands = (
        open_loop_result.environment_config.num_bands
        if open_loop_result.environment_config
        else 20
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), dpi=150, sharex=True)
    fig.patch.set_facecolor("#0f172a")

    for ax, res_obj, title, sweep_color in [
        (ax1, open_loop_result, "Phase 2: Open-Loop Baseline (Cyclic Sweep)", "#38bdf8"),
        (ax2, xgboost_result, "Phase 3: XGBoost + Action Optimizer (Adaptive)", "#a855f7"),
    ]:
        ax.set_facecolor("#1e293b")

        step_times = []
        step_bands = []
        hits_x, hits_y = [], []
        fa_x, fa_y = [], []
        miss_x, miss_y = [], []

        for rec in res_obj.step_records:
            if rec.end_time < t_min or rec.start_time >= t_max:
                continue

            d_start = max(rec.start_time, t_min)
            d_end = min(rec.end_time + 1, t_max)
            band = rec.action.frequency_band

            step_times.extend([d_start, d_end])
            step_bands.extend([band, band])

            # Draw dwell horizontal segment
            ax.hlines(
                y=band,
                xmin=d_start,
                xmax=d_end,
                colors=sweep_color,
                linewidth=3.2,
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

        # Plot connecting scan pattern
        if step_times:
            ax.plot(
                step_times,
                step_bands,
                color=sweep_color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.5,
                zorder=2,
            )

        # Plot detection markers
        if hits_x:
            ax.scatter(
                hits_x, hits_y,
                color="#22c55e", marker="*", s=160,
                edgecolors="white", linewidths=1.2,
                label="Result: HIT", zorder=5,
            )
        if fa_x:
            ax.scatter(
                fa_x, fa_y,
                color="#ef4444", marker="^", s=100,
                edgecolors="white", linewidths=1.0,
                label="Result: FALSE ALARM", zorder=5,
            )
        if miss_x:
            ax.scatter(
                miss_x, miss_y,
                color="#94a3b8", marker="x", s=45,
                linewidths=1.2, alpha=0.6,
                label="Result: MISS / Quiet", zorder=4,
            )

        ax.set_yticks(range(num_bands))
        ax.set_yticklabels([f"B{b}" for b in range(num_bands)], color="#e2e8f0", fontsize=8)
        ax.set_ylim(-0.5, num_bands - 0.5)
        ax.set_ylabel("Frequency Band", color="#f8fafc", fontsize=10, fontweight="bold")
        ax.set_title(title, color="#f8fafc", fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors="#cbd5e1", which="both")
        ax.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(
            loc="upper right",
            facecolor="#0f172a",
            edgecolor="#475569",
            labelcolor="#f1f5f9",
            fontsize=8,
            framealpha=0.9,
        )

    ax2.set_xlim(t_min, t_max)
    ax2.set_xlabel("Simulation Time Slot ($t$)", color="#f8fafc", fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(
            save_path,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig


def plot_feature_importances(
    importances: Dict[str, float],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot bar chart of XGBoost feature importances.
    """
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    scores = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, align="center", color="#38bdf8", edgecolor="white", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color="#e2e8f0", fontsize=9)
    ax.invert_yaxis()  # top feature on top

    ax.set_xlabel("Feature Importance (Gain)", color="#f8fafc", fontsize=10, fontweight="bold")
    ax.set_title("XGBoost Band Predictor: Feature Importance Ranking", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#cbd5e1", which="both")
    ax.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6, axis="x")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")

    if show:
        plt.show()

    return fig
