"""
Timeline visualization utility for the RF Environment (SIH26055).

This module provides functions to inspect and plot the RF simulation timeline,
clearly distinguishing between hidden ground truth emitter activity, ESM receiver
scan dwells, and resulting detection outcomes (HIT, FALSE_ALARM, MISS).
"""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from environment.rf_environment import RFEnvironment
from environment.types import DetectionResult


def plot_rf_timeline(
    env: RFEnvironment,
    time_range: Tuple[int, int] = (0, 200),
    save_path: Optional[str] = None,
    title: str = "SIH26055 RF Spectrum & Receiver Scan Timeline",
    show: bool = False,
) -> plt.Figure:
    """
    Plot the RF spectrum timeline comparing ground truth activity against receiver scans.
    
    Args:
        env: Simulated RFEnvironment instance after running steps.
        time_range: (t_min, t_max) interval to display on the horizontal axis.
        save_path: Optional file path to save the generated figure.
        title: Plot title.
        show: If True, display interactive window.
        
    Returns:
        plt.Figure: The generated matplotlib figure object.
    """
    t_min, t_max = time_range
    num_bands = env.num_bands
    num_slots = t_max - t_min

    if num_slots <= 0:
        raise ValueError(f"Invalid time range: {time_range}")

    # Build ground truth matrix: 0=idle, 1=transmitting but not observable, 2=transmitting and observable
    gt_matrix = np.zeros((num_bands, num_slots), dtype=int)
    for idx, t in enumerate(range(t_min, t_max)):
        for b in range(num_bands):
            gt_slot = env.emitter_registry.get_ground_truth_slot(t, b)
            if gt_slot.is_transmitting:
                if gt_slot.is_observable:
                    gt_matrix[b, idx] = 2  # Observable transmission
                else:
                    gt_matrix[b, idx] = 1  # Hidden / Unobservable transmission

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    fig.patch.set_facecolor("#121826")
    ax.set_facecolor("#1a2234")

    # Custom colormap for ground truth: Dark slate -> Dim orange (unobservable) -> Vibrant Cyan (observable)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#1a2234", "#4a3c28", "#00d2ff"])

    # Display ground truth raster
    ax.imshow(
        gt_matrix,
        aspect="auto",
        origin="lower",
        extent=[t_min, t_max, -0.5, num_bands - 0.5],
        cmap=cmap,
        interpolation="nearest",
        alpha=0.6,
    )

    # Overlay receiver dwells and detections
    hits_x, hits_y = [], []
    fa_x, fa_y = [], []
    miss_x, miss_y = [], []

    for dwell in env.episode_dwell_history:
        # Check if dwell overlaps with time_range
        if dwell.end_time < t_min or dwell.start_time >= t_max:
            continue

        d_start = max(dwell.start_time, t_min)
        d_end = min(dwell.end_time + 1, t_max)
        band = dwell.scanned_band

        # Draw receiver scan bounding box
        rect_color = "#38bdf8"
        ax.add_patch(
            plt.Rectangle(
                (d_start, band - 0.4),
                d_end - d_start,
                0.8,
                fill=False,
                edgecolor=rect_color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
            )
        )

        # Mark action detection result at dwell midpoint
        mid_x = (d_start + d_end) / 2.0
        if dwell.overall_result == DetectionResult.HIT:
            hits_x.append(mid_x)
            hits_y.append(band)
        elif dwell.overall_result == DetectionResult.FALSE_ALARM:
            fa_x.append(mid_x)
            fa_y.append(band)
        elif dwell.overall_result == DetectionResult.MISS:
            miss_x.append(mid_x)
            miss_y.append(band)

    # Plot detection markers
    if hits_x:
        ax.scatter(hits_x, hits_y, color="#22c55e", marker="*", s=140, edgecolors="white", linewidths=1.0, label="HIT (True Positive)", zorder=5)
    if fa_x:
        ax.scatter(fa_x, fa_y, color="#ef4444", marker="^", s=90, edgecolors="white", linewidths=1.0, label="FALSE ALARM (False Positive)", zorder=5)
    if miss_x:
        ax.scatter(miss_x, miss_y, color="#94a3b8", marker="x", s=60, linewidths=1.2, label="MISS / Inactive", zorder=4)

    # Formatting axes and labels
    ax.set_yticks(range(num_bands))
    ax.set_yticklabels([f"B{b}" for b in range(num_bands)], color="#e2e8f0", fontsize=9)
    ax.set_xlabel("Simulation Time Slot ($t$)", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency Band", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_title(title, color="#f8fafc", fontsize=13, fontweight="bold", pad=12)

    ax.tick_params(colors="#cbd5e1", which="both")
    ax.grid(color="#334155", linestyle=":", linewidth=0.5, alpha=0.7)

    # Legend creation
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#00d2ff", edgecolor="none", alpha=0.6, label="Ground Truth: Observable Signal"),
        Patch(facecolor="#4a3c28", edgecolor="none", alpha=0.6, label="Ground Truth: Unobservable/Sidelobe"),
        Line2D([0], [0], color="#38bdf8", linestyle="--", linewidth=1.5, label="ESM Receiver Scan Dwell"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#22c55e", markersize=12, label="Receiver Result: HIT"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#ef4444", markersize=9, label="Receiver Result: FALSE ALARM"),
        Line2D([0], [0], marker="x", color="#94a3b8", markersize=8, linestyle="None", label="Receiver Result: MISS"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f1f5f9", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")

    if show:
        plt.show()

    return fig
