"""
Phase 6 Visualization Utilities for SIH26055.

Generates presentation-quality figures for:
1. Six-way head-to-head benchmark comparison (Open-Loop vs XGBoost vs LinUCB vs Original PPO vs Hardened PPO vs Hybrid).
2. Six-way time-series scan trajectory raster plots.
3. Dynamic frequency-hopping acquisition latencies.
4. Hybrid exploration vs exploitation mode progression.
5. Hybrid arbitration weights and component contribution diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np


# Styling configuration
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})

SCHEDULER_COLORS_6WAY = {
    "Open-Loop Baseline": "#7f7f7f",       # Grey
    "XGBoost Adaptive": "#2ca02c",         # Green
    "Hardened LinUCB": "#1f77b4",          # Blue
    "Original PPO Baseline": "#d62728",    # Red
    "Hardened PPO": "#ff7f0e",             # Orange
    "Hybrid Adaptive Scheduler": "#9467bd",# Purple
}


def plot_6way_benchmark_comparison(
    summary_stats: Dict[str, Dict[str, Tuple[float, float]]],
    output_path: str = "phase6_sixway_comparison.png",
) -> None:
    """
    Plot 6-panel bar chart comparison across all 6 schedulers.
    """
    metrics = [
        ("interception_rate", "Interception Rate (%)", "%", 100.0),
        ("successful_interceptions", "Unique Opps Intercepted", " opps", 1.0),
        ("average_intercept_time", "Average Intercept Delay", " slots", 1.0),
        ("scenario_ttfd", "Scenario TTFD", " slots", 1.0),
        ("dwell_efficiency", "Dwell Efficiency (%)", "%", 100.0),
        ("tp_count", "Total TP Detections", " slots", 1.0),
    ]

    schedulers = list(summary_stats.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric_key, title, unit, scale) in enumerate(metrics):
        ax = axes[idx]
        means = []
        stds = []
        colors = []

        for name in schedulers:
            stats = summary_stats[name].get(metric_key, (0.0, 0.0))
            means.append(stats[0] * scale)
            stds.append(stats[1] * scale)
            colors.append(SCHEDULER_COLORS_6WAY.get(name, "#333333"))

        x = np.arange(len(schedulers))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=1.0)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" ", "\n") for s in schedulers], fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar, m in zip(bars, means):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02 * max(means),
                f"{m:.1f}{unit}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.suptitle("SIH26055 — Six-Way Head-to-Head Benchmark Evaluation (N=10 Unseen Test Seeds)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_6way_trajectories(
    step_records_dict: Dict[str, List[Any]],
    num_bands: int = 20,
    max_time: int = 200,
    output_path: str = "phase6_hybrid_trajectory_comparison.png",
) -> None:
    """
    Plot 6-panel time-frequency raster plot comparing all 6 schedulers over t=0..max_time.
    """
    fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True, sharey=True)
    schedulers = list(step_records_dict.keys())

    for idx, (name, records) in enumerate(step_records_dict.items()):
        ax = axes[idx]
        color = SCHEDULER_COLORS_6WAY.get(name, "#333333")

        times = []
        bands = []
        hits_t = []
        hits_b = []

        for r in records:
            if r.start_time > max_time:
                break
            band = r.action.frequency_band
            for t_slot in range(r.start_time, min(max_time, r.end_time + 1)):
                times.append(t_slot)
                bands.append(band)

            if hasattr(r, "observation") and r.observation.result.value == "HIT":
                hits_t.append(r.start_time)
                hits_b.append(band)

        ax.scatter(times, bands, color=color, s=8, alpha=0.6, label=f"{name} Scans")
        if hits_t:
            ax.scatter(hits_t, hits_b, color="gold", edgecolor="black", s=30, marker="*", label="HIT Detections", zorder=5)

        ax.set_ylabel("Band", fontweight="bold")
        ax.set_title(f"{name}", fontsize=11, fontweight="bold", loc="left")
        ax.set_ylim(-0.5, num_bands - 0.5)
        ax.set_yticks([0, 5, 10, 15, 19])
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[-1].set_xlabel("Simulation Time Slot (t)", fontweight="bold")
    axes[-1].set_xlim(0, max_time)
    plt.suptitle(f"Time-Frequency Scanning Raster Trajectories (t=0..{max_time})", fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_6way_frequency_hopping_adaptation(
    hopping_latencies: Dict[str, List[float]],
    output_path: str = "phase6_hybrid_frequency_hopping.png",
) -> None:
    """
    Plot bar chart of dynamic frequency-hopping discovery latency across schedulers.
    """
    scenarios = ["Hop 1 (t=1000 -> B14)", "Hop 2 (t=2000 -> B7)", "Hop 3 (t=3000 -> B18)"]
    schedulers = list(hopping_latencies.keys())

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(scenarios))
    width = 0.13

    for i, name in enumerate(schedulers):
        latencies = [min(1000.0, val) for val in hopping_latencies[name]]
        color = SCHEDULER_COLORS_6WAY.get(name, "#333333")
        offset = (i - len(schedulers) / 2.0 + 0.5) * width
        rects = ax.bar(x + offset, latencies, width, label=name, color=color, alpha=0.85, edgecolor="black")

        for rect, lat in zip(rects, latencies):
            height = rect.get_height()
            label_text = f"{int(lat)}s" if lat < 1000 else "FAIL (>1000s)"
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + 15,
                label_text,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
                fontweight="bold",
            )

    ax.set_ylabel("Discovery Latency (Slots to First Detection)", fontweight="bold")
    ax.set_title("Dynamic Frequency-Hopping Emitter Acquisition Latency Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontweight="bold")
    ax.set_ylim(0, 1150)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_hybrid_exploration_exploitation(
    step_logs: List[Any],
    output_path: str = "phase6_hybrid_exploration_exploitation.png",
) -> None:
    """
    Plot time series showing hybrid decision mode progression and cumulative entropy.
    """
    times = [log.current_time for log in step_logs[:500]]
    modes = [log.mode for log in step_logs[:500]]

    mode_map = {"COLD_START": 0, "EXPLORATION": 1, "ADAPTATION": 2, "EXPLOITATION": 3}
    mode_vals = [mode_map.get(m, 1) for m in modes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Operational Mode Progression
    ax1.step(times, mode_vals, where="post", color="#9467bd", linewidth=1.5)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["COLD START", "EXPLORATION", "ADAPTATION", "EXPLOITATION"], fontweight="bold")
    ax1.set_ylabel("Decision Mode", fontweight="bold")
    ax1.set_title("Hybrid Scheduler Operational Mode Transitions (t=0..500)", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Selected Band vs Model Probabilities
    xgb_p = [log.xgb_proba for log in step_logs[:500]]
    ppo_p = [log.ppo_proba for log in step_logs[:500]]
    lin_u = [log.linucb_uncertainty for log in step_logs[:500]]

    ax2.plot(times, xgb_p, label="XGBoost Predicted Prob", color="#2ca02c", alpha=0.7)
    ax2.plot(times, ppo_p, label="PPO Marginal Prob", color="#ff7f0e", alpha=0.7)
    ax2.plot(times, lin_u, label="LinUCB Uncertainty", color="#1f77b4", alpha=0.7, linestyle="--")

    ax2.set_xlabel("Simulation Time Slot (t)", fontweight="bold")
    ax2.set_ylabel("Signal Strength / Uncertainty", fontweight="bold")
    ax2.set_title("Component Signal Tracking & Uncertainty Evolution", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_hybrid_arbitration_diagnostics(
    step_logs: List[Any],
    output_path: str = "phase6_hybrid_arbitration_diagnostics.png",
) -> None:
    """
    Plot stacked area / component weight progression over time.
    """
    times = [log.current_time for log in step_logs[:300]]
    w_xgb = [log.weights.get("xgb", 0.0) for log in step_logs[:300]]
    w_ppo = [log.weights.get("ppo", 0.0) for log in step_logs[:300]]
    w_lin = [log.weights.get("linucb", 0.0) for log in step_logs[:300]]
    w_exp = [log.weights.get("explore", 0.0) for log in step_logs[:300]]
    w_sta = [log.weights.get("staleness", 0.0) for log in step_logs[:300]]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(
        times,
        w_xgb,
        w_ppo,
        w_lin,
        w_exp,
        w_sta,
        labels=["XGBoost Weight", "PPO Weight", "LinUCB Weight", "Exploration Weight", "Staleness Weight"],
        colors=["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b"],
        alpha=0.85,
    )
    ax.set_xlabel("Simulation Time Slot (t)", fontweight="bold")
    ax.set_ylabel("Arbitration Weight Contribution", fontweight="bold")
    ax.set_title("Dynamic Component Weight Arbitration Over Time (t=0..300)", fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
