"""
Phase 4 Visualizations (SIH26055):
1. 3-Panel Trajectory Comparison (Open-Loop vs XGBoost vs LinUCB).
2. LinUCB Online Diagnostics (Uncertainty, Cumulative Reward, Arm Pulls).
3. Tri-Scheduler Benchmark Summary Bar Chart.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from environment.types import DetectionResult
from runners.episode_runner import EpisodeResult
from schedulers.linucb_scheduler import LinUCBScheduler


def plot_tri_scheduler_trajectories(
    open_loop_result: EpisodeResult,
    xgboost_result: EpisodeResult,
    linucb_result: EpisodeResult,
    max_time: int = 200,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate 3-row stacked comparison of frequency scanning trajectories.
    """
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True, sharey=True)
    plt.style.use("dark_background")

    schedulers_data = [
        ("Phase 2: Open-Loop Baseline (Cyclic Sweep)", open_loop_result, axes[0], "#4A90E2"),
        ("Phase 3: XGBoost + Action Optimizer (Supervised)", xgboost_result, axes[1], "#9B51E0"),
        ("Phase 4: LinUCB Contextual Bandit (Online Learning)", linucb_result, axes[2], "#00E676"),
    ]

    for title, res, ax, line_color in schedulers_data:
        ax.set_facecolor("#121824")
        ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
        ax.set_title(title, fontsize=13, fontweight="bold", color="#ECEFF1", pad=8)
        ax.set_ylabel("Frequency Band", fontsize=11, color="#CFD8DC")
        ax.set_yticks(range(20))
        ax.set_yticklabels([f"B{b}" for b in range(20)], fontsize=8)
        ax.set_ylim(-0.8, 19.8)

        t_points = []
        band_points = []
        hit_t, hit_b = [], []
        fa_t, fa_b = [], []
        miss_t, miss_b = [], []

        for rec in res.step_records:
            if rec.start_time > max_time:
                break
            band = rec.action.frequency_band
            dwell = rec.action.dwell_time
            t_start = rec.start_time
            t_end = min(rec.end_time, max_time)

            ax.hlines(y=band, xmin=t_start, xmax=t_end, colors=line_color, linewidth=4, alpha=0.8)
            t_points.extend([t_start, t_end])
            band_points.extend([band, band])

            res_val = rec.observation.result
            t_mid = (t_start + t_end) / 2.0
            if res_val == DetectionResult.HIT:
                hit_t.append(t_mid)
                hit_b.append(band)
            elif res_val == DetectionResult.FALSE_ALARM:
                fa_t.append(t_mid)
                fa_b.append(band)
            else:
                miss_t.append(t_mid)
                miss_b.append(band)

        if t_points:
            ax.plot(t_points, band_points, linestyle=":", color=line_color, alpha=0.4, linewidth=1.2)

        if miss_t:
            ax.scatter(miss_t, miss_b, color="#78909C", marker="x", s=40, alpha=0.6, label="Result: MISS / Quiet")
        if fa_t:
            ax.scatter(fa_t, fa_b, color="#FF5252", marker="^", s=90, edgecolors="#FFF", linewidth=1.2, label="Result: FALSE ALARM", zorder=4)
        if hit_t:
            ax.scatter(hit_t, hit_b, color="#00E676", marker="*", s=140, edgecolors="#FFF", linewidth=1.2, label="Result: HIT", zorder=5)

        ax.legend(loc="upper right", fontsize=9, framealpha=0.8, facecolor="#1E2738")

    axes[2].set_xlabel("Simulation Time Slot ($t$)", fontsize=12, fontweight="bold", color="#ECEFF1")
    axes[2].set_xlim(0, max_time)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_linucb_diagnostics(
    scheduler: LinUCBScheduler,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot LinUCB online learning diagnostic telemetry:
    - Uncertainty evolution for candidate bands over decisions
    - Cumulative online reward growth
    - Frequency band pull count distribution
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("#0F141C")

    # 1. Cumulative Reward
    ax0 = axes[0]
    ax0.set_facecolor("#161F2E")
    ax0.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax0.set_title("LinUCB Online Cumulative Reward", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax0.set_xlabel("Scheduling Decision Step", fontsize=10, color="#CFD8DC")
    ax0.set_ylabel("Cumulative Reward", fontsize=10, color="#CFD8DC")

    decisions = range(len(scheduler.decision_history))
    cum_rewards = [d.get("cumulative_reward", 0.0) for d in scheduler.decision_history]
    ax0.plot(decisions, cum_rewards, color="#00E676", linewidth=2.0, label="LinUCB Reward")
    ax0.legend(loc="upper left", fontsize=9, facecolor="#1E2738")

    # 2. Uncertainty Decay
    ax1 = axes[1]
    ax1.set_facecolor("#161F2E")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax1.set_title(r"Selected Arm Parameter Uncertainty ($\sigma_{b^*}$)", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax1.set_xlabel("Scheduling Decision Step", fontsize=10, color="#CFD8DC")
    ax1.set_ylabel(r"Uncertainty $\sqrt{x^T A^{-1} x}$", fontsize=10, color="#CFD8DC")

    uncerts = [d.get("uncertainty", 1.0) for d in scheduler.decision_history]
    ax1.plot(decisions[:500], uncerts[:500], color="#29B6F6", linewidth=1.5, alpha=0.8, label="Decision Uncertainty")
    ax1.legend(loc="upper right", fontsize=9, facecolor="#1E2738")

    # 3. Arm Pull Distribution
    ax2 = axes[2]
    ax2.set_facecolor("#161F2E")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax2.set_title("LinUCB Frequency Band Pull Distribution", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax2.set_xlabel("Frequency Band", fontsize=10, color="#CFD8DC")
    ax2.set_ylabel("Total Pull Count", fontsize=10, color="#CFD8DC")

    pulls = scheduler.linucb.pull_counts
    bars = ax2.bar(range(len(pulls)), pulls, color="#AB47BC", edgecolor="#E1BEE7", alpha=0.85)
    ax2.set_xticks(range(20))
    ax2.set_xticklabels([f"B{b}" for b in range(20)], fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_tri_benchmark_summary(
    metrics_summary: Dict[str, Dict[str, Tuple[float, float]]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot bar chart comparisons of major metrics across Open Loop, XGBoost, and LinUCB.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0F141C")

    schedulers = ["Open-Loop", "XGBoost", "LinUCB"]
    colors = ["#4A90E2", "#9B51E0", "#00E676"]

    # 1. Interception Rate
    ax0 = axes[0, 0]
    ax0.set_facecolor("#161F2E")
    ax0.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax0.set_title("Interception Rate (%)", fontsize=12, fontweight="bold", color="#ECEFF1")
    ir_vals = [metrics_summary[s]["interception_rate"][0] * 100 for s in schedulers]
    ir_errs = [metrics_summary[s]["interception_rate"][1] * 100 for s in schedulers]
    ax0.bar(schedulers, ir_vals, yerr=ir_errs, color=colors, capsize=5, alpha=0.85)
    ax0.set_ylim(0, 100)

    # 2. Average Intercept Delay
    ax1 = axes[0, 1]
    ax1.set_facecolor("#161F2E")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax1.set_title("Average Intercept Delay (Slots)", fontsize=12, fontweight="bold", color="#ECEFF1")
    del_vals = [metrics_summary[s]["average_intercept_time"][0] for s in schedulers]
    del_errs = [metrics_summary[s]["average_intercept_time"][1] for s in schedulers]
    ax1.bar(schedulers, del_vals, yerr=del_errs, color=colors, capsize=5, alpha=0.85)

    # 3. Dwell Efficiency
    ax2 = axes[1, 0]
    ax2.set_facecolor("#161F2E")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax2.set_title("Dwell Efficiency (%)", fontsize=12, fontweight="bold", color="#ECEFF1")
    eff_vals = [metrics_summary[s]["dwell_efficiency"][0] * 100 for s in schedulers]
    eff_errs = [metrics_summary[s]["dwell_efficiency"][1] * 100 for s in schedulers]
    ax2.bar(schedulers, eff_vals, yerr=eff_errs, color=colors, capsize=5, alpha=0.85)
    ax2.set_ylim(0, 100)

    # 4. Unique Opportunities Intercepted
    ax3 = axes[1, 1]
    ax3.set_facecolor("#161F2E")
    ax3.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax3.set_title("Unique Opportunities Intercepted", fontsize=12, fontweight="bold", color="#ECEFF1")
    opp_vals = [metrics_summary[s]["unique_opportunities"][0] for s in schedulers]
    opp_errs = [metrics_summary[s]["unique_opportunities"][1] for s in schedulers]
    ax3.bar(schedulers, opp_vals, yerr=opp_errs, color=colors, capsize=5, alpha=0.85)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_before_after_hardening(
    before_stats: Dict[str, Tuple[float, float]],
    after_stats: Dict[str, Tuple[float, float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot comparative bar charts highlighting improvements from Phase 4 Hardening.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#0F141C")

    categories = ["Pre-Hardening", "Hardened LinUCB"]
    colors = ["#FF7043", "#00E676"]

    # 1. Max Consecutive Scans
    ax0 = axes[0]
    ax0.set_facecolor("#161F2E")
    ax0.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax0.set_title("Max Consecutive Scans (Anti-Camping)", fontsize=11, fontweight="bold", color="#ECEFF1")
    v_mc = [before_stats["max_consecutive"][0], after_stats["max_consecutive"][0]]
    e_mc = [before_stats["max_consecutive"][1], after_stats["max_consecutive"][1]]
    ax0.bar(categories, v_mc, yerr=e_mc, color=colors, capsize=5, alpha=0.85)
    ax0.axhline(3.0, color="#FF5252", linestyle="--", linewidth=1.5, label="Limit (3)")
    ax0.legend(loc="upper right", fontsize=8, facecolor="#1E2738")

    # 2. Interception Rate (%)
    ax1 = axes[1]
    ax1.set_facecolor("#161F2E")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax1.set_title("Interception Rate (%)", fontsize=11, fontweight="bold", color="#ECEFF1")
    v_ir = [before_stats["interception_rate"][0] * 100, after_stats["interception_rate"][0] * 100]
    e_ir = [before_stats["interception_rate"][1] * 100, after_stats["interception_rate"][1] * 100]
    ax1.bar(categories, v_ir, yerr=e_ir, color=colors, capsize=5, alpha=0.85)
    ax1.set_ylim(0, 60)

    # 3. Unique Opportunities Intercepted
    ax2 = axes[2]
    ax2.set_facecolor("#161F2E")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax2.set_title("Unique Opportunities Intercepted", fontsize=11, fontweight="bold", color="#ECEFF1")
    v_opp = [before_stats["unique_opportunities"][0], after_stats["unique_opportunities"][0]]
    e_opp = [before_stats["unique_opportunities"][1], after_stats["unique_opportunities"][1]]
    ax2.bar(categories, v_opp, yerr=e_opp, color=colors, capsize=5, alpha=0.85)

    # 4. Band Selection Entropy
    ax3 = axes[3]
    ax3.set_facecolor("#161F2E")
    ax3.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax3.set_title("Band-Selection Shannon Entropy", fontsize=11, fontweight="bold", color="#ECEFF1")
    v_ent = [before_stats["entropy"][0], after_stats["entropy"][0]]
    e_ent = [before_stats["entropy"][1], after_stats["entropy"][1]]
    ax3.bar(categories, v_ent, yerr=e_ent, color=colors, capsize=5, alpha=0.85)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_frequency_adaptation_timeline(
    adaptation_data: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot multi-scenario detection latency timeline for dynamic hopping emitters.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0F141C")
    ax.set_facecolor("#161F2E")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax.set_title("Dynamic Hopping Emitter Acquisition Latency Across Scenarios", fontsize=13, fontweight="bold", color="#ECEFF1")
    ax.set_xlabel("Scenario (Change Time & Destination Band)", fontsize=11, color="#CFD8DC")
    ax.set_ylabel("Detection Latency (Slots After Change)", fontsize=11, color="#CFD8DC")

    labels = [f"t={d['change_time']}, B{d['dest_band']}" for d in adaptation_data]
    x = np.arange(len(labels))
    width = 0.25

    ol_latencies = [d["Open-Loop"]["detection_latency"] for d in adaptation_data]
    xgb_latencies = [d["XGBoost"]["detection_latency"] for d in adaptation_data]
    lin_latencies = [d["LinUCB"]["detection_latency"] for d in adaptation_data]

    ax.bar(x - width, ol_latencies, width, label="Open-Loop Baseline", color="#4A90E2", alpha=0.85)
    ax.bar(x, xgb_latencies, width, label="XGBoost Adaptive", color="#9B51E0", alpha=0.85)
    ax.bar(x + width, lin_latencies, width, label="Hardened LinUCB", color="#00E676", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="upper right", fontsize=10, facecolor="#1E2738")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig

