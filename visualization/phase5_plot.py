"""
Phase 5 Visualizations for Reinforcement Learning (SIH26055 Phase 5):
1. PPO Training Curves (Reward, Policy Loss, Value Loss, Entropy).
2. PPO Exploration Diagnostics (Band Selection Distribution & Shannon Entropy).
3. 4-Way Benchmark Summary (Open-Loop vs XGBoost vs Hardened LinUCB vs PPO).
4. Dynamic Frequency-Hopping Adaptation Timeline (4 Schedulers).
5. 4-Panel Scan Trajectory Comparison.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from environment.types import DetectionResult
from runners.episode_runner import EpisodeResult


def plot_ppo_training_curves(
    training_history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot PPO training learning curves (Reward, Losses, Entropy).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0F141C")

    episodes = np.arange(1, len(training_history["rewards"]) + 1)
    rewards = np.array(training_history["rewards"])

    # 1. Episode Reward
    ax = axes[0, 0]
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax.plot(episodes, rewards, color="#388E3C", alpha=0.35, label="Raw Episode Reward")
    if len(rewards) >= 10:
        window = min(15, len(rewards) // 3)
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1:], smooth, color="#00E676", linewidth=2.5, label=f"Smoothed ({window}-ep MA)")
    ax.set_title("PPO Episode Return / Cumulative Reward", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax.set_xlabel("Training Episode", fontsize=10, color="#B0BEC5")
    ax.set_ylabel("Reward", fontsize=10, color="#B0BEC5")
    ax.legend(loc="lower right")

    # 2. Policy Loss
    ax = axes[0, 1]
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax.plot(episodes, training_history["policy_loss"], color="#FF7043", linewidth=2.0)
    ax.set_title("Clipped Surrogate Policy Loss", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax.set_xlabel("Training Episode", fontsize=10, color="#B0BEC5")
    ax.set_ylabel("Policy Loss", fontsize=10, color="#B0BEC5")

    # 3. Value Function Loss
    ax = axes[1, 0]
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax.plot(episodes, training_history["value_loss"], color="#42A5F5", linewidth=2.0)
    ax.set_title("Critic Value Function MSE Loss", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax.set_xlabel("Training Episode", fontsize=10, color="#B0BEC5")
    ax.set_ylabel("Value Loss", fontsize=10, color="#B0BEC5")

    # 4. Policy Entropy
    ax = axes[1, 1]
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    ax.plot(episodes, training_history["entropy"], color="#AB47BC", linewidth=2.0)
    ax.set_title("Policy Exploration Entropy", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax.set_xlabel("Training Episode", fontsize=10, color="#B0BEC5")
    ax.set_ylabel("Entropy (nats)", fontsize=10, color="#B0BEC5")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_ppo_exploration_diagnostics(
    band_selection_counts: np.ndarray,
    entropy_history: List[float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot PPO exploration diagnostics: band selection frequency & entropy evolution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0F141C")

    # Left: Band selection distribution
    ax = axes[0]
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    bands = np.arange(len(band_selection_counts))
    tot = max(1, np.sum(band_selection_counts))
    fractions = band_selection_counts / tot * 100.0
    colors = plt.cm.viridis(fractions / max(1.0, np.max(fractions)))

    bars = ax.bar(bands, fractions, color=colors, edgecolor="#00E676", linewidth=1.2, alpha=0.85)
    ax.axhline(100.0 / len(bands), color="#FF5252", linestyle="--", alpha=0.7, label=f"Uniform baseline ({100.0/len(bands):.1f}%)")
    ax.set_title("PPO Band Selection Distribution (%)", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax.set_xlabel("Frequency Band", fontsize=10, color="#B0BEC5")
    ax.set_ylabel("Allocation Percentage (%)", fontsize=10, color="#B0BEC5")
    ax.set_xticks(bands)
    ax.set_xticklabels([f"B{b}" for b in bands], fontsize=8)
    ax.legend(loc="upper right")

    # Right: Entropy over training
    ax = axes[1]
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    eps = np.arange(1, len(entropy_history) + 1)
    ax.plot(eps, entropy_history, color="#FFB300", linewidth=2.2, label="PPO Policy Entropy")
    max_h = np.log(len(band_selection_counts))
    ax.axhline(max_h, color="#00E676", linestyle="--", alpha=0.7, label=f"Theoretical Max ln(20)={max_h:.2f}")
    ax.set_title("Exploration Shannon Entropy Evolution", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax.set_xlabel("Training Episode", fontsize=10, color="#B0BEC5")
    ax.set_ylabel("Shannon Entropy H (nats)", fontsize=10, color="#B0BEC5")
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_4way_benchmark_comparison(
    metrics_summary: Dict[str, Dict[str, Tuple[float, float]]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate 4-way comparative bar charts for Open-Loop, XGBoost, LinUCB, and PPO.
    """
    schedulers = ["Open-Loop Baseline", "XGBoost Adaptive", "Hardened LinUCB", "PPO Policy"]
    colors = ["#4A90E2", "#9B51E0", "#00E676", "#FF9800"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0F141C")

    # 1. Interception Rate (%)
    ax = axes[0, 0]
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    means = [metrics_summary[s]["interception_rate"][0] * 100.0 for s in schedulers]
    stds = [metrics_summary[s]["interception_rate"][1] * 100.0 for s in schedulers]
    bars = ax.bar(schedulers, means, yerr=stds, capsize=5, color=colors, edgecolor="#FFFFFF", alpha=0.85, width=0.55)
    ax.set_title("Interception Rate (%)", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax.set_ylabel("Rate (%)", fontsize=10, color="#B0BEC5")
    ax.set_ylim(0, 60)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m + s + 1.2, f"{m:.1f}%", ha="center", va="bottom", color="#FFFFFF", fontweight="bold", fontsize=9)

    # 2. Average Intercept Time (slots)
    ax = axes[0, 1]
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    means = [metrics_summary[s]["avg_intercept_time"][0] for s in schedulers]
    stds = [metrics_summary[s]["avg_intercept_time"][1] for s in schedulers]
    bars = ax.bar(schedulers, means, yerr=stds, capsize=5, color=colors, edgecolor="#FFFFFF", alpha=0.85, width=0.55)
    ax.set_title("Average Intercept Delay (Slots) [Lower is Better]", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax.set_ylabel("Slots", fontsize=10, color="#B0BEC5")
    ax.set_ylim(0, 12)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m + s + 0.3, f"{m:.2f}s", ha="center", va="bottom", color="#FFFFFF", fontweight="bold", fontsize=9)

    # 3. Dwell Efficiency (%)
    ax = axes[1, 0]
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    means = [metrics_summary[s]["dwell_efficiency"][0] * 100.0 for s in schedulers]
    stds = [metrics_summary[s]["dwell_efficiency"][1] * 100.0 for s in schedulers]
    bars = ax.bar(schedulers, means, yerr=stds, capsize=5, color=colors, edgecolor="#FFFFFF", alpha=0.85, width=0.55)
    ax.set_title("Dwell Efficiency (%)", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax.set_ylabel("Efficiency (%)", fontsize=10, color="#B0BEC5")
    ax.set_ylim(0, 80)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m + s + 1.5, f"{m:.1f}%", ha="center", va="bottom", color="#FFFFFF", fontweight="bold", fontsize=9)

    # 4. Unique Opportunities Intercepted
    ax = axes[1, 1]
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
    means = [metrics_summary[s]["unique_opportunities"][0] for s in schedulers]
    stds = [metrics_summary[s]["unique_opportunities"][1] for s in schedulers]
    bars = ax.bar(schedulers, means, yerr=stds, capsize=5, color=colors, edgecolor="#FFFFFF", alpha=0.85, width=0.55)
    ax.set_title("Unique Burst Opportunities Intercepted", fontsize=12, fontweight="bold", color="#ECEFF1")
    ax.set_ylabel("Count", fontsize=10, color="#B0BEC5")
    ax.set_ylim(0, 1100)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m + s + 20, f"{m:.0f}", ha="center", va="bottom", color="#FFFFFF", fontweight="bold", fontsize=9)

    for ax in axes.flat:
        ax.set_xticklabels(schedulers, rotation=15, ha="right", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_frequency_hopping_adaptation(
    hop_results: Dict[str, Dict[str, Tuple[int, int]]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot detection latency across dynamic hopping scenarios for 4 schedulers.
    """
    scenarios = ["t=1000, B14", "t=2000, B7", "t=3000, B18"]
    schedulers = ["Open-Loop Baseline", "XGBoost Adaptive", "Hardened LinUCB", "PPO Policy"]
    colors = ["#4A90E2", "#9B51E0", "#00E676", "#FF9800"]

    fig, ax = plt.subplots(figsize=(14, 7))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0F141C")
    ax.set_facecolor("#151D2A")
    ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")

    x = np.arange(len(scenarios))
    width = 0.20

    for i, (sched, color) in enumerate(zip(schedulers, colors)):
        latencies = [hop_results[scen][sched][1] for scen in scenarios]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, latencies, width, label=sched, color=color, alpha=0.85, edgecolor="#FFFFFF")
        for rect, lat in zip(rects, latencies):
            ax.text(rect.get_x() + rect.get_width() / 2.0, lat + 6, f"{lat}s", ha="center", va="bottom", color="#FFFFFF", fontsize=8, fontweight="bold")

    ax.set_title("Dynamic Frequency Hopping Emitter Interception Latency", fontsize=14, fontweight="bold", color="#ECEFF1", pad=12)
    ax.set_xlabel("Dynamic Hopping Scenario (Event Time & Target Band)", fontsize=11, color="#B0BEC5")
    ax.set_ylabel("Detection Latency (Time Slots After Hop)", fontsize=11, color="#B0BEC5")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_ylim(0, max(500, max([hop_results[scen][s][1] for scen in scenarios for s in schedulers]) + 50))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_4way_trajectories(
    open_loop_res: EpisodeResult,
    xgboost_res: EpisodeResult,
    linucb_res: EpisodeResult,
    ppo_res: EpisodeResult,
    max_time: int = 200,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate 4-row stacked comparison of frequency scanning trajectories.
    """
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True, sharey=True)
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0F141C")

    schedulers_data = [
        ("Phase 2: Open-Loop Baseline (Cyclic Sweep)", open_loop_res, axes[0], "#4A90E2"),
        ("Phase 3: XGBoost + Action Optimizer (Supervised)", xgboost_res, axes[1], "#9B51E0"),
        ("Phase 4: LinUCB Contextual Bandit (Online Learning)", linucb_res, axes[2], "#00E676"),
        ("Phase 5: PPO Reinforcement Learning (Policy Gradient)", ppo_res, axes[3], "#FF9800"),
    ]

    for title, res, ax, line_color in schedulers_data:
        ax.set_facecolor("#151D2A")
        ax.grid(True, linestyle=":", alpha=0.3, color="#607D8B")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#ECEFF1", pad=6)
        ax.set_ylabel("Frequency Band", fontsize=10, color="#CFD8DC")
        ax.set_yticks(range(20))
        ax.set_yticklabels([f"B{b}" for b in range(20)], fontsize=8)
        ax.set_ylim(-0.8, 19.8)

        t_points, band_points = [], []
        hit_t, hit_b = [], []
        fa_t, fa_b = [], []
        miss_t, miss_b = [], []

        for rec in res.step_records:
            if rec.start_time > max_time:
                break
            band = rec.action.frequency_band
            t_start = rec.start_time
            t_end = min(rec.end_time, max_time)

            ax.hlines(y=band, xmin=t_start, xmax=t_end, colors=line_color, linewidth=4, alpha=0.8)
            mid_t = (t_start + t_end) / 2.0
            t_points.append(mid_t)
            band_points.append(band)

            if rec.observation.result == DetectionResult.HIT:
                hit_t.append(mid_t)
                hit_b.append(band)
            elif rec.observation.result == DetectionResult.FALSE_ALARM:
                fa_t.append(mid_t)
                fa_b.append(band)
            else:
                miss_t.append(mid_t)
                miss_b.append(band)

        if t_points:
            ax.plot(t_points, band_points, linestyle=":", color="#78909C", alpha=0.4, linewidth=1.0)
        if miss_t:
            ax.scatter(miss_t, miss_b, marker="x", color="#78909C", s=30, alpha=0.6, label="Result: MISS / Quiet")
        if fa_t:
            ax.scatter(fa_t, fa_b, marker="^", color="#FF5252", s=80, edgecolors="#FFFFFF", linewidth=1.2, zorder=5, label="Result: FALSE ALARM")
        if hit_t:
            ax.scatter(hit_t, hit_b, marker="*", color="#00E676", s=130, edgecolors="#FFFFFF", linewidth=1.2, zorder=6, label="Result: HIT")

        ax.legend(loc="upper right", framealpha=0.8, fontsize=8)

    axes[-1].set_xlabel("Simulation Time Slot (t)", fontsize=11, color="#ECEFF1")
    axes[-1].set_xlim(0, max_time)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
