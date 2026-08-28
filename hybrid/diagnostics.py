"""
Diagnostics and telemetry recording for Hybrid Scheduler (Phase 6).

Maintains decision logs and calculates exploration/exploitation mode breakdowns
without accessing any hidden environment ground truth.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class HybridStepLog:
    """Telemetry record for a single scheduling decision."""
    step_index: int
    current_time: int
    selected_band: int
    selected_dwell: int
    mode: str
    is_exploration: bool
    is_exploitation: bool
    is_adaptation: bool
    composite_score: float
    xgb_proba: float
    ppo_proba: float
    linucb_mean: float
    linucb_uncertainty: float
    staleness: float
    weights: Dict[str, float]


class HybridDiagnostics:
    """
    Tracks and summarizes hybrid decision metrics across a simulation episode.
    """

    def __init__(self, num_bands: int = 20) -> None:
        self.num_bands = num_bands
        self.step_logs: List[HybridStepLog] = []
        self.band_counts: np.ndarray = np.zeros(num_bands, dtype=np.int64)
        self.mode_counts: Dict[str, int] = {
            "COLD_START": 0,
            "EXPLOITATION": 0,
            "EXPLORATION": 0,
            "ADAPTATION": 0,
        }

    def reset(self) -> None:
        """Clear all step logs and counters for a new episode."""
        self.step_logs.clear()
        self.band_counts.fill(0)
        for k in self.mode_counts:
            self.mode_counts[k] = 0

    def record_step(
        self,
        step_index: int,
        current_time: int,
        action_band: int,
        action_dwell: int,
        mode_str: str,
        telemetry: Dict[str, Any],
    ) -> None:
        """Record a single step decision."""
        is_exploit = (mode_str == "EXPLOITATION")
        is_explore = (mode_str in ("EXPLORATION", "COLD_START"))
        is_adapt = (mode_str == "ADAPTATION")

        log = HybridStepLog(
            step_index=step_index,
            current_time=current_time,
            selected_band=action_band,
            selected_dwell=action_dwell,
            mode=mode_str,
            is_exploration=is_explore,
            is_exploitation=is_exploit,
            is_adaptation=is_adapt,
            composite_score=telemetry.get("composite_score", 0.0),
            xgb_proba=telemetry.get("xgb_proba", 0.0),
            ppo_proba=telemetry.get("ppo_proba", 0.0),
            linucb_mean=telemetry.get("linucb_mean", 0.0),
            linucb_uncertainty=telemetry.get("linucb_uncertainty", 0.0),
            staleness=telemetry.get("staleness", 0.0),
            weights=telemetry.get("weights", {}),
        )
        self.step_logs.append(log)

        if 0 <= action_band < self.num_bands:
            self.band_counts[action_band] += 1
        if mode_str in self.mode_counts:
            self.mode_counts[mode_str] += 1

    def compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate summary metrics over the recorded episode."""
        total = len(self.step_logs)
        if total == 0:
            return {
                "total_decisions": 0,
                "pct_exploitation": 0.0,
                "pct_exploration": 0.0,
                "pct_adaptation": 0.0,
                "mode_switches": 0,
                "entropy": 0.0,
            }

        pct_exploit = (self.mode_counts.get("EXPLOITATION", 0) / total) * 100.0
        pct_explore = ((self.mode_counts.get("EXPLORATION", 0) + self.mode_counts.get("COLD_START", 0)) / total) * 100.0
        pct_adapt = (self.mode_counts.get("ADAPTATION", 0) / total) * 100.0

        # Mode switches
        switches = 0
        for i in range(1, total):
            if self.step_logs[i].mode != self.step_logs[i - 1].mode:
                switches += 1

        # Shannon entropy of band selection
        probs = self.band_counts / float(total)
        non_zero = probs[probs > 0]
        entropy = float(-np.sum(non_zero * np.log(non_zero)))

        # Average weights
        avg_weights: Dict[str, float] = {}
        for k in ["xgb", "ppo", "linucb", "explore", "staleness"]:
            vals = [log.weights.get(k, 0.0) for log in self.step_logs if k in log.weights]
            avg_weights[f"avg_weight_{k}"] = float(np.mean(vals)) if vals else 0.0

        return {
            "total_decisions": total,
            "pct_exploitation": float(pct_exploit),
            "pct_exploration": float(pct_explore),
            "pct_adaptation": float(pct_adapt),
            "mode_switches": switches,
            "entropy": entropy,
            "unique_bands_scanned": int(np.count_nonzero(self.band_counts)),
            **avg_weights,
        }
