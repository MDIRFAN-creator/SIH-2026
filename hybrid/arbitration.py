"""
Observation-driven arbitration layer for Hybrid Scheduler (Phase 6).

Dynamically balances Exploitation (XGBoost/PPO), Exploration (LinUCB uncertainty),
and Adaptation (shock-triggered hopping discovery) using only legitimate observations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from environment.types import Action, Observation
from hybrid.config import HybridConfig
from hybrid.scoring import ComponentSignals


class DecisionMode(str, Enum):
    """Operational mode of the hybrid scheduler."""
    COLD_START = "COLD_START"
    EXPLOITATION = "EXPLOITATION"
    EXPLORATION = "EXPLORATION"
    ADAPTATION = "ADAPTATION"


class HybridArbitrator:
    """
    Arbitrates among XGBoost, LinUCB, PPO, and causal observation metrics
    to select the optimal frequency band and dwell time.
    """

    def __init__(self, config: Optional[HybridConfig] = None) -> None:
        self.config = config or HybridConfig()
        self.config.validate()

        # Dynamic state tracking
        self.consecutive_misses_on_target: int = 0
        self.last_target_band: Optional[int] = None
        self.total_switches: int = 0
        self.last_selected_band: Optional[int] = None

    def reset(self) -> None:
        """Reset internal tracking state at the start of an episode."""
        self.consecutive_misses_on_target = 0
        self.last_target_band = None
        self.total_switches = 0
        self.last_selected_band = None

    def update_observation_state(self, observation: Observation) -> None:
        """Track consecutive misses on high-confidence target bands to detect hopping."""
        if observation.result is not None:
            from environment.types import DetectionResult
            if observation.result == DetectionResult.MISS:
                self.consecutive_misses_on_target += 1
            elif observation.result == DetectionResult.HIT:
                self.consecutive_misses_on_target = 0


    def determine_mode(
        self, signals: ComponentSignals, eligible_mask: np.ndarray
    ) -> Tuple[DecisionMode, Dict[str, float]]:
        """
        Determine operational mode and compute adaptive component weights.
        
        Args:
            signals: Normalized component signals.
            eligible_mask: Boolean array indicating eligible arms under anti-camping.
            
        Returns:
            Tuple[DecisionMode, Dict[str, float]]: (Selected Mode, Weight Dictionary).
        """
        # 1. Cold start check: any eligible band with 0 pulls?
        unpulled = np.where(eligible_mask & (signals.pull_counts < self.config.min_initial_pulls))[0]
        if len(unpulled) > 0:
            weights = {
                "xgb": 0.0,
                "ppo": 0.0,
                "linucb": 0.50,
                "explore": 0.50,
                "staleness": 0.0,
            }
            return DecisionMode.COLD_START, weights

        # 2. Adaptation mode (Hopping Shock): consecutive misses on predicted active target
        if (
            self.consecutive_misses_on_target >= self.config.shock_miss_threshold
            and signals.xgb_max_proba >= self.config.confidence_threshold
        ):
            weights = {
                "xgb": 0.15,
                "ppo": 0.15,
                "linucb": 0.40,
                "explore": 0.20,
                "staleness": 0.10,
            }
            return DecisionMode.ADAPTATION, weights

        # 3. Exploitation mode: strong model confidence
        if (
            signals.xgb_max_proba >= self.config.confidence_threshold
            or np.max(signals.ppo_band_probas[eligible_mask]) >= self.config.confidence_threshold
        ):
            weights = {
                "xgb": 0.45,
                "ppo": 0.35,
                "linucb": 0.15,
                "explore": 0.05,
                "staleness": 0.00,
            }
            return DecisionMode.EXPLOITATION, weights

        # 4. Default: Exploration mode
        weights = {
            "xgb": 0.10,
            "ppo": 0.10,
            "linucb": 0.45,
            "explore": self.config.exploration_bonus_weight,
            "staleness": self.config.staleness_bonus_weight,
        }
        return DecisionMode.EXPLORATION, weights

    def arbitrate(
        self,
        signals: ComponentSignals,
        eligible_mask: np.ndarray,
        observation: Observation,
    ) -> Tuple[Action, DecisionMode, Dict[str, Any]]:
        """
        Execute arbitration across components to produce the final Action.
        
        Args:
            signals: Normalized component signals.
            eligible_mask: Boolean array for anti-camping compliance.
            observation: Current observation.
            
        Returns:
            Tuple[Action, DecisionMode, Dict[str, Any]]: Selected Action, Mode, and Telemetry.
        """
        num_bands = self.config.num_bands
        mode, weights = self.determine_mode(signals, eligible_mask)

        # 1. Cold-start direct selection
        if mode == DecisionMode.COLD_START:
            unpulled = np.where(eligible_mask & (signals.pull_counts < self.config.min_initial_pulls))[0]
            selected_band = int(unpulled[0])
            selected_dwell = int(self.config.allowed_dwells[0])
            composite_scores = np.zeros(num_bands, dtype=np.float64)
            composite_scores[selected_band] = 1.0
        else:
            # 2. Compute composite score for all eligible bands
            composite_scores = np.full(num_bands, -1e9, dtype=np.float64)
            for b in range(num_bands):
                if not eligible_mask[b]:
                    continue

                # Normalize LinUCB predicted mean to [0, 1]
                ucb_mean_norm = float(np.clip(signals.linucb_means[b], 0.0, 1.0))
                # Normalize LinUCB uncertainty to [0, 1]
                uncert_norm = float(np.clip(signals.linucb_uncertainties[b], 0.0, 1.0))

                score_b = (
                    weights["xgb"] * signals.xgb_probas[b]
                    + weights["ppo"] * signals.ppo_band_probas[b]
                    + weights["linucb"] * ucb_mean_norm
                    + weights["explore"] * uncert_norm
                    + weights["staleness"] * signals.staleness_scores[b]
                )
                composite_scores[b] = score_b

            selected_band = int(np.argmax(composite_scores))

            # 3. Dwell duration arbitration
            if mode == DecisionMode.EXPLOITATION and composite_scores[selected_band] > 0.65:
                # Confident exploitation: choose dwell = 2 to maximize burst intercept probability
                selected_dwell = 2 if 2 in self.config.allowed_dwells else self.config.allowed_dwells[0]
            else:
                # Fast agile scan: choose shortest dwell
                selected_dwell = int(self.config.allowed_dwells[0])

        action = Action(frequency_band=selected_band, dwell_time=selected_dwell)

        # Track switches
        if self.last_selected_band is not None and self.last_selected_band != selected_band:
            self.total_switches += 1
        self.last_selected_band = selected_band

        # Build telemetry record
        telemetry = {
            "mode": mode.value,
            "selected_band": selected_band,
            "selected_dwell": selected_dwell,
            "composite_score": float(composite_scores[selected_band]),
            "composite_scores": composite_scores.copy(),
            "weights": weights.copy(),
            "xgb_proba": float(signals.xgb_probas[selected_band]),
            "ppo_proba": float(signals.ppo_band_probas[selected_band]),
            "linucb_mean": float(signals.linucb_means[selected_band]),
            "linucb_uncertainty": float(signals.linucb_uncertainties[selected_band]),
            "staleness": float(signals.staleness_scores[selected_band]),
            "consecutive_misses": self.consecutive_misses_on_target,
        }

        return action, mode, telemetry
