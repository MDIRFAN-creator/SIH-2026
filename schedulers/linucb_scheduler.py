"""
LinUCB Contextual Bandit Adaptive Scheduler for SIH26055 (Phase 4 Hardened).

Online learning frequency-time scheduler using the Discounted Linear Upper Confidence Bound algorithm
with:
1. Configurable hard anti-camping constraint (max_consecutive_scans = 3).
2. Principled cold-start exploration guarantee (min_initial_pulls = 1).
3. Non-stationary discounted adaptation (gamma = 0.99) for fast frequency-hopping emitter acquisition.
4. Information-theoretic allocation entropy and run-length telemetry.

Strict Non-Leakage Guarantee:
- Consumes strictly `Observation` objects and causal history.
- Zero access to environment internals, EmitterRegistry, GroundTruthSlot, or diagnostic `info`.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from bandits.linucb import LinUCB
from environment.types import Action, DetectionResult, Observation
from features.linucb_features import LinUCBFeatureExtractor
from schedulers.base import BaseScheduler


class LinUCBScheduler(BaseScheduler):
    """
    Hardened Contextual Bandit Frequency-Time Scheduler implementing Discounted LinUCB
    with hard anti-camping and cold-start exploration.
    """

    def __init__(
        self,
        num_bands: int = 20,
        alpha: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.99,
        max_consecutive_scans: int = 3,
        min_initial_pulls: int = 1,
        allowed_dwells: Optional[List[int]] = None,
        dwell_cost: float = 0.05,
        model: Optional[LinUCB] = None,
        feature_extractor: Optional[LinUCBFeatureExtractor] = None,
        seed: Optional[int] = None,
        scheduler_name: str = "LinUCBAdaptiveScheduler",
    ) -> None:
        """
        Initialize the Hardened LinUCB Adaptive Scheduler.
        
        Args:
            num_bands: Number of frequency bands.
            alpha: Exploration coefficient scaling UCB confidence bonus.
            reg_lambda: Ridge regularization parameter.
            gamma: Non-stationary discount factor in (0, 1].
            max_consecutive_scans: Hard limit on consecutive decisions on the same band (e.g. 3).
            min_initial_pulls: Minimum initial pulls required per arm before pure exploitation.
            allowed_dwells: List of discrete dwell durations allowed (e.g. [1, 2, 3]).
            dwell_cost: Linear dwell time overhead penalty.
            model: Optional pre-configured LinUCB model instance.
            feature_extractor: Optional custom LinUCBFeatureExtractor.
            seed: Optional seed for reproducibility.
            scheduler_name: Display name.
        """
        super().__init__(scheduler_name=scheduler_name)
        if max_consecutive_scans < 1:
            raise ValueError(f"max_consecutive_scans must be >= 1, got {max_consecutive_scans}")
        if min_initial_pulls < 0:
            raise ValueError(f"min_initial_pulls must be >= 0, got {min_initial_pulls}")

        self.num_bands = num_bands
        self.alpha = float(alpha)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.max_consecutive_scans = int(max_consecutive_scans)
        self.min_initial_pulls = int(min_initial_pulls)
        self.allowed_dwells = allowed_dwells if allowed_dwells is not None else [1, 2, 3]
        self.dwell_cost = float(dwell_cost)

        self.feature_extractor = (
            feature_extractor if feature_extractor is not None else LinUCBFeatureExtractor(num_bands=num_bands)
        )
        self.linucb = (
            model
            if model is not None
            else LinUCB(
                num_arms=num_bands,
                feature_dim=self.feature_extractor.feature_dim,
                alpha=alpha,
                reg_lambda=reg_lambda,
                gamma=gamma,
                seed=seed,
            )
        )

        # Anti-camping consecutive scan tracker
        self._last_selected_band: Optional[int] = None
        self._consecutive_scans: int = 0

        # Pending update cache for synchronous online learning
        self._last_context: Optional[np.ndarray] = None
        self._last_action: Optional[Action] = None
        self._pending_update: bool = False

        # Diagnostic telemetry
        self.decision_history: List[Dict[str, Any]] = []
        self.cumulative_reward: float = 0.0

    def reset(self) -> None:
        """Reset internal online bandit model, feature extractor, anti-camping counters, and telemetry."""
        self.feature_extractor.reset()
        self.linucb.reset()
        self._last_selected_band = None
        self._consecutive_scans = 0
        self._last_context = None
        self._last_action = None
        self._pending_update = False
        self.decision_history.clear()
        self.cumulative_reward = 0.0

    def compute_reward(self, observation: Observation, dwell_time: int) -> float:
        """
        Compute scalar feedback reward strictly from scheduler-visible observation.
        
        Formula:
            - HIT:         +1.0 - dwell_cost * (dwell_time - 1)
            - MISS / NONE: -0.05 - dwell_cost * (dwell_time - 1)
            - FALSE_ALARM: -0.50 - dwell_cost * (dwell_time - 1)
        """
        res = observation.result
        cost = self.dwell_cost * max(0, dwell_time - 1)

        if res == DetectionResult.HIT:
            base_r = 1.0
        elif res == DetectionResult.FALSE_ALARM:
            base_r = -0.50
        else:
            base_r = -0.05

        return float(base_r - cost)

    def select_dwell(self, arm: int, pred_mean: float, uncertainty: float) -> int:
        """
        Select dwell duration based on predicted signal probability and parameter certainty.
        
        - High estimated signal presence (pred_mean > 0.40) & low uncertainty (uncertainty < 0.60) -> dwell = 3
        - Moderate estimated signal presence (pred_mean > 0.15) -> dwell = 2
        - Exploration or low certainty -> dwell = 1 (minimal time overhead)
        """
        if 3 in self.allowed_dwells and pred_mean > 0.40 and uncertainty < 0.60:
            return 3
        elif 2 in self.allowed_dwells and pred_mean > 0.15:
            return 2
        elif 1 in self.allowed_dwells:
            return 1
        return self.allowed_dwells[0]

    def _get_eligible_arms(self) -> List[int]:
        """
        Determine eligible arms enforcing:
        1. Cold-start exploration: prioritize arms with pull_counts < min_initial_pulls.
        2. Hard anti-camping constraint: exclude the last scanned arm if consecutive_scans >= max_consecutive_scans.
        """
        all_arms = list(range(self.num_bands))

        # Check anti-camping restriction
        banned_arm = None
        if self._last_selected_band is not None and self._consecutive_scans >= self.max_consecutive_scans:
            banned_arm = self._last_selected_band

        # 1. Cold-start check
        if self.min_initial_pulls > 0:
            unpulled_arms = [a for a in all_arms if self.linucb.pull_counts[a] < self.min_initial_pulls]
            if unpulled_arms:
                eligible_cold_start = [a for a in unpulled_arms if a != banned_arm]
                if eligible_cold_start:
                    return eligible_cold_start

        # 2. General candidate pool with anti-camping mask
        if banned_arm is not None:
            eligible = [a for a in all_arms if a != banned_arm]
            return eligible if eligible else all_arms

        return all_arms

    def select_action(self, observation: Observation) -> Action:
        """
        Perform online update on previous observation (if pending) and select next Action(band, dwell).
        
        Guarantees:
        - Strict hard anti-camping: maximum consecutive scans <= max_consecutive_scans.
        - Cold start: all arms sampled at least min_initial_pulls times.
        - Observation-only causal feedback loop.
        """
        # 1. Process pending update from preceding action outcome
        if self._pending_update and self._last_action is not None and self._last_context is not None:
            r = self.compute_reward(observation, self._last_action.dwell_time)
            self.linucb.update(self._last_action.frequency_band, self._last_context, r)
            self.cumulative_reward += r
            if self.decision_history:
                self.decision_history[-1]["reward_received"] = r
                self.decision_history[-1]["cumulative_reward"] = self.cumulative_reward
            self._pending_update = False

        # 2. Update feature history with the incoming observation
        self.feature_extractor.update(observation)

        # 3. Extract context feature matrix across all 20 candidate bands
        current_time = observation.current_time
        context_matrix = self.feature_extractor.extract_features_all_bands(current_time)

        # 4. Determine eligible arms under cold-start and hard anti-camping rules
        eligible_arms = self._get_eligible_arms()

        # 5. LinUCB arm selection from eligible set
        selected_band, arm_diag = self.linucb.select_arm(context_matrix, eligible_arms=eligible_arms)

        # 6. Update anti-camping consecutive counters
        if selected_band == self._last_selected_band:
            self._consecutive_scans += 1
        else:
            self._last_selected_band = selected_band
            self._consecutive_scans = 1

        # 7. Principled dwell selection
        selected_dwell = self.select_dwell(
            arm=selected_band,
            pred_mean=arm_diag["selected_mean"],
            uncertainty=arm_diag["selected_uncertainty"],
        )

        # 8. Cache action and context for the next online update
        action = Action(frequency_band=selected_band, dwell_time=selected_dwell)
        self._last_context = context_matrix[selected_band].copy()
        self._last_action = action
        self._pending_update = True

        # 9. Record telemetry
        telemetry = {
            "time": current_time,
            "selected_band": selected_band,
            "selected_dwell": selected_dwell,
            "consecutive_scans": self._consecutive_scans,
            "ucb_score": arm_diag["selected_ucb"],
            "pred_mean": arm_diag["selected_mean"],
            "uncertainty": arm_diag["selected_uncertainty"],
            "reward_received": None,  # Populated on next step
            "cumulative_reward": self.cumulative_reward,
            "eligible_arms": eligible_arms,
        }
        self.decision_history.append(telemetry)

        return action

    def update(
        self,
        observation: Observation,
        action: Action,
        reward: float,
    ) -> None:
        """
        Explicit update hook called by EpisodeRunner after environment step.
        """
        if self._pending_update and self._last_context is not None:
            r = self.compute_reward(observation, action.dwell_time)
            self.linucb.update(action.frequency_band, self._last_context, r)
            self.cumulative_reward += r
            if self.decision_history:
                self.decision_history[-1]["reward_received"] = r
                self.decision_history[-1]["cumulative_reward"] = self.cumulative_reward
            self._pending_update = False

    def compute_band_selection_entropy(self) -> float:
        r"""
        Compute Shannon entropy of the band selection distribution:
            H = -\sum_{b=0}^{19} p(b) * \ln(p(b))
        """
        if not self.decision_history:
            return 0.0
        pulls = np.zeros(self.num_bands, dtype=np.float64)
        for d in self.decision_history:
            pulls[d["selected_band"]] += 1.0
        total = np.sum(pulls)
        if total == 0:
            return 0.0
        probs = pulls / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))
