"""
Phase 6: Hybrid Adaptive RF Frequency-Time Scheduler.

An isolated adaptive scheduler that arbitrates among:
1. XGBoost Band Predictor (Supervised periodic threat exploitation)
2. Hardened LinUCB Contextual Bandit (Uncertainty-driven discovery & non-stationary agility)
3. Hardened PPO Actor-Critic Policy (Learned rapid threat tracking)
4. Observation-Derived Staleness & Anti-Camping Invariant

Strict Non-Leakage Guarantee:
- Consumes strictly `Observation` objects and causal scanning history.
- Zero access to environment internals, EmitterRegistry, GroundTruthSlot, or diagnostic `info`.
- Completely isolated from earlier phases without modifying any existing frozen algorithm.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from bandits.linucb import LinUCB
from environment.types import Action, DetectionResult, Observation
from hybrid.arbitration import DecisionMode, HybridArbitrator
from hybrid.config import HybridConfig
from hybrid.diagnostics import HybridDiagnostics
from hybrid.scoring import ComponentSignalExtractor, ComponentSignals
from models.xgboost_model import XGBoostBandPredictor
from rl.ppo_agent import PPOAgent
from schedulers.base import BaseScheduler


class HybridAdaptiveScheduler(BaseScheduler):
    """
    Phase 6 Isolated Hybrid Adaptive Scheduler.
    """

    def __init__(
        self,
        config: Optional[HybridConfig] = None,
        xgb_model: Optional[XGBoostBandPredictor] = None,
        linucb_model: Optional[LinUCB] = None,
        ppo_agent: Optional[PPOAgent] = None,
        ppo_model_path: Optional[str] = None,
        scheduler_name: str = "HybridAdaptiveScheduler",
    ) -> None:
        super().__init__(scheduler_name=scheduler_name)
        self.config = config or HybridConfig()
        self.config.validate()

        self.num_bands = self.config.num_bands
        self.allowed_dwells = self.config.allowed_dwells
        self.max_consecutive_scans = self.config.max_consecutive_scans

        # 1. Models
        self.xgb_model = xgb_model
        self.linucb_model = (
            linucb_model
            if linucb_model is not None
            else LinUCB(
                num_arms=self.num_bands,
                feature_dim=10,
                alpha=self.config.linucb_alpha,
                reg_lambda=self.config.linucb_reg_lambda,
                gamma=self.config.linucb_gamma,
                seed=self.config.seed,
            )
        )

        if ppo_agent is not None:
            self.ppo_agent = ppo_agent
        elif ppo_model_path is not None:
            from rl.action_encoding import ActionEncoder
            from rl.state_features import RLStateExtractor
            se = RLStateExtractor(num_bands=self.num_bands, max_dwell=max(self.allowed_dwells))
            ae = ActionEncoder(num_bands=self.num_bands, dwell_values=self.allowed_dwells)
            self.ppo_agent = PPOAgent(state_dim=se.state_dim, action_dim=ae.num_actions)
            self.ppo_agent.load(ppo_model_path)
        else:
            self.ppo_agent = None

        # 2. Hybrid sub-systems
        self.signal_extractor = ComponentSignalExtractor(
            num_bands=self.num_bands,
            allowed_dwells=self.allowed_dwells,
            xgb_model=self.xgb_model,
            linucb_model=self.linucb_model,
            ppo_agent=self.ppo_agent,
        )
        self.arbitrator = HybridArbitrator(config=self.config)
        self.diagnostics = HybridDiagnostics(num_bands=self.num_bands)

        # 3. Anti-camping & online bandit update caches
        self._last_selected_band: Optional[int] = None
        self._consecutive_scans: int = 0
        self._last_context: Optional[np.ndarray] = None
        self._last_action: Optional[Action] = None
        self._pending_update: bool = False
        self._decision_count: int = 0

    def reset(self) -> None:
        """Reset internal causal state at the start of a simulation episode."""
        self.signal_extractor.reset()
        self.arbitrator.reset()
        self.diagnostics.reset()
        self._last_selected_band = None
        self._consecutive_scans = 0
        self._last_context = None
        self._last_action = None
        self._pending_update = False
        self._decision_count = 0

    def _compute_linucb_reward(self, observation: Observation, dwell_time: int) -> float:
        """Compute scalar feedback reward strictly from observation."""
        res = observation.result
        cost = self.config.dwell_cost * max(0, dwell_time - 1)
        if res == DetectionResult.HIT:
            base_r = 1.0
        elif res == DetectionResult.FALSE_ALARM:
            base_r = -0.50
        else:
            base_r = -0.05
        return float(base_r - cost)


    def select_action(self, observation: Observation) -> Action:
        """
        Select frequency band and dwell duration given scheduler-facing Observation.

        Args:
            observation: Current scheduler observation from ESM receiver.

        Returns:
            Action: Selected Action(frequency_band, dwell_time).
        """
        # 1. Apply pending online bandit update from previous step outcome
        if self._pending_update and self._last_action is not None and self._last_context is not None:
            r = self._compute_linucb_reward(observation, self._last_action.dwell_time)
            self.linucb_model.update(
                arm=self._last_action.frequency_band,
                context=self._last_context,
                reward=r,
            )
            self._pending_update = False

        # 2. Update feature extractors and arbitration state with current observation
        self.signal_extractor.update_history(observation)
        self.arbitrator.update_observation_state(observation)

        # 3. Compute eligible arms mask enforcing strict anti-camping constraint
        eligible_mask = np.ones(self.num_bands, dtype=bool)
        if (
            self._last_selected_band is not None
            and self._consecutive_scans >= self.max_consecutive_scans
        ):
            eligible_mask[self._last_selected_band] = False

        # If all arms masked (edge case), reset mask
        if not np.any(eligible_mask):
            eligible_mask.fill(True)

        # 4. Extract normalized signals across all components
        signals = self.signal_extractor.extract_signals(observation, eligible_mask=eligible_mask)

        # 5. Arbitrate and select Action
        action, mode, telemetry = self.arbitrator.arbitrate(
            signals=signals,
            eligible_mask=eligible_mask,
            observation=observation,
        )

        # 6. Update anti-camping consecutive counters
        if action.frequency_band == self._last_selected_band:
            self._consecutive_scans += 1
        else:
            self._last_selected_band = action.frequency_band
            self._consecutive_scans = 1

        # 7. Cache context and action for online LinUCB learning
        X_linucb = self.signal_extractor.linucb_fe.extract_features_all_bands(
            current_time=observation.current_time
        )
        self._last_context = X_linucb[action.frequency_band].copy()

        self._last_action = action
        self._pending_update = True

        # 8. Record diagnostics telemetry
        self.diagnostics.record_step(
            step_index=self._decision_count,
            current_time=observation.current_time,
            action_band=action.frequency_band,
            action_dwell=action.dwell_time,
            mode_str=mode.value,
            telemetry=telemetry,
        )
        self._decision_count += 1

        return action

    def update(
        self,
        observation: Observation,
        action: Action,
        reward: float,
    ) -> None:
        """Post-step callback for online learning."""
        if self._pending_update and self._last_action is not None and self._last_context is not None:
            r = self._compute_linucb_reward(observation, self._last_action.dwell_time)
            self.linucb_model.update(
                arm=self._last_action.frequency_band,
                context=self._last_context,
                reward=r,
            )
            self._pending_update = False
