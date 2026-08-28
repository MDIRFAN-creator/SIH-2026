"""
Component scoring and feature normalization module for Hybrid Scheduler (Phase 6).

Extracts and normalizes legitimate observation-derived signals from:
1. XGBoost Band Predictor
2. LinUCB Contextual Bandit
3. Hardened PPO Actor-Critic Policy
4. Causal Observation History (Staleness & Novelty)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from environment.types import Action, Observation
from features.linucb_features import LinUCBFeatureExtractor
from features.rf_features import RFFeatureExtractor
from models.xgboost_model import XGBoostBandPredictor
from bandits.linucb import LinUCB
from rl.action_encoding import ActionEncoder
from rl.ppo_agent import PPOAgent
from rl.state_features import RLStateExtractor


@dataclass
class ComponentSignals:
    """
    Normalized signals extracted from each underlying component for all candidate bands.
    """
    # XGBoost signals (num_bands,)
    xgb_probas: np.ndarray
    xgb_max_proba: float
    xgb_argmax_band: int

    # LinUCB signals (num_bands,)
    linucb_scores: np.ndarray
    linucb_means: np.ndarray
    linucb_uncertainties: np.ndarray
    linucb_max_uncertainty: float
    linucb_argmax_band: int

    # PPO signals
    ppo_band_probas: np.ndarray      # Marginal probability per band (num_bands,)
    ppo_action_probas: np.ndarray    # Full discrete action distribution (num_actions,)
    ppo_entropy: float               # Policy entropy H(pi)
    ppo_value: float                 # Estimated state value V(s)
    ppo_argmax_action: int           # Discrete action index
    ppo_argmax_band: int
    ppo_argmax_dwell: int

    # Causal observation history
    staleness_scores: np.ndarray     # Normalized time since last scan (num_bands,)
    pull_counts: np.ndarray          # Pull counts per band (num_bands,)


class ComponentSignalExtractor:
    """
    Consumes scheduler-facing Observation and extracts normalized signals
    from all underlying sub-models without leaking any hidden ground truth.
    """

    def __init__(
        self,
        num_bands: int = 20,
        allowed_dwells: Optional[List[int]] = None,
        xgb_model: Optional[XGBoostBandPredictor] = None,
        linucb_model: Optional[LinUCB] = None,
        ppo_agent: Optional[PPOAgent] = None,
        xgb_feature_extractor: Optional[RFFeatureExtractor] = None,
        linucb_feature_extractor: Optional[LinUCBFeatureExtractor] = None,
        rl_state_extractor: Optional[RLStateExtractor] = None,
    ) -> None:
        self.num_bands = num_bands
        self.allowed_dwells = allowed_dwells if allowed_dwells is not None else [1, 2, 3]

        self.xgb_model = xgb_model
        self.linucb_model = linucb_model
        self.ppo_agent = ppo_agent

        self.xgb_fe = xgb_feature_extractor or RFFeatureExtractor(num_bands=num_bands)
        self.linucb_fe = linucb_feature_extractor or LinUCBFeatureExtractor(num_bands=num_bands)
        self.rl_se = rl_state_extractor or RLStateExtractor(
            num_bands=num_bands, max_dwell=max(self.allowed_dwells)
        )
        self.action_encoder = ActionEncoder(
            num_bands=num_bands, dwell_values=self.allowed_dwells
        )

        # Pull count tracking across the episode
        self.pull_counts = np.zeros(num_bands, dtype=np.int64)

    def reset(self) -> None:
        """Reset internal feature extractors and episode statistics."""
        self.xgb_fe.reset()
        self.linucb_fe.reset()
        self.rl_se.reset()
        if self.linucb_model is not None:
            self.linucb_model.reset()
        self.pull_counts.fill(0)

    def update_history(self, observation: Observation) -> None:
        """Update causal extractors with latest observation feedback."""
        self.xgb_fe.update(observation)
        self.linucb_fe.update(observation)
        if observation.scanned_band is not None and observation.dwell_time is not None:
            self.rl_se.update(observation)
            if 0 <= observation.scanned_band < self.num_bands:
                self.pull_counts[observation.scanned_band] += 1

    def extract_signals(
        self, observation: Observation, eligible_mask: Optional[np.ndarray] = None
    ) -> ComponentSignals:
        """
        Extract normalized signals across all components.
        
        Args:
            observation: Current scheduler-visible Observation.
            eligible_mask: Boolean array of shape (num_bands,) where True = eligible.
            
        Returns:
            ComponentSignals: Container with all computed signals.
        """
        mask = eligible_mask if eligible_mask is not None else np.ones(self.num_bands, dtype=bool)

        # -------------------------------------------------------------
        # 1. XGBoost Signals
        # -------------------------------------------------------------
        if self.xgb_model is not None:
            X_xgb = self.xgb_fe.extract_features_all_bands(current_time=observation.current_time)
            xgb_probas = self.xgb_model.predict_proba(X_xgb)
        else:
            xgb_probas = np.full(self.num_bands, 1.0 / self.num_bands, dtype=np.float64)

        masked_xgb = np.where(mask, xgb_probas, -1.0)
        xgb_max_p = float(np.max(masked_xgb)) if np.any(mask) else 0.0
        xgb_argmax = int(np.argmax(masked_xgb)) if np.any(mask) else 0

        # -------------------------------------------------------------
        # 2. LinUCB Signals
        # -------------------------------------------------------------
        if self.linucb_model is not None:
            X_linucb = self.linucb_fe.extract_features_all_bands(current_time=observation.current_time)
            ucb_scores, pred_means, uncertainties = self.linucb_model.predict_all_arms(X_linucb)
        else:

            ucb_scores = np.zeros(self.num_bands, dtype=np.float64)
            pred_means = np.zeros(self.num_bands, dtype=np.float64)
            uncertainties = np.ones(self.num_bands, dtype=np.float64)

        masked_ucb = np.where(mask, ucb_scores, -1e9)
        masked_uncert = np.where(mask, uncertainties, -1.0)
        linucb_max_u = float(np.max(masked_uncert)) if np.any(mask) else 0.0
        linucb_argmax = int(np.argmax(masked_ucb)) if np.any(mask) else 0

        # -------------------------------------------------------------
        # 3. Hardened PPO Signals
        # -------------------------------------------------------------
        if self.ppo_agent is not None:
            rl_state = self.rl_se.extract_state(observation)
            state_tensor = torch.tensor(rl_state, dtype=torch.float32).unsqueeze(0)
            
            # Action mask for PPO
            action_mask = np.ones(self.action_encoder.num_actions, dtype=bool)
            for a_idx in range(self.action_encoder.num_actions):
                b, _ = self.action_encoder.decode(a_idx)
                if not mask[b]:
                    action_mask[a_idx] = False
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)

            with torch.no_grad():
                dist, value = self.ppo_agent.network.forward(state_tensor, action_mask=mask_tensor)
                ppo_action_probas = dist.probs.squeeze(0).cpu().numpy()
                ppo_entropy = float(dist.entropy().squeeze(0).item())
                ppo_val = float(value.squeeze(0).item())

            # Marginalize action probabilities over dwell to get per-band probability
            ppo_band_probas = np.zeros(self.num_bands, dtype=np.float64)
            for a_idx, p in enumerate(ppo_action_probas):
                b, _ = self.action_encoder.decode(a_idx)
                ppo_band_probas[b] += p

            masked_act_probas = np.where(action_mask, ppo_action_probas, -1.0)
            ppo_argmax_act = int(np.argmax(masked_act_probas)) if np.any(action_mask) else 0
            ppo_argmax_b, ppo_argmax_d = self.action_encoder.decode(ppo_argmax_act)

        else:
            ppo_band_probas = np.full(self.num_bands, 1.0 / self.num_bands, dtype=np.float64)
            ppo_action_probas = np.full(
                self.action_encoder.num_actions,
                1.0 / self.action_encoder.num_actions,
                dtype=np.float64,
            )
            ppo_entropy = float(np.log(self.action_encoder.num_actions))
            ppo_val = 0.0
            ppo_argmax_act = 0
            ppo_argmax_b, ppo_argmax_d = 0, self.allowed_dwells[0]

        # -------------------------------------------------------------
        # 4. Causal Staleness Scores
        # -------------------------------------------------------------
        staleness = np.zeros(self.num_bands, dtype=np.float64)
        time_since_data = (
            observation.history_summary.get("time_since_last_scan", None)
            if observation.history_summary
            else None
        )
        for b in range(self.num_bands):
            if isinstance(time_since_data, (list, np.ndarray)) and b < len(time_since_data):
                time_since = float(time_since_data[b])
            elif isinstance(time_since_data, dict):
                time_since = float(time_since_data.get(b, observation.current_time))
            else:
                time_since = float(observation.current_time)
            # Normalize to [0, 1] over 100 slots
            staleness[b] = min(1.0, max(0.0, time_since) / 100.0)



        return ComponentSignals(
            xgb_probas=xgb_probas,
            xgb_max_proba=xgb_max_p,
            xgb_argmax_band=xgb_argmax,
            linucb_scores=ucb_scores,
            linucb_means=pred_means,
            linucb_uncertainties=uncertainties,
            linucb_max_uncertainty=linucb_max_u,
            linucb_argmax_band=linucb_argmax,
            ppo_band_probas=ppo_band_probas,
            ppo_action_probas=ppo_action_probas,
            ppo_entropy=ppo_entropy,
            ppo_value=ppo_val,
            ppo_argmax_action=ppo_argmax_act,
            ppo_argmax_band=ppo_argmax_b,
            ppo_argmax_dwell=ppo_argmax_d,
            staleness_scores=staleness,
            pull_counts=self.pull_counts.copy(),
        )
