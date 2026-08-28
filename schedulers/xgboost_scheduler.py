"""
XGBoost Adaptive Scheduler for SIH26055 (Phase 3).

Integrates feature extraction, XGBoost band probability prediction, and action optimization
into a clean, modular adaptive scheduler that implements BaseScheduler.
Strictly isolated from RF environment ground truth.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np

from environment.types import Action, Observation
from features.rf_features import RFFeatureExtractor
from models.xgboost_model import XGBoostBandPredictor
from optimizers.action_optimizer import ActionOptimizer
from schedulers.base import BaseScheduler


class XGBoostScheduler(BaseScheduler):
    """
    Adaptive frequency scheduler that extracts observation features, queries an XGBoost
    model to estimate signal presence probabilities across all frequency channels, and
    optimizes (band, dwell) actions with anti-camping constraints.
    """

    def __init__(
        self,
        model: XGBoostBandPredictor,
        num_bands: int = 20,
        optimizer: Optional[ActionOptimizer] = None,
        feature_extractor: Optional[RFFeatureExtractor] = None,
    ):
        super().__init__(scheduler_name="XGBoostScheduler")
        self.num_bands = num_bands
        self.model = model
        self.feature_extractor = feature_extractor or RFFeatureExtractor(num_bands=num_bands)
        self.optimizer = optimizer or ActionOptimizer(num_bands=num_bands)

        # Store last computed predictions for inspection/debugging
        self.last_predicted_probabilities: Optional[np.ndarray] = None
        self.last_action: Optional[Action] = None

    def reset(self) -> None:
        """Reset internal feature extractor and decision history."""
        self.feature_extractor.reset()
        self.last_predicted_probabilities = None
        self.last_action = None

    def select_action(self, observation: Observation) -> Action:
        """
        Select the next scanning action based on current observation.
        
        Args:
            observation: Scheduler-facing Observation object.
            
        Returns:
            Action: Command containing frequency_band and dwell_time.
        """
        # 1. Update feature history with latest observation feedback
        self.feature_extractor.update(observation)

        # 2. Extract feature matrix across all candidate frequency bands (N x D)
        X_candidates = self.feature_extractor.extract_features_all_bands(
            current_time=observation.current_time
        )

        # 3. Predict signal presence probability for each band using XGBoost
        predicted_probas = self.model.predict_proba(X_candidates)
        self.last_predicted_probabilities = predicted_probas

        # 4. Optimize (band, dwell) action considering anti-camping and dwell trade-offs
        action = self.optimizer.select_action(
            predicted_probabilities=predicted_probas,
            last_scanned_band=self.feature_extractor.last_scanned_band,
            consecutive_scans=self.feature_extractor.consecutive_scan_count,
        )

        self.last_action = action
        return action

    def update(self, observation: Observation, action: Action, reward: float) -> None:
        """Post-step callback for online updates (if applicable)."""
        pass
