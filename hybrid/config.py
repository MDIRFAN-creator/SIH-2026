"""
Configuration dataclass for Phase 6 Hybrid Adaptive RF Scheduler.

Defines arbitration weights, confidence thresholds, anti-camping limits,
and exploration hyperparameters without modifying any existing component configs.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HybridConfig:
    """
    Configuration hyperparameters for Hybrid Adaptive Scheduler (Phase 6).
    """
    num_bands: int = 20
    allowed_dwells: List[int] = field(default_factory=lambda: [1, 2, 3])
    max_consecutive_scans: int = 3
    min_initial_pulls: int = 1

    # Base component arbitration weights
    xgb_weight: float = 0.40
    linucb_weight: float = 0.35
    ppo_weight: float = 0.25

    # Adaptive exploration and novelty modifiers
    exploration_bonus_weight: float = 0.30
    staleness_bonus_weight: float = 0.10

    # Decision mode thresholds
    confidence_threshold: float = 0.45  # High-confidence threshold for exploitation mode
    uncertainty_threshold: float = 0.65 # High-uncertainty threshold for exploration mode
    shock_miss_threshold: int = 2       # Consecutive misses on predicted band to trigger adaptation mode

    # LinUCB hyperparams (reused safely)
    linucb_alpha: float = 1.0
    linucb_reg_lambda: float = 1.0
    linucb_gamma: float = 0.99

    # Dwell selection trade-offs
    dwell_cost: float = 0.02

    # Determinism / Seed
    seed: Optional[int] = None
    scheduler_name: str = "HybridAdaptiveScheduler"

    def validate(self) -> None:
        """Validate parameter boundaries."""
        if self.num_bands <= 0:
            raise ValueError(f"num_bands must be > 0, got {self.num_bands}")
        if self.max_consecutive_scans < 1:
            raise ValueError(f"max_consecutive_scans must be >= 1, got {self.max_consecutive_scans}")
        if not self.allowed_dwells:
            raise ValueError("allowed_dwells cannot be empty")
        if self.xgb_weight < 0 or self.linucb_weight < 0 or self.ppo_weight < 0:
            raise ValueError("Arbitration weights must be non-negative")
