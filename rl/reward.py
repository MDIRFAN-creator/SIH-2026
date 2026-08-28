"""
Observation-Only Reward Function for Reinforcement Learning (SIH26055 Phase 5).

Calculates a scalar reward feedback derived strictly from legitimate scheduler-visible
`Observation` outcomes and commanded actions.

Strict Non-Leakage Guarantee:
- Zero dependency on EmitterRegistry, GroundTruthSlot, DwellSummary, or hidden transmitter state.
- Bounded, numerically stable, and configurable.

Reward Formulation:
    r_t = R_det(result) - C_dwell * (dwell - 1) - C_repeat * max(0, consecutive - repeat_thresh)

Default Parameters:
- HIT: +1.0
- MISS / NONE: -0.05
- FALSE_ALARM: -0.50
- dwell_cost (C_dwell): 0.05 per extra dwell slot above 1
- repetition_penalty (C_repeat): 0.10 per scan above repeat_threshold (2)
"""

from dataclasses import dataclass
from typing import Optional
from environment.types import DetectionResult, Observation


@dataclass
class RLRewardConfig:
    """
    Configuration parameters for the RL reward module.
    """
    hit_reward: float = 1.0
    miss_penalty: float = -0.05
    false_alarm_penalty: float = -0.50
    dwell_cost: float = 0.05
    repetition_penalty: float = 0.10
    repetition_threshold: int = 2


class RLRewardCalculator:
    """
    Dedicated observation-only reward calculator for reinforcement learning.
    """

    def __init__(self, config: Optional[RLRewardConfig] = None) -> None:
        """
        Initialize the reward calculator with configuration.

        Args:
            config: Optional RLRewardConfig instance (defaults to standard values).
        """
        self.config = config if config is not None else RLRewardConfig()

    def compute_reward(
        self,
        observation: Observation,
        dwell_time: int,
        consecutive_scans: int = 1,
    ) -> float:
        """
        Compute the immediate scalar reward for a completed dwell step.

        Args:
            observation: Scheduler-visible observation containing DetectionResult.
            dwell_time: Commanded dwell duration (e.g. 1, 2, 3).
            consecutive_scans: Number of consecutive decisions spent on this band.

        Returns:
            float: Scalar reward value.
        """
        # 1. Base detection feedback
        if observation.result == DetectionResult.HIT:
            base_reward = self.config.hit_reward
        elif observation.result == DetectionResult.FALSE_ALARM:
            base_reward = self.config.false_alarm_penalty
        else:  # MISS or NONE
            base_reward = self.config.miss_penalty

        # 2. Dwell cost overhead (penalizes longer dwells unless justified by hits)
        dwell_overhead = self.config.dwell_cost * float(max(0, dwell_time - 1))

        # 3. Anti-camping / excessive repetition penalty
        excess_repeats = max(0, consecutive_scans - self.config.repetition_threshold)
        repetition_cost = self.config.repetition_penalty * float(excess_repeats)

        # Net reward
        total_reward = base_reward - dwell_overhead - repetition_cost
        return float(total_reward)
