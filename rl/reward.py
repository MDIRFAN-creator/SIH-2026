"""
Observation-Only Reward Function for Reinforcement Learning (SIH26055 Phase 5 & Pre-Phase 6A Hardening).

Calculates a scalar reward feedback derived strictly from legitimate scheduler-visible
`Observation` outcomes and causal scanning history.

Strict Non-Leakage Guarantee:
- Zero dependency on EmitterRegistry, GroundTruthSlot, DwellSummary, or hidden transmitter state.
- Bounded, numerically stable, and configurable.
- Encourages discovery and broad spectrum coverage while rewarding legitimate detection exploitation.

Reward Formulation:
    r_t = R_det(result, consecutive_hits) + R_explore(time_since_last_scan) - C_dwell * (dwell - 1) - C_repeat * max(0, consecutive - repeat_thresh)
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
    repetition_penalty: float = 0.15
    repetition_threshold: int = 2
    diminishing_hit_factor: float = 0.10
    min_hit_multiplier: float = 0.40
    stale_bonus_weight: float = 0.10
    max_stale_time: float = 200.0
    novelty_bonus: float = 0.10


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
        time_since_last_scan: float = 0.0,
        consecutive_hits: int = 1,
    ) -> float:
        """
        Compute the immediate scalar reward for a completed dwell step.

        Args:
            observation: Scheduler-visible observation containing DetectionResult.
            dwell_time: Commanded dwell duration (e.g. 1, 2, 3).
            consecutive_scans: Number of consecutive decisions spent on this band.
            time_since_last_scan: Time elapsed since this frequency band was last scanned (-1 if never).
            consecutive_hits: Number of consecutive hits recorded on this band.

        Returns:
            float: Scalar reward value bounded in [-3.0, +3.0].
        """
        # 1. Base detection feedback with diminishing returns for repeated hits
        if observation.result == DetectionResult.HIT:
            hit_decay = max(
                self.config.min_hit_multiplier,
                1.0 - self.config.diminishing_hit_factor * float(max(0, consecutive_hits - 1)),
            )
            base_reward = self.config.hit_reward * hit_decay
        elif observation.result == DetectionResult.FALSE_ALARM:
            base_reward = self.config.false_alarm_penalty
        else:  # MISS or NONE
            base_reward = self.config.miss_penalty

        # 2. Observation-derived exploration and staleness bonus
        if time_since_last_scan < 0:
            exploration_bonus = self.config.novelty_bonus
        else:
            staleness_fraction = min(1.0, max(0.0, float(time_since_last_scan)) / max(1.0, self.config.max_stale_time))
            exploration_bonus = self.config.stale_bonus_weight * staleness_fraction

        # 3. Dwell cost overhead (penalizes longer dwells unless justified by hits)
        dwell_overhead = self.config.dwell_cost * float(max(0, dwell_time - 1))

        # 4. Anti-camping / excessive repetition penalty
        excess_repeats = max(0, consecutive_scans - self.config.repetition_threshold)
        repetition_cost = self.config.repetition_penalty * float(excess_repeats)

        # Net reward
        total_reward = base_reward + exploration_bonus - dwell_overhead - repetition_cost
        # Clamp to guaranteed safe bounds
        clamped_reward = max(-3.0, min(3.0, float(total_reward)))
        return float(clamped_reward)

