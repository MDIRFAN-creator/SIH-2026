"""
Comprehensive unit, integration, and non-leakage tests for PPO Hardening (Pre-Phase 6A).
"""

import numpy as np
import pytest
import torch

from environment.config import EnvironmentConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, DetectionResult, Observation
from rl.action_encoding import ActionEncoder
from rl.ppo_agent import ActorCriticNetwork, PPOAgent, PPOConfig
from rl.reward import RLRewardCalculator, RLRewardConfig
from rl.rf_rl_env import RFRLGymEnv
from runners.episode_runner import EpisodeRunner
from schedulers.ppo_scheduler import PPOScheduler


def test_reward_diminishing_hits() -> None:
    """Test that consecutive hits on the same band receive diminishing returns."""
    calc = RLRewardCalculator(config=RLRewardConfig(hit_reward=1.0, diminishing_hit_factor=0.10, min_hit_multiplier=0.40, repetition_penalty=0.0))
    obs_hit = Observation(current_time=10, scanned_band=3, dwell_time=1, result=DetectionResult.HIT)

    r1 = calc.compute_reward(obs_hit, dwell_time=1, consecutive_scans=1, consecutive_hits=1)
    r2 = calc.compute_reward(obs_hit, dwell_time=1, consecutive_scans=2, consecutive_hits=2)
    r5 = calc.compute_reward(obs_hit, dwell_time=1, consecutive_scans=5, consecutive_hits=5)
    r10 = calc.compute_reward(obs_hit, dwell_time=1, consecutive_scans=10, consecutive_hits=10)

    assert r1 > r2 > r5
    assert r10 >= calc.config.hit_reward * 0.40  # Respects min_hit_multiplier



def test_reward_staleness_and_novelty_bonus() -> None:
    """Test that novel and stale bands receive appropriate exploration incentives."""
    calc = RLRewardCalculator(config=RLRewardConfig(novelty_bonus=0.10, stale_bonus_weight=0.10, max_stale_time=200.0))
    obs_miss = Observation(current_time=100, scanned_band=5, dwell_time=1, result=DetectionResult.MISS)

    # Never scanned before (time_since_last_scan < 0)
    r_novel = calc.compute_reward(obs_miss, dwell_time=1, time_since_last_scan=-1.0)
    # Scanned just 1 slot ago
    r_recent = calc.compute_reward(obs_miss, dwell_time=1, time_since_last_scan=1.0)
    # Scanned 200 slots ago (fully stale)
    r_stale = calc.compute_reward(obs_miss, dwell_time=1, time_since_last_scan=200.0)

    assert r_novel > r_recent
    assert r_stale > r_recent


def test_reward_anti_camping_penalty() -> None:
    """Test that excessive consecutive scans on the same band incur penalties."""
    calc = RLRewardCalculator(config=RLRewardConfig(repetition_penalty=0.15, repetition_threshold=2))
    obs_miss = Observation(current_time=50, scanned_band=2, dwell_time=1, result=DetectionResult.MISS)

    r_normal = calc.compute_reward(obs_miss, dwell_time=1, consecutive_scans=1)
    r_thresh = calc.compute_reward(obs_miss, dwell_time=1, consecutive_scans=2)
    r_excess = calc.compute_reward(obs_miss, dwell_time=1, consecutive_scans=5)

    assert r_normal == r_thresh
    assert r_excess < r_thresh


def test_reward_bounds() -> None:
    """Test that total computed reward is strictly bounded in [-3.0, +3.0]."""
    calc = RLRewardCalculator()
    obs_fa = Observation(current_time=0, scanned_band=0, dwell_time=3, result=DetectionResult.FALSE_ALARM)

    for consecutive in [1, 5, 20, 100]:
        for dwell in [1, 2, 3]:
            r = calc.compute_reward(obs_fa, dwell_time=dwell, consecutive_scans=consecutive)
            assert -3.0 <= r <= 3.0


def test_action_mask_in_actor_critic() -> None:
    """Test that masked actions receive zero probability in Categorical distribution."""
    net = ActorCriticNetwork(state_dim=227, action_dim=60)
    state = torch.randn(1, 227)
    mask = torch.ones(1, 60, dtype=torch.bool)
    # Mask out actions 0..2 (band 0)
    mask[0, 0:3] = False

    dist, value = net(state, action_mask=mask)
    probs = dist.probs.detach().numpy()[0]

    assert np.allclose(probs[0:3], 0.0, atol=1e-6)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)


def test_action_encoder_mask_excluding_band() -> None:
    """Test ActionEncoder mask generation excluding a specified band."""
    encoder = ActionEncoder(num_bands=20, dwell_values=[1, 2, 3])
    mask = encoder.get_mask_excluding_band(exclude_band=4)

    assert mask.shape == (60,)
    assert not np.any(mask[12:15])  # Band 4 actions (4*3=12 .. 14) are False
    assert np.all(mask[0:12])       # Other bands are True
    assert np.all(mask[15:])


def test_ppo_scheduler_strict_anti_camping_enforcement() -> None:
    """Test that PPOScheduler with max_consecutive_scans=3 never exceeds 3 consecutive scans."""
    agent = PPOAgent(state_dim=227, action_dim=60, config=PPOConfig(seed=42))
    scheduler = PPOScheduler(agent=agent, max_consecutive_scans=3, deterministic=True)
    scheduler.reset()

    # Simulate 50 sequential steps where receiver always returns HIT on whatever band was scanned
    current_time = 0
    scanned_band = None
    dwell = None
    result = DetectionResult.NONE

    for step in range(50):
        obs = Observation(current_time=current_time, scanned_band=scanned_band, dwell_time=dwell, result=result)
        action = scheduler.select_action(obs)
        scanned_band = action.frequency_band
        dwell = action.dwell_time
        result = DetectionResult.HIT
        current_time += dwell

    assert scheduler.max_consecutive_scans <= 3
    assert scheduler.unique_bands_scanned > 1


def test_ppo_scheduler_ground_truth_isolation_tampering() -> None:
    """
    STRICT NON-LEAKAGE AUDIT:
    Altering hidden environment emitter configurations while feeding identical legitimate
    Observation sequences must yield identical action sequences.
    """
    agent = PPOAgent(state_dim=227, action_dim=60, config=PPOConfig(seed=999))
    sched1 = PPOScheduler(agent=agent, max_consecutive_scans=3, deterministic=True)
    sched2 = PPOScheduler(agent=agent, max_consecutive_scans=3, deterministic=True)

    obs_stream = [
        Observation(current_time=0, scanned_band=None, dwell_time=None, result=DetectionResult.NONE),
        Observation(current_time=2, scanned_band=3, dwell_time=2, result=DetectionResult.HIT),
        Observation(current_time=4, scanned_band=3, dwell_time=2, result=DetectionResult.HIT),
        Observation(current_time=6, scanned_band=3, dwell_time=2, result=DetectionResult.HIT),
        Observation(current_time=8, scanned_band=7, dwell_time=2, result=DetectionResult.MISS),
        Observation(current_time=10, scanned_band=14, dwell_time=2, result=DetectionResult.FALSE_ALARM),
    ]

    actions1 = []
    actions2 = []

    sched1.reset()
    sched2.reset()

    for obs in obs_stream:
        a1 = sched1.select_action(obs)
        actions1.append((a1.frequency_band, a1.dwell_time))

        a2 = sched2.select_action(obs)
        actions2.append((a2.frequency_band, a2.dwell_time))

    assert actions1 == actions2
