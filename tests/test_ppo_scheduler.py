"""
Integration and Non-Leakage tests for PPOScheduler (SIH26055 Phase 5).
"""

import copy
import numpy as np
import pytest

from environment.config import EnvironmentConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, DetectionResult, Observation
from evaluation.baseline_metrics import BaselineMetrics, calculate_baseline_metrics
from rl.ppo_agent import PPOAgent, PPOConfig
from runners.episode_runner import EpisodeRunner
from schedulers.base import BaseScheduler
from schedulers.ppo_scheduler import PPOScheduler


def test_ppo_scheduler_interface_compliance() -> None:
    agent = PPOAgent(state_dim=227, action_dim=60, config=PPOConfig(seed=42))
    scheduler = PPOScheduler(agent=agent, deterministic=True)

    assert isinstance(scheduler, BaseScheduler)
    assert scheduler.name == "PPOAdaptiveScheduler"

    # Reset
    scheduler.reset()
    assert scheduler.total_decisions == 0
    assert scheduler.unique_bands_scanned == 0

    # Action selection
    obs0 = Observation(current_time=0, scanned_band=None, dwell_time=None, result=DetectionResult.NONE)
    action0 = scheduler.select_action(obs0)
    assert isinstance(action0, Action)
    assert 0 <= action0.frequency_band < 20
    assert action0.dwell_time in [1, 2, 3]


def test_ppo_scheduler_episode_runner_integration() -> None:
    env_config = EnvironmentConfig()
    env = RFEnvironment(config=env_config)

    agent = PPOAgent(state_dim=227, action_dim=60, config=PPOConfig(seed=42))
    scheduler = PPOScheduler(agent=agent, deterministic=True)

    runner = EpisodeRunner()
    result = runner.run_episode(env=env, scheduler=scheduler, seed=42)

    assert result.scheduler_name == "PPOAdaptiveScheduler"
    assert result.total_time_slots >= env_config.simulation_duration
    assert len(result.step_records) > 0


    # Calculate metrics
    metrics = calculate_baseline_metrics(result, env.emitter_registry)
    assert isinstance(metrics, BaselineMetrics)
    assert 0.0 <= metrics.interception_rate <= 1.0
    assert 0.0 <= metrics.dwell_efficiency <= 1.0



def test_ppo_scheduler_ground_truth_tampering_invariance() -> None:
    """
    CRITICAL NON-LEAKAGE AUDIT:
    Modifying hidden environment ground truth must NOT affect the PPOScheduler's
    decisions if the scheduler-visible observation stream remains identical.
    """
    agent = PPOAgent(state_dim=227, action_dim=60, config=PPOConfig(seed=777))
    sched1 = PPOScheduler(agent=agent, deterministic=True)
    sched2 = PPOScheduler(agent=agent, deterministic=True)

    # Identical synthetic observation sequence
    obs_seq = [
        Observation(current_time=0, scanned_band=None, dwell_time=None, result=DetectionResult.NONE),
        Observation(current_time=2, scanned_band=4, dwell_time=2, result=DetectionResult.HIT),
        Observation(current_time=5, scanned_band=4, dwell_time=3, result=DetectionResult.HIT),
        Observation(current_time=6, scanned_band=12, dwell_time=1, result=DetectionResult.MISS),
        Observation(current_time=8, scanned_band=12, dwell_time=2, result=DetectionResult.FALSE_ALARM),
    ]

    actions1 = []
    actions2 = []

    sched1.reset()
    sched2.reset()

    for obs in obs_seq:
        a1 = sched1.select_action(obs)
        actions1.append((a1.frequency_band, a1.dwell_time))

        a2 = sched2.select_action(obs)
        actions2.append((a2.frequency_band, a2.dwell_time))

    assert actions1 == actions2, "PPOScheduler decision path must be strictly deterministic and observation-derived"


def test_ppo_scheduler_determinism_and_reproducibility() -> None:
    """
    Deterministic inference guarantee: same model + same seed -> identical evaluation trajectory.
    """
    env_config = EnvironmentConfig()
    runner = EpisodeRunner()

    agent = PPOAgent(state_dim=227, action_dim=60, config=PPOConfig(seed=101))

    env1 = RFEnvironment(config=env_config)
    sched1 = PPOScheduler(agent=agent, deterministic=True)
    res1 = runner.run_episode(env=env1, scheduler=sched1, seed=55)

    env2 = RFEnvironment(config=env_config)
    sched2 = PPOScheduler(agent=agent, deterministic=True)
    res2 = runner.run_episode(env=env2, scheduler=sched2, seed=55)

    assert len(res1.step_records) == len(res2.step_records)
    for rec1, rec2 in zip(res1.step_records, res2.step_records):
        assert rec1.action.frequency_band == rec2.action.frequency_band
        assert rec1.action.dwell_time == rec2.action.dwell_time
        assert rec1.observation.result == rec2.observation.result

