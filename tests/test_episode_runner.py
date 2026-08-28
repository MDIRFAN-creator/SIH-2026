"""
Integration tests for EpisodeRunner (SIH26055 - Phase 2).
"""

import pytest
from environment.config import EnvironmentConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, Observation
from runners.episode_runner import EpisodeResult, EpisodeRunner
from schedulers.open_loop import OpenLoopScheduler


def test_runner_single_episode_execution():
    """Verify EpisodeRunner executes full episode to simulation_duration."""
    config = EnvironmentConfig(num_bands=20, simulation_duration=200, seed=42)
    env = RFEnvironment(config)
    scheduler = OpenLoopScheduler(num_bands=20, dwell_time=2)
    runner = EpisodeRunner()

    result = runner.run_episode(env=env, scheduler=scheduler, seed=42)

    assert isinstance(result, EpisodeResult)
    assert result.total_time_slots == 200
    assert result.total_decisions == 100  # 200 / 2 dwell = 100
    assert len(result.step_records) == 100
    assert len(result.dwell_history) == 100
    assert result.scheduler_name == "OpenLoopScheduler"

    # Verify chronological continuity
    for idx, rec in enumerate(result.step_records):
        assert rec.decision_index == idx
        assert rec.start_time == idx * 2
        assert rec.end_time == idx * 2 + 1
        assert rec.action.dwell_time == 2
        assert rec.action.frequency_band == idx % 20


def test_runner_multi_seed_execution():
    """Verify EpisodeRunner executes across multiple seeds."""
    config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=100,
        emitters=[
            {
                "emitter_id": "rand_agile",
                "emitter_type": "AGILE_RANDOM",
                "allowed_bands": [0, 5, 10, 15],
                "hop_period": 5,
            }
        ],
    )
    env = RFEnvironment(config)
    scheduler = OpenLoopScheduler(num_bands=20, dwell_time=1)
    runner = EpisodeRunner()

    seeds = [10, 20, 30]
    results = runner.run_episodes_multi_seed(env=env, scheduler=scheduler, seeds=seeds)

    assert len(results) == 3
    for idx, res in enumerate(results):
        assert res.seed == seeds[idx]
        assert res.total_time_slots == 100
        assert res.total_decisions == 100


class SpyScheduler(OpenLoopScheduler):
    """Spy scheduler to verify strict non-leakage during runner execution."""

    def __init__(self, num_bands: int = 20):
        super().__init__(num_bands=num_bands)
        self.received_objects = []

    def select_action(self, observation: Observation) -> Action:
        self.received_objects.append(observation)
        return super().select_action(observation)


def test_runner_strict_observation_isolation():
    """Verify that only Observation instances are passed into the scheduler."""
    config = EnvironmentConfig(num_bands=20, simulation_duration=50)
    env = RFEnvironment(config)
    spy = SpyScheduler(num_bands=20)
    runner = EpisodeRunner()

    runner.run_episode(env=env, scheduler=spy, seed=42)

    assert len(spy.received_objects) == 50
    for item in spy.received_objects:
        assert isinstance(item, Observation)
        assert not hasattr(item, "dwell_summary")
        assert not hasattr(item, "ground_truth")
        assert not hasattr(item, "active_emitter_ids")
