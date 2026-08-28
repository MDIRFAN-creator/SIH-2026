"""
Scheduler interface validation tests (SIH26055 - Phase 1).

Verifies that the RF Environment is completely decoupled from any scheduling algorithm
and can be operated by generic schedulers consuming strictly non-leaked observations.
"""

from typing import List, Optional
import numpy as np

from environment.config import EnvironmentConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, Observation


class DummyOpenLoopScheduler:
    """A simple round-robin baseline scanner demonstrating the scheduler contract."""

    def __init__(self, num_bands: int = 20, fixed_dwell: int = 2) -> None:
        self.num_bands = num_bands
        self.fixed_dwell = fixed_dwell
        self.current_band_idx = 0
        self.history: List[Observation] = []

    def reset(self) -> None:
        self.current_band_idx = 0
        self.history.clear()

    def select_action(self, observation: Observation) -> Action:
        # Scheduler consumes observation only
        self.history.append(observation)
        action = Action(frequency_band=self.current_band_idx, dwell_time=self.fixed_dwell)
        self.current_band_idx = (self.current_band_idx + 1) % self.num_bands
        return action


class DummyRandomScheduler:
    """A stochastic exploration scanner demonstrating the scheduler contract."""

    def __init__(self, num_bands: int = 20, allowed_dwells: Optional[List[int]] = None, seed: int = 42) -> None:
        self.num_bands = num_bands
        self.allowed_dwells = allowed_dwells if allowed_dwells else [1, 2, 3, 5]
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def select_action(self, observation: Observation) -> Action:
        band = int(self._rng.integers(0, self.num_bands))
        dwell = int(self._rng.choice(self.allowed_dwells))
        return Action(frequency_band=band, dwell_time=dwell)


def test_open_loop_scheduler_interaction():
    """Verify that a dummy open-loop scheduler can run an entire simulation episode."""
    config = EnvironmentConfig(num_bands=20, simulation_duration=500, seed=42)
    env = RFEnvironment(config)
    scheduler = DummyOpenLoopScheduler(num_bands=20, fixed_dwell=2)

    obs = env.reset(seed=42)
    scheduler.reset()

    step_count = 0
    terminated = False
    total_reward = 0.0

    while not terminated:
        action = scheduler.select_action(obs)
        obs, reward, terminated, info = env.step(action)
        total_reward += reward
        step_count += 1

    assert env.current_time >= 500
    assert terminated is True
    assert step_count == 250  # 500 duration / 2 dwell = 250 decisions
    assert len(scheduler.history) == 250


def test_random_scheduler_interaction():
    """Verify that a random scheduler can successfully complete an episode."""
    config = EnvironmentConfig(num_bands=20, simulation_duration=1000, seed=123)
    env = RFEnvironment(config)
    scheduler = DummyRandomScheduler(num_bands=20, seed=456)

    obs = env.reset(seed=123)
    scheduler.reset()

    step_count = 0
    terminated = False

    while not terminated:
        action = scheduler.select_action(obs)
        obs, reward, terminated, info = env.step(action)
        step_count += 1

    assert env.current_time >= 1000
    assert terminated is True
    assert step_count > 0
