"""
Unit tests for RFRLGymEnv and RLRewardCalculator (SIH26055 Phase 5).
"""

import numpy as np
import pytest

from environment.config import EnvironmentConfig
from environment.types import DetectionResult, Observation
from rl.reward import RLRewardCalculator, RLRewardConfig
from rl.rf_rl_env import RFRLGymEnv
from rl.state_features import RLStateExtractor


def test_rl_reward_calculator_outcomes() -> None:
    config = RLRewardConfig(
        hit_reward=1.0,
        miss_penalty=-0.05,
        false_alarm_penalty=-0.50,
        dwell_cost=0.05,
        repetition_penalty=0.10,
        repetition_threshold=2,
    )
    calculator = RLRewardCalculator(config=config)

    # 1. HIT with dwell 1, 1st scan -> +1.0
    obs_hit = Observation(current_time=10, scanned_band=5, dwell_time=1, result=DetectionResult.HIT)
    r_hit = calculator.compute_reward(obs_hit, dwell_time=1, consecutive_scans=1)
    assert r_hit == 1.0

    # 2. HIT with dwell 3 (2 extra slots), 1st scan -> 1.0 - 0.05 * 2 = 0.90
    r_hit_d3 = calculator.compute_reward(obs_hit, dwell_time=3, consecutive_scans=1)
    assert pytest.approx(r_hit_d3) == 0.90

    # 3. MISS with dwell 1 -> -0.05
    obs_miss = Observation(current_time=10, scanned_band=5, dwell_time=1, result=DetectionResult.MISS)
    r_miss = calculator.compute_reward(obs_miss, dwell_time=1, consecutive_scans=1)
    assert pytest.approx(r_miss) == -0.05

    # 4. FALSE_ALARM with dwell 1 -> -0.50
    obs_fa = Observation(current_time=10, scanned_band=5, dwell_time=1, result=DetectionResult.FALSE_ALARM)
    r_fa = calculator.compute_reward(obs_fa, dwell_time=1, consecutive_scans=1)
    assert pytest.approx(r_fa) == -0.50

    # 5. Repetition penalty: 4th scan on same band -> 2 excess repeats -> -0.20
    r_repeat = calculator.compute_reward(obs_hit, dwell_time=1, consecutive_scans=4)
    assert pytest.approx(r_repeat) == 1.0 - 0.20


def test_rl_state_extractor_dimension_and_bounds() -> None:
    extractor = RLStateExtractor(num_bands=20, max_time_slots=1000, max_dwell=3)
    assert extractor.state_dim == 227

    # Initial state before any steps
    init_obs = Observation(current_time=0, scanned_band=None, dwell_time=None, result=DetectionResult.NONE)
    state = extractor.extract_state(init_obs)
    assert state.shape == (227,)
    assert np.all(np.isfinite(state))
    assert np.all(state >= -1.0) and np.all(state <= 1.0)

    # Step update
    obs1 = Observation(current_time=5, scanned_band=3, dwell_time=2, result=DetectionResult.HIT)
    extractor.update(obs1)
    state1 = extractor.extract_state(obs1)
    assert state1.shape == (227,)
    assert np.all(np.isfinite(state1))
    assert np.all(state1 >= -1.0) and np.all(state1 <= 1.0)


def test_rf_rl_gym_env_lifecycle() -> None:
    env_config = EnvironmentConfig()
    env = RFRLGymEnv(env_config=env_config, seed=42)

    assert env.action_space.n == 60
    assert env.observation_space.shape == (227,)

    state, info = env.reset(seed=42)
    assert state.shape == (227,)
    assert isinstance(info, dict)
    assert np.all(np.isfinite(state))

    # Take 5 steps
    for step_i in range(5):
        action_id = step_i * 3  # Bands 0, 1, 2, 3, 4 with dwell 1
        next_state, reward, terminated, truncated, step_info = env.step(action_id)
        assert next_state.shape == (227,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
        assert np.all(np.isfinite(next_state))


def test_rf_rl_gym_env_episode_termination() -> None:
    env_config = EnvironmentConfig()
    env_config.simulation_duration = 20  # Short horizon for quick test
    env = RFRLGymEnv(env_config=env_config, seed=10)


    state, _ = env.reset(seed=10)
    terminated = False
    step_count = 0

    while not terminated and step_count < 50:
        _, _, terminated, _, _ = env.step(0)
        step_count += 1

    assert terminated is True
    assert step_count <= 20
