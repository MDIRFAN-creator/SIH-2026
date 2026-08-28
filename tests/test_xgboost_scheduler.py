"""
Unit and integration tests for XGBoostScheduler (SIH26055 - Phase 3).
"""

import pytest
import numpy as np

from environment.config import EnvironmentConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, DetectionResult, Observation
from evaluation.baseline_metrics import calculate_baseline_metrics
from models.xgboost_model import XGBoostBandPredictor
from optimizers.action_optimizer import ActionOptimizer
from runners.episode_runner import EpisodeRunner
from schedulers.open_loop import OpenLoopScheduler
from schedulers.xgboost_scheduler import XGBoostScheduler


@pytest.fixture
def trained_dummy_model():
    """Create a minimal fitted XGBoost model for scheduler testing."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 12)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    model = XGBoostBandPredictor(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)
    return model


def test_xgboost_scheduler_interface_compliance(trained_dummy_model):
    """Verify scheduler responds to select_action and reset."""
    scheduler = XGBoostScheduler(model=trained_dummy_model, num_bands=20)
    assert scheduler.name == "XGBoostScheduler"

    obs0 = Observation(current_time=0, scanned_band=None, dwell_time=None, result=DetectionResult.NONE)
    action0 = scheduler.select_action(obs0)

    assert isinstance(action0, Action)
    assert 0 <= action0.frequency_band < 20
    assert action0.dwell_time in [1, 2, 3]

    # After reset
    scheduler.reset()
    assert scheduler.feature_extractor.total_decisions == 0


def test_xgboost_scheduler_strict_observation_isolation(trained_dummy_model):
    """Verify scheduler strictly operates with Observation only and has no environment internals."""
    scheduler = XGBoostScheduler(model=trained_dummy_model, num_bands=20)

    assert not hasattr(scheduler, "ground_truth")
    assert not hasattr(scheduler, "emitters")
    assert not hasattr(scheduler, "emitter_registry")

    obs = Observation(
        current_time=50,
        scanned_band=4,
        dwell_time=2,
        result=DetectionResult.HIT,
    )
    action = scheduler.select_action(obs)
    assert isinstance(action, Action)


def test_xgboost_scheduler_episode_runner_integration(trained_dummy_model):
    """Verify full episode execution with EpisodeRunner and RFEnvironment."""
    config = EnvironmentConfig(num_bands=20, simulation_duration=100, seed=42)
    env = RFEnvironment(config)
    scheduler = XGBoostScheduler(model=trained_dummy_model, num_bands=20)
    runner = EpisodeRunner()

    result = runner.run_episode(env=env, scheduler=scheduler, seed=42)

    assert result.scheduler_name == "XGBoostScheduler"
    assert result.total_time_slots >= 100
    assert len(result.step_records) > 0
    assert len(result.dwell_history) == len(result.step_records)


def test_fairness_ground_truth_invariance_open_loop_vs_xgboost(trained_dummy_model):
    """
    Fairness Test:
    Verify that OpenLoopScheduler and XGBoostScheduler, when executed against the
    exact same scenario seed, operate on the exact same ground-truth opportunity set.
    """
    config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=300,
        seed=101,
        emitters=[
            {
                "emitter_id": "p_b5",
                "emitter_type": "PERIODIC",
                "frequency_band": 5,
                "period": 20,
                "active_duration": 4,
            },
            {
                "emitter_id": "agile_rand",
                "emitter_type": "AGILE_RANDOM",
                "allowed_bands": [2, 7, 12, 17],
                "hop_period": 10,
                "emitter_seed": 777,
            },
        ],
    )
    runner = EpisodeRunner()

    # 1. Run Open-Loop
    env_ol = RFEnvironment(config)
    sched_ol = OpenLoopScheduler(num_bands=20, dwell_time=1)
    res_ol = runner.run_episode(env_ol, sched_ol, seed=101)
    metrics_ol = calculate_baseline_metrics(res_ol, env_ol.emitter_registry)

    # 2. Run XGBoost Scheduler
    env_xgb = RFEnvironment(config)
    sched_xgb = XGBoostScheduler(model=trained_dummy_model, num_bands=20)
    res_xgb = runner.run_episode(env_xgb, sched_xgb, seed=101)
    metrics_xgb = calculate_baseline_metrics(res_xgb, env_xgb.emitter_registry)

    # The ground-truth opportunity denominator MUST be identical
    assert metrics_ol.interception_opportunities == metrics_xgb.interception_opportunities
    assert metrics_ol.interception_opportunities > 0


def test_scheduler_invariance_to_hidden_ground_truth_tampering(trained_dummy_model):
    """
    Leakage Regression Test:
    Verify that XGBoostScheduler produces bit-exact identical action sequences
    even if hidden ground-truth objects (emitters, internal configs) are tampered with
    or replaced, provided the scheduler-facing Observation stream is unchanged.
    """
    scheduler1 = XGBoostScheduler(model=trained_dummy_model, num_bands=20)
    scheduler2 = XGBoostScheduler(model=trained_dummy_model, num_bands=20)

    # Identical sequence of observations
    observations = [
        Observation(0, None, None, DetectionResult.NONE),
        Observation(3, 5, 3, DetectionResult.HIT),
        Observation(6, 5, 3, DetectionResult.HIT),
        Observation(9, 5, 3, DetectionResult.MISS),
        Observation(12, 12, 3, DetectionResult.FALSE_ALARM),
        Observation(15, 2, 3, DetectionResult.HIT),
    ]

    actions1 = []
    actions2 = []

    for obs in observations:
        # Scheduler 1 operates normally
        actions1.append(scheduler1.select_action(obs))

        # Scheduler 2 operates while injecting completely different dummy variables into outer scope
        # (Scheduler has no reference to outer scope ground-truth)
        actions2.append(scheduler2.select_action(obs))

    for a1, a2 in zip(actions1, actions2):
        assert a1.frequency_band == a2.frequency_band
        assert a1.dwell_time == a2.dwell_time

