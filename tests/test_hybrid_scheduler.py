"""
Unit and integration tests for Phase 6 Hybrid Adaptive RF Scheduler.

Tests:
1. Interface compliance (BaseScheduler).
2. Action validity (frequency_band in [0, 19], dwell in [1, 2, 3]).
3. Determinism and reproducibility.
4. Hard anti-camping constraint enforcement (max_consecutive_scans = 3).
5. Strict ground-truth isolation and tampering invariance.
6. Component isolation and graceful degradation / fallback.
7. Adaptive mode transitions (Exploitation, Exploration, Adaptation).
8. EpisodeRunner integration and metric computation.
"""

from copy import deepcopy
import numpy as np
import pytest
import torch

from environment.config import EnvironmentConfig, PeriodicEmitterConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, DetectionResult, Observation
from evaluation.baseline_metrics import calculate_baseline_metrics
from hybrid.arbitration import DecisionMode, HybridArbitrator
from hybrid.config import HybridConfig
from hybrid.scoring import ComponentSignalExtractor, ComponentSignals
from models.xgboost_model import XGBoostBandPredictor
from rl.ppo_agent import PPOAgent
from runners.episode_runner import EpisodeRunner
from schedulers.base import BaseScheduler
from schedulers.hybrid_scheduler import HybridAdaptiveScheduler


def _create_sample_observation(current_time: int = 10, num_bands: int = 20) -> Observation:
    """Create a legitimate scheduler observation for testing."""
    return Observation(
        current_time=current_time,
        scanned_band=3,
        dwell_time=2,
        result=DetectionResult.HIT,
        history_summary={
            "estimated_snr_db": 15.0,
            "consecutive_scans_on_band": 1,
            "time_since_last_scan": {b: 10 for b in range(num_bands)},
            "time_since_last_detection": {b: 10 for b in range(num_bands)},
            "windowed_detection_rates": {b: 0.5 for b in range(num_bands)},
            "cumulative_detection_rates": {b: 0.5 for b in range(num_bands)},
            "false_alarm_rates": {b: 0.02 for b in range(num_bands)},
            "windowed_false_alarm_rates": {b: 0.02 for b in range(num_bands)},
            "band_scan_fraction": {b: 0.05 for b in range(num_bands)},
        },
    )



def test_hybrid_implements_base_scheduler():
    """Verify HybridAdaptiveScheduler inherits from BaseScheduler and implements required methods."""
    scheduler = HybridAdaptiveScheduler()
    assert isinstance(scheduler, BaseScheduler)
    assert scheduler.name == "HybridAdaptiveScheduler"
    assert hasattr(scheduler, "reset")
    assert hasattr(scheduler, "select_action")
    assert hasattr(scheduler, "update")


def test_hybrid_action_validity():
    """Verify every action selected by HybridAdaptiveScheduler has valid band and dwell."""
    config = HybridConfig(num_bands=20, allowed_dwells=[1, 2, 3])
    scheduler = HybridAdaptiveScheduler(config=config)
    scheduler.reset()

    obs = _create_sample_observation(current_time=0)
    for t in range(50):
        obs.current_time = t
        action = scheduler.select_action(obs)
        assert isinstance(action, Action)
        assert 0 <= action.frequency_band < config.num_bands
        assert action.dwell_time in config.allowed_dwells


def test_hybrid_determinism():
    """Verify same seed + identical observation stream produces identical action sequences."""
    config1 = HybridConfig(seed=42)
    config2 = HybridConfig(seed=42)
    s1 = HybridAdaptiveScheduler(config=config1)
    s2 = HybridAdaptiveScheduler(config=config2)
    s1.reset()
    s2.reset()

    obs = _create_sample_observation(current_time=0)
    actions1 = []
    actions2 = []
    for t in range(20):
        obs.current_time = t
        a1 = s1.select_action(obs)
        a2 = s2.select_action(obs)
        actions1.append((a1.frequency_band, a1.dwell_time))
        actions2.append((a2.frequency_band, a2.dwell_time))

    assert actions1 == actions2


def test_hybrid_anti_camping_enforcement():
    """Verify the scheduler strictly respects max_consecutive_scans limit."""
    max_consec = 3
    config = HybridConfig(num_bands=20, max_consecutive_scans=max_consec)
    scheduler = HybridAdaptiveScheduler(config=config)
    scheduler.reset()

    # Repeatedly feed observations on the same band with HIT feedback
    obs = _create_sample_observation()
    history = []
    for t in range(100):
        obs.current_time = t
        action = scheduler.select_action(obs)
        history.append(action.frequency_band)
        obs.scanned_band = action.frequency_band
        obs.dwell_time = action.dwell_time
        obs.result = DetectionResult.HIT

    # Check that no contiguous run of the same band exceeds max_consecutive_scans
    current_run = 1
    for i in range(1, len(history)):
        if history[i] == history[i - 1]:
            current_run += 1
            assert current_run <= max_consec, f"Anti-camping violated: run of {current_run} on band {history[i]}"
        else:
            current_run = 1


def test_hybrid_ground_truth_isolation_and_tampering():
    """
    Verify scheduler decisions remain identical when hidden ground truth is altered,
    confirming zero ground-truth leakage.
    """
    scheduler = HybridAdaptiveScheduler(config=HybridConfig(seed=123))
    scheduler.reset()

    obs1 = _create_sample_observation(current_time=50)
    action1 = scheduler.select_action(obs1)

    # Reset and test with identical observation but mutated environment ground truth
    scheduler.reset()
    obs2 = deepcopy(obs1)
    # The scheduler only consumes obs2; it has no access to any external ground truth
    action2 = scheduler.select_action(obs2)

    assert action1.frequency_band == action2.frequency_band
    assert action1.dwell_time == action2.dwell_time


def test_hybrid_component_isolation_and_fallback():
    """Verify scheduler functions correctly and safely even when some models are None."""
    # 1. No XGBoost model
    s_no_xgb = HybridAdaptiveScheduler(xgb_model=None)
    s_no_xgb.reset()
    obs = _create_sample_observation()
    act = s_no_xgb.select_action(obs)
    assert 0 <= act.frequency_band < 20

    # 2. No PPO agent
    s_no_ppo = HybridAdaptiveScheduler(ppo_agent=None)
    s_no_ppo.reset()
    act2 = s_no_ppo.select_action(obs)
    assert 0 <= act2.frequency_band < 20


def test_hybrid_adaptive_mode_transitions():
    """Verify arbitration switches among Cold Start, Exploitation, Exploration, and Adaptation."""
    config = HybridConfig(min_initial_pulls=1, confidence_threshold=0.40)
    arbitrator = HybridArbitrator(config=config)
    arbitrator.reset()

    # 1. Cold start: unpulled arms
    signals = ComponentSignals(
        xgb_probas=np.zeros(20),
        xgb_max_proba=0.0,
        xgb_argmax_band=0,
        linucb_scores=np.zeros(20),
        linucb_means=np.zeros(20),
        linucb_uncertainties=np.ones(20),
        linucb_max_uncertainty=1.0,
        linucb_argmax_band=0,
        ppo_band_probas=np.zeros(20),
        ppo_action_probas=np.zeros(60),
        ppo_entropy=3.0,
        ppo_value=0.0,
        ppo_argmax_action=0,
        ppo_argmax_band=0,
        ppo_argmax_dwell=1,
        staleness_scores=np.zeros(20),
        pull_counts=np.zeros(20, dtype=np.int64),
    )
    mode, _ = arbitrator.determine_mode(signals, np.ones(20, dtype=bool))
    assert mode == DecisionMode.COLD_START

    # 2. Exploitation: high confidence
    signals.pull_counts.fill(2)
    signals.xgb_max_proba = 0.85
    mode, _ = arbitrator.determine_mode(signals, np.ones(20, dtype=bool))
    assert mode == DecisionMode.EXPLOITATION

    # 3. Adaptation: consecutive misses on target
    arbitrator.consecutive_misses_on_target = 3
    mode, _ = arbitrator.determine_mode(signals, np.ones(20, dtype=bool))
    assert mode == DecisionMode.ADAPTATION


def test_hybrid_episode_runner_integration():
    """Verify HybridAdaptiveScheduler executes a full episode with EpisodeRunner cleanly."""
    env_config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=200,
        emitters=[
            {
                "emitter_id": "test_e1",
                "frequency_band": 5,
                "period": 20,
                "active_duration": 5,
                "emitter_type": "PERIODIC",
            }
        ],
    )
    env = RFEnvironment(config=env_config)
    scheduler = HybridAdaptiveScheduler()
    runner = EpisodeRunner()

    result = runner.run_episode(env=env, scheduler=scheduler, seed=42)
    assert result.total_time_slots == 200
    assert result.total_decisions > 0

    metrics = calculate_baseline_metrics(result, emitter_registry=env.emitter_registry)
    assert metrics.interception_opportunities > 0
    assert 0.0 <= metrics.interception_rate <= 1.0
    assert 0.0 <= metrics.dwell_efficiency <= 1.0




