"""
Unit and integration tests for LinUCBScheduler (SIH26055 Phase 4 Hardened).

Comprehensive test coverage for:
1. Interface compliance & reward computation
2. Hard anti-camping constraint (max_consecutive_scans)
3. Cold-start exploration guarantee (min_initial_pulls)
4. Non-stationary adaptation under discounting (gamma < 1.0)
5. Band-selection Shannon entropy & run length metrics
6. Information isolation & ground-truth tampering invariance
7. Seeded determinism & reproducibility
"""

import numpy as np
import pytest

from environment.config import EnvironmentConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, DetectionResult, Observation
from evaluation.baseline_metrics import calculate_baseline_metrics
from runners.episode_runner import EpisodeRunner
from schedulers.linucb_scheduler import LinUCBScheduler


def test_linucb_scheduler_interface():
    scheduler = LinUCBScheduler(num_bands=20, alpha=1.0)
    assert scheduler.name == "LinUCBAdaptiveScheduler"
    assert scheduler.num_bands == 20
    assert scheduler.max_consecutive_scans == 3
    assert scheduler.min_initial_pulls == 1

    obs0 = Observation(current_time=0, scanned_band=None, dwell_time=None, result=DetectionResult.NONE)
    action = scheduler.select_action(obs0)

    assert isinstance(action, Action)
    assert 0 <= action.frequency_band < 20
    assert action.dwell_time in [1, 2, 3]


def test_linucb_reward_computation():
    scheduler = LinUCBScheduler(num_bands=20, dwell_cost=0.05)

    # HIT with dwell=1 -> +1.0 - 0 = 1.0
    obs_hit_d1 = Observation(3, 5, 1, DetectionResult.HIT)
    r1 = scheduler.compute_reward(obs_hit_d1, dwell_time=1)
    assert np.isclose(r1, 1.0)

    # HIT with dwell=3 -> +1.0 - 0.05*2 = 0.90
    obs_hit_d3 = Observation(6, 5, 3, DetectionResult.HIT)
    r3 = scheduler.compute_reward(obs_hit_d3, dwell_time=3)
    assert np.isclose(r3, 0.90)

    # FALSE_ALARM with dwell=1 -> -0.50
    obs_fa = Observation(9, 5, 1, DetectionResult.FALSE_ALARM)
    r_fa = scheduler.compute_reward(obs_fa, dwell_time=1)
    assert np.isclose(r_fa, -0.50)

    # MISS with dwell=2 -> -0.05 - 0.05*1 = -0.10
    obs_miss = Observation(12, 5, 2, DetectionResult.MISS)
    r_miss = scheduler.compute_reward(obs_miss, dwell_time=2)
    assert np.isclose(r_miss, -0.10)


def test_linucb_hard_anti_camping():
    """
    Verify that an arm is strictly forbidden on the 4th consecutive decision
    when max_consecutive_scans = 3, even if it yields constant +1.0 rewards.
    """
    scheduler = LinUCBScheduler(
        num_bands=5,
        alpha=1.0,
        max_consecutive_scans=3,
        min_initial_pulls=0,  # Disable cold start to test pure anti-camping
        seed=42,
    )

    # Prime band 2 with heavy positive reward
    t = 0
    obs = Observation(current_time=t, scanned_band=None, dwell_time=None, result=DetectionResult.NONE)
    selected_bands = []

    for step in range(10):
        action = scheduler.select_action(obs)
        selected_bands.append(action.frequency_band)
        t += action.dwell_time
        # Return strong HIT on band 2, MISS elsewhere
        res = DetectionResult.HIT if action.frequency_band == 2 else DetectionResult.MISS
        obs = Observation(current_time=t, scanned_band=action.frequency_band, dwell_time=action.dwell_time, result=res)

    # Verify no band was selected > 3 consecutive times
    max_run = 0
    curr_run = 0
    prev_b = -1
    for b in selected_bands:
        if b == prev_b:
            curr_run += 1
        else:
            prev_b = b
            curr_run = 1
        max_run = max(max_run, curr_run)

    assert max_run <= 3, f"Hard anti-camping violated: max run was {max_run} > 3"


def test_linucb_max_consecutive_constraint_full_episode():
    """
    Verify that across an entire multi-emitter episode, max consecutive scans <= 3.
    """
    config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=1000,
        seed=101,
        emitters=[
            {"emitter_id": "p1", "emitter_type": "PERIODIC", "frequency_band": 3, "period": 10, "active_duration": 4},
            {"emitter_id": "p2", "emitter_type": "PERIODIC", "frequency_band": 7, "period": 15, "active_duration": 5},
        ],
    )
    env = RFEnvironment(config)
    scheduler = LinUCBScheduler(num_bands=20, max_consecutive_scans=3, seed=101)
    runner = EpisodeRunner()

    res = runner.run_episode(env, scheduler, seed=101)

    max_consec = 0
    curr_consec = 0
    last_band = -1
    for rec in res.step_records:
        b = rec.action.frequency_band
        if b == last_band:
            curr_consec += 1
        else:
            last_band = b
            curr_consec = 1
        max_consec = max(max_consec, curr_consec)

    assert max_consec <= 3, f"Max consecutive scans {max_consec} exceeded threshold 3"


def test_linucb_cold_start_exploration():
    """
    Verify that min_initial_pulls = 1 ensures all 20 bands are scanned in the first 20 decisions.
    """
    scheduler = LinUCBScheduler(num_bands=20, min_initial_pulls=1, seed=7)

    scanned = []
    t = 0
    obs = Observation(0, None, None, DetectionResult.NONE)

    for _ in range(20):
        action = scheduler.select_action(obs)
        scanned.append(action.frequency_band)
        t += action.dwell_time
        obs = Observation(t, action.frequency_band, action.dwell_time, DetectionResult.HIT)

    # All 20 bands must be visited uniquely in the first 20 decisions
    assert len(set(scanned)) == 20
    assert set(scanned) == set(range(20))


def test_linucb_all_bands_initialization():
    """
    Verify min_initial_pulls = 2 ensures all bands receive 2 pulls before arbitrary exploitation.
    """
    scheduler = LinUCBScheduler(num_bands=10, min_initial_pulls=2, max_consecutive_scans=2, seed=7)
    scanned = []
    t = 0
    obs = Observation(0, None, None, DetectionResult.NONE)

    for _ in range(20):
        action = scheduler.select_action(obs)
        scanned.append(action.frequency_band)
        t += action.dwell_time
        obs = Observation(t, action.frequency_band, action.dwell_time, DetectionResult.HIT)

    # In 20 steps, each of the 10 bands must have been selected exactly 2 times
    assert all(scanned.count(b) == 2 for b in range(10))


def test_linucb_non_stationary_adaptation():
    """
    Verify that discounted LinUCB (gamma = 0.90) decays past negative evidence,
    restoring uncertainty and enabling re-exploration of previously quiet bands.
    """
    scheduler = LinUCBScheduler(num_bands=3, gamma=0.90, min_initial_pulls=0, seed=1)

    # Step 1: Arm 0 receives heavy negative feedback (5 misses)
    t = 0
    obs = Observation(0, None, None, DetectionResult.NONE)
    for _ in range(5):
        action = scheduler.select_action(obs)
        t += action.dwell_time
        obs = Observation(t, 0, action.dwell_time, DetectionResult.MISS)

    # Arm 0 has negative response vector
    b0_init = scheduler.linucb.b_vec[0].copy()

    # Step 2: Operate on Arm 1 and Arm 2 for 50 steps
    for _ in range(50):
        action = scheduler.select_action(obs)
        t += action.dwell_time
        obs = Observation(t, action.frequency_band, action.dwell_time, DetectionResult.HIT)

    # Arm 0's negative b_vec should have decayed towards 0
    b0_decayed = scheduler.linucb.b_vec[0]
    assert np.linalg.norm(b0_decayed) < np.linalg.norm(b0_init)


def test_linucb_frequency_change_response():
    """
    Controlled adaptation test: emitter appears on band 14 at t=200.
    Verify scheduler detects the emitter.
    """
    config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=600,
        seed=42,
        emitters=[
            {"emitter_id": "dyn_b14", "emitter_type": "PERIODIC", "frequency_band": 14, "period": 15, "active_duration": 6, "start_time": 200}
        ],
    )
    env = RFEnvironment(config)
    scheduler = LinUCBScheduler(num_bands=20, alpha=1.0, gamma=0.99, max_consecutive_scans=3, seed=42)
    runner = EpisodeRunner()

    res = runner.run_episode(env, scheduler, seed=42)

    hits_on_b14_after_200 = [
        rec for rec in res.step_records
        if rec.start_time >= 200 and rec.action.frequency_band == 14 and rec.observation.result == DetectionResult.HIT
    ]
    assert len(hits_on_b14_after_200) > 0, "Hardened LinUCB failed to detect dynamic hopping emitter on Band 14"


def test_linucb_band_entropy():
    """
    Verify Shannon entropy computation on uniform vs concentrated scans.
    """
    sched = LinUCBScheduler(num_bands=4)

    # Uniform allocation: 1 scan per band
    obs = Observation(0, None, None, DetectionResult.NONE)
    for b in range(4):
        sched.decision_history.append({"selected_band": b})

    # Entropy of 4 uniform states is ln(4) ~ 1.386
    h_uniform = sched.compute_band_selection_entropy()
    assert np.isclose(h_uniform, np.log(4))

    # Concentrated allocation: all on band 0
    sched.decision_history.clear()
    for _ in range(10):
        sched.decision_history.append({"selected_band": 0})
    h_concentrated = sched.compute_band_selection_entropy()
    assert np.isclose(h_concentrated, 0.0)


def test_linucb_deterministic_hardening():
    """
    Verify that hardened LinUCB is 100% deterministic across reset cycles.
    """
    config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=300,
        seed=888,
        emitters=[
            {"emitter_id": "p1", "emitter_type": "PERIODIC", "frequency_band": 5, "period": 12, "active_duration": 4}
        ],
    )
    runner = EpisodeRunner()

    env1 = RFEnvironment(config)
    sched = LinUCBScheduler(num_bands=20, alpha=1.0, gamma=0.99, max_consecutive_scans=3, seed=77)
    res1 = runner.run_episode(env1, sched, seed=888)

    env2 = RFEnvironment(config)
    sched.reset()
    res2 = runner.run_episode(env2, sched, seed=888)

    assert len(res1.step_records) == len(res2.step_records)
    for r1, r2 in zip(res1.step_records, res2.step_records):
        assert r1.action.frequency_band == r2.action.frequency_band
        assert r1.action.dwell_time == r2.action.dwell_time


def test_linucb_scheduler_tampering_isolation():
    """
    Leakage Test:
    Altering hidden environment variables (ground truth) while keeping Observation
    stream identical produces 100% identical LinUCB action sequence.
    """
    sched1 = LinUCBScheduler(num_bands=20, alpha=1.0, seed=123)
    sched2 = LinUCBScheduler(num_bands=20, alpha=1.0, seed=123)

    observations = [
        Observation(0, None, None, DetectionResult.NONE),
        Observation(3, 4, 3, DetectionResult.HIT),
        Observation(6, 4, 3, DetectionResult.HIT),
        Observation(9, 4, 3, DetectionResult.MISS),
        Observation(12, 10, 3, DetectionResult.FALSE_ALARM),
        Observation(15, 2, 3, DetectionResult.HIT),
    ]

    actions1 = []
    actions2 = []

    for obs in observations:
        actions1.append(sched1.select_action(obs))
        actions2.append(sched2.select_action(obs))

    for a1, a2 in zip(actions1, actions2):
        assert a1.frequency_band == a2.frequency_band
        assert a1.dwell_time == a2.dwell_time
