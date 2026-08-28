"""
Unit and deterministic verification tests for BaselineMetrics (SIH26055 - Phase 2).
"""

import pytest
from environment.config import EnvironmentConfig
from environment.emitters import EmitterRegistry, PeriodicEmitter
from environment.rf_environment import RFEnvironment
from environment.types import (
    Action,
    DetectionResult,
    DwellSlotOutcome,
    DwellSummary,
    Observation,
)
from evaluation.baseline_metrics import (
    BaselineMetrics,
    aggregate_metrics_across_seeds,
    calculate_baseline_metrics,
    extract_emitter_opportunities,
)
from runners.episode_runner import EpisodeResult, EpisodeRunner, EpisodeStepRecord
from schedulers.open_loop import OpenLoopScheduler


def test_extract_emitter_opportunities():
    """Verify opportunity extraction from emitter models across simulation duration."""
    e1 = PeriodicEmitter("p_b5", frequency_band=5, period=20, active_duration=4, start_time=0)
    e2 = PeriodicEmitter("p_b10", frequency_band=10, period=50, active_duration=10, start_time=100)
    registry = EmitterRegistry([e1, e2])

    opps = extract_emitter_opportunities(registry, simulation_duration=100, num_bands=20)

    # e1 has 5 opportunities in 100 slots: [0..3], [20..23], [40..43], [60..63], [80..83]
    # e2 starts at 100, so in [0, 100) e2 has 0 opportunities
    assert len(opps) == 5
    for idx, opp in enumerate(opps):
        assert opp.emitter_id == "p_b5"
        assert opp.frequency_band == 5
        assert opp.start_time == idx * 20
        assert opp.end_time == idx * 20 + 3
        assert opp.duration == 4


def test_deterministic_metric_calculation():
    """
    Verify exact mathematical calculation of Interception Rate, TTFD, Pd, Pfa,
    and Dwell Efficiency using hand-crafted episode records.
    """
    # 1 emitter on Band 5: 2 bursts at [0..3] and [20..23]
    emitter = PeriodicEmitter("test_e", frequency_band=5, period=20, active_duration=4, start_time=0)
    registry = EmitterRegistry([emitter])

    # Construct synthetic EpisodeResult:
    # Step 0: Scan B5 for dwell 2 at t=0..1 -> 2 True Positives
    # Step 1: Scan B0 for dwell 2 at t=2..3 -> 2 True Negatives
    # Step 2: Scan B5 for dwell 2 at t=20..21 -> 1 TP, 1 FN (miss)
    # Step 3: Scan B1 for dwell 2 at t=22..23 -> 1 FP (false alarm), 1 TN
    records = []
    dwells = []

    # Step 0: [0, 1] on B5 (TP, TP)
    s0_outcomes = [
        DwellSlotOutcome(0, 5, is_transmitting=True, is_observable=True, detected=True, result=DetectionResult.HIT),
        DwellSlotOutcome(1, 5, is_transmitting=True, is_observable=True, detected=True, result=DetectionResult.HIT),
    ]
    d0 = DwellSummary(0, 1, 2, 5, DetectionResult.HIT, s0_outcomes)
    obs0 = Observation(2, 5, 2, DetectionResult.HIT)
    records.append(EpisodeStepRecord(0, 0, 1, Action(5, 2), obs0, d0, 0.0))
    dwells.append(d0)

    # Step 1: [2, 3] on B0 (TN, TN)
    s1_outcomes = [
        DwellSlotOutcome(2, 0, is_transmitting=False, is_observable=False, detected=False, result=DetectionResult.MISS),
        DwellSlotOutcome(3, 0, is_transmitting=False, is_observable=False, detected=False, result=DetectionResult.MISS),
    ]
    d1 = DwellSummary(2, 3, 2, 0, DetectionResult.MISS, s1_outcomes)
    obs1 = Observation(4, 0, 2, DetectionResult.MISS)
    records.append(EpisodeStepRecord(1, 2, 3, Action(0, 2), obs1, d1, 0.0))
    dwells.append(d1)

    # Step 2: [20, 21] on B5 (TP, FN)
    s2_outcomes = [
        DwellSlotOutcome(20, 5, is_transmitting=True, is_observable=True, detected=True, result=DetectionResult.HIT),
        DwellSlotOutcome(21, 5, is_transmitting=True, is_observable=True, detected=False, result=DetectionResult.MISS),
    ]
    d2 = DwellSummary(20, 21, 2, 5, DetectionResult.HIT, s2_outcomes)
    obs2 = Observation(22, 5, 2, DetectionResult.HIT)
    records.append(EpisodeStepRecord(2, 20, 21, Action(5, 2), obs2, d2, 0.0))
    dwells.append(d2)

    # Step 3: [22, 23] on B1 (FP, TN)
    s3_outcomes = [
        DwellSlotOutcome(22, 1, is_transmitting=False, is_observable=False, detected=True, result=DetectionResult.FALSE_ALARM),
        DwellSlotOutcome(23, 1, is_transmitting=False, is_observable=False, detected=False, result=DetectionResult.MISS),
    ]
    d3 = DwellSummary(22, 23, 2, 1, DetectionResult.FALSE_ALARM, s3_outcomes)
    obs3 = Observation(24, 1, 2, DetectionResult.FALSE_ALARM)
    records.append(EpisodeStepRecord(3, 22, 23, Action(1, 2), obs3, d3, 0.0))
    dwells.append(d3)

    ep_result = EpisodeResult(
        scheduler_name="MockScheduler",
        seed=42,
        total_time_slots=30,
        total_decisions=4,
        step_records=records,
        dwell_history=dwells,
        environment_config=EnvironmentConfig(num_bands=20, simulation_duration=30),
    )

    metrics = calculate_baseline_metrics(ep_result, registry)

    # Confusion matrix checks
    assert metrics.tp_count == 3   # slot 0, slot 1, slot 20
    assert metrics.fn_count == 1   # slot 21
    assert metrics.fp_count == 1   # slot 22
    assert metrics.tn_count == 3   # slot 2, slot 3, slot 23

    # Empirical Pd = 3 / (3 + 1) = 0.75
    assert metrics.empirical_pd == pytest.approx(0.75)

    # Empirical Pfa = 1 / (1 + 3) = 0.25
    assert metrics.empirical_pfa == pytest.approx(0.25)

    # Opportunities: [0..3] and [20..23] (2 total, both intercepted at t=0 and t=20)
    assert metrics.interception_opportunities == 2
    assert metrics.successful_interceptions == 2
    assert metrics.interception_rate == pytest.approx(1.0)

    # TTFD checks (PRD Section 25.6 & Per-Emitter)
    assert metrics.scenario_ttfd == 0
    assert metrics.emitter_ttfd["test_e"] == 0
    assert metrics.time_to_first_detection["test_e"] == 0

    # Average intercept delay: opp1 intercepted at t=0 (delay=0), opp2 intercepted at t=20 (delay=0) -> avg = 0.0
    assert metrics.average_intercept_time == pytest.approx(0.0)
    assert metrics.average_intercept_delay == pytest.approx(0.0)

    # Dwell efficiency: (TP + FN) / total slots = (3 + 1) / 8 = 4 / 8 = 0.50
    assert metrics.dwell_efficiency == pytest.approx(0.50)


def test_multi_seed_metrics_aggregation():
    """Verify statistical aggregation (mean, std) across multiple seed metric objects."""
    m1 = BaselineMetrics(
        scheduler_name="OpenLoop", seed=1, total_simulation_slots=100, total_decisions=100,
        action_hits=10, action_misses=88, action_false_alarms=2,
        tp_count=10, fn_count=1, fp_count=2, tn_count=87,
        empirical_pd=0.90, empirical_pfa=0.02,
        interception_opportunities=20, successful_interceptions=10, interception_rate=0.50,
        scenario_ttfd=5, emitter_ttfd={"e1": 5}, average_intercept_time=3.0, dwell_efficiency=0.11,
    )
    m2 = BaselineMetrics(
        scheduler_name="OpenLoop", seed=2, total_simulation_slots=100, total_decisions=100,
        action_hits=12, action_misses=86, action_false_alarms=2,
        tp_count=12, fn_count=1, fp_count=2, tn_count=85,
        empirical_pd=0.92, empirical_pfa=0.02,
        interception_opportunities=20, successful_interceptions=12, interception_rate=0.60,
        scenario_ttfd=4, emitter_ttfd={"e1": 4}, average_intercept_time=2.0, dwell_efficiency=0.13,
    )

    agg = aggregate_metrics_across_seeds([m1, m2])

    mean_ir, std_ir = agg["interception_rate"]
    assert mean_ir == pytest.approx(0.55)
    assert std_ir == pytest.approx(0.07071, rel=1e-3)

    mean_delay, std_delay = agg["average_intercept_time"]
    assert mean_delay == pytest.approx(2.5)
    assert std_delay == pytest.approx(0.7071, rel=1e-3)

    mean_ttfd, std_ttfd = agg["scenario_ttfd"]
    assert mean_ttfd == pytest.approx(4.5)
    assert std_ttfd == pytest.approx(0.7071, rel=1e-3)


class CampScheduler(OpenLoopScheduler):
    """Camped scheduler that stays on a single band with fixed dwell."""
    def __init__(self, target_band: int = 7, dwell_time: int = 2):
        super().__init__(num_bands=20, dwell_time=dwell_time)
        self.target_band = target_band

    def select_action(self, observation: Observation) -> Action:
        return Action(frequency_band=self.target_band, dwell_time=self.dwell_time)


class RandomHopScheduler(OpenLoopScheduler):
    """Scheduler that hops randomly across bands with fixed dwell."""
    def __init__(self, seed: int = 999, dwell_time: int = 2):
        super().__init__(num_bands=20, dwell_time=dwell_time)
        import numpy as np
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: Observation) -> Action:
        band = int(self._rng.integers(0, self.num_bands))
        return Action(frequency_band=band, dwell_time=self.dwell_time)


def test_interception_opportunities_scheduler_independence():
    """
    Regression Test:
    Verify that for a fixed scenario, environment configuration, and seed,
    the total number and exact sequence of ground-truth interception opportunities
    is 100% invariant to the scheduler's actions, scan sequence, and dwell duration (1, 2, 3, 5).
    """
    config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=1000,
        seed=42,
        emitters=[
            {
                "emitter_id": "p_b5",
                "emitter_type": "PERIODIC",
                "frequency_band": 5,
                "period": 20,
                "active_duration": 4,
            },
            {
                "emitter_id": "agile_pred",
                "emitter_type": "AGILE_PREDICTABLE",
                "band_sequence": [2, 7, 12, 17],
                "hop_period": 15,
            },
            {
                "emitter_id": "agile_rand",
                "emitter_type": "AGILE_RANDOM",
                "allowed_bands": [1, 3, 8, 14],
                "hop_period": 20,
                "emitter_seed": 555,
            },
            {
                "emitter_id": "intermittent_b9",
                "emitter_type": "INTERMITTENT",
                "frequency_band": 9,
                "scan_period": 50,
                "observable_duration": 5,
            },
            {
                "emitter_id": "dynamic_b15",
                "emitter_type": "PERIODIC",
                "frequency_band": 15,
                "period": 30,
                "active_duration": 5,
                "start_time": 500,
            },
        ],
    )

    runner = EpisodeRunner()

    # Schedulers with diverse policies and dwell durations:
    # 1. Sweep Dwell 1
    env1 = RFEnvironment(config)
    res1 = runner.run_episode(env1, OpenLoopScheduler(num_bands=20, dwell_time=1), seed=42)
    m1 = calculate_baseline_metrics(res1, env1.emitter_registry)

    # 2. Sweep Dwell 2
    env2 = RFEnvironment(config)
    res2 = runner.run_episode(env2, OpenLoopScheduler(num_bands=20, dwell_time=2), seed=42)
    m2 = calculate_baseline_metrics(res2, env2.emitter_registry)

    # 3. Sweep Dwell 3 (previously caused overshoot due to 10002 time slots)
    env3 = RFEnvironment(config)
    res3 = runner.run_episode(env3, OpenLoopScheduler(num_bands=20, dwell_time=3), seed=42)
    m3 = calculate_baseline_metrics(res3, env3.emitter_registry)

    # 4. Sweep Dwell 5
    env4 = RFEnvironment(config)
    res4 = runner.run_episode(env4, OpenLoopScheduler(num_bands=20, dwell_time=5), seed=42)
    m4 = calculate_baseline_metrics(res4, env4.emitter_registry)

    # 5. Camped Scanner on Band 7 with dwell 2
    env5 = RFEnvironment(config)
    res5 = runner.run_episode(env5, CampScheduler(target_band=7, dwell_time=2), seed=42)
    m5 = calculate_baseline_metrics(res5, env5.emitter_registry)

    # 6. Random Hopping Scanner
    env6 = RFEnvironment(config)
    res6 = runner.run_episode(env6, RandomHopScheduler(seed=123, dwell_time=3), seed=42)
    m6 = calculate_baseline_metrics(res6, env6.emitter_registry)

    # Opportunity count must be identical across all 6 runs
    opp_counts = [
        m1.interception_opportunities,
        m2.interception_opportunities,
        m3.interception_opportunities,
        m4.interception_opportunities,
        m5.interception_opportunities,
        m6.interception_opportunities,
    ]
    assert len(set(opp_counts)) == 1, f"Opportunity counts differed across schedulers/dwells: {opp_counts}"

    # Extract raw opportunities directly and assert 100% equivalence
    raw_opps1 = extract_emitter_opportunities(env1.emitter_registry, config.simulation_duration, 20)
    raw_opps3 = extract_emitter_opportunities(env3.emitter_registry, config.simulation_duration, 20)
    raw_opps5 = extract_emitter_opportunities(env5.emitter_registry, config.simulation_duration, 20)

    assert len(raw_opps1) == len(raw_opps3) == len(raw_opps5) == opp_counts[0]
    for o1, o3, o5 in zip(raw_opps1, raw_opps3, raw_opps5):
        assert o1.emitter_id == o3.emitter_id == o5.emitter_id
        assert o1.frequency_band == o3.frequency_band == o5.frequency_band
        assert o1.start_time == o3.start_time == o5.start_time
        assert o1.end_time == o3.end_time == o5.end_time

