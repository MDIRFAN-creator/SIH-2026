"""
Unit and statistical tests for ESM Receiver detection model (SIH26055 - Phase 1).
"""

import pytest
import numpy as np

from environment.config import ReceiverConfig
from environment.emitters import EmitterRegistry, PeriodicEmitter
from environment.receiver import ESMReceiver
from environment.types import Action, DetectionResult


def test_action_validation():
    """Verify ESMReceiver action validation rules."""
    receiver = ESMReceiver(num_bands=20)

    # Valid actions
    receiver.validate_action(Action(frequency_band=0, dwell_time=1))
    receiver.validate_action(Action(frequency_band=19, dwell_time=5))

    # Invalid band
    with pytest.raises(ValueError, match="Invalid frequency_band"):
        receiver.validate_action(Action(frequency_band=-1, dwell_time=1))
    with pytest.raises(ValueError, match="Invalid frequency_band"):
        receiver.validate_action(Action(frequency_band=20, dwell_time=1))

    # Invalid dwell
    with pytest.raises(ValueError, match="Invalid dwell_time"):
        receiver.validate_action(Action(frequency_band=5, dwell_time=4))
    with pytest.raises(ValueError, match="Invalid dwell_time"):
        receiver.validate_action(Action(frequency_band=5, dwell_time=10))


def test_pd_statistical_accuracy():
    """
    Statistically verify Probability of Detection (Pd = 0.90) across a large sample (N = 20,000).
    Expected standard deviation: sqrt(0.9 * 0.1 / 20000) ~ 0.00212.
    """
    receiver = ESMReceiver(config=ReceiverConfig(pd=0.90, pfa=0.02), seed=42)
    
    # Continuous active emitter on band 5
    emitter = PeriodicEmitter("const_emitter", frequency_band=5, period=1, active_duration=1)
    registry = EmitterRegistry([emitter])

    trials = 20000
    hits = 0
    action = Action(frequency_band=5, dwell_time=1)

    for t in range(trials):
        summary = receiver.scan_dwell(action, start_time=t, emitter_registry=registry)
        if summary.overall_result == DetectionResult.HIT:
            hits += 1

    measured_pd = hits / trials
    print(f"\n[Test Result] Measured Pd: {measured_pd:.4f} (Target: 0.9000, N={trials})")
    assert 0.885 <= measured_pd <= 0.915, f"Measured Pd {measured_pd} outside expected range [0.885, 0.915]"


def test_pfa_statistical_accuracy():
    """
    Statistically verify Probability of False Alarm (Pfa = 0.02) on an idle channel (N = 50,000).
    Expected standard deviation: sqrt(0.02 * 0.98 / 50000) ~ 0.000626.
    """
    receiver = ESMReceiver(config=ReceiverConfig(pd=0.90, pfa=0.02), seed=99)
    
    # Empty registry -> no signals present
    registry = EmitterRegistry([])

    trials = 50000
    false_alarms = 0
    action = Action(frequency_band=3, dwell_time=1)

    for t in range(trials):
        summary = receiver.scan_dwell(action, start_time=t, emitter_registry=registry)
        if summary.overall_result == DetectionResult.FALSE_ALARM:
            false_alarms += 1

    measured_pfa = false_alarms / trials
    print(f"\n[Test Result] Measured Pfa: {measured_pfa:.4f} (Target: 0.0200, N={trials})")
    assert 0.016 <= measured_pfa <= 0.024, f"Measured Pfa {measured_pfa} outside expected range [0.016, 0.024]"


def test_multi_slot_dwell_aggregation():
    """Verify multi-slot dwell evaluation and aggregation policy."""
    receiver = ESMReceiver(config=ReceiverConfig(pd=1.0, pfa=0.0), seed=42)

    # Emitter active ONLY at t = 11 on band 7
    # (period=100, active_duration=1, start_time=11)
    emitter = PeriodicEmitter("sparse", frequency_band=7, period=100, active_duration=1, start_time=11)
    registry = EmitterRegistry([emitter])

    # Scan dwell of 3 slots starting at t = 10 (covers slots 10, 11, 12)
    # Slot 11 has signal and Pd=1.0 -> should yield HIT
    action = Action(frequency_band=7, dwell_time=3)
    summary = receiver.scan_dwell(action, start_time=10, emitter_registry=registry)

    assert summary.dwell_time == 3
    assert len(summary.slot_outcomes) == 3
    assert summary.overall_result == DetectionResult.HIT
    assert summary.slot_outcomes[0].detected is False  # Slot 10 (quiet)
    assert summary.slot_outcomes[1].detected is True   # Slot 11 (active)
    assert summary.slot_outcomes[2].detected is False  # Slot 12 (quiet)

    # Scan dwell starting at t = 20 (covers slots 20, 21, 22 - all quiet)
    summary_quiet = receiver.scan_dwell(action, start_time=20, emitter_registry=registry)
    assert summary_quiet.overall_result == DetectionResult.MISS
    assert all(not slot.detected for slot in summary_quiet.slot_outcomes)
