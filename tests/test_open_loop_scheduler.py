"""
Unit tests for OpenLoopScheduler (SIH26055 - Phase 2).
"""

import pytest
from environment.types import Action, DetectionResult, Observation
from schedulers.open_loop import OpenLoopScheduler


def test_open_loop_sequential_sweep_ordering():
    """Verify exact cyclic sweep: B0 -> B1 -> ... -> B19 -> B0."""
    scheduler = OpenLoopScheduler(num_bands=20, dwell_time=1, start_band=0)
    dummy_obs = Observation(current_time=0, scanned_band=None, dwell_time=None, result=DetectionResult.NONE)

    # First full cycle
    for expected_band in range(20):
        action = scheduler.select_action(dummy_obs)
        assert action.frequency_band == expected_band
        assert action.dwell_time == 1

    # Second cycle wraps to 0
    for expected_band in range(20):
        action = scheduler.select_action(dummy_obs)
        assert action.frequency_band == expected_band


def test_open_loop_fixed_dwell():
    """Verify that every commanded action uses the configured fixed dwell."""
    for dwell in [1, 2, 3, 5]:
        scheduler = OpenLoopScheduler(num_bands=20, dwell_time=dwell)
        dummy_obs = Observation(current_time=0, scanned_band=None, dwell_time=None, result=DetectionResult.NONE)
        
        for _ in range(40):
            action = scheduler.select_action(dummy_obs)
            assert action.dwell_time == dwell


def test_open_loop_observation_independence():
    """
    Verify that OpenLoopScheduler is strictly non-adaptive and ignores observation content.
    Feeding diverse observations (HIT, MISS, FALSE_ALARM) must not change the scan sequence.
    """
    scheduler = OpenLoopScheduler(num_bands=20, dwell_time=1, start_band=0)

    # Feed HIT
    obs_hit = Observation(current_time=10, scanned_band=0, dwell_time=1, result=DetectionResult.HIT)
    a0 = scheduler.select_action(obs_hit)
    assert a0.frequency_band == 0

    # Feed FALSE ALARM
    obs_fa = Observation(current_time=11, scanned_band=0, dwell_time=1, result=DetectionResult.FALSE_ALARM)
    a1 = scheduler.select_action(obs_fa)
    assert a1.frequency_band == 1

    # Feed MISS with arbitrary history
    obs_miss = Observation(
        current_time=12,
        scanned_band=1,
        dwell_time=1,
        result=DetectionResult.MISS,
        history_summary={"recent_hit_rate": [0.9] * 20},
    )
    a2 = scheduler.select_action(obs_miss)
    assert a2.frequency_band == 2


def test_open_loop_reset():
    """Verify reset() restarts the scan sequence from start_band."""
    scheduler = OpenLoopScheduler(num_bands=10, dwell_time=2, start_band=3)
    dummy_obs = Observation(current_time=0, scanned_band=None, dwell_time=None, result=DetectionResult.NONE)

    a0 = scheduler.select_action(dummy_obs)
    assert a0.frequency_band == 3
    a1 = scheduler.select_action(dummy_obs)
    assert a1.frequency_band == 4

    scheduler.reset()
    a_reset = scheduler.select_action(dummy_obs)
    assert a_reset.frequency_band == 3


def test_open_loop_invalid_params():
    """Verify validation of scheduler initialization parameters."""
    with pytest.raises(ValueError):
        OpenLoopScheduler(num_bands=0)
    with pytest.raises(ValueError):
        OpenLoopScheduler(dwell_time=0)
    with pytest.raises(ValueError):
        OpenLoopScheduler(num_bands=20, start_band=20)
    with pytest.raises(ValueError):
        OpenLoopScheduler(num_bands=20, start_band=-1)
