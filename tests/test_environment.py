"""
Integration and API contract tests for RFEnvironment (SIH26055 - Phase 1).
"""

import pytest
from environment.config import EnvironmentConfig, ReceiverConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, DetectionResult, Observation


def test_env_reset():
    """Verify reset() behavior and initial observation structure."""
    config = EnvironmentConfig(num_bands=20, simulation_duration=1000, seed=42)
    env = RFEnvironment(config)

    obs = env.reset(seed=123)
    assert isinstance(obs, Observation)
    assert obs.current_time == 0
    assert obs.scanned_band is None
    assert obs.dwell_time is None
    assert obs.result == DetectionResult.NONE
    assert obs.history_summary["total_decisions"] == 0
    assert obs.history_summary["total_scans_per_band"] == [0] * 20
    assert env.current_time == 0
    assert env.is_terminated is False


def test_env_step_time_progression():
    """Verify step() advances discrete simulation time by dwell_time."""
    config = EnvironmentConfig(num_bands=20, simulation_duration=100)
    env = RFEnvironment(config)
    env.reset(seed=42)

    obs, reward, term, info = env.step(Action(frequency_band=3, dwell_time=2))
    assert env.current_time == 2
    assert obs.current_time == 2
    assert obs.scanned_band == 3
    assert obs.dwell_time == 2
    assert term is False

    obs2, reward2, term2, info2 = env.step(Action(frequency_band=7, dwell_time=5))
    assert env.current_time == 7
    assert obs2.current_time == 7
    assert obs2.scanned_band == 7
    assert obs2.dwell_time == 5
    assert term2 is False


def test_env_termination():
    """Verify episode termination at simulation_duration limit."""
    config = EnvironmentConfig(num_bands=20, simulation_duration=10)
    env = RFEnvironment(config)
    env.reset()

    # Step 5 slots -> t=5 (not terminated)
    _, _, term1, _ = env.step(Action(frequency_band=0, dwell_time=5))
    assert term1 is False
    assert env.is_terminated is False

    # Step 5 slots -> t=10 (terminated)
    _, _, term2, _ = env.step(Action(frequency_band=0, dwell_time=5))
    assert term2 is True
    assert env.is_terminated is True

    # Calling step() on terminated episode must raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot call step\\(\\) on a terminated episode"):
        env.step(Action(frequency_band=0, dwell_time=1))


def test_observation_ground_truth_isolation():
    """Verify that Observation presents strictly non-leaked ESM data."""
    config = EnvironmentConfig(
        num_bands=20,
        simulation_duration=1000,
        emitters=[
            {
                "emitter_id": "secret_radar",
                "emitter_type": "PERIODIC",
                "frequency_band": 4,
                "period": 10,
                "active_duration": 5,
            }
        ],
    )
    env = RFEnvironment(config)
    env.reset()

    obs, _, _, _ = env.step(Action(frequency_band=4, dwell_time=3))

    # Inspect all attributes of Observation
    allowed_obs_attrs = {"current_time", "scanned_band", "dwell_time", "result", "history_summary"}
    assert set(obs.__dict__.keys()) == allowed_obs_attrs

    # Inspect history summary dictionary
    forbidden_keys = {
        "emitter_ids", "active_emitter_ids", "is_transmitting", "is_observable",
        "ground_truth", "gt_matrix", "secret_radar"
    }
    for k in obs.history_summary.keys():
        assert k not in forbidden_keys, f"Found leaked ground-truth key in observation: {k}"

    summary_str = str(obs.history_summary)
    assert "secret_radar" not in summary_str


def test_invalid_action_rejection():
    """Verify that invalid actions fail immediately and clearly."""
    env = RFEnvironment(EnvironmentConfig(num_bands=20, simulation_duration=100))
    env.reset()

    # Out of range band
    with pytest.raises(ValueError):
        env.step(Action(frequency_band=25, dwell_time=1))

    # Unsupported dwell
    with pytest.raises(ValueError):
        env.step(Action(frequency_band=5, dwell_time=4))

    # Non-action input
    with pytest.raises(TypeError):
        env.step((5, 1))  # type: ignore


def test_observation_history_tracking():
    """Verify cumulative scan counts and recency metrics in ObservationMemory."""
    env = RFEnvironment(EnvironmentConfig(num_bands=20, simulation_duration=100))
    env.reset()

    # Scan band 5 twice, band 2 once
    env.step(Action(frequency_band=5, dwell_time=2))  # t=2
    env.step(Action(frequency_band=2, dwell_time=3))  # t=5
    obs, _, _, _ = env.step(Action(frequency_band=5, dwell_time=1))  # t=6

    hist = obs.history_summary
    assert hist["total_decisions"] == 3
    assert hist["total_scans_per_band"][5] == 2
    assert hist["total_scans_per_band"][2] == 1
    assert hist["total_scans_per_band"][0] == 0

    assert hist["last_scan_time_per_band"][5] == 6
    assert hist["last_scan_time_per_band"][2] == 5
    assert hist["time_since_last_scan"][5] == 0
    assert hist["time_since_last_scan"][2] == 1  # 6 - 5


def test_phase1_placeholder_reward():
    """Verify that Phase 1 returns neutral 0.0 reward without arbitrary formulas."""
    env = RFEnvironment(EnvironmentConfig(num_bands=20, simulation_duration=100))
    env.reset(seed=42)

    _, reward1, _, _ = env.step(Action(frequency_band=0, dwell_time=1))
    assert reward1 == 0.0

    _, reward2, _, _ = env.step(Action(frequency_band=5, dwell_time=3))
    assert reward2 == 0.0


def test_detection_semantics_4_cases():
    """
    Verify that DwellSlotOutcome and SlotEvaluationCategory clearly distinguish:
    a) TP: transmission + observable + detection
    b) FN: transmission + observable + no detection
    c) FP: no observable transmission + false alarm
    d) TN: no observable transmission + no false alarm
    """
    from environment.types import DetectionResult, DwellSlotOutcome, SlotEvaluationCategory

    # Case a: True Positive
    slot_tp = DwellSlotOutcome(0, 5, is_transmitting=True, is_observable=True, detected=True, result=DetectionResult.HIT)
    assert slot_tp.evaluation_category == SlotEvaluationCategory.TRUE_POSITIVE
    assert slot_tp.is_true_positive is True
    assert slot_tp.is_false_negative is False

    # Case b: False Negative
    slot_fn = DwellSlotOutcome(1, 5, is_transmitting=True, is_observable=True, detected=False, result=DetectionResult.MISS)
    assert slot_fn.evaluation_category == SlotEvaluationCategory.FALSE_NEGATIVE
    assert slot_fn.is_false_negative is True
    assert slot_fn.is_true_positive is False

    # Case c: False Positive (False Alarm)
    slot_fp = DwellSlotOutcome(2, 5, is_transmitting=False, is_observable=False, detected=True, result=DetectionResult.FALSE_ALARM)
    assert slot_fp.evaluation_category == SlotEvaluationCategory.FALSE_POSITIVE
    assert slot_fp.is_false_positive is True
    assert slot_fp.is_true_negative is False

    # Case d: True Negative (Quiet channel)
    slot_tn = DwellSlotOutcome(3, 5, is_transmitting=False, is_observable=False, detected=False, result=DetectionResult.MISS)
    assert slot_tn.evaluation_category == SlotEvaluationCategory.TRUE_NEGATIVE
    assert slot_tn.is_true_negative is True
    assert slot_fp.is_false_negative is False

