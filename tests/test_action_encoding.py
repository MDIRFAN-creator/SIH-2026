"""
Unit tests for ActionEncoder in rl/action_encoding.py (SIH26055 Phase 5).
"""

import pytest
from environment.types import Action
from rl.action_encoding import ActionEncoder


def test_action_encoder_initialization() -> None:
    encoder = ActionEncoder(num_bands=20, dwell_values=[1, 2, 3])
    assert encoder.num_bands == 20
    assert encoder.num_dwells == 3
    assert encoder.num_actions == 60


def test_action_encoder_round_trip_all_actions() -> None:
    encoder = ActionEncoder(num_bands=20, dwell_values=[1, 2, 3])
    for band in range(20):
        for dwell in [1, 2, 3]:
            action_id = encoder.encode(band, dwell)
            assert 0 <= action_id < 60
            dec_band, dec_dwell = encoder.decode(action_id)
            assert dec_band == band
            assert dec_dwell == dwell


def test_action_encoder_to_action() -> None:
    encoder = ActionEncoder(num_bands=20, dwell_values=[1, 2, 3])
    for action_id in range(60):
        act = encoder.to_action(action_id)
        assert isinstance(act, Action)
        expected_band, expected_dwell = encoder.decode(action_id)
        assert act.frequency_band == expected_band
        assert act.dwell_time == expected_dwell


def test_action_encoder_invalid_inputs() -> None:
    encoder = ActionEncoder(num_bands=20, dwell_values=[1, 2, 3])

    with pytest.raises(ValueError):
        encoder.encode(-1, 1)

    with pytest.raises(ValueError):
        encoder.encode(20, 1)

    with pytest.raises(ValueError):
        encoder.encode(5, 4)  # Invalid dwell

    with pytest.raises(ValueError):
        encoder.decode(-1)

    with pytest.raises(ValueError):
        encoder.decode(60)
