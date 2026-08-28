"""
Behavioral tests for all simulated RF emitter models (SIH26055 - Phase 1).
"""

import pytest
from environment.emitters import (
    EmitterRegistry,
    FrequencyAgileEmitter,
    IntermittentEmitter,
    PeriodicEmitter,
)
from environment.types import EmitterType


def test_periodic_emitter_timing():
    """Verify that PeriodicEmitter transmits precisely during its active window."""
    emitter = PeriodicEmitter(
        emitter_id="p1",
        frequency_band=5,
        period=10,
        active_duration=3,
        start_time=0,
        offset=0,
    )

    # In period=10, active_duration=3, slots 0, 1, 2 should be active; slots 3..9 inactive
    for t in range(30):
        state = emitter.get_state(t)
        cycle_pos = t % 10
        if cycle_pos < 3:
            assert state.is_transmitting is True, f"Slot {t} should be transmitting"
            assert state.is_observable is True
            assert state.frequency_band == 5
        else:
            assert state.is_transmitting is False, f"Slot {t} should NOT be transmitting"
            assert state.is_observable is False


def test_periodic_emitter_offset_and_lifespan():
    """Verify offset and start/end time boundaries for PeriodicEmitter."""
    emitter = PeriodicEmitter(
        emitter_id="p2",
        frequency_band=3,
        period=20,
        active_duration=5,
        start_time=100,
        end_time=300,
        offset=10,
    )

    # Before start_time (t < 100) -> inactive
    assert emitter.get_state(50).is_transmitting is False
    assert emitter.get_state(99).is_transmitting is False

    # At start_time + offset = 110, slots 110..114 should be active
    assert emitter.get_state(109).is_transmitting is False
    assert emitter.get_state(110).is_transmitting is True
    assert emitter.get_state(114).is_transmitting is True
    assert emitter.get_state(115).is_transmitting is False

    # After end_time (t >= 300) -> inactive
    assert emitter.get_state(300).is_transmitting is False
    assert emitter.get_state(310).is_transmitting is False


def test_agile_predictable_emitter():
    """Verify that predictable agile emitter hops through its sequence accurately."""
    seq = [2, 7, 14, 9]
    emitter = FrequencyAgileEmitter(
        emitter_id="agile_pred",
        band_sequence=seq,
        hop_period=5,
        start_time=0,
        mode="predictable",
    )

    # Slots 0..4 -> band 2
    for t in range(0, 5):
        st = emitter.get_state(t)
        assert st.is_transmitting is True
        assert st.frequency_band == 2

    # Slots 5..9 -> band 7
    for t in range(5, 10):
        st = emitter.get_state(t)
        assert st.is_transmitting is True
        assert st.frequency_band == 7

    # Slots 10..14 -> band 14
    for t in range(10, 15):
        st = emitter.get_state(t)
        assert st.is_transmitting is True
        assert st.frequency_band == 14

    # Slots 15..19 -> band 9
    for t in range(15, 20):
        st = emitter.get_state(t)
        assert st.is_transmitting is True
        assert st.frequency_band == 9

    # Slots 20..24 -> wraps back to band 2
    for t in range(20, 25):
        st = emitter.get_state(t)
        assert st.is_transmitting is True
        assert st.frequency_band == 2


def test_agile_random_emitter_reproducibility():
    """Verify that random agile emitter hops randomly but reproducibly under fixed seed."""
    allowed = [1, 3, 5, 7, 9, 11]
    emitter1 = FrequencyAgileEmitter(
        emitter_id="agile_rand_1",
        allowed_bands=allowed,
        hop_period=10,
        seed=12345,
        mode="random",
    )
    emitter2 = FrequencyAgileEmitter(
        emitter_id="agile_rand_2",
        allowed_bands=allowed,
        hop_period=10,
        seed=12345,
        mode="random",
    )

    bands1 = [emitter1.get_state(t).frequency_band for t in range(0, 200, 10)]
    bands2 = [emitter2.get_state(t).frequency_band for t in range(0, 200, 10)]

    assert bands1 == bands2
    assert all(b in allowed for b in bands1)
    # Check that multiple distinct bands were visited (not stuck on one)
    assert len(set(bands1)) > 1


def test_intermittent_emitter_observability():
    """Verify that IntermittentEmitter transmits continuously but is only intermittently observable."""
    emitter = IntermittentEmitter(
        emitter_id="intermittent_1",
        frequency_band=8,
        scan_period=50,
        observable_duration=5,
        scan_offset=10,
        start_time=0,
    )

    # In [0, 50), scan_offset=10, observable window is [10, 15)
    for t in range(50):
        st = emitter.get_state(t)
        assert st.is_transmitting is True, f"Slot {t} should always be transmitting"
        assert st.frequency_band == 8
        if 10 <= t < 15:
            assert st.is_observable is True, f"Slot {t} should be observable"
        else:
            assert st.is_observable is False, f"Slot {t} should NOT be observable"


def test_dynamic_emitter_appearance():
    """Verify dynamic appearance of an emitter starting at t = 5000."""
    emitter = PeriodicEmitter(
        emitter_id="dynamic_e",
        frequency_band=12,
        period=20,
        active_duration=5,
        start_time=5000,
    )

    # Before 5000
    st_pre = emitter.get_state(4999)
    assert st_pre.is_transmitting is False
    assert st_pre.is_observable is False

    # At 5000 -> active cycle starts
    st_post = emitter.get_state(5000)
    assert st_post.is_transmitting is True
    assert st_post.is_observable is True
    assert st_post.frequency_band == 12


def test_emitter_registry_multi_emitter_coexistence():
    """Verify multiple emitters coexisting on different and identical bands."""
    e1 = PeriodicEmitter("e1", frequency_band=3, period=10, active_duration=4)
    e2 = PeriodicEmitter("e2", frequency_band=7, period=10, active_duration=4)
    e3 = PeriodicEmitter("e3", frequency_band=3, period=20, active_duration=6, offset=5)

    registry = EmitterRegistry([e1, e2, e3])

    # At t = 2: e1 is active on B3, e2 is active on B7, e3 is inactive
    gt_b3_t2 = registry.get_ground_truth_slot(2, 3)
    assert gt_b3_t2.is_transmitting is True
    assert gt_b3_t2.is_observable is True
    assert "e1" in gt_b3_t2.active_emitter_ids

    gt_b7_t2 = registry.get_ground_truth_slot(2, 7)
    assert gt_b7_t2.is_transmitting is True
    assert "e2" in gt_b7_t2.active_emitter_ids

    gt_b0_t2 = registry.get_ground_truth_slot(2, 0)
    assert gt_b0_t2.is_transmitting is False
    assert gt_b0_t2.active_emitter_ids == []

    # At t = 7: e1 is inactive (cycle_pos=7 >= 4), e3 is active on B3 (offset=5 -> cycle_pos=2 < 6)
    gt_b3_t7 = registry.get_ground_truth_slot(7, 3)
    assert gt_b3_t7.is_transmitting is True
    assert "e3" in gt_b3_t7.active_emitter_ids
    assert "e1" not in gt_b3_t7.active_emitter_ids
