"""
Unit tests for ActionOptimizer (SIH26055 - Phase 3).
"""

import pytest
import numpy as np

from environment.types import Action
from optimizers.action_optimizer import ActionOptimizer


def test_optimizer_selects_highest_utility_band():
    """Verify optimizer selects the band with highest probability when no repeat penalties apply."""
    optimizer = ActionOptimizer(num_bands=5, allowed_dwells=[1, 2], repeat_penalty_weight=0.1, dwell_penalty_weight=0.0)
    probas = np.array([0.1, 0.2, 0.85, 0.3, 0.05])

    action = optimizer.select_action(predicted_probabilities=probas)
    assert action.frequency_band == 2


def test_optimizer_respects_allowed_dwells():
    """Verify optimizer only commands dwell durations from allowed_dwells."""
    allowed = [2, 5]
    optimizer = ActionOptimizer(num_bands=5, allowed_dwells=allowed)
    probas = np.array([0.9, 0.1, 0.1, 0.1, 0.1])

    action = optimizer.select_action(predicted_probabilities=probas)
    assert action.dwell_time in allowed


def test_optimizer_anti_camping_penalty():
    """
    Verify that consecutive scanning on the same band triggers anti-camping suppression,
    causing the optimizer to switch to the next-best alternative band.
    """
    optimizer = ActionOptimizer(
        num_bands=4,
        allowed_dwells=[1],
        repeat_penalty_weight=0.3,
        max_consecutive_scans=3,
    )
    # Band 1 has highest prob (0.80), Band 2 has 0.70
    probas = np.array([0.10, 0.80, 0.70, 0.10])

    # First scan on B1
    a1 = optimizer.select_action(probas, last_scanned_band=None, consecutive_scans=0)
    assert a1.frequency_band == 1

    # Second scan on B1 (1 consecutive scan) -> still selects B1
    a2 = optimizer.select_action(probas, last_scanned_band=1, consecutive_scans=1)
    assert a2.frequency_band == 1

    # Fourth scan attempt on B1 (3 consecutive scans, hitting max_consecutive_scans)
    # Anti-camping penalty heavily penalizes B1 -> optimizer must switch to B2
    a4 = optimizer.select_action(probas, last_scanned_band=1, consecutive_scans=3)
    assert a4.frequency_band == 2


def test_optimizer_invalid_inputs():
    """Verify validation of optimizer parameters."""
    with pytest.raises(ValueError):
        ActionOptimizer(num_bands=5, allowed_dwells=[])
    with pytest.raises(ValueError):
        ActionOptimizer(num_bands=5, allowed_dwells=[-1, 2])
    with pytest.raises(ValueError):
        optimizer = ActionOptimizer(num_bands=5)
        optimizer.select_action(np.array([0.1, 0.2]))  # mismatched length (2 != 5)
