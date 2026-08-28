"""
Unit tests for LinUCB contextual bandit algorithm (SIH26055 Phase 4).
"""

import numpy as np
import pytest

from bandits.linucb import LinUCB


def test_linucb_initialization():
    bandit = LinUCB(num_arms=5, feature_dim=4, alpha=1.5, reg_lambda=2.0)
    assert bandit.num_arms == 5
    assert bandit.feature_dim == 4
    assert bandit.alpha == 1.5
    assert bandit.reg_lambda == 2.0
    assert np.all(bandit.pull_counts == 0)

    # Verify initial A_b = lambda * I
    for a in range(5):
        np.testing.assert_array_equal(bandit.A[a], 2.0 * np.eye(4))
        np.testing.assert_array_equal(bandit.b_vec[a], np.zeros(4))


def test_linucb_invalid_initialization():
    with pytest.raises(ValueError):
        LinUCB(num_arms=0)
    with pytest.raises(ValueError):
        LinUCB(feature_dim=-1)
    with pytest.raises(ValueError):
        LinUCB(alpha=-0.5)
    with pytest.raises(ValueError):
        LinUCB(reg_lambda=0.0)


def test_linucb_predict_arm_cold_start():
    bandit = LinUCB(num_arms=3, feature_dim=2, alpha=1.0, reg_lambda=1.0)
    context = np.array([1.0, 0.0], dtype=np.float32)

    ucb, mean, uncert = bandit.predict_arm(0, context)
    # With b_vec = 0, theta = 0, pred_mean = 0.0
    assert mean == 0.0
    # A = I, v = [1, 0], x^T v = 1.0, uncertainty = 1.0
    assert uncert == 1.0
    # ucb = mean + alpha * uncertainty = 0 + 1 * 1 = 1.0
    assert ucb == 1.0


def test_linucb_update_and_prediction():
    bandit = LinUCB(num_arms=2, feature_dim=2, alpha=1.0, reg_lambda=1.0)
    x = np.array([1.0, 0.0], dtype=np.float32)

    # Arm 0 gets positive reward +1.0
    bandit.update(arm=0, context=x, reward=1.0)
    assert bandit.pull_counts[0] == 1
    assert bandit.pull_counts[1] == 0

    # For Arm 0: A_0 = [[2, 0], [0, 1]], b_0 = [1, 0]
    # theta_0 = [0.5, 0.0]
    # mean_0 = 0.5
    # v = [0.5, 0.0], uncert = sqrt(0.5) ~ 0.7071
    ucb0, mean0, uncert0 = bandit.predict_arm(0, x)
    assert np.isclose(mean0, 0.5)
    assert np.isclose(uncert0, np.sqrt(0.5))
    assert np.isclose(ucb0, 0.5 + np.sqrt(0.5))

    # For Arm 1 (unpulled): mean_1 = 0, uncert_1 = 1.0, ucb_1 = 1.0
    ucb1, mean1, uncert1 = bandit.predict_arm(1, x)
    assert mean1 == 0.0
    assert uncert1 == 1.0
    assert ucb1 == 1.0


def test_linucb_select_arm_exploration_bonus():
    # Show that high uncertainty drives exploration
    bandit = LinUCB(num_arms=2, feature_dim=2, alpha=2.0, reg_lambda=1.0)
    contexts = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    # Update arm 0 heavily with low reward
    for _ in range(10):
        bandit.update(arm=0, context=contexts[0], reward=0.1)

    # Arm 0 has mean ~ 0.1, but uncertainty is very low ~ 1/sqrt(11) ~ 0.3
    # Arm 1 is unpulled: mean = 0, uncertainty = 1.0, ucb = 0 + 2.0 * 1.0 = 2.0
    selected, diag = bandit.select_arm(contexts)
    assert selected == 1  # Should explore Arm 1 due to high uncertainty


def test_linucb_reset():
    bandit = LinUCB(num_arms=3, feature_dim=2, alpha=1.0, reg_lambda=1.0)
    x = np.array([1.0, 0.0], dtype=np.float32)
    bandit.update(0, x, 5.0)
    assert bandit.pull_counts[0] == 1

    bandit.reset()
    assert np.all(bandit.pull_counts == 0)
    assert np.all(bandit.b_vec == 0)
    for a in range(3):
        np.testing.assert_array_equal(bandit.A[a], np.eye(2))


def test_linucb_statistics():
    bandit = LinUCB(num_arms=2, feature_dim=2, alpha=1.0, reg_lambda=1.0)
    x = np.array([1.0, 1.0], dtype=np.float32)
    bandit.update(0, x, 1.0)

    stats = bandit.get_arm_statistics()
    assert len(stats) == 2
    assert stats[0]["arm"] == 0
    assert stats[0]["pull_count"] == 1
    assert stats[1]["pull_count"] == 0
