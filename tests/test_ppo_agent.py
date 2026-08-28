"""
Unit tests for PPOAgent and ActorCriticNetwork in rl/ppo_agent.py (SIH26055 Phase 5).
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from rl.ppo_agent import ActorCriticNetwork, PPOAgent, PPOConfig


def test_actor_critic_network_forward() -> None:
    net = ActorCriticNetwork(state_dim=227, action_dim=60, hidden_dim=128)
    state = torch.randn(4, 227)
    dist, value = net(state)

    assert dist.logits.shape == (4, 60)
    assert value.shape == (4,)

    action = dist.sample()
    assert action.shape == (4,)
    log_prob = dist.log_prob(action)
    assert log_prob.shape == (4,)


def test_ppo_agent_select_action() -> None:
    agent = PPOAgent(state_dim=227, action_dim=60, config=PPOConfig(seed=42))
    state = np.random.randn(227).astype(np.float32)

    # Stochastic action selection
    action, log_prob, val = agent.select_action(state, deterministic=False)
    assert 0 <= action < 60
    assert isinstance(log_prob, float)
    assert isinstance(val, float)

    # Deterministic action selection (must be invariant for the same state)
    action_det1, _, _ = agent.select_action(state, deterministic=True)
    action_det2, _, _ = agent.select_action(state, deterministic=True)
    assert action_det1 == action_det2


def test_ppo_agent_gae_and_update() -> None:
    agent = PPOAgent(state_dim=227, action_dim=60, config=PPOConfig(seed=123, batch_size=16, n_epochs=2))

    # Generate synthetic rollout of 32 steps
    states = np.random.randn(32, 227).astype(np.float32)
    actions = np.random.randint(0, 60, size=32, dtype=np.int64)
    old_log_probs = np.random.uniform(-4.0, -1.0, size=32).astype(np.float32)
    rewards = [1.0 if i % 4 == 0 else -0.05 for i in range(32)]
    values = [0.2 for _ in range(32)]
    dones = [False] * 31 + [True]

    advs, returns = agent.compute_gae(rewards, values, dones, last_value=0.0)
    assert len(advs) == 32
    assert len(returns) == 32

    # Perform update step
    metrics = agent.update(states, actions, old_log_probs, returns, advs)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])


def test_ppo_agent_save_and_load_invariance() -> None:
    agent1 = PPOAgent(state_dim=227, action_dim=60, config=PPOConfig(seed=999))
    state = np.random.randn(227).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_ppo.pt")
        agent1.save(model_path)
        assert os.path.exists(model_path)

        agent2 = PPOAgent(state_dim=227, action_dim=60)
        agent2.load(model_path)

        # Confirm exact identical logits and value estimates
        with torch.no_grad():
            s_t = torch.as_tensor(state).unsqueeze(0)
            d1, v1 = agent1.network(s_t)
            d2, v2 = agent2.network(s_t)

            np.testing.assert_allclose(d1.logits.numpy(), d2.logits.numpy(), rtol=1e-5)
            np.testing.assert_allclose(v1.numpy(), v2.numpy(), rtol=1e-5)
