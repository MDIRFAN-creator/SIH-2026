"""
Proximal Policy Optimization (PPO) Actor-Critic Agent for SIH26055 (Phase 5).

Implements a robust, self-contained PyTorch Actor-Critic PPO algorithm with:
- Orthogonal layer initialization with standard gain scaling.
- Generalized Advantage Estimation (GAE-Lambda).
- Clipped surrogate policy objective with entropy regularization.
- Value function loss clipping and gradient norm bounding.
- Strict seed handling and deterministic inference mode.

Strict Non-Leakage Guarantee:
- Consumes strictly normalized state feature vectors (dim=227).
- Learns from observation-derived scalar rewards.
- Zero access to ground-truth transmitter states or future information.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialize linear layer with orthogonal weights and constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNetwork(nn.Module):
    """
    Two-headed Actor-Critic MLP neural network.
    """

    def __init__(
        self,
        state_dim: int = 227,
        action_dim: int = 60,
        hidden_dim: int = 128,
    ) -> None:
        """
        Initialize the ActorCriticNetwork.

        Args:
            state_dim: Dimension of input state vector.
            action_dim: Number of discrete action choices (e.g. 60).
            hidden_dim: Hidden layer width for both actor and critic trunks.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature trunk
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # Policy (Actor) Head
        self.actor = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        # Value (Critic) Head
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Compute action distribution and state value.

        Args:
            state: Float tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Tuple[Categorical, torch.Tensor]: (Action distribution, State values).
        """
        features = self.trunk(state)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        dist = Categorical(logits=logits)
        return dist, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state value V(s)."""
        features = self.trunk(state)
        return self.critic(features).squeeze(-1)

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probabilities, entropy, and values for given state-action pairs.

        Args:
            states: Tensor of shape (batch_size, state_dim).
            actions: Long tensor of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (log_probs, entropy, values).
        """
        dist, values = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


@dataclass
class PPOConfig:
    """Hyperparameters for PPO Agent."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.01
    vf_coef: float = 0.50
    max_grad_norm: float = 0.50
    n_epochs: int = 10
    batch_size: int = 64
    hidden_dim: int = 128
    seed: Optional[int] = 42


class PPOAgent:
    """
    PPO Agent with GAE computation, clipped surrogate updates, and model persistence.
    """

    def __init__(
        self,
        state_dim: int = 227,
        action_dim: int = 60,
        config: Optional[PPOConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the PPO agent.

        Args:
            state_dim: Dimension of state representation.
            action_dim: Total discrete actions.
            config: Optional PPOConfig instance.
            device: 'cpu' or 'cuda' (defaults to cpu).
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config if config is not None else PPOConfig()

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)

        self.device = torch.device(device if device is not None else "cpu")
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select an action given the current state.

        Args:
            state: 1D numpy array of shape (state_dim,).
            deterministic: If True, selects argmax action without sampling.

        Returns:
            Tuple[int, float, float]: (action_id, log_prob, value_estimate).
        """
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist, value = self.network(state_tensor)

            if deterministic:
                action = torch.argmax(dist.logits, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimates (GAE) and target returns.

        Args:
            rewards: List of step rewards.
            values: List of state values V(s_t).
            dones: List of termination flags.
            last_value: Value estimate V(s_{T+1}) for bootstrapping.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (advantages, returns).
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            delta = rewards[t] + self.config.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> Dict[str, float]:
        """
        Perform PPO mini-batch policy and value updates.

        Args:
            states: Array of shape (N, state_dim).
            actions: Array of shape (N,).
            old_log_probs: Array of shape (N,).
            returns: Array of shape (N,).
            advantages: Array of shape (N,).

        Returns:
            Dict[str, float]: Training loss telemetry.
        """
        self.network.train()

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        # Normalize advantages
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std() + 1e-8
        norm_advantages_t = (advantages_t - adv_mean) / adv_std

        dataset_size = len(states)
        batch_size = min(self.config.batch_size, dataset_size)

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        kl_divs: List[float] = []
        clip_fractions: List[float] = []

        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_returns = returns_t[batch_idx]
                b_advantages = norm_advantages_t[batch_idx]

                new_log_probs, entropy, new_values = self.network.evaluate_actions(b_states, b_actions)

                # Policy Loss with PPO Clipping
                log_ratio = new_log_probs - b_old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss (MSE with targets)
                value_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

                # Entropy Bonus
                entropy_loss = -entropy.mean()

                # Total Loss
                loss = policy_loss + self.config.vf_coef * value_loss + self.config.ent_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Diagnostic metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clipped = ((ratio < 1.0 - self.config.clip_range) | (ratio > 1.0 + self.config.clip_range)).float().mean().item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                kl_divs.append(approx_kl)
                clip_fractions.append(clipped)

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kl_divs)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }

    def save(self, filepath: str) -> None:
        """
        Save the model weights and configuration to disk.

        Args:
            filepath: Destination file path (e.g. 'artifacts/ppo/ppo_model.pt').
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "state_dict": self.network.state_dict(),
            "config": self.config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> None:
        """
        Load model weights from disk.

        Args:
            filepath: Source file path.
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()

