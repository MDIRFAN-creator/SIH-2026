r"""
LinUCB (Linear Upper Confidence Bound) Contextual Bandit Implementation (SIH26055 Phase 4 Hardened).

Implements the Discounted Linear Upper Confidence Bound (D-LinUCB) contextual bandit algorithm:
For each arm b in [0, num_arms - 1]:
    A_b = \gamma A_b + (1 - \gamma) \lambda I + \sum x_t x_t^T
    b_b = \gamma b_b + \sum r_t x_t
    \theta_b = A_b^{-1} b_b
    p_b = \theta_b^T x_b + \alpha \sqrt{x_b^T A_b^{-1} x_b}

Features:
- Discounted decay factor \gamma \in (0, 1] for non-stationary RF adaptation.
- Numerically stable linear system solves (np.linalg.solve).
- Explicit eligible arm masking for hard anti-camping and cold-start exploration.
- Guaranteed positive-definite covariance matrices (A_b \succeq \lambda I).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class LinUCB:
    """
    Disjoint Linear Upper Confidence Bound (LinUCB) contextual bandit model with
    non-stationary exponential decay and candidate arm masking.
    """

    def __init__(
        self,
        num_arms: int = 20,
        feature_dim: int = 10,
        alpha: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.99,
        seed: Optional[int] = None,
    ) -> None:
        r"""
        Initialize the LinUCB contextual bandit.
        
        Args:
            num_arms: Number of distinct bandit arms (e.g. 20 frequency bands).
            feature_dim: Dimension of context feature vectors.
            alpha: Exploration coefficient scaling the confidence bound (\alpha >= 0).
            reg_lambda: Ridge regression regularization parameter (\lambda > 0).
            gamma: Non-stationary discount factor \gamma \in (0, 1].
            seed: Optional random seed for tie-breaking.
        """
        if num_arms <= 0:
            raise ValueError(f"num_arms must be positive, got {num_arms}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if reg_lambda <= 0:
            raise ValueError(f"reg_lambda must be positive, got {reg_lambda}")
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError(f"gamma must be in (0.0, 1.0], got {gamma}")

        self.num_arms = num_arms
        self.feature_dim = feature_dim
        self.alpha = float(alpha)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

        # Per-arm state matrices
        self.A = np.zeros((num_arms, feature_dim, feature_dim), dtype=np.float64)
        self.b_vec = np.zeros((num_arms, feature_dim), dtype=np.float64)
        self.pull_counts = np.zeros(num_arms, dtype=np.int64)

        self.reset()

    def reset(self) -> None:
        r"""Reset all arm design matrices to \lambda * I and response vectors to 0."""
        for a in range(self.num_arms):
            self.A[a] = self.reg_lambda * np.eye(self.feature_dim, dtype=np.float64)
            self.b_vec[a] = np.zeros(self.feature_dim, dtype=np.float64)
        self.pull_counts.fill(0)

    def predict_arm(
        self, arm: int, context: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute the UCB score, predicted mean, and uncertainty bonus for a specific arm.
        
        Uses numerically stable linear system solves (np.linalg.solve) rather than
        explicit matrix inversion.
        
        Args:
            arm: Arm index in [0, num_arms - 1].
            context: 1D context feature vector of shape (feature_dim,).
            
        Returns:
            Tuple[float, float, float]: (ucb_score, predicted_mean, uncertainty_bonus)
        """
        if arm < 0 or arm >= self.num_arms:
            raise IndexError(f"Arm index {arm} out of bounds for {self.num_arms} arms")
        if context.shape != (self.feature_dim,):
            raise ValueError(f"Context shape {context.shape} does not match feature_dim ({self.feature_dim},)")

        A_a = self.A[arm]
        b_a = self.b_vec[arm]
        x = context.astype(np.float64)

        # 1. Estimate parameter \theta_a = A_a^{-1} b_a
        theta_a = np.linalg.solve(A_a, b_a)

        # 2. Predicted expected reward \hat{\mu}_a = x^T \theta_a
        pred_mean = float(np.dot(x, theta_a))

        # 3. Variance / uncertainty term v = x^T A_a^{-1} x
        v_vec = np.linalg.solve(A_a, x)
        uncertainty_var = float(np.dot(x, v_vec))
        uncertainty = float(np.sqrt(max(0.0, uncertainty_var)))

        # 4. Confidence bonus and UCB score
        confidence_bonus = self.alpha * uncertainty
        ucb_score = pred_mean + confidence_bonus

        return ucb_score, pred_mean, uncertainty

    def predict_all_arms(
        self, contexts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute UCB scores for all candidate arms given their respective context vectors.
        
        Args:
            contexts: 2D array of shape (num_arms, feature_dim).
            
        Returns:
            Tuple of (ucb_scores, predicted_means, uncertainties) each of shape (num_arms,).
        """
        if contexts.shape != (self.num_arms, self.feature_dim):
            raise ValueError(
                f"Contexts shape {contexts.shape} must be ({self.num_arms}, {self.feature_dim})"
            )

        ucb_scores = np.zeros(self.num_arms, dtype=np.float64)
        pred_means = np.zeros(self.num_arms, dtype=np.float64)
        uncertainties = np.zeros(self.num_arms, dtype=np.float64)

        for a in range(self.num_arms):
            ucb, mean, uncert = self.predict_arm(a, contexts[a])
            ucb_scores[a] = ucb
            pred_means[a] = mean
            uncertainties[a] = uncert

        return ucb_scores, pred_means, uncertainties

    def select_arm(
        self, contexts: np.ndarray, eligible_arms: Optional[List[int]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select the optimal arm maximizing the LinUCB objective score among eligible candidates.
        
        Args:
            contexts: 2D array of shape (num_arms, feature_dim).
            eligible_arms: Optional list of arm indices allowed to be selected (for anti-camping / cold-start).
            
        Returns:
            Tuple[int, Dict[str, Any]]: (selected_arm, diagnostics_dict)
        """
        ucb_scores, pred_means, uncertainties = self.predict_all_arms(contexts)

        if eligible_arms is not None and len(eligible_arms) > 0:
            candidates = list(eligible_arms)
        else:
            candidates = list(range(self.num_arms))

        # Filter candidate scores
        candidate_ucbs = np.array([ucb_scores[a] for a in candidates], dtype=np.float64)
        max_val = np.max(candidate_ucbs)
        best_indices = [candidates[i] for i in np.where(np.isclose(candidate_ucbs, max_val, atol=1e-9))[0]]

        if len(best_indices) == 1:
            selected_arm = int(best_indices[0])
        else:
            selected_arm = int(best_indices[0])  # Deterministic lowest index on tie

        diagnostics = {
            "selected_arm": selected_arm,
            "selected_ucb": float(ucb_scores[selected_arm]),
            "selected_mean": float(pred_means[selected_arm]),
            "selected_uncertainty": float(uncertainties[selected_arm]),
            "all_ucb_scores": ucb_scores.tolist(),
            "all_pred_means": pred_means.tolist(),
            "all_uncertainties": uncertainties.tolist(),
            "pull_counts": self.pull_counts.tolist(),
            "eligible_arms": candidates,
        }

        return selected_arm, diagnostics

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        r"""
        Online incremental update of the selected arm's design matrix and response vector.
        
        Applies non-stationary exponential decay \gamma to all arms:
            A_i \leftarrow \gamma A_i + (1 - \gamma) \lambda I
            b_i \leftarrow \gamma b_i
        And adds the empirical feedback to the selected arm:
            A_a \leftarrow A_a + x x^T
            b_a \leftarrow b_a + r * x
        
        Args:
            arm: Selected arm index.
            context: Context feature vector associated with the arm at selection time.
            reward: Observed scalar feedback reward.
        """
        if arm < 0 or arm >= self.num_arms:
            raise IndexError(f"Arm index {arm} out of bounds for {self.num_arms} arms")
        if context.shape != (self.feature_dim,):
            raise ValueError(f"Context shape {context.shape} does not match feature_dim ({self.feature_dim},)")

        x = context.astype(np.float64)
        r = float(reward)

        # 1. Non-stationary discount step
        if self.gamma < 1.0:
            reg_identity = self.reg_lambda * np.eye(self.feature_dim, dtype=np.float64)
            for a in range(self.num_arms):
                self.A[a] = self.gamma * self.A[a] + (1.0 - self.gamma) * reg_identity
                self.b_vec[a] = self.gamma * self.b_vec[a]

        # 2. Outer product update for the selected arm
        self.A[arm] += np.outer(x, x)

        # 3. Response vector update
        self.b_vec[arm] += r * x

        # 4. Increment pull count
        self.pull_counts[arm] += 1

    def get_arm_statistics(self) -> List[Dict[str, Any]]:
        """Return diagnostic summary for each arm."""
        stats = []
        for a in range(self.num_arms):
            theta = np.linalg.solve(self.A[a], self.b_vec[a])
            stats.append(
                {
                    "arm": a,
                    "pull_count": int(self.pull_counts[a]),
                    "theta_norm": float(np.linalg.norm(theta)),
                    "matrix_cond": float(np.linalg.cond(self.A[a])),
                }
            )
        return stats
