"""
Action optimization layer for SIH26055 (Phase 3).

Optimizes frequency band and dwell duration selection given XGBoost predicted probabilities
and scanning constraints (anti-camping repeat penalties and dwell trade-offs).
"""

from typing import List, Optional, Tuple
import numpy as np

from environment.types import Action


class ActionOptimizer:
    """
    Evaluates candidate frequency-time actions and selects the action that maximizes
    expected utility while penalizing repetitive camping and excessive dwell.
    """

    def __init__(
        self,
        num_bands: int = 20,
        allowed_dwells: Optional[List[int]] = None,
        repeat_penalty_weight: float = 0.15,
        dwell_penalty_weight: float = 0.05,
        max_consecutive_scans: int = 3,
    ):
        self.num_bands = num_bands
        self.allowed_dwells = [1, 2, 3] if allowed_dwells is None else allowed_dwells
        self.repeat_penalty_weight = repeat_penalty_weight
        self.dwell_penalty_weight = dwell_penalty_weight
        self.max_consecutive_scans = max_consecutive_scans

        if not self.allowed_dwells:
            raise ValueError("allowed_dwells cannot be empty")
        for d in self.allowed_dwells:
            if d <= 0:
                raise ValueError(f"Dwell must be positive integer, got {d}")

    def compute_utility_matrix(
        self,
        predicted_probabilities: np.ndarray,
        last_scanned_band: Optional[int] = None,
        consecutive_scans: int = 0,
    ) -> np.ndarray:
        """
        Compute the 2D utility matrix for all (band, dwell) candidate pairs.
        
        Args:
            predicted_probabilities: 1D float array of length num_bands in [0, 1].
            last_scanned_band: The band scanned in the immediate previous step.
            consecutive_scans: Number of consecutive times last_scanned_band was scanned.
            
        Returns:
            np.ndarray: 2D array of shape (num_bands, len(allowed_dwells)) containing utilities.
        """
        if len(predicted_probabilities) != self.num_bands:
            raise ValueError(
                f"Expected {self.num_bands} probabilities, got {len(predicted_probabilities)}"
            )

        utility_matrix = np.zeros((self.num_bands, len(self.allowed_dwells)), dtype=float)

        for b in range(self.num_bands):
            # 1. Anti-camping repeat factor
            if last_scanned_band is not None and b == last_scanned_band:
                if consecutive_scans >= self.max_consecutive_scans:
                    # Hard anti-camping penalty: strongly suppress staying on the same band
                    repeat_factor = 10.0
                else:
                    repeat_factor = float(consecutive_scans) / float(self.max_consecutive_scans)
            else:
                repeat_factor = 0.0

            p_hat = float(predicted_probabilities[b])

            for d_idx, d in enumerate(self.allowed_dwells):
                # Utility = Expected hit gain (scaled with dwell sensitivity) - Repeat Penalty - Dwell Cost
                # sqrt(d) models diminishing detection return for longer dwell in dynamic RF environment
                gain = p_hat * np.sqrt(float(d))
                penalty_repeat = self.repeat_penalty_weight * repeat_factor
                penalty_dwell = self.dwell_penalty_weight * float(d - 1)

                utility_matrix[b, d_idx] = gain - penalty_repeat - penalty_dwell

        return utility_matrix

    def select_action(
        self,
        predicted_probabilities: np.ndarray,
        last_scanned_band: Optional[int] = None,
        consecutive_scans: int = 0,
    ) -> Action:
        """
        Select the optimal (band, dwell) action maximizing the utility function.
        
        Args:
            predicted_probabilities: 1D array of shape (num_bands,).
            last_scanned_band: The band index scanned in the previous step.
            consecutive_scans: Consecutive count on last_scanned_band.
            
        Returns:
            Action: Chosen frequency band and dwell duration.
        """
        utilities = self.compute_utility_matrix(
            predicted_probabilities=predicted_probabilities,
            last_scanned_band=last_scanned_band,
            consecutive_scans=consecutive_scans,
        )

        # Find index of maximum utility (flattened argmax)
        best_flat_idx = int(np.argmax(utilities))
        best_band, best_dwell_idx = np.unravel_index(best_flat_idx, utilities.shape)
        best_dwell = self.allowed_dwells[best_dwell_idx]

        return Action(frequency_band=int(best_band), dwell_time=int(best_dwell))
