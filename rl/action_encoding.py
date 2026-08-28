"""
Action Space Encoding and Decoding for Reinforcement Learning (SIH26055 Phase 5).

Provides deterministic, bi-directional, round-trip mappings between:
    Flat discrete action_id ∈ [0, num_actions - 1]
    and
    (frequency_band, dwell_time) tuple.

Supported Default Configuration:
- num_bands = 20 (bands 0..19)
- dwell_values = [1, 2, 3] (dwell durations 1, 2, 3 slots)
- total_actions = 20 * 3 = 60 discrete actions.

Action Mapping Convention:
    action_id = band * len(dwell_values) + dwell_idx
    band = action_id // len(dwell_values)
    dwell = dwell_values[action_id % len(dwell_values)]
"""

from typing import List, Optional, Sequence, Tuple
import numpy as np
from environment.types import Action



class ActionEncoder:
    """
    Bi-directional converter between discrete action IDs and (band, dwell) tuples.
    """

    def __init__(
        self,
        num_bands: int = 20,
        dwell_values: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Initialize the ActionEncoder.

        Args:
            num_bands: Total number of RF frequency bands.
            dwell_values: Sequence of allowable dwell durations (default: [1, 2, 3]).
        """
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        
        self.num_bands = num_bands
        self.dwell_values = list(dwell_values) if dwell_values is not None else [1, 2, 3]
        if not self.dwell_values:
            raise ValueError("dwell_values cannot be empty")
        for d in self.dwell_values:
            if d <= 0:
                raise ValueError(f"dwell duration must be positive, got {d}")

        self.num_dwells = len(self.dwell_values)
        self.num_actions = self.num_bands * self.num_dwells
        self._dwell_to_idx = {d: i for i, d in enumerate(self.dwell_values)}

    def encode(self, band: int, dwell: int) -> int:
        """
        Encode (band, dwell) into a flat discrete action_id.

        Args:
            band: Frequency band index in [0, num_bands - 1].
            dwell: Dwell duration present in self.dwell_values.

        Returns:
            int: Discrete action ID in [0, num_actions - 1].
        """
        if not (0 <= band < self.num_bands):
            raise ValueError(f"band {band} out of range [0, {self.num_bands - 1}]")
        if dwell not in self._dwell_to_idx:
            raise ValueError(f"dwell {dwell} not in allowed dwell_values {self.dwell_values}")

        dwell_idx = self._dwell_to_idx[dwell]
        return band * self.num_dwells + dwell_idx

    def decode(self, action_id: int) -> Tuple[int, int]:
        """
        Decode a discrete action_id into (band, dwell).

        Args:
            action_id: Integer in [0, num_actions - 1].

        Returns:
            Tuple[int, int]: (frequency_band, dwell_time).
        """
        if not (0 <= action_id < self.num_actions):
            raise ValueError(f"action_id {action_id} out of range [0, {self.num_actions - 1}]")

        band = action_id // self.num_dwells
        dwell_idx = action_id % self.num_dwells
        dwell = self.dwell_values[dwell_idx]
        return band, dwell

    def to_action(self, action_id: int) -> Action:
        """
        Convert a discrete action_id directly to an environment Action object.

        Args:
            action_id: Discrete action ID.

        Returns:
            Action: Action(frequency_band=band, dwell_time=dwell).
        """
        band, dwell = self.decode(action_id)
        return Action(frequency_band=band, dwell_time=dwell)

    def get_mask_excluding_band(self, exclude_band: Optional[int]) -> np.ndarray:
        """
        Generate a boolean action mask excluding actions for a given frequency band.

        Args:
            exclude_band: Frequency band index to forbid (or None to allow all).

        Returns:
            np.ndarray: Boolean array of shape (num_actions,) where True = valid, False = masked.
        """
        mask = np.ones(self.num_actions, dtype=bool)
        if exclude_band is not None and 0 <= exclude_band < self.num_bands:
            start_idx = exclude_band * self.num_dwells
            end_idx = start_idx + self.num_dwells
            mask[start_idx:end_idx] = False
        return mask

