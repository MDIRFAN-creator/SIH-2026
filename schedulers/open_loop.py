"""
Open-Loop Baseline Scheduler for ESM frequency-time scanning (SIH26055 - Phase 2).

This module implements a conventional, non-adaptive frequency sweep scheduler
that cyclically steps through frequency bands with a fixed dwell duration:
B0 -> B1 -> B2 -> ... -> B(N-1) -> B0 ...
"""

from typing import Optional
from environment.types import Action, Observation
from schedulers.base import BaseScheduler


class OpenLoopScheduler(BaseScheduler):
    """
    Open-Loop Cyclic Frequency Sweep Scheduler.
    
    Operates without learning or feedback from receiver observations.
    Maintains only the minimal internal pointer needed to sequence through bands.
    
    Attributes:
        num_bands: Total number of frequency bands in the spectrum.
        dwell_time: Configurable fixed dwell duration in slots (e.g. 1, 2, 3, 5).
        start_band: Initial frequency band to scan on episode start.
    """

    def __init__(
        self,
        num_bands: int = 20,
        dwell_time: int = 1,
        start_band: int = 0,
        scheduler_name: str = "OpenLoopScheduler",
    ) -> None:
        super().__init__(scheduler_name=scheduler_name)
        if num_bands <= 0:
            raise ValueError(f"num_bands must be > 0, got {num_bands}")
        if dwell_time <= 0:
            raise ValueError(f"dwell_time must be a positive integer, got {dwell_time}")
        if not (0 <= start_band < num_bands):
            raise ValueError(f"start_band {start_band} out of range [0, {num_bands - 1}]")

        self.num_bands = num_bands
        self.dwell_time = dwell_time
        self.start_band = start_band
        self._current_band: int = self.start_band

    def reset(self) -> None:
        """Reset the scan pointer back to start_band."""
        self._current_band = self.start_band

    def select_action(self, observation: Observation) -> Action:
        """
        Select the next frequency band in the fixed cyclic sweep.
        
        Note: The observation object is accepted to fulfill the BaseScheduler contract,
        but its contents are intentionally ignored to maintain open-loop behavior.
        
        Args:
            observation: Scheduler-visible observation (ignored).
            
        Returns:
            Action: Action(frequency_band=current_band, dwell_time=self.dwell_time)
        """
        selected_band = self._current_band
        action = Action(frequency_band=selected_band, dwell_time=self.dwell_time)
        
        # Advance scan pointer cyclically
        self._current_band = (self._current_band + 1) % self.num_bands
        return action

    def update(
        self,
        observation: Observation,
        action: Action,
        reward: float,
    ) -> None:
        """No-op for open-loop baseline."""
        pass
