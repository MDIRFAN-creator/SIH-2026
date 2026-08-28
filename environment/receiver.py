"""
ESM Receiver model for the RF Environment (SIH26055).

This module simulates an Electronic Support Measures (ESM) receiver with:
- Configurable instantaneous bandwidth (1 band)
- Configurable Probability of Detection (Pd) and Probability of False Alarm (Pfa)
- Multi-slot dwell evaluation with dedicated pseudo-random number generator
- Action validation and detailed dwell event logging
"""

from typing import List, Optional
import numpy as np

from environment.config import ReceiverConfig
from environment.emitters import EmitterRegistry
from environment.types import (
    Action,
    DetectionResult,
    DwellSlotOutcome,
    DwellSummary,
)


class ESMReceiver:
    """
    Simulated Electronic Support Measures (ESM) receiver.
    
    The receiver evaluates RF detection over a commanded frequency band and dwell duration.
    All random detection draws are performed using a dedicated NumPy Generator.
    """

    def __init__(
        self,
        config: Optional[ReceiverConfig] = None,
        num_bands: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config if config is not None else ReceiverConfig()
        self.config.validate()
        self.num_bands = num_bands
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the internal random number generator with an optional seed."""
        self._rng = np.random.default_rng(seed)

    def validate_action(self, action: Action) -> None:
        """
        Validate whether the given action complies with receiver limits.
        
        Raises:
            ValueError: If band is out of bounds or dwell time is not in allowed_dwell_times.
        """
        if not (0 <= action.frequency_band < self.num_bands):
            raise ValueError(
                f"Invalid frequency_band {action.frequency_band}. Must be in [0, {self.num_bands - 1}]."
            )
        if action.dwell_time not in self.config.allowed_dwell_times:
            raise ValueError(
                f"Invalid dwell_time {action.dwell_time}. Must be one of {self.config.allowed_dwell_times}."
            )

    def scan_dwell(
        self,
        action: Action,
        start_time: int,
        emitter_registry: EmitterRegistry,
    ) -> DwellSummary:
        """
        Execute receiver dwell scan over [start_time, start_time + dwell_time - 1].
        
        Args:
            action: Selected Action(frequency_band, dwell_time).
            start_time: Current simulation time slot.
            emitter_registry: Ground truth emitter registry.
            
        Returns:
            DwellSummary: Complete summary with per-slot outcomes and action-level result.
        """
        self.validate_action(action)
        
        dwell_len = action.dwell_time
        band = action.frequency_band
        slot_outcomes: List[DwellSlotOutcome] = []
        has_hit = False
        has_false_alarm = False

        for offset in range(dwell_len):
            t = start_time + offset
            gt = emitter_registry.get_ground_truth_slot(t, band)

            # Evaluate detection logic per slot
            if gt.is_transmitting and gt.is_observable:
                # Observable signal present -> evaluate with Pd
                detected = bool(self._rng.random() < self.config.pd)
                slot_result = DetectionResult.HIT if detected else DetectionResult.MISS
                if detected:
                    has_hit = True
            else:
                # No observable signal -> evaluate noise with Pfa
                detected = bool(self._rng.random() < self.config.pfa)
                slot_result = DetectionResult.FALSE_ALARM if detected else DetectionResult.MISS
                if detected:
                    has_false_alarm = True

            slot_outcomes.append(
                DwellSlotOutcome(
                    time_slot=t,
                    frequency_band=band,
                    is_transmitting=gt.is_transmitting,
                    is_observable=gt.is_observable,
                    detected=detected,
                    result=slot_result,
                )
            )

        # Action-level outcome aggregation:
        # 1. Any True positive detection during dwell => HIT
        # 2. Else if any False positive detection occurred => FALSE_ALARM
        # 3. Else => MISS
        if has_hit:
            overall_result = DetectionResult.HIT
        elif has_false_alarm:
            overall_result = DetectionResult.FALSE_ALARM
        else:
            overall_result = DetectionResult.MISS

        return DwellSummary(
            start_time=start_time,
            end_time=start_time + dwell_len - 1,
            dwell_time=dwell_len,
            scanned_band=band,
            overall_result=overall_result,
            slot_outcomes=slot_outcomes,
        )
