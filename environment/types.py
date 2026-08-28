"""
Data contracts and type definitions for the RF Simulation Environment (SIH26055).

This module defines common dataclasses and enums used across the RF environment,
emitter models, receiver models, and observation interfaces.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class DetectionResult(str, Enum):
    """
    Outcome of an ESM receiver scan.
    
    HIT: True positive detection of an active, observable emitter.
    MISS: No detection occurred despite transmission, or scan was idle and correctly quiet.
    FALSE_ALARM: False positive detection triggered by receiver noise.
    NONE: Initial observation prior to any receiver scan action.
    """
    HIT = "HIT"
    MISS = "MISS"
    FALSE_ALARM = "FALSE_ALARM"
    NONE = "NONE"


class EmitterType(str, Enum):
    """Supported emitter behavioral categories."""
    PERIODIC = "PERIODIC"
    AGILE_PREDICTABLE = "AGILE_PREDICTABLE"
    AGILE_RANDOM = "AGILE_RANDOM"
    INTERMITTENT = "INTERMITTENT"
    DYNAMIC = "DYNAMIC"


@dataclass(frozen=True)
class Action:
    """
    Scheduler decision specifying which band to scan and for how many time slots.
    
    Attributes:
        frequency_band: 0-indexed integer representing the logical frequency band (0 to num_bands - 1).
        dwell_time: Positive integer duration in time slots to observe the selected band.
    """
    frequency_band: int
    dwell_time: int

    def __post_init__(self) -> None:
        if not isinstance(self.frequency_band, (int,)):
            raise TypeError(f"frequency_band must be an integer, got {type(self.frequency_band).__name__}")
        if not isinstance(self.dwell_time, (int,)):
            raise TypeError(f"dwell_time must be an integer, got {type(self.dwell_time).__name__}")
        if self.dwell_time <= 0:
            raise ValueError(f"dwell_time must be a positive integer, got {self.dwell_time}")


@dataclass
class EmitterState:
    """
    State of an individual emitter at a specific time slot.
    
    Attributes:
        emitter_id: Unique string identifier for the emitter.
        is_transmitting: True if the emitter is actively emitting RF energy.
        frequency_band: Logical frequency band where emission occurs.
        is_observable: True if emission is interceptable (e.g. main-beam illumination).
    """
    emitter_id: str
    is_transmitting: bool
    frequency_band: int
    is_observable: bool


@dataclass
class GroundTruthSlot:
    """
    Hidden ground-truth RF state for a specific (time_slot, frequency_band) bin.
    
    The scheduler must never directly access this object.
    
    Attributes:
        time_slot: Discrete time index.
        frequency_band: 0-indexed frequency band.
        is_transmitting: True if at least one emitter is emitting in this band at this slot.
        is_observable: True if at least one transmitting emitter in this band is interceptable.
        active_emitter_ids: List of emitter IDs currently emitting in this band.
    """
    time_slot: int
    frequency_band: int
    is_transmitting: bool
    is_observable: bool
    active_emitter_ids: List[str] = field(default_factory=list)


class SlotEvaluationCategory(str, Enum):
    """
    Ground-truth diagnostic classification of an individual slot scan.
    Used exclusively for offline evaluation metrics (Pd, Pfa, Interception Rate).
    """
    TRUE_POSITIVE = "TRUE_POSITIVE"     # Signal present & observable, receiver detected (HIT)
    FALSE_NEGATIVE = "FALSE_NEGATIVE"   # Signal present & observable, receiver missed (MISS)
    FALSE_POSITIVE = "FALSE_POSITIVE"   # Signal absent/unobservable, receiver triggered (FALSE_ALARM)
    TRUE_NEGATIVE = "TRUE_NEGATIVE"     # Signal absent/unobservable, receiver quiet (MISS/Quiet)


@dataclass
class DwellSlotOutcome:
    """
    Detailed receiver outcome for a single slot within a dwell duration (for diagnostics/evaluation).
    
    Attributes:
        time_slot: Discrete time index.
        frequency_band: Frequency band scanned.
        is_transmitting: Ground truth transmission flag.
        is_observable: Ground truth observability flag.
        detected: Whether receiver generated a detection signal.
        result: Evaluated outcome (HIT, MISS, FALSE_ALARM).
    """
    time_slot: int
    frequency_band: int
    is_transmitting: bool
    is_observable: bool
    detected: bool
    result: DetectionResult

    @property
    def evaluation_category(self) -> SlotEvaluationCategory:
        """Categorize slot into one of the four confusion-matrix states."""
        if self.is_transmitting and self.is_observable:
            return SlotEvaluationCategory.TRUE_POSITIVE if self.detected else SlotEvaluationCategory.FALSE_NEGATIVE
        else:
            return SlotEvaluationCategory.FALSE_POSITIVE if self.detected else SlotEvaluationCategory.TRUE_NEGATIVE

    @property
    def is_true_positive(self) -> bool:
        return self.is_transmitting and self.is_observable and self.detected

    @property
    def is_false_negative(self) -> bool:
        return self.is_transmitting and self.is_observable and not self.detected

    @property
    def is_false_positive(self) -> bool:
        return (not self.is_transmitting or not self.is_observable) and self.detected

    @property
    def is_true_negative(self) -> bool:
        return (not self.is_transmitting or not self.is_observable) and not self.detected



@dataclass
class DwellSummary:
    """
    Complete summary of a multi-slot receiver dwell operation.
    
    Attributes:
        start_time: Start time slot of the dwell (inclusive).
        end_time: End time slot of the dwell (inclusive).
        dwell_time: Total duration in slots.
        scanned_band: Scanned frequency band.
        overall_result: Action-level aggregated detection result (HIT, FALSE_ALARM, MISS).
        slot_outcomes: List of individual slot outcomes.
    """
    start_time: int
    end_time: int
    dwell_time: int
    scanned_band: int
    overall_result: DetectionResult
    slot_outcomes: List[DwellSlotOutcome] = field(default_factory=list)


@dataclass
class Observation:
    """
    Legitimately observable ESM data presented to the scheduler.
    
    Contains strictly no hidden ground-truth information (no emitter IDs,
    no future state, no hidden observability status).
    
    Attributes:
        current_time: Current simulation time slot (at the decision point).
        scanned_band: Frequency band scanned during the last action (None on reset).
        dwell_time: Dwell duration of the last action (None on reset).
        result: Detection result of the last action (DetectionResult.NONE on reset).
        history_summary: Non-leaking historical scan statistics and context metadata.
    """
    current_time: int
    scanned_band: Optional[int]
    dwell_time: Optional[int]
    result: DetectionResult
    history_summary: dict[str, Any] = field(default_factory=dict)
