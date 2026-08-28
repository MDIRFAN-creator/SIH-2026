"""
SIH26055 RF Simulation Environment.

Package exports for RF spectrum, emitter models, receiver model, observation system,
and configuration contracts.
"""

from environment.config import (
    AgilePredictableConfig,
    AgileRandomConfig,
    DynamicEmitterConfig,
    EnvironmentConfig,
    IntermittentConfig,
    PeriodicEmitterConfig,
    ReceiverConfig,
    load_config,
)
from environment.emitters import (
    BaseEmitter,
    EmitterRegistry,
    FrequencyAgileEmitter,
    IntermittentEmitter,
    PeriodicEmitter,
)
from environment.observation import ObservationMemory
from environment.receiver import ESMReceiver
from environment.rf_environment import RFEnvironment
from environment.types import (
    Action,
    DetectionResult,
    DwellSlotOutcome,
    DwellSummary,
    EmitterState,
    EmitterType,
    GroundTruthSlot,
    Observation,
    SlotEvaluationCategory,
)

__all__ = [
    "Action",
    "AgilePredictableConfig",
    "AgileRandomConfig",
    "BaseEmitter",
    "DetectionResult",
    "DwellSlotOutcome",
    "DwellSummary",
    "DynamicEmitterConfig",
    "ESMReceiver",
    "EmitterRegistry",
    "EmitterState",
    "EmitterType",
    "EnvironmentConfig",
    "FrequencyAgileEmitter",
    "GroundTruthSlot",
    "IntermittentConfig",
    "IntermittentEmitter",
    "Observation",
    "ObservationMemory",
    "PeriodicEmitter",
    "PeriodicEmitterConfig",
    "RFEnvironment",
    "ReceiverConfig",
    "SlotEvaluationCategory",
    "load_config",
]

