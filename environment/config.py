"""
Configuration data models and validation for the RF Environment (SIH26055).

This module defines configuration structures for the environment, receiver,
and various emitter models, supporting loading from and saving to YAML.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from environment.types import EmitterType


@dataclass
class PeriodicEmitterConfig:
    """Configuration for a periodic pulse/radar emitter."""
    emitter_id: str
    frequency_band: int
    period: int
    active_duration: int
    start_time: int = 0
    end_time: Optional[int] = None
    offset: int = 0
    emitter_type: str = EmitterType.PERIODIC.value

    def validate(self, num_bands: int) -> None:
        if not (0 <= self.frequency_band < num_bands):
            raise ValueError(f"Emitter '{self.emitter_id}': frequency_band {self.frequency_band} out of range [0, {num_bands-1}]")
        if self.period <= 0:
            raise ValueError(f"Emitter '{self.emitter_id}': period must be > 0, got {self.period}")
        if self.active_duration <= 0:
            raise ValueError(f"Emitter '{self.emitter_id}': active_duration must be > 0, got {self.active_duration}")
        if self.active_duration > self.period:
            raise ValueError(f"Emitter '{self.emitter_id}': active_duration ({self.active_duration}) cannot exceed period ({self.period})")
        if self.start_time < 0:
            raise ValueError(f"Emitter '{self.emitter_id}': start_time must be >= 0, got {self.start_time}")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError(f"Emitter '{self.emitter_id}': end_time ({self.end_time}) must be > start_time ({self.start_time})")


@dataclass
class AgilePredictableConfig:
    """Configuration for a frequency-agile emitter with a deterministic hop sequence."""
    emitter_id: str
    band_sequence: List[int]
    hop_period: int = 1
    start_time: int = 0
    end_time: Optional[int] = None
    emitter_type: str = EmitterType.AGILE_PREDICTABLE.value

    def validate(self, num_bands: int) -> None:
        if not self.band_sequence:
            raise ValueError(f"Emitter '{self.emitter_id}': band_sequence cannot be empty")
        for b in self.band_sequence:
            if not (0 <= b < num_bands):
                raise ValueError(f"Emitter '{self.emitter_id}': band {b} in sequence out of range [0, {num_bands-1}]")
        if self.hop_period <= 0:
            raise ValueError(f"Emitter '{self.emitter_id}': hop_period must be > 0, got {self.hop_period}")
        if self.start_time < 0:
            raise ValueError(f"Emitter '{self.emitter_id}': start_time must be >= 0, got {self.start_time}")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError(f"Emitter '{self.emitter_id}': end_time ({self.end_time}) must be > start_time ({self.start_time})")


@dataclass
class AgileRandomConfig:
    """Configuration for a frequency-agile emitter with pseudo-random frequency hopping."""
    emitter_id: str
    allowed_bands: List[int]
    hop_period: int = 1
    start_time: int = 0
    end_time: Optional[int] = None
    emitter_seed: Optional[int] = None
    emitter_type: str = EmitterType.AGILE_RANDOM.value

    def validate(self, num_bands: int) -> None:
        if not self.allowed_bands:
            raise ValueError(f"Emitter '{self.emitter_id}': allowed_bands cannot be empty")
        for b in self.allowed_bands:
            if not (0 <= b < num_bands):
                raise ValueError(f"Emitter '{self.emitter_id}': band {b} in allowed_bands out of range [0, {num_bands-1}]")
        if self.hop_period <= 0:
            raise ValueError(f"Emitter '{self.emitter_id}': hop_period must be > 0, got {self.hop_period}")
        if self.start_time < 0:
            raise ValueError(f"Emitter '{self.emitter_id}': start_time must be >= 0, got {self.start_time}")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError(f"Emitter '{self.emitter_id}': end_time ({self.end_time}) must be > start_time ({self.start_time})")


@dataclass
class IntermittentConfig:
    """Configuration for an intermittent or spatially scanning radar emitter (main-beam illumination abstraction)."""
    emitter_id: str
    frequency_band: int
    scan_period: int
    observable_duration: int
    scan_offset: int = 0
    start_time: int = 0
    end_time: Optional[int] = None
    emitter_type: str = EmitterType.INTERMITTENT.value

    def validate(self, num_bands: int) -> None:
        if not (0 <= self.frequency_band < num_bands):
            raise ValueError(f"Emitter '{self.emitter_id}': frequency_band {self.frequency_band} out of range [0, {num_bands-1}]")
        if self.scan_period <= 0:
            raise ValueError(f"Emitter '{self.emitter_id}': scan_period must be > 0, got {self.scan_period}")
        if self.observable_duration <= 0:
            raise ValueError(f"Emitter '{self.emitter_id}': observable_duration must be > 0, got {self.observable_duration}")
        if self.observable_duration > self.scan_period:
            raise ValueError(f"Emitter '{self.emitter_id}': observable_duration ({self.observable_duration}) cannot exceed scan_period ({self.scan_period})")
        if self.start_time < 0:
            raise ValueError(f"Emitter '{self.emitter_id}': start_time must be >= 0, got {self.start_time}")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError(f"Emitter '{self.emitter_id}': end_time ({self.end_time}) must be > start_time ({self.start_time})")


@dataclass
class DynamicEmitterConfig:
    """Configuration for an emitter that dynamically appears after simulation starts."""
    emitter_id: str
    underlying_config: Dict[str, Any]
    start_time: int
    end_time: Optional[int] = None
    emitter_type: str = EmitterType.DYNAMIC.value

    def validate(self, num_bands: int) -> None:
        if self.start_time <= 0:
            raise ValueError(f"Dynamic emitter '{self.emitter_id}': start_time should be > 0, got {self.start_time}")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError(f"Dynamic emitter '{self.emitter_id}': end_time must be > start_time")


@dataclass
class ReceiverConfig:
    """Configuration for the simulated ESM receiver."""
    pd: float = 0.90
    pfa: float = 0.02
    allowed_dwell_times: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    bandwidth_bands: int = 1

    def validate(self) -> None:
        if not (0.0 <= self.pd <= 1.0):
            raise ValueError(f"Pd must be in [0.0, 1.0], got {self.pd}")
        if not (0.0 <= self.pfa <= 1.0):
            raise ValueError(f"Pfa must be in [0.0, 1.0], got {self.pfa}")
        if not self.allowed_dwell_times:
            raise ValueError("allowed_dwell_times cannot be empty")
        for d in self.allowed_dwell_times:
            if not isinstance(d, int) or d <= 0:
                raise ValueError(f"All dwell times must be positive integers, got {d}")
        if self.bandwidth_bands != 1:
            if self.bandwidth_bands <= 0:
                raise ValueError(f"bandwidth_bands must be positive, got {self.bandwidth_bands}")


@dataclass
class EnvironmentConfig:
    """Master configuration for the RF Simulation Environment."""
    num_bands: int = 20
    simulation_duration: int = 10000
    seed: Optional[int] = 42
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    emitters: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        """Validate all environment, receiver, and emitter configuration parameters."""
        if self.num_bands <= 0:
            raise ValueError(f"num_bands must be > 0, got {self.num_bands}")
        if self.simulation_duration <= 0:
            raise ValueError(f"simulation_duration must be > 0, got {self.simulation_duration}")
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError(f"seed must be an integer or None, got {type(self.seed).__name__}")
        
        self.receiver.validate()

        for emitter_dict in self.emitters:
            e_type = emitter_dict.get("emitter_type")
            e_id = emitter_dict.get("emitter_id", "unknown")
            if not e_type:
                raise ValueError(f"Emitter '{e_id}' missing 'emitter_type'")
            
            # Construct and validate corresponding config object
            if e_type == EmitterType.PERIODIC.value:
                PeriodicEmitterConfig(**emitter_dict).validate(self.num_bands)
            elif e_type == EmitterType.AGILE_PREDICTABLE.value:
                AgilePredictableConfig(**emitter_dict).validate(self.num_bands)
            elif e_type == EmitterType.AGILE_RANDOM.value:
                AgileRandomConfig(**emitter_dict).validate(self.num_bands)
            elif e_type == EmitterType.INTERMITTENT.value:
                IntermittentConfig(**emitter_dict).validate(self.num_bands)
            elif e_type == EmitterType.DYNAMIC.value:
                # Dynamic emitter validation
                start_time = emitter_dict.get("start_time", 0)
                if start_time < 0:
                    raise ValueError(f"Dynamic emitter '{e_id}': start_time must be >= 0")
            else:
                raise ValueError(f"Unknown emitter_type '{e_type}' for emitter '{e_id}'")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentConfig":
        """Create EnvironmentConfig from dictionary."""
        data_copy = dict(data)
        if "receiver" in data_copy and isinstance(data_copy["receiver"], dict):
            data_copy["receiver"] = ReceiverConfig(**data_copy["receiver"])
        config = cls(**data_copy)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EnvironmentConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a plain dictionary."""
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(path: Union[str, Path]) -> EnvironmentConfig:
    """Convenience function to load and validate an EnvironmentConfig from a YAML file."""
    return EnvironmentConfig.from_yaml(path)
