"""
Emitter models and ground truth generation engine for the RF Environment (SIH26055).

This module implements:
- PeriodicEmitter: regular pulse/interval transmissions
- FrequencyAgileEmitter: predictable sequence hops or stochastic seeded frequency hops
- IntermittentEmitter: spatial scanning / beam illumination abstraction (transmitting vs observable)
- Dynamic appearance: time-bounded activation (start_time, end_time)
- EmitterRegistry: multi-emitter aggregation and hidden ground truth querying
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from environment.types import EmitterState, EmitterType, GroundTruthSlot


class BaseEmitter(ABC):
    """Abstract base class for all simulated RF emitters."""

    def __init__(
        self,
        emitter_id: str,
        start_time: int = 0,
        end_time: Optional[int] = None,
        emitter_type: EmitterType = EmitterType.PERIODIC,
    ) -> None:
        self.emitter_id = emitter_id
        self.start_time = max(0, start_time)
        self.end_time = end_time
        self.emitter_type = emitter_type

    def is_active_in_scenario(self, t: int) -> bool:
        """Check if time slot t falls within the emitter's active lifespan [start_time, end_time)."""
        if t < self.start_time:
            return False
        if self.end_time is not None and t >= self.end_time:
            return False
        return True

    @abstractmethod
    def get_state(self, t: int) -> EmitterState:
        """
        Compute emitter transmission and observability state at time slot t.
        
        Returns:
            EmitterState: indicating whether transmitting, active frequency band, and observability.
        """
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state/RNG if applicable."""
        pass


class PeriodicEmitter(BaseEmitter):
    """
    Periodic emitter emitting pulses on a fixed frequency band at regular intervals.
    
    Formula:
        Transmitting when: ((t - start_time - offset) % period) < active_duration
    """

    def __init__(
        self,
        emitter_id: str,
        frequency_band: int,
        period: int,
        active_duration: int,
        start_time: int = 0,
        end_time: Optional[int] = None,
        offset: int = 0,
    ) -> None:
        super().__init__(
            emitter_id=emitter_id,
            start_time=start_time,
            end_time=end_time,
            emitter_type=EmitterType.PERIODIC,
        )
        self.frequency_band = frequency_band
        self.period = period
        self.active_duration = active_duration
        self.offset = offset

    def get_state(self, t: int) -> EmitterState:
        if not self.is_active_in_scenario(t):
            return EmitterState(
                emitter_id=self.emitter_id,
                is_transmitting=False,
                frequency_band=self.frequency_band,
                is_observable=False,
            )

        cycle_pos = (t - self.start_time - self.offset) % self.period
        is_transmitting = (0 <= cycle_pos < self.active_duration)
        # For a standard periodic emitter without spatial scanning, transmission is directly observable
        return EmitterState(
            emitter_id=self.emitter_id,
            is_transmitting=is_transmitting,
            frequency_band=self.frequency_band,
            is_observable=is_transmitting,
        )


class FrequencyAgileEmitter(BaseEmitter):
    """
    Frequency-agile emitter that hops across frequency bands over time.
    
    Supports two modes:
    1. 'predictable': Follows a deterministic cyclic sequence of frequency bands.
    2. 'random': Pseudorandom frequency selection from an allowed set, seeded deterministically.
    """

    def __init__(
        self,
        emitter_id: str,
        band_sequence: Optional[List[int]] = None,
        allowed_bands: Optional[List[int]] = None,
        hop_period: int = 1,
        start_time: int = 0,
        end_time: Optional[int] = None,
        seed: Optional[int] = None,
        mode: str = "predictable",
    ) -> None:
        e_type = EmitterType.AGILE_PREDICTABLE if mode == "predictable" else EmitterType.AGILE_RANDOM
        super().__init__(
            emitter_id=emitter_id,
            start_time=start_time,
            end_time=end_time,
            emitter_type=e_type,
        )
        self.mode = mode
        self.hop_period = max(1, hop_period)
        self.band_sequence = list(band_sequence) if band_sequence is not None else []
        self.allowed_bands = list(allowed_bands) if allowed_bands is not None else []
        self.base_seed = seed if seed is not None else 42
        self._current_seed = self.base_seed

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._current_seed = seed
        else:
            self._current_seed = self.base_seed

    def _get_random_band_for_hop(self, hop_index: int) -> int:
        """
        Deterministically derive the frequency band for hop_index.
        Stateless calculation ensures random-access reproducibility across any query order.
        """
        # Mix base seed with hop_index using 64-bit integer hashing
        mix = (self._current_seed * 6364136223846793005 + hop_index * 1442695040888963407 + 1) & 0xFFFFFFFFFFFFFFFF
        idx = mix % len(self.allowed_bands)
        return self.allowed_bands[idx]

    def get_state(self, t: int) -> EmitterState:
        if not self.is_active_in_scenario(t):
            fallback_band = self.band_sequence[0] if self.band_sequence else (self.allowed_bands[0] if self.allowed_bands else 0)
            return EmitterState(
                emitter_id=self.emitter_id,
                is_transmitting=False,
                frequency_band=fallback_band,
                is_observable=False,
            )

        hop_index = (t - self.start_time) // self.hop_period

        if self.mode == "predictable":
            band_idx = hop_index % len(self.band_sequence)
            current_band = self.band_sequence[band_idx]
        else:
            current_band = self._get_random_band_for_hop(hop_index)

        return EmitterState(
            emitter_id=self.emitter_id,
            is_transmitting=True,
            frequency_band=current_band,
            is_observable=True,
        )


class IntermittentEmitter(BaseEmitter):
    """
    Intermittent / Spatial Scanning Emitter.
    
    Models an emitter (e.g. rotating radar antenna) that is continuously or periodically
    transmitting, but whose main-beam is only pointed toward the ESM receiver during
    specific interceptable intervals.
    
    Formula:
        Transmitting: True while t in [start_time, end_time)
        Observable: ((t - start_time - scan_offset) % scan_period) < observable_duration
    """

    def __init__(
        self,
        emitter_id: str,
        frequency_band: int,
        scan_period: int,
        observable_duration: int,
        scan_offset: int = 0,
        start_time: int = 0,
        end_time: Optional[int] = None,
    ) -> None:
        super().__init__(
            emitter_id=emitter_id,
            start_time=start_time,
            end_time=end_time,
            emitter_type=EmitterType.INTERMITTENT,
        )
        self.frequency_band = frequency_band
        self.scan_period = scan_period
        self.observable_duration = observable_duration
        self.scan_offset = scan_offset

    def get_state(self, t: int) -> EmitterState:
        if not self.is_active_in_scenario(t):
            return EmitterState(
                emitter_id=self.emitter_id,
                is_transmitting=False,
                frequency_band=self.frequency_band,
                is_observable=False,
            )

        scan_pos = (t - self.start_time - self.scan_offset) % self.scan_period
        is_observable = (0 <= scan_pos < self.observable_duration)
        
        # In spatial scanning radar, energy is actively transmitted at all times,
        # but receiver can only intercept during the main-beam window.
        return EmitterState(
            emitter_id=self.emitter_id,
            is_transmitting=True,
            frequency_band=self.frequency_band,
            is_observable=is_observable,
        )


class EmitterRegistry:
    """
    Aggregates and queries multiple coexisting emitters to construct hidden ground truth.
    """

    def __init__(self, emitters: Optional[List[BaseEmitter]] = None) -> None:
        self._emitters: List[BaseEmitter] = list(emitters) if emitters is not None else []

    def add_emitter(self, emitter: BaseEmitter) -> None:
        """Add an emitter to the registry."""
        self._emitters.append(emitter)

    def get_emitters(self) -> List[BaseEmitter]:
        """Return list of all registered emitters."""
        return list(self._emitters)

    def reset(self, base_seed: Optional[int] = None) -> None:
        """Reset all registered emitters with deterministic seeds."""
        for idx, emitter in enumerate(self._emitters):
            emitter_seed = (base_seed + idx * 7919) if base_seed is not None else None
            emitter.reset(emitter_seed)

    def get_active_emitter_states(self, t: int) -> List[EmitterState]:
        """Query state of all registered emitters at time slot t."""
        return [e.get_state(t) for e in self._emitters if e.is_active_in_scenario(t)]

    def get_ground_truth_slot(self, t: int, frequency_band: int) -> GroundTruthSlot:
        """
        Compute hidden ground truth for a specific (time, band) coordinate.
        """
        active_ids = []
        is_transmitting = False
        is_observable = False

        for emitter in self._emitters:
            if emitter.is_active_in_scenario(t):
                state = emitter.get_state(t)
                if state.frequency_band == frequency_band and state.is_transmitting:
                    is_transmitting = True
                    active_ids.append(emitter.emitter_id)
                    if state.is_observable:
                        is_observable = True

        return GroundTruthSlot(
            time_slot=t,
            frequency_band=frequency_band,
            is_transmitting=is_transmitting,
            is_observable=is_observable,
            active_emitter_ids=active_ids,
        )

    def get_full_spectrum_ground_truth(self, t: int, num_bands: int) -> Dict[int, GroundTruthSlot]:
        """Compute hidden ground truth for all bands at time slot t."""
        return {b: self.get_ground_truth_slot(t, b) for b in range(num_bands)}

    @classmethod
    def from_config_list(
        cls,
        emitter_configs: List[Dict[str, Any]],
        num_bands: int,
        base_seed: Optional[int] = None,
    ) -> "EmitterRegistry":
        """Instantiate an EmitterRegistry from a list of emitter configuration dicts."""
        registry = cls()
        for idx, cfg in enumerate(emitter_configs):
            e_type = cfg.get("emitter_type")
            e_id = cfg.get("emitter_id", f"emitter_{idx}")
            start_time = cfg.get("start_time", 0)
            end_time = cfg.get("end_time", None)

            if e_type == EmitterType.PERIODIC.value:
                registry.add_emitter(
                    PeriodicEmitter(
                        emitter_id=e_id,
                        frequency_band=cfg["frequency_band"],
                        period=cfg["period"],
                        active_duration=cfg["active_duration"],
                        start_time=start_time,
                        end_time=end_time,
                        offset=cfg.get("offset", 0),
                    )
                )
            elif e_type == EmitterType.AGILE_PREDICTABLE.value:
                registry.add_emitter(
                    FrequencyAgileEmitter(
                        emitter_id=e_id,
                        band_sequence=cfg["band_sequence"],
                        hop_period=cfg.get("hop_period", 1),
                        start_time=start_time,
                        end_time=end_time,
                        mode="predictable",
                    )
                )
            elif e_type == EmitterType.AGILE_RANDOM.value:
                e_seed = cfg.get("emitter_seed", (base_seed + idx * 1000) if base_seed is not None else None)
                registry.add_emitter(
                    FrequencyAgileEmitter(
                        emitter_id=e_id,
                        allowed_bands=cfg["allowed_bands"],
                        hop_period=cfg.get("hop_period", 1),
                        start_time=start_time,
                        end_time=end_time,
                        seed=e_seed,
                        mode="random",
                    )
                )
            elif e_type == EmitterType.INTERMITTENT.value:
                registry.add_emitter(
                    IntermittentEmitter(
                        emitter_id=e_id,
                        frequency_band=cfg["frequency_band"],
                        scan_period=cfg["scan_period"],
                        observable_duration=cfg["observable_duration"],
                        scan_offset=cfg.get("scan_offset", 0),
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
            else:
                raise ValueError(f"Unknown emitter_type: {e_type}")

        registry.reset(base_seed)
        return registry
