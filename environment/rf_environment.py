"""
RF Simulation Environment for Electronic Warfare (SIH26055 - Phase 1).

This module implements the primary RFEnvironment class providing a modular,
Gym-like frequency-time scanning simulation environment.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from environment.config import EnvironmentConfig, load_config
from environment.emitters import EmitterRegistry
from environment.observation import ObservationMemory
from environment.receiver import ESMReceiver
from environment.types import (
    Action,
    DetectionResult,
    DwellSummary,
    Observation,
)


class RFEnvironment:
    """
    Simulated RF spectrum and ESM receiver environment.
    
    Architecture:
        Environment (owns hidden ground truth)
          -> EmitterRegistry (multi-emitter state)
          -> ESMReceiver (Pd/Pfa stochastic detection)
          -> ObservationMemory (history tracking)
          -> Scheduler-facing Observation (strictly filtered, non-leaking)
    """

    def __init__(
        self,
        config: Optional[Union[EnvironmentConfig, str, Path]] = None,
    ) -> None:
        if config is None:
            self.config = EnvironmentConfig()
        elif isinstance(config, (str, Path)):
            self.config = load_config(config)
        elif isinstance(config, EnvironmentConfig):
            self.config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        self.config.validate()
        self.num_bands = self.config.num_bands
        self.simulation_duration = self.config.simulation_duration
        self.seed = self.config.seed

        # Internal state
        self.current_time: int = 0
        self._terminated: bool = False
        self._step_count: int = 0

        # Subsystems
        self.emitter_registry: EmitterRegistry = EmitterRegistry.from_config_list(
            self.config.emitters,
            num_bands=self.num_bands,
            base_seed=self.seed,
        )
        self.receiver: ESMReceiver = ESMReceiver(
            config=self.config.receiver,
            num_bands=self.num_bands,
            seed=(self.seed + 200) if self.seed is not None else None,
        )
        self.observation_memory: ObservationMemory = ObservationMemory(
            num_bands=self.num_bands,
        )

        # Detailed episode log for metrics and visualization
        self.episode_dwell_history: List[DwellSummary] = []
        self.episode_action_history: List[Action] = []

    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset the simulation to t = 0.
        
        Args:
            seed: Optional random seed to re-initialize environment determinism.
            
        Returns:
            Observation: Initial scheduler-facing observation at t = 0.
        """
        if seed is not None:
            self.seed = seed
            self.config.seed = seed

        self.current_time = 0
        self._terminated = False
        self._step_count = 0
        self.episode_dwell_history.clear()
        self.episode_action_history.clear()

        # Deterministically seed subsystems
        emitter_seed = (self.seed + 100) if self.seed is not None else None
        receiver_seed = (self.seed + 200) if self.seed is not None else None

        self.emitter_registry.reset(emitter_seed)
        self.receiver.reset(receiver_seed)
        self.observation_memory.reset()

        return self.observation_memory.build_observation(
            current_time=self.current_time,
            last_band=None,
            last_dwell=None,
            last_result=DetectionResult.NONE,
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Advance the simulation by executing receiver scan dwell for the commanded action.
        
        Args:
            action: Commanded Action(frequency_band, dwell_time).
            
        Returns:
            Tuple of:
                observation (Observation): Scheduler-facing observation (non-leaked).
                reward (float): Immediate reward (Phase 1 placeholder).
                terminated (bool): True if simulation duration has been reached.
                info (dict): Internal diagnostics/evaluation info (for tests/metrics).
        """
        if self._terminated:
            raise RuntimeError("Cannot call step() on a terminated episode. Please call reset().")

        # 1. Validate action
        if not isinstance(action, Action):
            raise TypeError(f"action must be an Action instance, got {type(action).__name__}")
        self.receiver.validate_action(action)

        # 2. Execute receiver dwell over hidden ground truth
        dwell_summary = self.receiver.scan_dwell(
            action=action,
            start_time=self.current_time,
            emitter_registry=self.emitter_registry,
        )

        # 3. Phase 1 Placeholder Reward
        # Reward shaping (interception bonuses, false alarm penalties, dwell costs)
        # belongs to subsequent scheduler/evaluation phases and will be tuned experimentally.
        reward: float = 0.0

        # 4. Advance time
        dwell_start = self.current_time
        self.current_time += action.dwell_time
        self._step_count += 1

        # 5. Log history internally
        self.episode_dwell_history.append(dwell_summary)
        self.episode_action_history.append(action)

        # 6. Update observation memory
        self.observation_memory.update(
            band=action.frequency_band,
            dwell=action.dwell_time,
            result=dwell_summary.overall_result,
            event_time=self.current_time,
        )

        # 7. Check termination
        if self.current_time >= self.simulation_duration:
            self._terminated = True

        # 8. Build scheduler-visible observation (strictly no ground truth)
        observation = self.observation_memory.build_observation(
            current_time=self.current_time,
            last_band=action.frequency_band,
            last_dwell=action.dwell_time,
            last_result=dwell_summary.overall_result,
        )

        # 9. Build diagnostic info (EXCLUSIVELY for offline evaluation, metrics, and testing).
        # IMPORTANT: Schedulers must NEVER consume `info` or its nested `dwell_summary`
        # as it contains hidden ground truth (actual transmission and observability states).
        info = {
            "step_count": self._step_count,
            "dwell_start_time": dwell_start,
            "dwell_end_time": self.current_time - 1,
            "scanned_band": action.frequency_band,
            "dwell_time": action.dwell_time,
            "overall_result": dwell_summary.overall_result.value,
            "dwell_summary": dwell_summary,
        }

        return observation, reward, self._terminated, info

    @property
    def is_terminated(self) -> bool:
        """Whether the current simulation episode has reached simulation_duration."""
        return self._terminated

    def get_ground_truth_at(self, t: int, band: int):
        """Internal access to ground truth for debugging/visualization."""
        return self.emitter_registry.get_ground_truth_slot(t, band)
