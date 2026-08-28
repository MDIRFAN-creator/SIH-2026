"""
Episode Runner and Execution Recorder for SIH26055 (Phase 2).

Connects a frequency-time scheduler to the RFEnvironment while maintaining strict
architectural separation between scheduler-visible Observations and offline evaluation data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from environment.config import EnvironmentConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, DwellSummary, Observation
from schedulers.base import BaseScheduler


@dataclass
class EpisodeStepRecord:
    """
    Complete audit record for a single scheduling decision and dwell execution.
    
    Attributes:
        decision_index: 0-indexed sequence counter of scheduling decisions.
        start_time: Time slot when dwell began (inclusive).
        end_time: Time slot when dwell ended (inclusive).
        action: Action commanded by the scheduler.
        observation: Scheduler-facing observation received after step.
        dwell_summary: Detailed evaluation diagnostics (including ground truth).
        reward: Immediate reward value.
    """
    decision_index: int
    start_time: int
    end_time: int
    action: Action
    observation: Observation
    dwell_summary: DwellSummary
    reward: float


@dataclass
class EpisodeResult:
    """
    Complete outcome container for an executed simulation episode.
    
    Preserves all necessary data for offline evaluation, metric computation,
    and visualization across subsequent phases.
    """
    scheduler_name: str
    seed: Optional[int]
    total_time_slots: int
    total_decisions: int
    step_records: List[EpisodeStepRecord] = field(default_factory=list)
    dwell_history: List[DwellSummary] = field(default_factory=list)
    environment_config: Optional[EnvironmentConfig] = None


class EpisodeRunner:
    """
    Executes and records simulation episodes between any BaseScheduler and RFEnvironment.
    
    Strict Isolation Guarantee:
    - The scheduler receives ONLY `Observation` objects via `select_action(observation)` and `update(observation, action, reward)`.
    - Diagnostic `info` dictionaries and hidden `GroundTruthSlot` objects are NEVER passed to the scheduler.
    """

    def __init__(self) -> None:
        pass

    def run_episode(
        self,
        env: RFEnvironment,
        scheduler: BaseScheduler,
        seed: Optional[int] = None,
    ) -> EpisodeResult:
        """
        Execute a complete simulation episode from t = 0 to simulation_duration.
        
        Args:
            env: Initialized RFEnvironment instance.
            scheduler: Initialized BaseScheduler instance.
            seed: Optional random seed for reproducible episode generation.
            
        Returns:
            EpisodeResult: Comprehensive episode execution container.
        """
        # 1. Reset environment and scheduler
        initial_obs = env.reset(seed=seed)
        scheduler.reset()

        step_records: List[EpisodeStepRecord] = []
        current_obs = initial_obs
        decision_idx = 0

        # 2. Main simulation loop
        while not env.is_terminated:
            # Schedulers receive ONLY observation
            action = scheduler.select_action(current_obs)

            start_t = env.current_time
            next_obs, reward, terminated, info = env.step(action)
            end_t = env.current_time - 1

            dwell_summary: DwellSummary = info["dwell_summary"]

            # Record step audit
            record = EpisodeStepRecord(
                decision_index=decision_idx,
                start_time=start_t,
                end_time=end_t,
                action=action,
                observation=next_obs,
                dwell_summary=dwell_summary,
                reward=reward,
            )
            step_records.append(record)

            # Optional scheduler update hook
            scheduler.update(next_obs, action, reward)

            current_obs = next_obs
            decision_idx += 1

        return EpisodeResult(
            scheduler_name=scheduler.scheduler_name,
            seed=seed if seed is not None else env.seed,
            total_time_slots=env.current_time,
            total_decisions=len(step_records),
            step_records=step_records,
            dwell_history=list(env.episode_dwell_history),
            environment_config=env.config,
        )

    def run_episodes_multi_seed(
        self,
        env: RFEnvironment,
        scheduler: BaseScheduler,
        seeds: List[int],
    ) -> List[EpisodeResult]:
        """
        Execute multiple independent episodes across a list of seeds.
        
        Args:
            env: RFEnvironment instance.
            scheduler: BaseScheduler instance.
            seeds: List of integer random seeds.
            
        Returns:
            List[EpisodeResult]: List of episode results, one per seed.
        """
        results: List[EpisodeResult] = []
        for s in seeds:
            res = self.run_episode(env=env, scheduler=scheduler, seed=s)
            results.append(res)
        return results
