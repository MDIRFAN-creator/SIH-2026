"""
Abstract Base Scheduler interface for ESM frequency-time scanning (SIH26055).

All present and future scheduling algorithms (Open-Loop, XGBoost, Contextual Bandit,
and Reinforcement Learning) implement this common interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
from environment.types import Action, Observation


class BaseScheduler(ABC):
    """
    Abstract base class for all frequency-time schedulers.
    
    A scheduler consumes strictly non-leaked ESM observations and emits
    an Action(frequency_band, dwell_time).
    """

    def __init__(self, scheduler_name: str = "BaseScheduler") -> None:
        self.scheduler_name = scheduler_name

    @property
    def name(self) -> str:
        return self.scheduler_name

    @abstractmethod
    def reset(self) -> None:
        """Reset internal scheduler state at the beginning of an episode."""
        pass

    @abstractmethod
    def select_action(self, observation: Observation) -> Action:
        """
        Choose the next frequency band to observe and the dwell duration.
        
        Args:
            observation: Current scheduler-visible observation from the ESM receiver.
            
        Returns:
            Action: Selected frequency_band and dwell_time.
        """
        pass

    def update(
        self,
        observation: Observation,
        action: Action,
        reward: float,
    ) -> None:
        """
        Optional learning/update hook called after action execution.
        
        Default implementation is a no-op (for non-learning / open-loop schedulers).
        """
        pass
