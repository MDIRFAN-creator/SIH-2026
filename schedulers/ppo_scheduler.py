"""
Reinforcement Learning PPO Adaptive Scheduler for SIH26055 (Phase 5).

Connects a trained PPO policy to the standard `BaseScheduler` interface, enabling
plug-and-play evaluation with `EpisodeRunner` and the multi-scheduler benchmark framework.

Strict Non-Leakage Guarantee:
- Consumes strictly `Observation` objects and causal scanning history.
- Zero access to environment internals, EmitterRegistry, GroundTruthSlot, or diagnostic `info`.
- Deterministic inference during evaluation.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from environment.types import Action, Observation
from rl.action_encoding import ActionEncoder
from rl.ppo_agent import PPOAgent
from rl.state_features import RLStateExtractor
from schedulers.base import BaseScheduler


class PPOScheduler(BaseScheduler):
    """
    PPO Reinforcement Learning Adaptive Scheduler.
    """

    def __init__(
        self,
        agent: Optional[PPOAgent] = None,
        model_path: Optional[str] = None,
        num_bands: int = 20,
        allowed_dwells: Optional[List[int]] = None,
        max_consecutive_scans: int = 3,
        deterministic: bool = True,
        scheduler_name: str = "PPOAdaptiveScheduler",
    ) -> None:
        """
        Initialize the PPOScheduler.

        Args:
            agent: Optional initialized PPOAgent instance.
            model_path: Optional file path to load trained PPO weights from.
            num_bands: Total frequency bands (default: 20).
            allowed_dwells: Allowed dwell choices (default: [1, 2, 3]).
            max_consecutive_scans: Maximum consecutive scans allowed on any single frequency band (default: 3).
            deterministic: If True, uses argmax action selection at inference time.
            scheduler_name: Display name.
        """
        super().__init__(scheduler_name=scheduler_name)
        self.num_bands = num_bands
        self.allowed_dwells = allowed_dwells if allowed_dwells is not None else [1, 2, 3]
        self.max_consecutive_scans_limit = max_consecutive_scans
        self.deterministic = deterministic

        self.action_encoder = ActionEncoder(
            num_bands=self.num_bands,
            dwell_values=self.allowed_dwells,
        )

        self.state_extractor = RLStateExtractor(
            num_bands=self.num_bands,
            max_dwell=max(self.allowed_dwells),
        )

        if agent is not None:
            self.agent = agent
        else:
            self.agent = PPOAgent(
                state_dim=self.state_extractor.state_dim,
                action_dim=self.action_encoder.num_actions,
            )

        if model_path is not None:
            self.agent.load(model_path)

        # Telemetry and tracking
        self.total_decisions: int = 0
        self.band_selection_counts: np.ndarray = np.zeros(self.num_bands, dtype=np.int64)
        self.consecutive_runs: List[int] = []
        self._current_run_length: int = 0
        self._last_selected_band: Optional[int] = None

    def reset(self) -> None:
        """Reset internal causal state at the start of a simulation episode."""
        self.state_extractor.reset()
        self.total_decisions = 0
        self.band_selection_counts.fill(0)
        self.consecutive_runs.clear()
        self._current_run_length = 0
        self._last_selected_band = None

    def select_action(self, observation: Observation) -> Action:
        """
        Select frequency band and dwell duration given scheduler-facing Observation.

        Args:
            observation: Current scheduler observation from ESM receiver.

        Returns:
            Action: Selected Action(frequency_band, dwell_time).
        """
        # 1. Update state extractor causal history with observation
        if observation.scanned_band is not None and observation.dwell_time is not None:
            self.state_extractor.update(observation)

        # 2. Extract normalized state representation
        state = self.state_extractor.extract_state(observation)

        # 3. Compute observation-derived action mask for anti-camping
        if (
            self.max_consecutive_scans_limit > 0
            and self._current_run_length >= self.max_consecutive_scans_limit
            and self._last_selected_band is not None
        ):
            action_mask = self.action_encoder.get_mask_excluding_band(self._last_selected_band)
        else:
            action_mask = None

        # 4. Policy inference with optional action mask
        action_id, _, _ = self.agent.select_action(
            state,
            action_mask=action_mask,
            deterministic=self.deterministic,
        )

        # 5. Decode to discrete band and dwell
        band, dwell = self.action_encoder.decode(action_id)

        # 6. Telemetry updates
        self.total_decisions += 1
        self.band_selection_counts[band] += 1

        if self._last_selected_band is not None and band == self._last_selected_band:
            self._current_run_length += 1
        else:
            if self._current_run_length > 0:
                self.consecutive_runs.append(self._current_run_length)
            self._current_run_length = 1

        self._last_selected_band = band
        return Action(frequency_band=band, dwell_time=dwell)


    def update(
        self,
        observation: Observation,
        action: Action,
        reward: float,
    ) -> None:
        """
        Evaluation update hook. The policy is evaluated in frozen mode during benchmark.
        """
        pass

    def compute_band_selection_entropy(self) -> float:
        """
        Compute Shannon entropy of the band selection distribution:
            H = - sum(p_b * ln(p_b)) for all bands b.

        Returns:
            float: Shannon entropy in nats (bounded in [0, ln(num_bands)]).
        """
        total = np.sum(self.band_selection_counts)
        if total == 0:
            return 0.0
        probs = self.band_selection_counts / total
        non_zero = probs[probs > 0]
        return float(-np.sum(non_zero * np.log(non_zero)))

    @property
    def max_consecutive_scans(self) -> int:
        """Maximum consecutive scans on any single frequency band during the episode."""
        all_runs = self.consecutive_runs + ([self._current_run_length] if self._current_run_length > 0 else [])
        return max(all_runs) if all_runs else 0

    @property
    def mean_consecutive_run_length(self) -> float:
        """Mean run length of consecutive scans on the same band."""
        all_runs = self.consecutive_runs + ([self._current_run_length] if self._current_run_length > 0 else [])
        return float(np.mean(all_runs)) if all_runs else 0.0

    @property
    def unique_bands_scanned(self) -> int:
        """Count of unique frequency bands scanned at least once."""
        return int(np.count_nonzero(self.band_selection_counts))
