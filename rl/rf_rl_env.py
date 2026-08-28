"""
Gymnasium-Compatible Reinforcement Learning Environment for RF Spectrum Scanning (SIH26055 Phase 5).

Wraps the core `RFEnvironment` to provide standard `reset(seed=None)` and `step(action_id)` methods
with observation-only states and rewards.

Strict Non-Leakage Guarantee:
- The RL agent receives strictly observation-derived state vectors (dim=227).
- Rewards are computed strictly from scheduler-visible observation outcomes.
- No simulator internals or ground-truth opportunities are exposed in observations or rewards.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Fallback to minimal class structure if gymnasium is unavailable
    class gym:  # type: ignore
        class Env:
            pass

    class spaces:  # type: ignore
        class Discrete:
            def __init__(self, n: int):
                self.n = n

        class Box:
            def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype


from environment.config import EnvironmentConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, Observation
from rl.action_encoding import ActionEncoder
from rl.reward import RLRewardCalculator, RLRewardConfig
from rl.state_features import RLStateExtractor


class RFRLGymEnv(gym.Env):
    """
    Gymnasium environment wrapper for RF spectrum scanning.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        reward_config: Optional[RLRewardConfig] = None,
        dwell_values: Optional[List[int]] = None,
        max_consecutive_scans: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Gymnasium RL environment.

        Args:
            env_config: Optional RF EnvironmentConfig.
            reward_config: Optional RLRewardConfig.
            dwell_values: Allowed dwell durations (default: [1, 2, 3]).
            max_consecutive_scans: Max consecutive scans before masking (0 = unconstrained).
            seed: Initial random seed.
        """
        super().__init__()
        self.rf_config = env_config if env_config is not None else EnvironmentConfig()
        self.reward_calculator = RLRewardCalculator(config=reward_config)
        self.dwell_values = dwell_values if dwell_values is not None else [1, 2, 3]
        self.max_consecutive_scans = max_consecutive_scans

        self.num_bands = self.rf_config.num_bands
        self.action_encoder = ActionEncoder(
            num_bands=self.num_bands,
            dwell_values=self.dwell_values,
        )

        self.state_extractor = RLStateExtractor(
            num_bands=self.num_bands,
            max_time_slots=self.rf_config.simulation_duration,
            max_dwell=max(self.dwell_values),
        )

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(self.action_encoder.num_actions)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.state_extractor.state_dim,),
            dtype=np.float32,
        )

        if seed is not None:
            self.rf_config.seed = seed
        self.rf_env = RFEnvironment(config=self.rf_config)
        self.last_observation: Optional[Observation] = None
        self.consecutive_scans_tracker: int = 0
        self.consecutive_hits_tracker: int = 0
        self.last_scanned_band: Optional[int] = None

    def get_action_mask(self) -> np.ndarray:
        """
        Compute valid action mask based on observation-derived anti-camping constraints.

        Returns:
            np.ndarray: Boolean array of shape (num_actions,) where True = valid, False = masked.
        """
        if self.max_consecutive_scans > 0 and self.consecutive_scans_tracker >= self.max_consecutive_scans:
            return self.action_encoder.get_mask_excluding_band(self.last_scanned_band)
        return np.ones(self.action_encoder.num_actions, dtype=bool)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the RF environment and state extractor for a new episode.

        Args:
            seed: Optional integer seed for reproducibility.
            options: Optional gymnasium options dictionary.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (initial_state_vector, info_dict).
        """
        initial_obs = self.rf_env.reset(seed=seed)
        self.state_extractor.reset()
        self.last_observation = initial_obs
        self.consecutive_scans_tracker = 0
        self.consecutive_hits_tracker = 0
        self.last_scanned_band = None

        state = self.state_extractor.extract_state(initial_obs)
        info: Dict[str, Any] = {
            "current_time": self.rf_env.current_time,
            "seed": seed if seed is not None else self.rf_env.seed,
            "action_mask": self.get_action_mask(),
        }
        return state, info

    def step(
        self,
        action_id: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one scheduling decision in the RF simulation.

        Args:
            action_id: Integer action in [0, num_actions - 1].

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                (next_state, reward, terminated, truncated, info)
        """
        band, dwell = self.action_encoder.decode(action_id)
        action = Action(frequency_band=band, dwell_time=dwell)

        # Observation-derived staleness prior to executing the action
        last_t = self.state_extractor.last_scanned_time[band]
        time_since_last_scan = float(self.rf_env.current_time - last_t) if last_t >= 0 else -1.0

        # Track consecutive scans on the same band
        if band == self.last_scanned_band:
            self.consecutive_scans_tracker += 1
        else:
            self.consecutive_scans_tracker = 1
        self.last_scanned_band = band

        # Step underlying RF Environment
        obs, _, terminated, raw_info = self.rf_env.step(action)
        self.last_observation = obs

        # Track consecutive hits
        from environment.types import DetectionResult
        if obs.result == DetectionResult.HIT:
            self.consecutive_hits_tracker += 1
        else:
            self.consecutive_hits_tracker = 0

        # Calculate observation-derived reward
        reward = self.reward_calculator.compute_reward(
            observation=obs,
            dwell_time=dwell,
            consecutive_scans=self.consecutive_scans_tracker,
            time_since_last_scan=time_since_last_scan,
            consecutive_hits=self.consecutive_hits_tracker,
        )

        # Update state extractor
        self.state_extractor.update(obs)
        next_state = self.state_extractor.extract_state(obs)

        truncated = False
        info: Dict[str, Any] = {
            "scanned_band": band,
            "dwell_time": dwell,
            "result": obs.result.name if obs.result else "NONE",
            "current_time": self.rf_env.current_time,
            "terminated": terminated,
            "action_mask": self.get_action_mask(),
        }

        return next_state, reward, terminated, truncated, info

