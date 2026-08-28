"""
Observation-Only Reinforcement Learning State Feature Extractor (SIH26055 Phase 5).

Constructs a compact, fully normalized, fixed-dimensional state vector from scheduler-visible
`Observation` streams and causal scanning history.

Strict Non-Leakage Guarantee:
- Zero access to EmitterRegistry, GroundTruthSlot, DwellSummary, or simulator internals.
- All features are derived strictly from legitimate scheduler inputs (time, action, detection result).
- Deterministic cold-start initialization.

State Vector Structure (Total Dimension = 227):
1. Global Context Features (7 dimensions):
   - normalized_time: t / max_time_slots ∈ [0.0, 1.0]
   - prev_band_norm: last_scanned_band / (num_bands - 1) ∈ [0.0, 1.0] (-1.0 if initial)
   - prev_dwell_norm: (last_dwell - 1) / (max_dwell - 1) ∈ [0.0, 1.0]
   - prev_result_hit: 1.0 if last result was HIT else 0.0
   - prev_result_miss: 1.0 if last result was MISS or NONE else 0.0
   - prev_result_fa: 1.0 if last result was FALSE_ALARM else 0.0
   - consecutive_scans_norm: min(consecutive_scans, 10) / 10.0 ∈ [0.0, 1.0]

2. Per-Band Context Features (11 features × 20 bands = 220 dimensions):
   For each frequency band b ∈ [0, 19]:
   - band_norm: b / (num_bands - 1)
   - time_since_scan: min(t - last_scanned_t, max_idle) / max_idle ∈ [0.0, 1.0] (1.0 if never scanned)
   - time_since_hit: min(t - last_hit_t, max_idle_hit) / max_idle_hit ∈ [0.0, 1.0] (1.0 if never hit)
   - cumulative_hit_rate: hits[b] / max(1, scans[b]) ∈ [0.0, 1.0]
   - windowed_hit_rate: window_hits[b] / max(1, window_len[b]) ∈ [0.0, 1.0]
   - false_alarm_rate: fas[b] / max(1, scans[b]) ∈ [0.0, 1.0]
   - windowed_fa_rate: window_fas[b] / max(1, window_len[b]) ∈ [0.0, 1.0]
   - scan_fraction: scans[b] / max(1, total_decisions) ∈ [0.0, 1.0]
   - is_last_scanned: 1.0 if b == last_scanned_band else 0.0
   - consecutive_scans: min(consecutive_scans[b], 10) / 10.0 ∈ [0.0, 1.0]
   - recent_dwell_norm: (last_dwell[b] - 1) / (max_dwell - 1) ∈ [0.0, 1.0]
"""

from collections import deque
from typing import Dict, List, Optional
import numpy as np

from environment.types import DetectionResult, Observation


GLOBAL_FEATURE_NAMES = [
    "normalized_time",
    "prev_band_norm",
    "prev_dwell_norm",
    "prev_result_hit",
    "prev_result_miss",
    "prev_result_fa",
    "consecutive_scans_norm",
]

PER_BAND_FEATURE_NAMES = [
    "band_norm",
    "time_since_scan",
    "time_since_hit",
    "cumulative_hit_rate",
    "windowed_hit_rate",
    "false_alarm_rate",
    "windowed_fa_rate",
    "scan_fraction",
    "is_last_scanned",
    "consecutive_scans",
    "recent_dwell_norm",
]


class RLStateExtractor:
    """
    State representation builder for reinforcement learning agents in the RF environment.
    """

    def __init__(
        self,
        num_bands: int = 20,
        max_time_slots: int = 1000,
        max_dwell: int = 3,
        window_size: int = 10,
        max_idle_scan: int = 200,
        max_idle_hit: int = 500,
    ) -> None:
        """
        Initialize the RLStateExtractor.

        Args:
            num_bands: Total frequency bands.
            max_time_slots: Normalization horizon for episode time.
            max_dwell: Maximum dwell duration for normalization.
            window_size: Rolling observation window length per band.
            max_idle_scan: Recency normalization cap for scans.
            max_idle_hit: Recency normalization cap for hits.
        """
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        if max_time_slots <= 0:
            raise ValueError(f"max_time_slots must be positive, got {max_time_slots}")
        if max_dwell <= 1:
            raise ValueError(f"max_dwell must be > 1, got {max_dwell}")

        self.num_bands = num_bands
        self.max_time_slots = max_time_slots
        self.max_dwell = max_dwell
        self.window_size = window_size
        self.max_idle_scan = float(max_idle_scan)
        self.max_idle_hit = float(max_idle_hit)

        self.num_global_features = len(GLOBAL_FEATURE_NAMES)
        self.num_per_band_features = len(PER_BAND_FEATURE_NAMES)
        self.state_dim = self.num_global_features + self.num_bands * self.num_per_band_features

        # Internal causal tracking state
        self.total_decisions: int = 0
        self.current_time_slot: int = 0
        self.scan_counts: np.ndarray = np.zeros(num_bands, dtype=np.int64)
        self.hit_counts: np.ndarray = np.zeros(num_bands, dtype=np.int64)
        self.fa_counts: np.ndarray = np.zeros(num_bands, dtype=np.int64)
        self.last_scanned_time: np.ndarray = np.full(num_bands, -1, dtype=np.int64)
        self.last_hit_time: np.ndarray = np.full(num_bands, -1, dtype=np.int64)
        self.last_dwell_duration: np.ndarray = np.zeros(num_bands, dtype=np.float32)
        self.consecutive_scans_count: np.ndarray = np.zeros(num_bands, dtype=np.int64)

        self.last_scanned_band: Optional[int] = None
        self.last_dwell: Optional[int] = None
        self.last_result: Optional[DetectionResult] = None
        self.current_consecutive: int = 0

        self.recent_results: Dict[int, deque] = {b: deque(maxlen=window_size) for b in range(num_bands)}

    def reset(self) -> None:
        """Reset internal history state at the start of a new episode."""
        self.total_decisions = 0
        self.current_time_slot = 0
        self.scan_counts.fill(0)
        self.hit_counts.fill(0)
        self.fa_counts.fill(0)
        self.last_scanned_time.fill(-1)
        self.last_hit_time.fill(-1)
        self.last_dwell_duration.fill(0.0)
        self.consecutive_scans_count.fill(0)

        self.last_scanned_band = None
        self.last_dwell = None
        self.last_result = None
        self.current_consecutive = 0

        for b in range(self.num_bands):
            self.recent_results[b].clear()

    def update(self, observation: Observation) -> None:
        """
        Incorporate a new observation into internal causal tracking.

        Args:
            observation: Legitimate scheduler-visible observation.
        """
        self.current_time_slot = observation.current_time

        if observation.scanned_band is None or observation.dwell_time is None:
            return  # Initial reset observation before any action execution

        band = observation.scanned_band
        dwell = observation.dwell_time
        result = observation.result
        t = observation.current_time

        self.total_decisions += 1
        self.scan_counts[band] += 1
        self.last_scanned_time[band] = t
        self.last_dwell_duration[band] = float(dwell)

        if band == self.last_scanned_band:
            self.current_consecutive += 1
            self.consecutive_scans_count[band] = self.current_consecutive
        else:
            self.current_consecutive = 1
            self.consecutive_scans_count.fill(0)
            self.consecutive_scans_count[band] = 1

        self.last_scanned_band = band
        self.last_dwell = dwell
        self.last_result = result

        if result == DetectionResult.HIT:
            self.hit_counts[band] += 1
            self.last_hit_time[band] = t
        elif result == DetectionResult.FALSE_ALARM:
            self.fa_counts[band] += 1

        self.recent_results[band].append(result)

    def extract_state(self, observation: Optional[Observation] = None) -> np.ndarray:
        """
        Extract the complete normalized state vector.

        Args:
            observation: Optional current observation. If provided, updates time slot.

        Returns:
            np.ndarray: 1D float32 array of length self.state_dim (227).
        """
        if observation is not None:
            self.current_time_slot = observation.current_time

        state = np.zeros(self.state_dim, dtype=np.float32)
        idx = 0

        # 1. Global features (7)
        t_norm = min(1.0, float(self.current_time_slot) / float(self.max_time_slots))
        state[idx] = t_norm
        idx += 1

        if self.last_scanned_band is not None and self.num_bands > 1:
            state[idx] = float(self.last_scanned_band) / float(self.num_bands - 1)
        else:
            state[idx] = -1.0
        idx += 1

        if self.last_dwell is not None and self.max_dwell > 1:
            state[idx] = float(self.last_dwell - 1) / float(self.max_dwell - 1)
        else:
            state[idx] = 0.0
        idx += 1

        state[idx] = 1.0 if self.last_result == DetectionResult.HIT else 0.0
        idx += 1
        state[idx] = 1.0 if self.last_result in (DetectionResult.MISS, DetectionResult.NONE) else 0.0
        idx += 1
        state[idx] = 1.0 if self.last_result == DetectionResult.FALSE_ALARM else 0.0
        idx += 1

        state[idx] = min(1.0, float(self.current_consecutive) / 10.0)
        idx += 1

        # 2. Per-band features (20 * 11 = 220)
        t = self.current_time_slot
        tot_dec = max(1, self.total_decisions)

        for b in range(self.num_bands):
            # band_norm
            state[idx] = float(b) / float(max(1, self.num_bands - 1))
            idx += 1

            # time_since_scan
            last_s = self.last_scanned_time[b]
            if last_s < 0:
                state[idx] = 1.0
            else:
                state[idx] = min(1.0, float(t - last_s) / self.max_idle_scan)
            idx += 1

            # time_since_hit
            last_h = self.last_hit_time[b]
            if last_h < 0:
                state[idx] = 1.0
            else:
                state[idx] = min(1.0, float(t - last_h) / self.max_idle_hit)
            idx += 1

            # cumulative_hit_rate
            scans_b = self.scan_counts[b]
            state[idx] = float(self.hit_counts[b]) / float(max(1, scans_b))
            idx += 1

            # windowed_hit_rate
            win = self.recent_results[b]
            if len(win) == 0:
                state[idx] = 0.0
            else:
                state[idx] = float(sum(1 for r in win if r == DetectionResult.HIT)) / float(len(win))
            idx += 1

            # false_alarm_rate
            state[idx] = float(self.fa_counts[b]) / float(max(1, scans_b))
            idx += 1

            # windowed_fa_rate
            if len(win) == 0:
                state[idx] = 0.0
            else:
                state[idx] = float(sum(1 for r in win if r == DetectionResult.FALSE_ALARM)) / float(len(win))
            idx += 1

            # scan_fraction
            state[idx] = float(scans_b) / float(tot_dec)
            idx += 1

            # is_last_scanned
            state[idx] = 1.0 if (self.last_scanned_band == b) else 0.0
            idx += 1

            # consecutive_scans
            state[idx] = min(1.0, float(self.consecutive_scans_count[b]) / 10.0)
            idx += 1

            # recent_dwell_norm
            d_prev = self.last_dwell_duration[b]
            if d_prev > 0 and self.max_dwell > 1:
                state[idx] = float(d_prev - 1.0) / float(self.max_dwell - 1)
            else:
                state[idx] = 0.0
            idx += 1

        assert idx == self.state_dim, f"Feature dimension mismatch: expected {self.state_dim}, got {idx}"
        return state
