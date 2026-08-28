"""
LinUCB Contextual Feature Extractor for SIH26055 (Phase 4).

Extracts a compact, explainable 10-dimensional context feature vector for each candidate
frequency band derived strictly from scheduler-visible `Observation` objects and
scheduler-maintained causal history.

Strict Non-Leakage Guarantee:
- No access to EmitterRegistry, GroundTruthSlot, DwellSummary, or hidden simulation internals.
- Fully deterministic cold-start defaults.
"""

from collections import deque
from typing import Dict, List, Optional
import numpy as np

from environment.types import DetectionResult, Observation

LINUCB_FEATURE_NAMES = [
    "band_norm",
    "time_since_scan",
    "time_since_hit",
    "cumulative_hit_rate",
    "windowed_hit_rate",
    "false_alarm_rate",
    "scan_fraction",
    "is_last_scanned",
    "consecutive_scans",
    "recent_dwell_norm",
]


class LinUCBFeatureExtractor:
    """
    Observation-only causal feature extractor for LinUCB contextual bandits.
    """

    def __init__(self, num_bands: int = 20, window_size: int = 10) -> None:
        """
        Initialize the LinUCB feature extractor.
        
        Args:
            num_bands: Number of RF frequency bands.
            window_size: Rolling window size for short-term hit rate.
        """
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.num_bands = num_bands
        self.window_size = window_size
        self.feature_names = list(LINUCB_FEATURE_NAMES)
        self.feature_dim = len(self.feature_names)

        # Internal tracking state
        self.total_decisions: int = 0
        self.scan_counts: np.ndarray = np.zeros(num_bands, dtype=np.int64)
        self.hit_counts: np.ndarray = np.zeros(num_bands, dtype=np.int64)
        self.fa_counts: np.ndarray = np.zeros(num_bands, dtype=np.int64)
        self.last_scanned_time: np.ndarray = np.full(num_bands, -1, dtype=np.int64)
        self.last_hit_time: np.ndarray = np.full(num_bands, -1, dtype=np.int64)
        self.last_dwell_duration: np.ndarray = np.zeros(num_bands, dtype=np.float32)
        self.consecutive_scans_count: np.ndarray = np.zeros(num_bands, dtype=np.int64)
        self.last_scanned_band: Optional[int] = None
        self.recent_results: Dict[int, deque] = {b: deque(maxlen=window_size) for b in range(num_bands)}

    def reset(self) -> None:
        """Reset internal history state at the beginning of an episode."""
        self.total_decisions = 0
        self.scan_counts.fill(0)
        self.hit_counts.fill(0)
        self.fa_counts.fill(0)
        self.last_scanned_time.fill(-1)
        self.last_hit_time.fill(-1)
        self.last_dwell_duration.fill(0.0)
        self.consecutive_scans_count.fill(0)
        self.last_scanned_band = None
        for b in range(self.num_bands):
            self.recent_results[b].clear()

    def update(self, observation: Observation) -> None:
        """
        Update feature extractor history from the received Observation.
        
        Args:
            observation: Legitimate scheduler-visible observation.
        """
        if observation.scanned_band is None or observation.dwell_time is None:
            return  # Initial reset observation with no preceding action

        band = observation.scanned_band
        dwell = observation.dwell_time
        result = observation.result
        t = observation.current_time

        self.total_decisions += 1
        self.scan_counts[band] += 1
        self.last_scanned_time[band] = t
        self.last_dwell_duration[band] = float(dwell)

        if band == self.last_scanned_band:
            self.consecutive_scans_count[band] += 1
        else:
            self.consecutive_scans_count.fill(0)
            self.consecutive_scans_count[band] = 1
            self.last_scanned_band = band

        is_hit = (result == DetectionResult.HIT)
        is_fa = (result == DetectionResult.FALSE_ALARM)

        if is_hit:
            self.hit_counts[band] += 1
            self.last_hit_time[band] = t

        if is_fa:
            self.fa_counts[band] += 1

        self.recent_results[band].append(1 if is_hit else 0)

    def extract_feature_single_band(self, current_time: int, band: int) -> np.ndarray:
        """
        Extract the 10-dimensional context feature vector for candidate band `band`.
        
        Args:
            current_time: Current simulation time slot.
            band: Candidate frequency band index in [0, num_bands - 1].
            
        Returns:
            np.ndarray: 1D float32 array of shape (10,).
        """
        if band < 0 or band >= self.num_bands:
            raise IndexError(f"Band index {band} out of bounds for {self.num_bands} bands")

        # 1. Normalized band index
        band_norm = float(band) / max(1, self.num_bands - 1)

        # 2. Time since last scan
        last_scan_t = self.last_scanned_time[band]
        if last_scan_t < 0:
            time_since_scan = 10.0  # Cold-start prior for unscanned band
        else:
            time_since_scan = min(float(current_time - last_scan_t) / 100.0, 10.0)

        # 3. Time since last hit
        last_hit_t = self.last_hit_time[band]
        if last_hit_t < 0:
            time_since_hit = 10.0  # Cold-start prior for un-hit band
        else:
            time_since_hit = min(float(current_time - last_hit_t) / 100.0, 10.0)

        # 4. Cumulative hit rate
        n_scans = self.scan_counts[band]
        cumulative_hit_rate = float(self.hit_counts[band]) / max(1, n_scans) if n_scans > 0 else 0.0

        # 5. Windowed hit rate
        recents = self.recent_results[band]
        windowed_hit_rate = float(sum(recents)) / len(recents) if len(recents) > 0 else 0.0

        # 6. False alarm rate
        false_alarm_rate = float(self.fa_counts[band]) / max(1, n_scans) if n_scans > 0 else 0.0

        # 7. Scan fraction
        scan_fraction = float(n_scans) / max(1, self.total_decisions) if self.total_decisions > 0 else 0.0

        # 8. Is last scanned band
        is_last_scanned = 1.0 if (self.last_scanned_band is not None and band == self.last_scanned_band) else 0.0

        # 9. Consecutive scans count
        consecutive_scans = min(float(self.consecutive_scans_count[band]), 10.0) / 10.0

        # 10. Recent dwell duration
        recent_dwell_norm = self.last_dwell_duration[band] / 5.0

        return np.array(
            [
                band_norm,
                time_since_scan,
                time_since_hit,
                cumulative_hit_rate,
                windowed_hit_rate,
                false_alarm_rate,
                scan_fraction,
                is_last_scanned,
                consecutive_scans,
                recent_dwell_norm,
            ],
            dtype=np.float32,
        )

    def extract_features_all_bands(self, current_time: int) -> np.ndarray:
        """
        Extract feature matrix for all candidate frequency bands.
        
        Args:
            current_time: Current simulation time slot.
            
        Returns:
            np.ndarray: 2D float32 array of shape (num_bands, 10).
        """
        rows = [
            self.extract_feature_single_band(current_time=current_time, band=b)
            for b in range(self.num_bands)
        ]
        return np.vstack(rows).astype(np.float32)
