"""
Feature extraction module for SIH26055 (Phase 3).

Extracts explainable, causally valid RF scanning features from scheduler Observations.
Strictly isolated: Contains NO ground-truth data, NO emitter configurations, and NO future information.
"""

from collections import deque
from typing import Dict, List, Optional
import numpy as np

from environment.types import DetectionResult, Observation


FEATURE_NAMES = [
    "band_norm",              # Normalized frequency band index in [0, 1]
    "time_sin",               # Sinusoidal periodic time encoding (period 100)
    "time_cos",               # Cosine periodic time encoding (period 100)
    "time_since_scan",        # Normalized slots elapsed since band was last scanned
    "time_since_hit",         # Normalized slots elapsed since a HIT was detected on band
    "scan_fraction",          # Fraction of all decisions allocated to this band
    "cumulative_hit_rate",    # Cumulative hits / scans on this band
    "windowed_hit_rate",      # Hit rate over the last 10 scans of this band
    "false_alarm_rate",       # Cumulative false alarms / scans on this band
    "is_last_scanned",        # Binary indicator (1.0 if this band was scanned on previous step)
    "consecutive_scans",      # Normalized count of consecutive scans on this band
    "recent_dwell_norm",      # Dwell duration used on most recent scan of this band
]


class RFFeatureExtractor:
    """
    Stateful feature extractor that maintains scheduler-level observation history.
    
    Computes per-band feature vectors at each decision epoch without accessing
    environment ground truth or leaking future observations.
    """

    def __init__(self, num_bands: int = 20, window_size: int = 10):
        self.num_bands = num_bands
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        """Reset all historical trackers to cold-start initial state."""
        self.total_decisions: int = 0
        self.last_scanned_band: Optional[int] = None
        self.consecutive_scan_count: int = 0

        # Per-band statistics
        self.scan_counts: np.ndarray = np.zeros(self.num_bands, dtype=int)
        self.hit_counts: np.ndarray = np.zeros(self.num_bands, dtype=int)
        self.miss_counts: np.ndarray = np.zeros(self.num_bands, dtype=int)
        self.fa_counts: np.ndarray = np.zeros(self.num_bands, dtype=int)

        self.last_scan_time: Dict[int, Optional[int]] = {b: None for b in range(self.num_bands)}
        self.last_hit_time: Dict[int, Optional[int]] = {b: None for b in range(self.num_bands)}
        self.last_dwell_duration: Dict[int, int] = {b: 0 for b in range(self.num_bands)}

        # Sliding window history per band
        self.recent_results_window: Dict[int, deque] = {
            b: deque(maxlen=self.window_size) for b in range(self.num_bands)
        }

    def update(self, observation: Observation) -> None:
        """
        Update historical state from the incoming scheduler observation.
        
        Args:
            observation: The scheduler-facing Observation received from the environment.
        """
        if observation.scanned_band is None or observation.result == DetectionResult.NONE:
            # Initial cold-start observation before any actions have executed
            return

        band = observation.scanned_band
        res = observation.result
        t = observation.current_time
        dwell = observation.dwell_time if observation.dwell_time is not None else 1

        self.total_decisions += 1
        self.scan_counts[band] += 1
        self.last_scan_time[band] = t
        self.last_dwell_duration[band] = dwell

        # Update consecutive scans
        if self.last_scanned_band == band:
            self.consecutive_scan_count += 1
        else:
            self.last_scanned_band = band
            self.consecutive_scan_count = 1

        # Update detection outcome counts
        is_hit = (res == DetectionResult.HIT)
        is_fa = (res == DetectionResult.FALSE_ALARM)
        is_miss = (res == DetectionResult.MISS)

        if is_hit:
            self.hit_counts[band] += 1
            self.last_hit_time[band] = t
            self.recent_results_window[band].append(1)
        elif is_fa:
            self.fa_counts[band] += 1
            self.recent_results_window[band].append(0)
        elif is_miss:
            self.miss_counts[band] += 1
            self.recent_results_window[band].append(0)

    def extract_feature_single_band(self, current_time: int, band: int) -> np.ndarray:
        """
        Extract the 1D feature vector for a specific candidate band.
        
        Args:
            current_time: Current simulation time slot.
            band: Candidate frequency band index (0 <= band < num_bands).
            
        Returns:
            np.ndarray: 1D float array of length len(FEATURE_NAMES).
        """
        if not (0 <= band < self.num_bands):
            raise ValueError(f"Invalid band {band}; must be in [0, {self.num_bands - 1}]")

        # 1. Band norm
        band_norm = band / max(1, self.num_bands - 1)

        # 2-3. Periodic time encoding
        cycle_pos = (current_time % 100) / 100.0
        time_sin = np.sin(2.0 * np.pi * cycle_pos)
        time_cos = np.cos(2.0 * np.pi * cycle_pos)

        # 4. Time since last scan (cold-start prior: 10.0)
        if self.last_scan_time[band] is not None:
            time_since_scan = min((current_time - self.last_scan_time[band]) / 100.0, 10.0)
        else:
            time_since_scan = 10.0

        # 5. Time since last hit (cold-start prior: 10.0)
        if self.last_hit_time[band] is not None:
            time_since_hit = min((current_time - self.last_hit_time[band]) / 100.0, 10.0)
        else:
            time_since_hit = 10.0

        # 6. Scan fraction
        scan_fraction = self.scan_counts[band] / max(1, self.total_decisions)

        # 7. Cumulative hit rate
        cumulative_hit_rate = self.hit_counts[band] / max(1, self.scan_counts[band])

        # 8. Windowed hit rate
        window = self.recent_results_window[band]
        windowed_hit_rate = (sum(window) / len(window)) if len(window) > 0 else 0.0

        # 9. False alarm rate
        false_alarm_rate = self.fa_counts[band] / max(1, self.scan_counts[band])

        # 10. Is last scanned
        is_last_scanned = 1.0 if (self.last_scanned_band == band) else 0.0

        # 11. Consecutive scans
        consecutive_scans = (
            min(float(self.consecutive_scan_count), 10.0) / 10.0
            if (self.last_scanned_band == band)
            else 0.0
        )

        # 12. Recent dwell duration
        recent_dwell_norm = self.last_dwell_duration[band] / 5.0

        return np.array(
            [
                band_norm,
                time_sin,
                time_cos,
                time_since_scan,
                time_since_hit,
                scan_fraction,
                cumulative_hit_rate,
                windowed_hit_rate,
                false_alarm_rate,
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
            np.ndarray: 2D float array of shape (num_bands, num_features).
        """
        rows = [
            self.extract_feature_single_band(current_time=current_time, band=b)
            for b in range(self.num_bands)
        ]
        return np.vstack(rows).astype(np.float32)
