"""
Observation builder and non-leaking history tracker for the RF Environment (SIH26055).

This module manages receiver observation memory and computes legitimate historical
features (hit rates, scan recency, scan frequencies) without leaking any hidden
ground-truth information.
"""

from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np

from environment.types import DetectionResult, Observation


class ObservationMemory:
    """
    Tracks scheduler-visible observation history across frequency bands.
    
    Maintains:
    - Cumulative scan counts, hits, misses, and false alarms per band
    - Timestamps of last scan and last hit per band
    - Sliding-window buffers for recency-weighted hit rate estimates
    """

    def __init__(self, num_bands: int = 20, window_size: int = 20) -> None:
        self.num_bands = num_bands
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        """Clear all historical observation counters and buffers."""
        self.total_scans = np.zeros(self.num_bands, dtype=np.int64)
        self.hits = np.zeros(self.num_bands, dtype=np.int64)
        self.misses = np.zeros(self.num_bands, dtype=np.int64)
        self.false_alarms = np.zeros(self.num_bands, dtype=np.int64)
        self.last_scan_time = np.full(self.num_bands, -1, dtype=np.int64)
        self.last_hit_time = np.full(self.num_bands, -1, dtype=np.int64)
        self.sliding_window: List[deque] = [deque(maxlen=self.window_size) for _ in range(self.num_bands)]
        self.total_decisions: int = 0

    def update(
        self,
        band: int,
        dwell: int,
        result: DetectionResult,
        event_time: int,
    ) -> None:
        """
        Record a new scan outcome into observation memory.
        
        Args:
            band: Frequency band scanned.
            dwell: Dwell duration.
            result: Result observed (HIT, MISS, or FALSE_ALARM).
            event_time: Simulation time slot when the scan completed.
        """
        self.total_decisions += 1
        self.total_scans[band] += 1
        self.last_scan_time[band] = event_time

        is_hit = (result == DetectionResult.HIT)
        self.sliding_window[band].append(1 if is_hit else 0)

        if result == DetectionResult.HIT:
            self.hits[band] += 1
            self.last_hit_time[band] = event_time
        elif result == DetectionResult.FALSE_ALARM:
            self.false_alarms[band] += 1
        elif result == DetectionResult.MISS:
            self.misses[band] += 1

    def get_history_summary(self, current_time: int) -> Dict[str, Any]:
        """
        Generate a snapshot dictionary of historical statistics at current_time.
        
        All metrics are derived strictly from receiver observations.
        """
        # Time since last scan
        time_since_last_scan = np.where(
            self.last_scan_time >= 0,
            current_time - self.last_scan_time,
            current_time,
        )

        # Time since last hit
        time_since_last_hit = np.where(
            self.last_hit_time >= 0,
            current_time - self.last_hit_time,
            current_time,
        )

        # Cumulative hit rate
        cumulative_hit_rate = np.zeros(self.num_bands, dtype=np.float64)
        scanned_mask = self.total_scans > 0
        cumulative_hit_rate[scanned_mask] = self.hits[scanned_mask] / self.total_scans[scanned_mask]

        # Windowed hit rate
        recent_hit_rate = np.zeros(self.num_bands, dtype=np.float64)
        for b in range(self.num_bands):
            buf = self.sliding_window[b]
            if len(buf) > 0:
                recent_hit_rate[b] = sum(buf) / len(buf)

        return {
            "total_decisions": self.total_decisions,
            "total_scans_per_band": self.total_scans.tolist(),
            "hits_per_band": self.hits.tolist(),
            "misses_per_band": self.misses.tolist(),
            "false_alarms_per_band": self.false_alarms.tolist(),
            "last_scan_time_per_band": self.last_scan_time.tolist(),
            "last_hit_time_per_band": self.last_hit_time.tolist(),
            "time_since_last_scan": time_since_last_scan.tolist(),
            "time_since_last_hit": time_since_last_hit.tolist(),
            "cumulative_hit_rate": cumulative_hit_rate.tolist(),
            "recent_hit_rate": recent_hit_rate.tolist(),
        }

    def build_observation(
        self,
        current_time: int,
        last_band: Optional[int] = None,
        last_dwell: Optional[int] = None,
        last_result: DetectionResult = DetectionResult.NONE,
    ) -> Observation:
        """
        Construct a clean Observation dataclass instance.
        """
        return Observation(
            current_time=current_time,
            scanned_band=last_band,
            dwell_time=last_dwell,
            result=last_result,
            history_summary=self.get_history_summary(current_time),
        )
