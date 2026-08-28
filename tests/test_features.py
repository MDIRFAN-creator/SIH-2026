"""
Unit tests for RFFeatureExtractor (SIH26055 - Phase 3).
"""

import pytest
import numpy as np

from environment.types import DetectionResult, Observation
from features.rf_features import FEATURE_NAMES, RFFeatureExtractor


def test_feature_extractor_dimensions():
    """Verify feature vector length and matrix dimensions."""
    extractor = RFFeatureExtractor(num_bands=20)
    assert len(FEATURE_NAMES) == 12

    vec = extractor.extract_feature_single_band(current_time=0, band=5)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (12,)
    assert vec.dtype == np.float32

    matrix = extractor.extract_features_all_bands(current_time=0)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (20, 12)


def test_feature_cold_start_defaults():
    """Verify deterministic default values when no previous observations exist."""
    extractor = RFFeatureExtractor(num_bands=20)

    # Prior to any updates (cold start)
    vec = extractor.extract_feature_single_band(current_time=0, band=7)
    
    # band_norm = 7 / 19
    assert vec[0] == pytest.approx(7.0 / 19.0)
    # time_since_scan = 10.0 (unscanned default)
    assert vec[3] == 10.0
    # time_since_hit = 10.0 (never hit default)
    assert vec[4] == 10.0
    # scan_fraction = 0.0
    assert vec[5] == 0.0
    # cumulative_hit_rate = 0.0
    assert vec[6] == 0.0
    # windowed_hit_rate = 0.0
    assert vec[7] == 0.0
    # false_alarm_rate = 0.0
    assert vec[8] == 0.0
    # is_last_scanned = 0.0
    assert vec[9] == 0.0
    # consecutive_scans = 0.0
    assert vec[10] == 0.0


def test_feature_updates_and_tracking():
    """Verify feature extractor updates accurately on sequences of observations."""
    extractor = RFFeatureExtractor(num_bands=10, window_size=5)

    # Step 1: Scan B3 at t=10 with HIT (dwell 2)
    obs1 = Observation(current_time=10, scanned_band=3, dwell_time=2, result=DetectionResult.HIT)
    extractor.update(obs1)

    vec3 = extractor.extract_feature_single_band(current_time=20, band=3)
    assert vec3[3] == pytest.approx((20 - 10) / 100.0)  # time_since_scan = 0.10
    assert vec3[4] == pytest.approx((20 - 10) / 100.0)  # time_since_hit = 0.10
    assert vec3[5] == 1.0  # 1 scan out of 1 decision
    assert vec3[6] == 1.0  # 1 hit out of 1 scan
    assert vec3[7] == 1.0  # windowed hit rate 1.0
    assert vec3[8] == 0.0  # false alarm rate 0.0
    assert vec3[9] == 1.0  # is_last_scanned = 1.0
    assert vec3[10] == 0.1 # 1 consecutive scan / 10 = 0.1
    assert vec3[11] == pytest.approx(2.0 / 5.0)  # dwell 2 / 5 = 0.40

    # Step 2: Scan B3 at t=20 with MISS (dwell 1) -> 2 consecutive scans on B3
    obs2 = Observation(current_time=20, scanned_band=3, dwell_time=1, result=DetectionResult.MISS)
    extractor.update(obs2)

    vec3_after = extractor.extract_feature_single_band(current_time=30, band=3)
    assert vec3_after[5] == 1.0  # 2 scans out of 2 decisions
    assert vec3_after[6] == 0.5  # 1 hit out of 2 scans = 0.50
    assert vec3_after[7] == 0.5  # windowed hit rate: [1, 0] = 0.50
    assert vec3_after[10] == 0.2 # 2 consecutive scans / 10 = 0.2

    # Step 3: Scan B8 at t=30 with FALSE_ALARM -> B3 is no longer last scanned
    obs3 = Observation(current_time=30, scanned_band=8, dwell_time=1, result=DetectionResult.FALSE_ALARM)
    extractor.update(obs3)

    vec3_step3 = extractor.extract_feature_single_band(current_time=40, band=3)
    assert vec3_step3[9] == 0.0  # is_last_scanned = 0.0
    assert vec3_step3[10] == 0.0 # consecutive_scans on B3 reset to 0.0

    vec8_step3 = extractor.extract_feature_single_band(current_time=40, band=8)
    assert vec8_step3[8] == 1.0  # false alarm rate on B8 = 1.0
    assert vec8_step3[9] == 1.0  # B8 is last scanned


def test_feature_extractor_reset():
    """Verify reset() completely restores initial state."""
    extractor = RFFeatureExtractor(num_bands=10)
    extractor.update(Observation(10, 2, 1, DetectionResult.HIT))
    assert extractor.total_decisions == 1

    extractor.reset()
    assert extractor.total_decisions == 0
    assert extractor.last_scanned_band is None
    vec = extractor.extract_feature_single_band(current_time=0, band=2)
    assert vec[5] == 0.0  # scan_fraction is 0.0
