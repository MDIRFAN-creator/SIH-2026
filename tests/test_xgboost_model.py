"""
Unit tests for XGBoostBandPredictor wrapper (SIH26055 - Phase 3).
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from models.xgboost_model import XGBoostBandPredictor


def test_xgboost_fit_and_predict():
    """Verify model fitting, probability predictions in [0, 1], and metric computation."""
    rng = np.random.default_rng(42)
    N = 200
    D = 12

    X_train = rng.normal(size=(N, D)).astype(np.float32)
    # Target correlated with first two features
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0.0).astype(int)

    X_val = rng.normal(size=(50, D)).astype(np.float32)
    y_val = ((X_val[:, 0] + X_val[:, 1]) > 0.0).astype(int)

    model = XGBoostBandPredictor(n_estimators=30, max_depth=3, random_state=42)
    metrics = model.fit(X_train, y_train, X_val, y_val)

    assert model.is_fitted
    assert "roc_auc" in metrics
    assert "accuracy" in metrics
    assert metrics["roc_auc"] > 0.60

    # Predictions
    probs = model.predict_proba(X_val)
    assert probs.shape == (50,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds = model.predict(X_val)
    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_xgboost_single_class_error():
    """Verify informative error if training data contains only 1 class."""
    X = np.ones((50, 12), dtype=np.float32)
    y = np.zeros(50, dtype=int)

    model = XGBoostBandPredictor()
    with pytest.raises(ValueError, match="must contain both classes"):
        model.fit(X, y)


def test_xgboost_save_and_load_invariance():
    """Verify save and load generates identical predictions."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(100, 12)).astype(np.float32)
    y = (X[:, 2] > 0.0).astype(int)

    model = XGBoostBandPredictor(n_estimators=20, max_depth=2, random_state=123)
    model.fit(X, y)

    orig_probs = model.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.json"
        model.save(model_path)

        loaded_model = XGBoostBandPredictor.load(model_path)
        assert loaded_model.is_fitted

        loaded_probs = loaded_model.predict_proba(X)
        np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5, atol=1e-5)


def test_xgboost_feature_importances():
    """Verify feature importances are returned for all feature names."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 12)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    model = XGBoostBandPredictor(n_estimators=20, max_depth=2, random_state=42)
    model.fit(X, y)

    importances = model.get_feature_importances()
    assert len(importances) == 12
    # Feature 0 should have the highest importance score
    assert importances["band_norm"] >= 0.0
