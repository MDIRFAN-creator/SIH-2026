"""
XGBoost model wrapper for SIH26055 (Phase 3).

Provides a clean, modular wrapper around XGBoost for predicting candidate frequency band utility.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

from features.rf_features import FEATURE_NAMES


class XGBoostBandPredictor:
    """
    Supervised XGBoost binary classifier predicting the probability that scanning a
    candidate frequency band will yield an observable emitter signal detection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.feature_names = feature_names or FEATURE_NAMES

        self.model: Optional[xgb.XGBClassifier] = None
        self.is_fitted: bool = False

    def _init_model(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="logloss",
            tree_method="hist",
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Fit the XGBoost classifier on training data and evaluate validation metrics.
        
        Args:
            X_train: Training feature matrix of shape (N, D).
            y_train: Binary labels of shape (N,).
            X_val: Optional validation feature matrix of shape (M, D).
            y_val: Optional validation binary labels of shape (M,).
            
        Returns:
            Dict[str, float]: Performance metrics (ROC-AUC, PR-AUC, Accuracy, F1, Precision, Recall).
        """
        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"Training dataset must contain both classes (0 and 1). Found unique labels: {np.unique(y_train)}"
            )

        self.model = self._init_model()

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set if eval_set else None,
            verbose=False,
        )
        self.is_fitted = True

        # Evaluate performance on validation set (or train if no val provided)
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train

        preds_proba = self.predict_proba(eval_X)
        preds_binary = (preds_proba >= 0.5).astype(int)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(eval_y, preds_binary)),
            "precision": float(precision_score(eval_y, preds_binary, zero_division=0)),
            "recall": float(recall_score(eval_y, preds_binary, zero_division=0)),
            "f1": float(f1_score(eval_y, preds_binary, zero_division=0)),
        }

        # ROC-AUC and PR-AUC require at least one sample of each class
        if len(np.unique(eval_y)) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(eval_y, preds_proba))
            metrics["pr_auc"] = float(average_precision_score(eval_y, preds_proba))
        else:
            metrics["roc_auc"] = 0.5
            metrics["pr_auc"] = 0.0

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of finding an observable signal (class 1).
        
        Args:
            X: Feature matrix of shape (N, D) or single vector (D,).
            
        Returns:
            np.ndarray: 1D float array of predicted probabilities in [0.0, 1.0].
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been fitted or loaded. Call fit() or load() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        probas = self.model.predict_proba(X)
        # Return probability of positive class (index 1)
        return probas[:, 1].astype(float)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.
        
        Args:
            X: Feature matrix.
            threshold: Probability threshold for positive class.
            
        Returns:
            np.ndarray: Binary array of 0 or 1.
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Return mapping of feature names to their importance scores (gain-based).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been fitted or loaded.")

        importances = self.model.feature_importances_
        return {
            name: float(importances[idx]) if idx < len(importances) else 0.0
            for idx, name in enumerate(self.feature_names)
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save fitted model artifact to JSON file."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        feature_names: Optional[List[str]] = None,
    ) -> "XGBoostBandPredictor":
        """Load fitted model artifact from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")

        instance = cls(feature_names=feature_names)
        instance.model = instance._init_model()
        instance.model.load_model(str(path))
        instance.is_fitted = True
        return instance
