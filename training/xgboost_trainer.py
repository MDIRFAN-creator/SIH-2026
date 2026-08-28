"""
Offline supervised dataset generator and trainer for XGBoostBandPredictor (SIH26055 - Phase 3).

Collects training samples via exploratory scanning policies and trains the XGBoost model.
Maintains strict separation: Ground truth is used ONLY to generate training labels offline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from environment.config import EnvironmentConfig
from environment.rf_environment import RFEnvironment
from environment.types import Action, DetectionResult, Observation
from features.rf_features import FEATURE_NAMES, RFFeatureExtractor
from models.xgboost_model import XGBoostBandPredictor


@dataclass
class DatasetSplit:
    """Supervised feature matrix and label vector container."""
    X: np.ndarray
    y: np.ndarray
    seeds: List[int]
    total_samples: int
    positive_samples: int
    negative_samples: int

    @property
    def positive_ratio(self) -> float:
        return (self.positive_samples / self.total_samples) if self.total_samples > 0 else 0.0


class XGBoostDatasetGenerator:
    """
    Offline training data generator that runs exploration policies against the RF environment
    to generate causally-valid feature vectors and corresponding binary ground-truth detection labels.
    """

    def __init__(self, config: EnvironmentConfig, allowed_dwells: Optional[List[int]] = None):
        self.config = config
        self.allowed_dwells = allowed_dwells or [1, 2, 3]

    def collect_episode_samples(
        self,
        seed: int,
        policy: str = "mixed_exploration",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect supervised (X, y) samples from a single exploratory episode.
        
        Args:
            seed: RNG seed for the RF environment and policy.
            policy: Exploration policy ('mixed_exploration', 'random', 'sweep').
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix (N, D) and binary labels (N,).
        """
        rng = np.random.default_rng(seed)
        env = RFEnvironment(self.config)
        obs = env.reset(seed=seed)

        extractor = RFFeatureExtractor(num_bands=self.config.num_bands)
        samples_X: List[np.ndarray] = []
        samples_y: List[int] = []

        step_idx = 0
        while not env.is_terminated:
            # 1. Update extractor with previous observation
            extractor.update(obs)

            # 2. Choose exploratory action
            if policy == "random":
                band = int(rng.integers(0, self.config.num_bands))
                dwell = int(rng.choice(self.allowed_dwells))
            elif policy == "sweep":
                band = step_idx % self.config.num_bands
                dwell = int(rng.choice(self.allowed_dwells))
            else:  # mixed_exploration: 50% random, 50% sweep
                if rng.random() < 0.5:
                    band = int(rng.integers(0, self.config.num_bands))
                else:
                    band = step_idx % self.config.num_bands
                dwell = int(rng.choice(self.allowed_dwells))

            action = Action(frequency_band=band, dwell_time=dwell)

            # 3. Extract causal feature vector for the chosen candidate band BEFORE step executes
            feat_vec = extractor.extract_feature_single_band(
                current_time=obs.current_time,
                band=band,
            )

            # 4. Execute action in environment
            obs, reward, terminated, info = env.step(action)

            # 5. Determine supervised label from step outcome:
            # Target = 1 if the scan yielded a True Positive detection on an observable signal; 0 otherwise.
            dwell_summary = info.get("dwell_summary")
            if dwell_summary is not None:
                has_tp = any(slot.is_true_positive for slot in dwell_summary.slot_outcomes)
                label = 1 if has_tp else 0
            else:
                label = 1 if obs.result == DetectionResult.HIT else 0

            samples_X.append(feat_vec)
            samples_y.append(label)
            step_idx += 1

        return np.array(samples_X, dtype=np.float32), np.array(samples_y, dtype=int)

    def generate_dataset(
        self,
        seeds: List[int],
        policy: str = "mixed_exploration",
    ) -> DatasetSplit:
        """
        Generate aggregated dataset across multiple independent scenario seeds.
        """
        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []

        for s in seeds:
            X_ep, y_ep = self.collect_episode_samples(seed=s, policy=policy)
            all_X.append(X_ep)
            all_y.append(y_ep)

        X_concat = np.vstack(all_X)
        y_concat = np.concatenate(all_y)

        pos_count = int(np.sum(y_concat == 1))
        neg_count = int(np.sum(y_concat == 0))

        return DatasetSplit(
            X=X_concat,
            y=y_concat,
            seeds=seeds,
            total_samples=len(y_concat),
            positive_samples=pos_count,
            negative_samples=neg_count,
        )


def train_xgboost_pipeline(
    config: EnvironmentConfig,
    train_seeds: List[int],
    val_seeds: List[int],
    save_path: Optional[Union[str, Path]] = None,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Tuple[XGBoostBandPredictor, Dict[str, Any]]:
    """
    Complete training pipeline: generates dataset, fits XGBoostBandPredictor,
    evaluates classification metrics, and optionally saves model artifact.
    
    Args:
        config: Scenario EnvironmentConfig.
        train_seeds: List of seeds for training set.
        val_seeds: List of seeds for validation set.
        save_path: Optional file path to save trained model artifact.
        
    Returns:
        Tuple[XGBoostBandPredictor, Dict[str, Any]]: Trained model and evaluation report.
    """
    generator = XGBoostDatasetGenerator(config)

    # 1. Collect training and validation splits
    train_split = generator.generate_dataset(seeds=train_seeds)
    val_split = generator.generate_dataset(seeds=val_seeds)

    # 2. Instantiate and train model
    model = XGBoostBandPredictor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        feature_names=FEATURE_NAMES,
    )

    val_metrics = model.fit(
        X_train=train_split.X,
        y_train=train_split.y,
        X_val=val_split.X,
        y_val=val_split.y,
    )

    if save_path is not None:
        model.save(save_path)

    report = {
        "train_samples": train_split.total_samples,
        "train_positive_ratio": train_split.positive_ratio,
        "val_samples": val_split.total_samples,
        "val_positive_ratio": val_split.positive_ratio,
        "validation_metrics": val_metrics,
        "feature_importances": model.get_feature_importances(),
    }

    return model, report
