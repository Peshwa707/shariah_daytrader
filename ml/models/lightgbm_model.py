"""
LightGBM Model for Trading Signal Generation.

LightGBM is a gradient boosting framework that is:
- Faster than traditional gradient boosting
- Memory efficient
- Handles categorical features natively
- Often achieves better accuracy than Random Forest

This is the "advanced" model after proving the concept with Random Forest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class LGBMMetrics:
    """Container for LightGBM model metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None = None
    log_loss_value: float | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    best_iteration: int = 0
    training_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
            "log_loss": self.log_loss_value,
            "cv_mean": np.mean(self.cv_scores) if self.cv_scores else None,
            "best_iteration": self.best_iteration,
            "training_date": self.training_date.isoformat(),
        }

    def __str__(self) -> str:
        auc_str = f"{self.auc_roc:.4f}" if self.auc_roc is not None else "N/A"
        return (
            f"Accuracy: {self.accuracy:.4f}, "
            f"Precision: {self.precision:.4f}, "
            f"Recall: {self.recall:.4f}, "
            f"F1: {self.f1:.4f}, "
            f"AUC-ROC: {auc_str}"
        )


class LightGBMSignalModel:
    """
    LightGBM model for trading signal generation.

    Key advantages over Random Forest:
    - Handles larger datasets efficiently
    - Better with imbalanced classes
    - Built-in early stopping
    - Native handling of missing values

    This model includes:
    - Early stopping to prevent overfitting
    - Native time-series aware cross-validation
    - Optuna hyperparameter tuning
    - Multiple feature importance methods
    """

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        probability_threshold: float = 0.55,
        early_stopping_rounds: int = 50,
    ):
        """
        Initialize the LightGBM model.

        Args:
            params: Model hyperparameters
            probability_threshold: Min probability for signal generation
            early_stopping_rounds: Rounds for early stopping
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.probability_threshold = probability_threshold
        self.early_stopping_rounds = early_stopping_rounds

        self.model: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []
        self.metrics: LGBMMetrics | None = None
        self.is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        test_size: float = 0.2,
        use_early_stopping: bool = True,
    ) -> LGBMMetrics:
        """
        Train the model with time-series cross-validation.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits
            test_size: Fraction for final test set
            use_early_stopping: Whether to use early stopping

        Returns:
            LGBMMetrics with training results
        """
        self.feature_names = list(X.columns)

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Time series CV for model selection
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**self.params)

            if use_early_stopping:
                model.fit(
                    X_cv_train, y_cv_train,
                    eval_set=[(X_cv_val, y_cv_val)],
                    callbacks=[
                        lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
            else:
                model.fit(X_cv_train, y_cv_train)

            val_pred = model.predict(X_cv_val)
            fold_score = f1_score(y_cv_val, val_pred, zero_division=0)
            cv_scores.append(fold_score)
            logger.info(f"Fold {fold + 1}: F1 = {fold_score:.4f}")

        logger.info(f"CV Mean F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Train final model with early stopping on validation set
        # Split training data to create validation set for early stopping
        val_split_idx = int(len(X_train) * 0.85)
        X_train_final = X_train.iloc[:val_split_idx]
        X_val_final = X_train.iloc[val_split_idx:]
        y_train_final = y_train.iloc[:val_split_idx]
        y_val_final = y_train.iloc[val_split_idx:]

        self.model = lgb.LGBMClassifier(**self.params)

        if use_early_stopping:
            self.model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        else:
            self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        self.metrics = LGBMMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else None,
            log_loss_value=log_loss(y_test, y_prob),
            feature_importances=self._get_feature_importances(),
            cv_scores=cv_scores,
            best_iteration=self.model.best_iteration_ if hasattr(self.model, "best_iteration_") else 0,
        )

        self.is_trained = True
        logger.info(f"Model trained: {self.metrics}")

        return self.metrics

    def predict(
        self,
        X: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> list[dict[str, Any]]:
        """
        Generate predictions for input features.

        Args:
            X: Feature DataFrame
            symbol: Stock ticker symbol

        Returns:
            List of prediction dictionaries
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Ensure feature alignment
        X_aligned = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X

        # Get predictions
        probabilities = self.model.predict_proba(X_aligned)

        results = []
        for i in range(len(X_aligned)):
            prob_up = probabilities[i, 1]
            prob_down = probabilities[i, 0]

            if prob_up >= self.probability_threshold:
                signal = "buy"
                probability = prob_up
            elif prob_down >= self.probability_threshold:
                signal = "sell"
                probability = prob_down
            else:
                signal = "hold"
                probability = max(prob_up, prob_down)

            confidence = "high" if probability >= 0.7 else "medium" if probability >= 0.6 else "low"

            timestamp = X_aligned.index[i] if isinstance(X_aligned.index[i], datetime) else datetime.now()

            results.append({
                "symbol": symbol,
                "timestamp": timestamp,
                "signal": signal,
                "probability": probability,
                "confidence": confidence,
                "prob_up": prob_up,
                "prob_down": prob_down,
            })

        return results

    def predict_single(
        self,
        features: pd.DataFrame | dict,
        symbol: str = "UNKNOWN",
    ) -> dict[str, Any]:
        """Generate a single prediction."""
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        return self.predict(features, symbol)[0]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, 2) with [prob_down, prob_up]
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Ensure feature alignment
        if set(self.feature_names).issubset(X.columns):
            X_aligned = X[self.feature_names]
        else:
            X_aligned = X

        return self.model.predict_proba(X_aligned)

    def _get_feature_importances(
        self,
        importance_type: str = "gain",
    ) -> dict[str, float]:
        """
        Get feature importances.

        Args:
            importance_type: "gain", "split", or "shap"

        Returns:
            Dict mapping feature names to importance scores
        """
        if self.model is None:
            return {}

        if importance_type == "gain":
            importances = self.model.feature_importances_
        elif importance_type == "split":
            importances = self.model.booster_.feature_importance(importance_type="split")
        else:
            importances = self.model.feature_importances_

        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        ))

    def get_top_features(self, n: int = 15) -> list[tuple[str, float]]:
        """Get top N important features."""
        if self.metrics is None:
            return []
        return list(self.metrics.feature_importances.items())[:n]

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        n_splits: int = 3,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_trials: Number of trials
            n_splits: CV splits
            timeout: Maximum time in seconds

        Returns:
            Best parameters found
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for hyperparameter tuning")

        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(30, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )

                y_pred = model.predict(X_val)
                scores.append(f1_score(y_val, y_pred, zero_division=0))

            return np.mean(scores)

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        logger.info(f"Best trial F1: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        # Update model params
        self.params.update(study.best_params)

        return study.best_params

    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        train_period: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform walk-forward validation.

        This simulates real-world model retraining by:
        1. Training on historical data
        2. Predicting on future data
        3. Rolling forward and repeating

        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of forward steps
            train_period: Fixed training period (or expanding window if None)

        Returns:
            List of metrics for each forward step
        """
        results = []
        n_samples = len(X)
        test_size = n_samples // (n_splits + 1)

        for i in range(n_splits):
            if train_period:
                # Fixed window
                train_start = max(0, (i + 1) * test_size - train_period)
            else:
                # Expanding window
                train_start = 0

            train_end = (i + 1) * test_size
            test_end = min((i + 2) * test_size, n_samples)

            X_train = X.iloc[train_start:train_end]
            X_test = X.iloc[train_end:test_end]
            y_train = y.iloc[train_start:train_end]
            y_test = y.iloc[train_end:test_end]

            # Train model
            model = lgb.LGBMClassifier(**self.params)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            step_metrics = {
                "step": i + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }

            results.append(step_metrics)
            logger.info(f"Step {i + 1}: F1 = {step_metrics['f1']:.4f}")

        # Summary
        avg_f1 = np.mean([r["f1"] for r in results])
        logger.info(f"Walk-forward average F1: {avg_f1:.4f}")

        return results

    def save(self, filepath: str | Path) -> None:
        """Save the model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "probability_threshold": self.probability_threshold,
            "early_stopping_rounds": self.early_stopping_rounds,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "LightGBMSignalModel":
        """Load a model from file."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(
            params=model_data["params"],
            probability_threshold=model_data["probability_threshold"],
            early_stopping_rounds=model_data["early_stopping_rounds"],
        )
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.metrics = model_data["metrics"]
        instance.is_trained = model_data["is_trained"]

        return instance
