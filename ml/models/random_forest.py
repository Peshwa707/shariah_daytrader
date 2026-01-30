"""
Random Forest Model for Trading Signal Generation.

This module implements a Random Forest classifier for predicting
price direction (up/down) based on technical and price action features.

Random Forest is chosen as the baseline model because:
- Handles non-linear relationships well
- Built-in feature importance
- Robust to overfitting with proper tuning
- No feature scaling required
- Works well with time series data
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    training_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "cv_scores": self.cv_scores,
            "cv_mean": np.mean(self.cv_scores) if self.cv_scores else None,
            "cv_std": np.std(self.cv_scores) if self.cv_scores else None,
            "training_date": self.training_date.isoformat(),
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Accuracy: {self.accuracy:.4f}, "
            f"Precision: {self.precision:.4f}, "
            f"Recall: {self.recall:.4f}, "
            f"F1: {self.f1:.4f}"
        )


@dataclass
class PredictionResult:
    """Container for model prediction."""

    symbol: str
    timestamp: datetime
    signal: str  # "buy", "sell", "hold"
    probability: float
    confidence: str  # "high", "medium", "low"
    features_used: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "signal": self.signal,
            "probability": self.probability,
            "confidence": self.confidence,
            "features_used": self.features_used,
        }


class RandomForestSignalModel:
    """
    Random Forest model for trading signal generation.

    This model predicts whether the price will go up or down over
    a specified horizon, which is then converted to trading signals.

    Features:
    - Time-series aware cross-validation
    - Walk-forward validation support
    - Hyperparameter tuning with Optuna
    - Feature importance analysis
    - Probability calibration for confidence levels
    """

    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        probability_threshold: float = 0.55,
    ):
        """
        Initialize the Random Forest model.

        Args:
            params: Model hyperparameters (uses defaults if not provided)
            probability_threshold: Minimum probability for generating a signal
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.probability_threshold = probability_threshold

        self.model: RandomForestClassifier | None = None
        self.feature_names: list[str] = []
        self.metrics: ModelMetrics | None = None
        self.is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        test_size: float = 0.2,
    ) -> ModelMetrics:
        """
        Train the model with time-series cross-validation.

        Args:
            X: Feature DataFrame
            y: Target Series (binary: 1 for up, 0 for down)
            n_splits: Number of CV splits
            test_size: Fraction of data to use for final test

        Returns:
            ModelMetrics with training results
        """
        self.feature_names = list(X.columns)

        # Split into train and test (respecting time order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Time series cross-validation on training data
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(**self.params)
            model.fit(X_cv_train, y_cv_train)

            val_pred = model.predict(X_cv_val)
            cv_scores.append(accuracy_score(y_cv_val, val_pred))

        logger.info(f"CV Scores: {cv_scores}")
        logger.info(f"CV Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Train final model on all training data
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        self.metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            feature_importances=self._get_feature_importances(),
            cv_scores=cv_scores,
        )

        self.is_trained = True
        logger.info(f"Model trained: {self.metrics}")

        return self.metrics

    def predict(
        self,
        X: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> list[PredictionResult]:
        """
        Generate predictions for input features.

        Args:
            X: Feature DataFrame
            symbol: Stock ticker symbol

        Returns:
            List of PredictionResult objects
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Ensure feature alignment
        X_aligned = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X

        # Get predictions and probabilities
        probabilities = self.model.predict_proba(X_aligned)

        results = []
        for i in range(len(X_aligned)):
            prob_up = probabilities[i, 1]
            prob_down = probabilities[i, 0]

            # Determine signal
            if prob_up >= self.probability_threshold:
                signal = "buy"
                probability = prob_up
            elif prob_down >= self.probability_threshold:
                signal = "sell"
                probability = prob_down
            else:
                signal = "hold"
                probability = max(prob_up, prob_down)

            # Determine confidence
            if probability >= 0.7:
                confidence = "high"
            elif probability >= 0.6:
                confidence = "medium"
            else:
                confidence = "low"

            # Get timestamp
            if hasattr(X_aligned, "index"):
                timestamp = X_aligned.index[i] if isinstance(X_aligned.index[i], datetime) else datetime.now()
            else:
                timestamp = datetime.now()

            results.append(PredictionResult(
                symbol=symbol,
                timestamp=timestamp,
                signal=signal,
                probability=probability,
                confidence=confidence,
                features_used=len(self.feature_names),
            ))

        return results

    def predict_single(
        self,
        features: pd.DataFrame | dict,
        symbol: str = "UNKNOWN",
    ) -> PredictionResult:
        """
        Generate a single prediction.

        Args:
            features: Single row of features
            symbol: Stock ticker symbol

        Returns:
            PredictionResult
        """
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        results = self.predict(features, symbol)
        return results[0]

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

    def _get_feature_importances(self) -> dict[str, float]:
        """Get feature importances as a dictionary."""
        if self.model is None:
            return {}

        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        ))

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features."""
        if self.metrics is None:
            return []

        return list(self.metrics.feature_importances.items())[:n]

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        n_splits: int = 3,
    ) -> dict[str, Any]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_trials: Number of Optuna trials
            n_splits: Number of CV splits

        Returns:
            Best hyperparameters found
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install with: pip install optuna")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 5, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }

            # Time series CV
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                scores.append(f1_score(y_val, y_pred, zero_division=0))

            return np.mean(scores)

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        # Update model params
        self.params.update(study.best_params)

        return study.best_params

    def save(self, filepath: str | Path) -> None:
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "probability_threshold": self.probability_threshold,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "RandomForestSignalModel":
        """
        Load a model from a file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded RandomForestSignalModel
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(
            params=model_data["params"],
            probability_threshold=model_data["probability_threshold"],
        )
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.metrics = model_data["metrics"]
        instance.is_trained = model_data["is_trained"]

        return instance

    def print_classification_report(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> str:
        """
        Print detailed classification report.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Classification report string
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")

        y_pred = self.model.predict(X_test[self.feature_names])

        report = classification_report(
            y_test, y_pred,
            target_names=["Down", "Up"],
        )
        print(report)
        return report
