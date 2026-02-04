"""
Momentum Continuation Model using LightGBM.

A multi-model system for predicting momentum continuation vs reversal:
- Direction Model: Classifies continuation/reversal/neutral
- Magnitude Model: Predicts expected return percentage
- Duration Model: Predicts expected move duration in bars

This is designed for intraday momentum trading strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import logging
import pickle
import uuid

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

logger = logging.getLogger(__name__)

# Database manager for model registry (lazy loaded)
_db_manager = None


def _get_db_manager():
    """Get or create database manager for model registry."""
    global _db_manager
    if _db_manager is None:
        try:
            from data.storage import DatabaseManager
            _db_manager = DatabaseManager()
        except Exception as e:
            logger.warning(f"Could not initialize database manager: {e}")
    return _db_manager


@dataclass
class MomentumPredictionResult:
    """Result from momentum continuation prediction."""
    symbol: str
    timestamp: datetime
    direction: str  # "continuation", "reversal", "neutral"
    direction_probability: float
    expected_magnitude: float  # % return
    expected_duration_bars: int
    momentum_score: float  # 0-100 composite score
    suggested_stop_loss_pct: float
    suggested_take_profit_pct: float
    raw_probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class MomentumModelMetrics:
    """Training metrics for momentum model."""
    direction_accuracy: float
    direction_precision: float
    direction_recall: float
    direction_f1: float
    magnitude_mae: float
    magnitude_directional_accuracy: float  # Did we predict sign correctly
    duration_mae: float
    feature_importances: dict[str, float] = field(default_factory=dict)
    cv_scores: dict[str, list[float]] = field(default_factory=dict)
    training_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction_accuracy": self.direction_accuracy,
            "direction_precision": self.direction_precision,
            "direction_recall": self.direction_recall,
            "direction_f1": self.direction_f1,
            "magnitude_mae": self.magnitude_mae,
            "magnitude_directional_accuracy": self.magnitude_directional_accuracy,
            "duration_mae": self.duration_mae,
            "cv_scores": self.cv_scores,
            "training_date": self.training_date.isoformat(),
        }

    def __str__(self) -> str:
        return (
            f"Direction: Acc={self.direction_accuracy:.4f}, F1={self.direction_f1:.4f} | "
            f"Magnitude MAE={self.magnitude_mae:.4f}, DirAcc={self.magnitude_directional_accuracy:.4f} | "
            f"Duration MAE={self.duration_mae:.2f} bars"
        )


# Direction labels mapping
DIRECTION_LABELS = {
    0: "continuation",
    1: "reversal",
    2: "neutral",
}
DIRECTION_TO_INT = {v: k for k, v in DIRECTION_LABELS.items()}


class MomentumContinuationModel:
    """
    Multi-model system for momentum continuation prediction.

    Uses three LightGBM models:
    - Direction classifier: Predicts continuation/reversal/neutral
    - Magnitude regressor: Predicts expected % return
    - Duration regressor: Predicts expected number of bars

    The models work together to generate a comprehensive momentum score
    that can be used for trading decisions.
    """

    DIRECTION_PARAMS = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    REGRESSION_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self,
        direction_params: dict[str, Any] | None = None,
        magnitude_params: dict[str, Any] | None = None,
        duration_params: dict[str, Any] | None = None,
        min_momentum_score: float = 60.0,
    ):
        """
        Initialize the Momentum Continuation Model.

        Args:
            direction_params: Override params for direction classifier
            magnitude_params: Override params for magnitude regressor
            duration_params: Override params for duration regressor
            min_momentum_score: Minimum score (0-100) for actionable signals
        """
        self.direction_params = {**self.DIRECTION_PARAMS, **(direction_params or {})}
        self.magnitude_params = {**self.REGRESSION_PARAMS, **(magnitude_params or {})}
        self.duration_params = {**self.REGRESSION_PARAMS, **(duration_params or {})}
        self.min_momentum_score = min_momentum_score

        self.direction_model: LGBMClassifier | None = None
        self.magnitude_model: LGBMRegressor | None = None
        self.duration_model: LGBMRegressor | None = None

        self.feature_names: list[str] = []
        self.metrics: MomentumModelMetrics | None = None
        self.is_trained = False

        # Model registry tracking
        self.model_id: int | None = None
        self.model_version: str | None = None
        self.training_data_start: datetime | None = None
        self.training_data_end: datetime | None = None
        self.training_samples: int | None = None

    def train(
        self,
        X: pd.DataFrame,
        y_direction: pd.Series,
        y_magnitude: pd.Series,
        y_duration: pd.Series,
        n_splits: int = 5,
    ) -> MomentumModelMetrics:
        """
        Train all three models with TimeSeriesSplit CV.

        Args:
            X: Feature DataFrame
            y_direction: Direction labels (0=continuation, 1=reversal, 2=neutral)
            y_magnitude: Expected magnitude (% return)
            y_duration: Expected duration (bars)
            n_splits: Number of CV splits

        Returns:
            MomentumModelMetrics with training results
        """
        self.feature_names = list(X.columns)
        logger.info(f"Training momentum model with {len(X)} samples, {len(self.feature_names)} features")

        # Time-based train/test split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_dir_train, y_dir_test = y_direction.iloc[:split_idx], y_direction.iloc[split_idx:]
        y_mag_train, y_mag_test = y_magnitude.iloc[:split_idx], y_magnitude.iloc[split_idx:]
        y_dur_train, y_dur_test = y_duration.iloc[:split_idx], y_duration.iloc[split_idx:]

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores: dict[str, list[float]] = {
            "direction_f1": [],
            "magnitude_mae": [],
            "duration_mae": [],
        }

        logger.info("Performing cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]

            # Direction model CV
            y_cv_train = y_dir_train.iloc[train_idx]
            y_cv_val = y_dir_train.iloc[val_idx]
            dir_model = LGBMClassifier(**self.direction_params)
            dir_model.fit(X_cv_train, y_cv_train)
            dir_pred = dir_model.predict(X_cv_val)
            dir_f1 = f1_score(y_cv_val, dir_pred, average="weighted", zero_division=0)
            cv_scores["direction_f1"].append(dir_f1)

            # Magnitude model CV
            y_cv_train = y_mag_train.iloc[train_idx]
            y_cv_val = y_mag_train.iloc[val_idx]
            mag_model = LGBMRegressor(**self.magnitude_params)
            mag_model.fit(X_cv_train, y_cv_train)
            mag_pred = mag_model.predict(X_cv_val)
            mag_mae = mean_absolute_error(y_cv_val, mag_pred)
            cv_scores["magnitude_mae"].append(mag_mae)

            # Duration model CV
            y_cv_train = y_dur_train.iloc[train_idx]
            y_cv_val = y_dur_train.iloc[val_idx]
            dur_model = LGBMRegressor(**self.duration_params)
            dur_model.fit(X_cv_train, y_cv_train)
            dur_pred = dur_model.predict(X_cv_val)
            dur_mae = mean_absolute_error(y_cv_val, dur_pred)
            cv_scores["duration_mae"].append(dur_mae)

            logger.info(
                f"Fold {fold + 1}: Dir F1={dir_f1:.4f}, Mag MAE={mag_mae:.4f}, Dur MAE={dur_mae:.2f}"
            )

        # Log CV summary
        logger.info(
            f"CV Summary: Dir F1={np.mean(cv_scores['direction_f1']):.4f} "
            f"(+/- {np.std(cv_scores['direction_f1']):.4f})"
        )

        # Train final models on full training set
        logger.info("Training final models...")

        self.direction_model = LGBMClassifier(**self.direction_params)
        self.direction_model.fit(X_train, y_dir_train)

        self.magnitude_model = LGBMRegressor(**self.magnitude_params)
        self.magnitude_model.fit(X_train, y_mag_train)

        self.duration_model = LGBMRegressor(**self.duration_params)
        self.duration_model.fit(X_train, y_dur_train)

        # Evaluate on test set
        logger.info("Evaluating on test set...")

        # Direction metrics
        dir_pred = self.direction_model.predict(X_test)
        dir_prob = self.direction_model.predict_proba(X_test)

        direction_accuracy = accuracy_score(y_dir_test, dir_pred)
        direction_precision = precision_score(y_dir_test, dir_pred, average="weighted", zero_division=0)
        direction_recall = recall_score(y_dir_test, dir_pred, average="weighted", zero_division=0)
        direction_f1 = f1_score(y_dir_test, dir_pred, average="weighted", zero_division=0)

        # Magnitude metrics
        mag_pred = self.magnitude_model.predict(X_test)
        magnitude_mae = mean_absolute_error(y_mag_test, mag_pred)
        # Directional accuracy: did we predict the sign correctly?
        magnitude_directional_accuracy = np.mean(np.sign(mag_pred) == np.sign(y_mag_test))

        # Duration metrics
        dur_pred = self.duration_model.predict(X_test)
        duration_mae = mean_absolute_error(y_dur_test, dur_pred)

        # Combine feature importances from all models
        feature_importances = self._combine_feature_importances()

        self.metrics = MomentumModelMetrics(
            direction_accuracy=direction_accuracy,
            direction_precision=direction_precision,
            direction_recall=direction_recall,
            direction_f1=direction_f1,
            magnitude_mae=magnitude_mae,
            magnitude_directional_accuracy=magnitude_directional_accuracy,
            duration_mae=duration_mae,
            feature_importances=feature_importances,
            cv_scores=cv_scores,
        )

        self.is_trained = True
        self.training_samples = len(X)

        # Track training data range from index if available
        try:
            if hasattr(X.index, 'min') and hasattr(X.index, 'max'):
                idx_min = X.index.min()
                idx_max = X.index.max()

                # Handle various datetime types
                if isinstance(idx_min, datetime):
                    self.training_data_start = idx_min
                    self.training_data_end = idx_max
                elif hasattr(idx_min, 'to_pydatetime'):
                    # pandas Timestamp
                    self.training_data_start = idx_min.to_pydatetime()
                    self.training_data_end = idx_max.to_pydatetime()
                elif isinstance(idx_min, (int, float)):
                    # Numeric index - use current time as fallback
                    self.training_data_start = datetime.now()
                    self.training_data_end = datetime.now()
        except Exception as e:
            logger.debug(f"Could not extract training date range: {e}")
            # Set to current time as fallback
            self.training_data_start = datetime.now()
            self.training_data_end = datetime.now()

        logger.info(f"Model trained: {self.metrics}")

        # Note: Feature importances are saved after model registration in save()
        # because model_id is not available until then

        return self.metrics

    def _save_feature_importances_to_db(self) -> None:
        """Save feature importances to database for drift tracking."""
        if not self.model_id or not self.metrics or not self.metrics.feature_importances:
            return

        db = _get_db_manager()
        if not db:
            return

        try:
            # Save combined importances
            db.save_feature_importance(
                model_id=self.model_id,
                importances=self.metrics.feature_importances,
                importance_type="gain",
                model_component="combined",
            )

            # Save per-component importances
            if self.direction_model and self.feature_names:
                dir_imp = dict(zip(self.feature_names, self.direction_model.feature_importances_))
                db.save_feature_importance(
                    model_id=self.model_id,
                    importances=dir_imp,
                    importance_type="gain",
                    model_component="direction",
                )

            if self.magnitude_model and self.feature_names:
                mag_imp = dict(zip(self.feature_names, self.magnitude_model.feature_importances_))
                db.save_feature_importance(
                    model_id=self.model_id,
                    importances=mag_imp,
                    importance_type="gain",
                    model_component="magnitude",
                )

            if self.duration_model and self.feature_names:
                dur_imp = dict(zip(self.feature_names, self.duration_model.feature_importances_))
                db.save_feature_importance(
                    model_id=self.model_id,
                    importances=dur_imp,
                    importance_type="gain",
                    model_component="duration",
                )

            logger.info(f"Feature importances saved to database for model {self.model_id}")
        except Exception as e:
            logger.warning(f"Failed to save feature importances to database: {e}")

    def predict(
        self,
        X: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> list[MomentumPredictionResult]:
        """
        Generate predictions from all three models.

        Args:
            X: Feature DataFrame (must include 'atr' column for stop/take-profit calc)
            symbol: Stock ticker symbol

        Returns:
            List of MomentumPredictionResult objects
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Ensure feature alignment
        X_aligned = self._align_features(X)

        # Get predictions from all models
        dir_probs = self.direction_model.predict_proba(X_aligned)
        dir_pred = self.direction_model.predict(X_aligned)
        mag_pred = self.magnitude_model.predict(X_aligned)
        dur_pred = self.duration_model.predict(X_aligned)

        # Get ATR for stop/take-profit calculation (default to magnitude if not present)
        if "atr" in X.columns:
            atr_values = X["atr"].values
        elif "atr_14" in X.columns:
            atr_values = X["atr_14"].values
        else:
            # Use predicted magnitude as proxy for ATR
            atr_values = np.abs(mag_pred)

        results = []
        for i in range(len(X_aligned)):
            direction = DIRECTION_LABELS[int(dir_pred[i])]
            direction_prob = float(dir_probs[i, int(dir_pred[i])])
            magnitude = float(mag_pred[i])
            duration = max(1, int(round(dur_pred[i])))
            atr = float(atr_values[i])

            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(
                direction_prob=direction_prob,
                magnitude=magnitude,
                duration=duration,
                atr=atr,
            )

            # Calculate suggested stop loss and take profit
            stop_loss, take_profit = self._calculate_stop_take_profit(
                direction=direction,
                magnitude=magnitude,
                atr=atr,
            )

            # Get timestamp
            if hasattr(X_aligned.index, "__getitem__"):
                idx = X_aligned.index[i]
                timestamp = idx if isinstance(idx, datetime) else datetime.now()
            else:
                timestamp = datetime.now()

            # Raw probabilities
            raw_probs = {
                DIRECTION_LABELS[j]: float(dir_probs[i, j])
                for j in range(3)
            }

            results.append(MomentumPredictionResult(
                symbol=symbol,
                timestamp=timestamp,
                direction=direction,
                direction_probability=direction_prob,
                expected_magnitude=magnitude,
                expected_duration_bars=duration,
                momentum_score=momentum_score,
                suggested_stop_loss_pct=stop_loss,
                suggested_take_profit_pct=take_profit,
                raw_probabilities=raw_probs,
            ))

        return results

    def predict_single(
        self,
        features: pd.DataFrame | dict,
        symbol: str = "UNKNOWN",
    ) -> MomentumPredictionResult:
        """
        Generate a single prediction.

        Args:
            features: Single row DataFrame or dict of features
            symbol: Stock ticker symbol

        Returns:
            MomentumPredictionResult
        """
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        return self.predict(features, symbol)[0]

    def _calculate_momentum_score(
        self,
        direction_prob: float,
        magnitude: float,
        duration: int,
        atr: float,
    ) -> float:
        """
        Calculate composite momentum score (0-100).

        Weighting:
        - Direction confidence: 40%
        - Magnitude vs ATR: 30%
        - Duration optimality: 15%
        - Expected R:R: 15%

        Args:
            direction_prob: Probability of predicted direction
            magnitude: Expected % return
            duration: Expected bars
            atr: Average True Range (or proxy)

        Returns:
            Score from 0 to 100
        """
        # Direction confidence (40% weight)
        # Scale probability 0.33-1.0 to 0-100
        direction_score = max(0, min(100, (direction_prob - 0.33) / 0.67 * 100))

        # Magnitude vs ATR (30% weight)
        # Higher magnitude relative to ATR is better
        if atr > 0:
            mag_atr_ratio = abs(magnitude) / atr
            # Ratio of 1.5+ is excellent, 0.5 is poor
            magnitude_score = max(0, min(100, (mag_atr_ratio - 0.5) / 1.5 * 100))
        else:
            magnitude_score = 50.0  # Neutral if no ATR

        # Duration optimality (15% weight)
        # Sweet spot is 3-8 bars for intraday; too short = noise, too long = risk
        if duration < 2:
            duration_score = 30.0  # Too short, likely noise
        elif duration <= 3:
            duration_score = 60.0
        elif duration <= 8:
            duration_score = 100.0  # Optimal range
        elif duration <= 15:
            duration_score = 70.0
        else:
            duration_score = 40.0  # Too long, higher risk

        # Expected R:R (15% weight)
        # Magnitude should exceed expected reversal threshold
        # Assume reversal threshold is ~0.5 * ATR
        if atr > 0:
            expected_risk = 0.5 * atr
            expected_reward = abs(magnitude)
            rr_ratio = expected_reward / expected_risk if expected_risk > 0 else 1.0
            rr_score = max(0, min(100, (rr_ratio - 0.5) / 2.0 * 100))
        else:
            rr_score = 50.0

        # Weighted combination
        score = (
            direction_score * 0.40 +
            magnitude_score * 0.30 +
            duration_score * 0.15 +
            rr_score * 0.15
        )

        return round(max(0, min(100, score)), 2)

    def _calculate_stop_take_profit(
        self,
        direction: str,
        magnitude: float,
        atr: float,
    ) -> tuple[float, float]:
        """
        Calculate suggested stop loss and take profit percentages.

        Args:
            direction: "continuation", "reversal", or "neutral"
            magnitude: Expected % return
            atr: Average True Range

        Returns:
            Tuple of (stop_loss_pct, take_profit_pct)
        """
        # Stop loss based on ATR and reversal threshold
        # Use 1.5x ATR as stop loss (can be adjusted)
        if atr > 0:
            stop_loss = atr * 1.5
        else:
            stop_loss = abs(magnitude) * 0.5  # Fallback

        # Take profit based on expected magnitude
        if direction == "continuation":
            take_profit = abs(magnitude) * 1.0  # Full expected move
        elif direction == "reversal":
            take_profit = abs(magnitude) * 0.7  # More conservative on reversals
        else:  # neutral
            take_profit = abs(magnitude) * 0.5  # Very conservative

        # Ensure minimum stop loss to avoid getting stopped out on noise
        stop_loss = max(stop_loss, 0.1)  # At least 0.1%

        # Ensure reasonable take profit
        take_profit = max(take_profit, stop_loss * 1.5)  # At least 1.5:1 R:R

        return round(stop_loss, 4), round(take_profit, 4)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure features are aligned with training features."""
        if set(self.feature_names).issubset(X.columns):
            return X[self.feature_names]

        # Check for missing features
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            logger.warning(f"Missing features: {missing}. Using available features.")
            available = [f for f in self.feature_names if f in X.columns]
            return X[available]

        return X

    def _combine_feature_importances(self) -> dict[str, float]:
        """
        Combine feature importances from all three models.

        Uses weighted average: direction 50%, magnitude 30%, duration 20%
        """
        if not all([self.direction_model, self.magnitude_model, self.duration_model]):
            return {}

        dir_imp = dict(zip(self.feature_names, self.direction_model.feature_importances_))
        mag_imp = dict(zip(self.feature_names, self.magnitude_model.feature_importances_))
        dur_imp = dict(zip(self.feature_names, self.duration_model.feature_importances_))

        # Normalize each to sum to 1
        dir_total = sum(dir_imp.values()) or 1
        mag_total = sum(mag_imp.values()) or 1
        dur_total = sum(dur_imp.values()) or 1

        combined = {}
        for feature in self.feature_names:
            combined[feature] = (
                (dir_imp.get(feature, 0) / dir_total) * 0.50 +
                (mag_imp.get(feature, 0) / mag_total) * 0.30 +
                (dur_imp.get(feature, 0) / dur_total) * 0.20
            )

        # Sort by importance
        return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))

    def save(self, filepath: str | Path, register: bool = True, activate: bool = False) -> int | None:
        """
        Save all three models and configuration.

        Args:
            filepath: Path to save the model
            register: Register model in database registry
            activate: Activate model after registration (deactivates other versions)

        Returns:
            Model registry ID if registered, None otherwise
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Generate version if not set
        if not self.model_version:
            self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

        model_data = {
            "direction_model": self.direction_model,
            "magnitude_model": self.magnitude_model,
            "duration_model": self.duration_model,
            "direction_params": self.direction_params,
            "magnitude_params": self.magnitude_params,
            "duration_params": self.duration_params,
            "min_momentum_score": self.min_momentum_score,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
            "model_version": self.model_version,
            "model_id": self.model_id,
            "training_data_start": self.training_data_start,
            "training_data_end": self.training_data_end,
            "training_samples": self.training_samples,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Momentum model saved to {filepath}")

        # Register in database
        if register:
            model_id = self._register_in_db(str(filepath), activate)
            if model_id:
                self.model_id = model_id
                # Update pickle with model_id
                model_data["model_id"] = model_id
                with open(filepath, "wb") as f:
                    pickle.dump(model_data, f)

                # Now save feature importances (model_id is available)
                self._save_feature_importances_to_db()

            return model_id

        return None

    def _register_in_db(self, filepath: str, activate: bool = False) -> int | None:
        """
        Register the model in the database registry.

        Args:
            filepath: Path where model is saved
            activate: Whether to activate this model

        Returns:
            Model registry ID or None
        """
        db = _get_db_manager()
        if not db:
            logger.warning("Database manager not available, skipping registration")
            return None

        try:
            # Prepare hyperparameters
            hyperparameters = {
                "direction": self.direction_params,
                "magnitude": self.magnitude_params,
                "duration": self.duration_params,
                "min_momentum_score": self.min_momentum_score,
            }

            # Prepare metrics
            metrics = self.metrics.to_dict() if self.metrics else {}

            # Get parent model ID if this is a retrain
            parent_id = None
            active_model = db.get_active_model("momentum_continuation")
            if active_model:
                parent_id = active_model.get("id")

            # Register
            model_id = db.register_model(
                model_name="momentum_continuation",
                model_version=self.model_version,
                model_type="lightgbm_multi",
                hyperparameters=hyperparameters,
                feature_names=self.feature_names,
                metrics=metrics,
                model_path=filepath,
                parent_model_id=parent_id,
                training_data_start=self.training_data_start,
                training_data_end=self.training_data_end,
                training_samples=self.training_samples,
            )

            # Update status to validated
            db.update_model_status(model_id, "validated", metrics)

            # Activate if requested
            if activate:
                db.activate_model(model_id)
                logger.info(f"Model {model_id} activated")

            logger.info(f"Model registered in database with ID {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to register model in database: {e}")
            return None

    @classmethod
    def load(cls, filepath: str | Path) -> "MomentumContinuationModel":
        """
        Load a saved model.

        Args:
            filepath: Path to the saved model

        Returns:
            MomentumContinuationModel instance
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(
            direction_params=model_data["direction_params"],
            magnitude_params=model_data["magnitude_params"],
            duration_params=model_data["duration_params"],
            min_momentum_score=model_data["min_momentum_score"],
        )
        instance.direction_model = model_data["direction_model"]
        instance.magnitude_model = model_data["magnitude_model"]
        instance.duration_model = model_data["duration_model"]
        instance.feature_names = model_data["feature_names"]
        instance.metrics = model_data["metrics"]
        instance.is_trained = model_data["is_trained"]

        # Load model registry information
        instance.model_id = model_data.get("model_id")
        instance.model_version = model_data.get("model_version")
        instance.training_data_start = model_data.get("training_data_start")
        instance.training_data_end = model_data.get("training_data_end")
        instance.training_samples = model_data.get("training_samples")

        return instance

    @classmethod
    def load_active(cls, model_name: str = "momentum_continuation") -> "MomentumContinuationModel | None":
        """
        Load the currently active model from the registry.

        Args:
            model_name: Name of the model in registry

        Returns:
            MomentumContinuationModel instance or None if no active model
        """
        db = _get_db_manager()
        if not db:
            logger.warning("Database manager not available")
            return None

        try:
            active = db.get_active_model(model_name)
            if not active:
                logger.warning(f"No active model found for {model_name}")
                return None

            model_path = active.get("model_path")
            if not model_path:
                logger.warning(f"Active model {active.get('id')} has no model_path")
                return None

            return cls.load(model_path)

        except Exception as e:
            logger.error(f"Failed to load active model: {e}")
            return None

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get top N most important features across all models.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        if self.metrics is None or not self.metrics.feature_importances:
            return []
        return list(self.metrics.feature_importances.items())[:n]
