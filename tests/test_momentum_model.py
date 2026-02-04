"""
Unit tests for MomentumContinuationModel.

Tests model training, prediction, save/load, and metrics.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.models.momentum_continuation_model import (
    MomentumContinuationModel,
    MomentumPredictionResult,
    MomentumModelMetrics,
)
from ml.labeling.momentum_labeler import MomentumLabeler, MomentumLabelerConfig
from ml.features.momentum_features import MomentumContinuationFeatures


def generate_sample_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)

    # Create realistic price movement with trend
    base_price = 100.0

    # Create a mix of trending and mean-reverting behavior
    trend = np.linspace(0, 0.3, n)
    noise = np.random.normal(0, 0.015, n)
    returns = trend / n + noise

    close = base_price * np.cumprod(1 + returns)

    # Create OHLCV
    df = pd.DataFrame({
        "open": close * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "close": close,
        "volume": np.random.randint(100000, 5000000, n),
    })

    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


def prepare_training_data(n: int = 300, seed: int = 42):
    """Prepare labeled training data for model testing."""
    df = generate_sample_ohlcv(n, seed)

    # Calculate features
    feature_calc = MomentumContinuationFeatures()
    features_df = feature_calc.calculate_all(df)

    # Create labels
    labeler = MomentumLabeler()
    X, y_dir, y_mag, y_dur = labeler.prepare_labeled_data(df, features_df)

    # Drop non-feature columns from X
    exclude = {"open", "high", "low", "close", "volume", "_atr", "timestamp", "symbol"}
    feature_cols = [c for c in X.columns if c not in exclude]
    X = X[feature_cols]

    return X, y_dir, y_mag, y_dur


class TestMomentumContinuationModel:
    """Test suite for MomentumContinuationModel class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        model = MomentumContinuationModel()

        assert model.min_momentum_score == 60.0
        assert model.direction_model is None
        assert model.magnitude_model is None
        assert model.duration_model is None
        assert not model.is_trained

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        model = MomentumContinuationModel(min_momentum_score=70.0)

        assert model.min_momentum_score == 70.0

    def test_train_returns_metrics(self):
        """Test that training returns MomentumModelMetrics."""
        X, y_dir, y_mag, y_dur = prepare_training_data(300)

        model = MomentumContinuationModel()
        metrics = model.train(X, y_dir, y_mag, y_dur, n_splits=3)

        assert isinstance(metrics, MomentumModelMetrics)
        assert model.is_trained

    def test_train_metrics_values(self):
        """Test that trained model has reasonable metric values."""
        X, y_dir, y_mag, y_dur = prepare_training_data(300)

        model = MomentumContinuationModel()
        metrics = model.train(X, y_dir, y_mag, y_dur, n_splits=3)

        # Direction accuracy should be better than random (33%)
        assert metrics.direction_accuracy > 0.30, f"Direction accuracy {metrics.direction_accuracy} too low"

        # F1 should be positive
        assert metrics.direction_f1 > 0

        # MAE should be reasonable
        assert metrics.magnitude_mae < 20  # Less than 20% MAE
        assert metrics.duration_mae < 10  # Less than 10 bars MAE

    def test_predict_returns_list(self):
        """Test that predict returns list of MomentumPredictionResult."""
        X, y_dir, y_mag, y_dur = prepare_training_data(300)

        model = MomentumContinuationModel()
        model.train(X, y_dir, y_mag, y_dur, n_splits=3)

        # Predict on subset
        predictions = model.predict(X.iloc[-10:], symbol="TEST")

        assert isinstance(predictions, list)
        assert len(predictions) == 10
        assert all(isinstance(p, MomentumPredictionResult) for p in predictions)

    def test_predict_result_fields(self):
        """Test that prediction results have all expected fields."""
        X, y_dir, y_mag, y_dur = prepare_training_data(300)

        model = MomentumContinuationModel()
        model.train(X, y_dir, y_mag, y_dur, n_splits=3)

        predictions = model.predict(X.iloc[-5:], symbol="TEST")
        pred = predictions[0]

        assert pred.symbol == "TEST"
        assert pred.direction in ["continuation", "reversal", "neutral"]
        assert 0 <= pred.direction_probability <= 1
        assert isinstance(pred.expected_magnitude, float)
        assert isinstance(pred.expected_duration_bars, int)
        assert 0 <= pred.momentum_score <= 100
        assert pred.suggested_stop_loss_pct > 0
        assert pred.suggested_take_profit_pct > 0

    def test_predict_single(self):
        """Test predict_single method."""
        X, y_dir, y_mag, y_dur = prepare_training_data(300)

        model = MomentumContinuationModel()
        model.train(X, y_dir, y_mag, y_dur, n_splits=3)

        # Single prediction with DataFrame
        pred = model.predict_single(X.iloc[[-1]], symbol="AAPL")
        assert isinstance(pred, MomentumPredictionResult)
        assert pred.symbol == "AAPL"

        # Single prediction with dict
        features_dict = X.iloc[-1].to_dict()
        pred = model.predict_single(features_dict, symbol="GOOGL")
        assert isinstance(pred, MomentumPredictionResult)
        assert pred.symbol == "GOOGL"

    def test_predict_before_training_raises(self):
        """Test that predicting before training raises error."""
        X, _, _, _ = prepare_training_data(300)

        model = MomentumContinuationModel()

        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X.iloc[-5:])

    def test_save_and_load(self):
        """Test model save and load functionality."""
        X, y_dir, y_mag, y_dur = prepare_training_data(300)

        model = MomentumContinuationModel()
        model.train(X, y_dir, y_mag, y_dur, n_splits=3)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "momentum_model.pkl"
            model.save(filepath)

            assert filepath.exists()

            # Load model
            loaded_model = MomentumContinuationModel.load(filepath)

            assert loaded_model.is_trained
            assert loaded_model.min_momentum_score == model.min_momentum_score
            assert loaded_model.feature_names == model.feature_names

            # Predictions should match
            orig_preds = model.predict(X.iloc[-3:])
            loaded_preds = loaded_model.predict(X.iloc[-3:])

            for orig, loaded in zip(orig_preds, loaded_preds):
                assert orig.direction == loaded.direction
                assert abs(orig.momentum_score - loaded.momentum_score) < 0.01

    def test_get_top_features(self):
        """Test get_top_features returns feature importances."""
        X, y_dir, y_mag, y_dur = prepare_training_data(300)

        model = MomentumContinuationModel()
        model.train(X, y_dir, y_mag, y_dur, n_splits=3)

        top_features = model.get_top_features(n=5)

        assert isinstance(top_features, list)
        assert len(top_features) <= 5
        assert all(isinstance(f, tuple) and len(f) == 2 for f in top_features)
        assert all(isinstance(f[0], str) and isinstance(f[1], float) for f in top_features)

    def test_momentum_score_calculation(self):
        """Test that momentum score is calculated correctly."""
        X, y_dir, y_mag, y_dur = prepare_training_data(300)

        model = MomentumContinuationModel()
        model.train(X, y_dir, y_mag, y_dur, n_splits=3)

        predictions = model.predict(X.iloc[-20:])

        # All scores should be between 0 and 100
        for pred in predictions:
            assert 0 <= pred.momentum_score <= 100, f"Score {pred.momentum_score} out of range"

    def test_stop_loss_take_profit_reasonable(self):
        """Test that stop/take profit suggestions are reasonable."""
        X, y_dir, y_mag, y_dur = prepare_training_data(300)

        model = MomentumContinuationModel()
        model.train(X, y_dir, y_mag, y_dur, n_splits=3)

        predictions = model.predict(X.iloc[-20:])

        for pred in predictions:
            # Stop loss should be positive
            assert pred.suggested_stop_loss_pct > 0

            # Take profit should be positive
            assert pred.suggested_take_profit_pct > 0

            # R:R should be at least 1.0 (take profit >= stop loss)
            rr = pred.suggested_take_profit_pct / pred.suggested_stop_loss_pct
            assert rr >= 1.0, f"R:R {rr} too low"


class TestMomentumPredictionResult:
    """Test suite for MomentumPredictionResult dataclass."""

    def test_dataclass_creation(self):
        """Test creating a MomentumPredictionResult."""
        from datetime import datetime

        result = MomentumPredictionResult(
            symbol="AAPL",
            timestamp=datetime.now(),
            direction="continuation",
            direction_probability=0.75,
            expected_magnitude=2.5,
            expected_duration_bars=5,
            momentum_score=72.5,
            suggested_stop_loss_pct=1.0,
            suggested_take_profit_pct=2.5,
        )

        assert result.symbol == "AAPL"
        assert result.direction == "continuation"
        assert result.momentum_score == 72.5


class TestMomentumModelMetrics:
    """Test suite for MomentumModelMetrics dataclass."""

    def test_dataclass_creation(self):
        """Test creating a MomentumModelMetrics."""
        metrics = MomentumModelMetrics(
            direction_accuracy=0.55,
            direction_precision=0.54,
            direction_recall=0.55,
            direction_f1=0.54,
            magnitude_mae=3.2,
            magnitude_directional_accuracy=0.58,
            duration_mae=2.1,
            feature_importances={"feat1": 0.1, "feat2": 0.05},
            cv_scores={"direction_f1": [0.5, 0.55, 0.52]},
        )

        assert metrics.direction_accuracy == 0.55
        assert metrics.magnitude_mae == 3.2
        assert len(metrics.feature_importances) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
