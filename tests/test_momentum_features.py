"""
Unit tests for MomentumContinuationFeatures.

Tests feature calculation, output shape, and feature names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.momentum_features import MomentumContinuationFeatures, prepare_momentum_features


def generate_sample_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)

    # Create realistic price movement
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n)
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


class TestMomentumContinuationFeatures:
    """Test suite for MomentumContinuationFeatures class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        calc = MomentumContinuationFeatures()
        assert calc.config["atr_period"] == 14
        assert calc.config["move_periods"] == [5, 10, 20]
        assert calc.config["volume_ma_period"] == 20

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        custom_config = {"atr_period": 10, "move_periods": [3, 7, 14]}
        calc = MomentumContinuationFeatures(config=custom_config)
        assert calc.config["atr_period"] == 10
        assert calc.config["move_periods"] == [3, 7, 14]
        # Default values still present
        assert calc.config["volume_ma_period"] == 20

    def test_calculate_all_output_shape(self):
        """Test that calculate_all returns correct shape."""
        df = generate_sample_ohlcv(100)
        calc = MomentumContinuationFeatures()
        result = calc.calculate_all(df)

        # Should have same number of rows
        assert len(result) == len(df)

        # Should have more columns than input
        assert len(result.columns) > len(df.columns)

    def test_calculate_all_preserves_ohlcv(self):
        """Test that original OHLCV columns are preserved."""
        df = generate_sample_ohlcv(100)
        calc = MomentumContinuationFeatures()
        result = calc.calculate_all(df)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns
            pd.testing.assert_series_equal(result[col], df[col])

    def test_feature_count(self):
        """Test that we generate 20+ features."""
        df = generate_sample_ohlcv(100)
        calc = MomentumContinuationFeatures()
        result = calc.calculate_all(df)

        # Count new feature columns (excluding OHLCV and internal)
        exclude = {"open", "high", "low", "close", "volume", "_atr"}
        new_features = [c for c in result.columns if c not in exclude]

        assert len(new_features) >= 20, f"Expected 20+ features, got {len(new_features)}"

    def test_move_strength_features(self):
        """Test move strength feature columns exist."""
        df = generate_sample_ohlcv(100)
        calc = MomentumContinuationFeatures()
        result = calc.calculate_all(df)

        expected_features = [
            "move_strength_5", "move_strength_10", "move_strength_20",
            "move_velocity_5", "move_velocity_10", "move_velocity_20",
            "direction_consistency_5", "direction_consistency_10", "direction_consistency_20",
            "move_acceleration",
        ]

        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_breakout_features(self):
        """Test breakout feature columns exist."""
        df = generate_sample_ohlcv(100)
        calc = MomentumContinuationFeatures()
        result = calc.calculate_all(df)

        expected_features = [
            "consolidation_ratio",
            "breakout_up", "breakout_down",
            "breakout_strength",
            "range_position_5", "range_position_10", "range_position_20",
        ]

        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_volume_features(self):
        """Test volume confirmation feature columns exist."""
        df = generate_sample_ohlcv(100)
        calc = MomentumContinuationFeatures()
        result = calc.calculate_all(df)

        expected_features = [
            "volume_ratio", "volume_momentum",
            "buying_pressure", "volume_price_divergence",
            "ad_momentum",
        ]

        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_exhaustion_features(self):
        """Test exhaustion feature columns exist."""
        df = generate_sample_ohlcv(100)
        calc = MomentumContinuationFeatures()
        result = calc.calculate_all(df)

        expected_features = [
            "volatility_contraction",
            "pullback_from_high", "pullback_from_low",
            "retracement_ratio",
            "momentum_persistence_5", "momentum_persistence_10", "momentum_persistence_20",
        ]

        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_multi_timeframe_features(self):
        """Test multi-timeframe feature columns exist."""
        df = generate_sample_ohlcv(200)  # Need more data for higher TF
        calc = MomentumContinuationFeatures()
        result = calc.calculate_all(df)

        expected_features = [
            "higher_tf_trend_5", "higher_tf_trend_10", "higher_tf_trend_20",
            "higher_tf_momentum_5", "higher_tf_momentum_10", "higher_tf_momentum_20",
            "mtf_alignment_5", "mtf_alignment_10", "mtf_alignment_20",
        ]

        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_get_feature_names(self):
        """Test get_feature_names returns consistent list."""
        calc = MomentumContinuationFeatures()
        names = calc.get_feature_names()

        assert isinstance(names, list)
        assert len(names) >= 20
        assert all(isinstance(n, str) for n in names)

    def test_calculate_for_prediction(self):
        """Test calculate_for_prediction returns single row."""
        df = generate_sample_ohlcv(100)
        calc = MomentumContinuationFeatures()
        result = calc.calculate_for_prediction(df)

        assert len(result) == 1
        assert len(result.columns) > 5  # More than just OHLCV

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        calc = MomentumContinuationFeatures()

        with pytest.raises(ValueError, match="Missing required columns"):
            calc.calculate_all(df)

    def test_feature_values_reasonable(self):
        """Test that feature values are within reasonable ranges."""
        df = generate_sample_ohlcv(200)
        calc = MomentumContinuationFeatures()
        result = calc.calculate_all(df)

        # Volume ratio should be positive
        assert (result["volume_ratio"].dropna() > 0).all()

        # Range position should be between 0 and 1
        assert (result["range_position_10"].dropna() >= 0).all()
        assert (result["range_position_10"].dropna() <= 1).all()

        # Direction consistency should be between 0 and 1
        assert (result["direction_consistency_10"].dropna() >= 0).all()
        assert (result["direction_consistency_10"].dropna() <= 1).all()

        # MTF alignment should be 0, 0.5, or 1
        valid_values = {0, 0.5, 1}
        mtf_values = result["mtf_alignment_10"].dropna().unique()
        assert all(v in valid_values for v in mtf_values)


class TestPrepareMomentumFeatures:
    """Test suite for prepare_momentum_features function."""

    def test_direction_target(self):
        """Test direction target creation."""
        df = generate_sample_ohlcv(100)
        X, y = prepare_momentum_features(df, target_type="direction")

        assert len(X) == len(y)
        assert set(y.unique()).issubset({0, 1})

    def test_return_target(self):
        """Test return target creation."""
        df = generate_sample_ohlcv(100)
        X, y = prepare_momentum_features(df, target_type="return")

        assert len(X) == len(y)
        assert y.dtype == float


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
