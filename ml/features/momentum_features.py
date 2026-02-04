"""
Momentum Continuation Features for ML Models.

This module calculates features for momentum continuation prediction,
detecting strong price moves and calculating features to predict
continuation vs reversal with magnitude and duration estimates.

Feature Categories:
- Move Strength: Measures strength and velocity of price moves
- Breakout Detection: Identifies and measures breakout characteristics
- Volume Confirmation: Volume-based confirmation signals
- Exhaustion: Signs of trend exhaustion or continuation
- Multi-Timeframe: Higher timeframe alignment features
"""
from __future__ import annotations

from typing import Any
import logging

import numpy as np
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


class MomentumContinuationFeatures:
    """
    Features for momentum continuation prediction.

    Detects strong price moves and calculates features to predict
    continuation vs reversal with magnitude and duration estimates.
    """

    DEFAULT_CONFIG = {
        "move_periods": [5, 10, 20],
        "atr_period": 14,
        "volume_ma_period": 20,
        "consolidation_period": 10,
        "higher_tf_multiplier": 5,  # 5x period for higher timeframe
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize momentum continuation features calculator.

        Args:
            config: Custom configuration (uses defaults if not provided)
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all momentum continuation features.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            DataFrame with original columns plus feature columns
        """
        df = df.copy()

        # Ensure required columns
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Calculate ATR first (used by many features)
        df = self._add_atr(df)

        # Calculate all feature categories
        df = self._add_move_strength_features(df)
        df = self._add_breakout_features(df)
        df = self._add_volume_confirmation_features(df)
        df = self._add_exhaustion_features(df)
        df = self._add_multi_timeframe_features(df)

        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        period = self.config["atr_period"]

        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        df["_atr"] = tr.rolling(window=period).mean()

        return df

    def _add_move_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add move strength features.

        Features:
        - move_strength_N: Price change normalized by ATR
        - move_velocity_N: Move strength per period
        - move_acceleration: Change in velocity
        - direction_consistency_N: Consistency of direction
        """
        close = df["close"]
        atr = df["_atr"]

        for period in self.config["move_periods"]:
            # Move strength: (close - close.shift(N)) / ATR
            price_change = close - close.shift(period)
            df[f"move_strength_{period}"] = price_change / atr.replace(0, np.nan)

            # Move velocity: move_strength / N (normalized by period)
            df[f"move_velocity_{period}"] = df[f"move_strength_{period}"] / period

            # Direction consistency: count of same-direction closes / N
            # Calculate daily direction
            daily_direction = np.sign(close.diff())
            # Compare each day's direction to the overall move direction
            overall_direction = np.sign(price_change)
            same_direction = (daily_direction == overall_direction.shift(1)).astype(float)
            df[f"direction_consistency_{period}"] = same_direction.rolling(window=period).sum() / period

        # Move acceleration: change in velocity
        # (move_velocity_5 - move_velocity_10.shift(5))
        df["move_acceleration"] = df["move_velocity_5"] - df["move_velocity_10"].shift(5)

        return df

    def _add_breakout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add breakout detection features.

        Features:
        - consolidation_ratio: Range / ATR (low = consolidation)
        - breakout_up: Binary breakout above resistance
        - breakout_down: Binary breakout below support
        - breakout_strength: Distance from breakout level / ATR
        - range_position: Position of close within recent range
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        atr = df["_atr"]
        consol_period = self.config["consolidation_period"]

        # Consolidation ratio: range / ATR (< 1 = tight consolidation)
        rolling_high = high.rolling(window=consol_period).max()
        rolling_low = low.rolling(window=consol_period).min()
        range_size = rolling_high - rolling_low
        df["consolidation_ratio"] = range_size / atr.replace(0, np.nan)

        # Breakout detection (using 20-period high/low)
        prev_high_20 = high.rolling(window=20).max().shift(1)
        prev_low_20 = low.rolling(window=20).min().shift(1)

        # Breakout up: close > previous 20-period high
        df["breakout_up"] = (close > prev_high_20).astype(int)

        # Breakout down: close < previous 20-period low
        df["breakout_down"] = (close < prev_low_20).astype(int)

        # Breakout strength: distance from breakout level / ATR
        # If breakout up, distance from prev high; if down, distance from prev low
        breakout_distance = np.where(
            df["breakout_up"] == 1,
            close - prev_high_20,
            np.where(
                df["breakout_down"] == 1,
                prev_low_20 - close,
                0
            )
        )
        df["breakout_strength"] = breakout_distance / atr.replace(0, np.nan)

        # Range position: position within recent range [0, 1]
        # (close - low) / (high - low) for rolling period
        for period in self.config["move_periods"]:
            period_high = high.rolling(window=period).max()
            period_low = low.rolling(window=period).min()
            period_range = period_high - period_low
            df[f"range_position_{period}"] = (close - period_low) / period_range.replace(0, np.nan)

        return df

    def _add_volume_confirmation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume confirmation features.

        Features:
        - volume_ratio: Current volume vs average
        - volume_momentum: Change in volume ratio
        - buying_pressure: Volume-weighted close position
        - volume_price_divergence: Price/volume divergence signal
        - ad_momentum: Accumulation/Distribution momentum
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_ = df["open"]
        volume = df["volume"]
        vol_ma_period = self.config["volume_ma_period"]

        # Volume ratio: volume / volume.rolling(20).mean()
        volume_ma = volume.rolling(window=vol_ma_period).mean()
        df["volume_ratio"] = volume / volume_ma.replace(0, np.nan)

        # Volume momentum: change in volume ratio over 5 periods
        df["volume_momentum"] = df["volume_ratio"] - df["volume_ratio"].shift(5)

        # Buying pressure: (close - low) / (high - low) * volume_ratio
        day_range = high - low
        close_position = (close - low) / day_range.replace(0, np.nan)
        df["buying_pressure"] = close_position * df["volume_ratio"]

        # Volume price divergence:
        # 1 if price up but volume down (bearish divergence)
        # -1 if price down but volume up (bullish divergence)
        # 0 otherwise
        price_up = close > close.shift(1)
        volume_up = volume > volume.shift(1)

        df["volume_price_divergence"] = np.where(
            price_up & ~volume_up, 1,  # Price up, volume down (bearish)
            np.where(
                ~price_up & volume_up, -1,  # Price down, volume up (bullish)
                0
            )
        )

        # A/D momentum: cumulative (close - open) / (high - low) * volume
        # Then take 5-period momentum
        ad_raw = ((close - open_) / day_range.replace(0, np.nan)) * volume
        df["_ad_cumulative"] = ad_raw.cumsum()
        df["ad_momentum"] = df["_ad_cumulative"] - df["_ad_cumulative"].shift(5)

        # Clean up temporary column
        df = df.drop(columns=["_ad_cumulative"])

        return df

    def _add_exhaustion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add exhaustion and continuation features.

        Features:
        - volatility_contraction: ATR vs average ATR
        - pullback_from_high: Distance from 20-period high / ATR
        - pullback_from_low: Distance from 20-period low / ATR
        - retracement_ratio: How much has retraced vs move
        - momentum_persistence: Persistence of price direction
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        atr = df["_atr"]

        # Volatility contraction: ATR / ATR.rolling(20).mean()
        # < 1 indicates contraction (often precedes breakout)
        atr_ma = atr.rolling(window=20).mean()
        df["volatility_contraction"] = atr / atr_ma.replace(0, np.nan)

        # Pullback from high: (high.rolling(20).max() - close) / ATR
        high_20 = high.rolling(window=20).max()
        df["pullback_from_high"] = (high_20 - close) / atr.replace(0, np.nan)

        # Pullback from low: (close - low.rolling(20).min()) / ATR
        low_20 = low.rolling(window=20).min()
        df["pullback_from_low"] = (close - low_20) / atr.replace(0, np.nan)

        # Retracement ratio: pullback / move_strength
        # How much of the move has been retraced
        move_up = high_20 - low_20
        pullback = high_20 - close
        df["retracement_ratio"] = pullback / move_up.replace(0, np.nan)

        # Momentum persistence: sum of sign(close.diff()) over last N periods / N
        # Ranges from -1 (all down) to +1 (all up)
        close_direction = np.sign(close.diff())
        for period in self.config["move_periods"]:
            df[f"momentum_persistence_{period}"] = (
                close_direction.rolling(window=period).sum() / period
            )

        return df

    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add multi-timeframe features.

        Features:
        - higher_tf_trend: Trend direction on higher timeframe
        - higher_tf_momentum: Move strength on higher timeframe
        - mtf_alignment: Alignment of current vs higher TF trend
        """
        close = df["close"]
        atr = df["_atr"]
        multiplier = self.config["higher_tf_multiplier"]

        # Higher timeframe periods
        for period in self.config["move_periods"]:
            htf_period = period * multiplier

            # Higher TF trend: sign of linear regression slope
            df[f"higher_tf_trend_{period}"] = close.rolling(window=htf_period).apply(
                self._calculate_trend_sign, raw=True
            )

            # Higher TF momentum: move strength on higher timeframe
            htf_price_change = close - close.shift(htf_period)
            htf_atr = atr.rolling(window=htf_period).mean()
            df[f"higher_tf_momentum_{period}"] = htf_price_change / htf_atr.replace(0, np.nan)

            # MTF alignment:
            # 1 if current momentum aligns with higher tf trend
            # 0.5 if neutral (higher tf trend is 0)
            # 0 if contrary
            current_momentum = df.get(f"move_strength_{period}", pd.Series(0, index=df.index))
            htf_trend = df[f"higher_tf_trend_{period}"]

            current_direction = np.sign(current_momentum)
            htf_direction = htf_trend

            df[f"mtf_alignment_{period}"] = np.where(
                htf_direction == 0, 0.5,  # Neutral higher TF
                np.where(
                    current_direction == htf_direction, 1,  # Aligned
                    0  # Contrary
                )
            )

        return df

    @staticmethod
    def _calculate_trend_sign(x: np.ndarray) -> float:
        """
        Calculate the sign of linear regression slope.

        Args:
            x: Array of prices

        Returns:
            1 for uptrend, -1 for downtrend, 0 for flat
        """
        if len(x) < 2:
            return 0

        n = len(x)
        x_vals = np.arange(n)

        try:
            slope, _, _, _, _ = stats.linregress(x_vals, x)
            # Threshold for considering slope as significant
            # Using 0.1% of mean price as threshold
            threshold = np.mean(x) * 0.001
            if abs(slope) < threshold:
                return 0
            return 1 if slope > 0 else -1
        except Exception:
            return 0

    def get_feature_names(self) -> list[str]:
        """
        Get list of all feature column names.

        Returns:
            List of feature column names
        """
        # Create a sample DataFrame to get feature names
        sample_df = pd.DataFrame({
            "open": np.linspace(100, 110, 100).tolist(),
            "high": np.linspace(101, 111, 100).tolist(),
            "low": np.linspace(99, 109, 100).tolist(),
            "close": np.linspace(100.5, 110.5, 100).tolist(),
            "volume": [1000000] * 100,
        })

        result = self.calculate_all(sample_df)

        # Exclude original OHLCV columns and internal columns
        exclude_cols = {"open", "high", "low", "close", "volume", "timestamp", "symbol", "_atr"}
        feature_cols = [col for col in result.columns if col not in exclude_cols]

        return sorted(feature_cols)

    def calculate_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for making predictions (latest data point).

        This method is optimized for real-time prediction where we only
        need the most recent feature values.

        Args:
            df: DataFrame with recent OHLCV data (at least 100 bars recommended)

        Returns:
            DataFrame with single row of features for the latest timestamp
        """
        # Calculate all features
        features_df = self.calculate_all(df)

        # Return only the last row (most recent)
        return features_df.iloc[[-1]].copy()


def prepare_momentum_features(
    df: pd.DataFrame,
    target_horizon: int = 1,
    target_type: str = "direction",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare momentum features and target for ML training.

    Args:
        df: DataFrame with OHLCV data
        target_horizon: Number of periods ahead for target
        target_type: "direction" for binary, "return" for continuous

    Returns:
        Tuple of (features_df, target_series)
    """
    # Calculate features
    calculator = MomentumContinuationFeatures()
    features_df = calculator.calculate_all(df)

    # Create target variable
    future_return = features_df["close"].pct_change(target_horizon).shift(-target_horizon)

    if target_type == "direction":
        # Binary: 1 if positive return, 0 if negative
        target = (future_return > 0).astype(int)
    else:
        # Continuous return
        target = future_return

    # Remove rows with NaN
    valid_idx = features_df.notna().all(axis=1) & target.notna()

    # Get feature columns (exclude OHLCV and metadata)
    exclude_cols = {"open", "high", "low", "close", "volume", "timestamp", "symbol", "_atr"}
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    return features_df.loc[valid_idx, feature_cols], target.loc[valid_idx]
