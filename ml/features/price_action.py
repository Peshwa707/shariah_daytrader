"""
Price Action Features for ML Models.

This module calculates features derived directly from price movements,
focusing on patterns and behaviors that may predict future moves:

- Returns and momentum
- Support/resistance levels
- Price patterns
- Relative strength
"""
from __future__ import annotations

from typing import Any
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class PriceActionFeatures:
    """
    Calculates price action features for ML models.

    These features capture market dynamics and patterns that
    may not be captured by traditional technical indicators.
    """

    DEFAULT_CONFIG = {
        "return_periods": [1, 2, 3, 5, 10, 20],
        "momentum_periods": [5, 10, 20],
        "lookback_periods": [5, 10, 20, 50],
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize price action features calculator.

        Args:
            config: Custom configuration
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all price action features.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()

        df = self._add_return_features(df)
        df = self._add_momentum_features(df)
        df = self._add_range_features(df)
        df = self._add_pattern_features(df)
        df = self._add_level_features(df)

        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        close = df["close"]

        # Simple returns over various periods
        for period in self.config["return_periods"]:
            df[f"return_{period}d"] = close.pct_change(period)

        # Log returns (more normally distributed)
        df["log_return_1d"] = np.log(close / close.shift(1))
        df["log_return_5d"] = np.log(close / close.shift(5))

        # Cumulative returns
        for period in [5, 10, 20]:
            df[f"cum_return_{period}d"] = (1 + df["return_1d"]).rolling(period).apply(
                lambda x: x.prod() - 1, raw=True
            )

        # Return consistency (% of positive days)
        for period in [5, 10, 20]:
            df[f"positive_days_{period}d"] = (
                (df["return_1d"] > 0).rolling(period).mean()
            )

        # Return volatility (std of returns)
        for period in [5, 10, 20]:
            df[f"return_std_{period}d"] = df["return_1d"].rolling(period).std()

        # Sharpe-like ratio (return / volatility)
        df["sharpe_10d"] = (
            df["cum_return_10d"] / df["return_std_10d"].replace(0, np.nan)
        )

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features."""
        close = df["close"]

        # Price momentum (position relative to past)
        for period in self.config["momentum_periods"]:
            past_close = close.shift(period)
            df[f"momentum_{period}d"] = (close - past_close) / past_close

        # Acceleration (change in momentum)
        df["momentum_accel_5d"] = df["momentum_5d"] - df["momentum_5d"].shift(1)
        df["momentum_accel_10d"] = df["momentum_10d"] - df["momentum_10d"].shift(1)

        # Momentum persistence (autocorrelation of returns)
        df["return_autocorr_5d"] = (
            df["return_1d"].rolling(5).apply(
                lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
            )
        )

        # Trend strength (linear regression slope)
        for period in [10, 20]:
            df[f"trend_slope_{period}d"] = (
                close.rolling(period).apply(self._calculate_slope, raw=True)
            )

        # R-squared of trend (how linear is the trend)
        for period in [10, 20]:
            df[f"trend_r2_{period}d"] = (
                close.rolling(period).apply(self._calculate_r2, raw=True)
            )

        return df

    def _add_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add range and volatility features."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_ = df["open"]

        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        df["true_range"] = tr

        # Range expansion/contraction
        for period in [5, 10]:
            avg_range = (high - low).rolling(period).mean()
            df[f"range_vs_avg_{period}d"] = (high - low) / avg_range

        # Intraday range percentile
        for period in [20, 50]:
            df[f"range_percentile_{period}d"] = (
                (high - low).rolling(period).apply(
                    lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5,
                    raw=False
                )
            )

        # Close position in day's range
        day_range = high - low
        df["close_position_in_range"] = (close - low) / day_range.replace(0, np.nan)

        # Gap analysis
        df["gap_size"] = (open_ - close.shift(1)) / close.shift(1)
        df["gap_filled"] = (
            ((df["gap_size"] > 0) & (low <= close.shift(1))) |
            ((df["gap_size"] < 0) & (high >= close.shift(1)))
        ).astype(int)

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_ = df["open"]

        # Consecutive up/down days
        up_day = (close > close.shift(1)).astype(int)
        df["consecutive_up"] = up_day.groupby((up_day != up_day.shift()).cumsum()).cumsum()
        df["consecutive_down"] = (1 - up_day).groupby(
            ((1 - up_day) != (1 - up_day).shift()).cumsum()
        ).cumsum()

        # Higher highs / lower lows
        df["higher_high"] = (high > high.shift(1)).astype(int)
        df["lower_low"] = (low < low.shift(1)).astype(int)

        # Count of higher highs in last N days
        for period in [5, 10]:
            df[f"higher_highs_{period}d"] = df["higher_high"].rolling(period).sum()
            df[f"lower_lows_{period}d"] = df["lower_low"].rolling(period).sum()

        # Doji-like days (small body)
        body_size = abs(close - open_)
        day_range = high - low
        df["is_doji"] = (body_size < day_range * 0.1).astype(int)

        # Reversal patterns (simplified)
        # Hammer: small body, long lower shadow
        lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
        upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
        df["hammer_like"] = (
            (lower_shadow > 2 * body_size) &
            (upper_shadow < body_size)
        ).astype(int)

        # Engulfing pattern (simplified)
        prev_body = abs(close.shift(1) - open_.shift(1))
        df["bullish_engulf"] = (
            (close.shift(1) < open_.shift(1)) &  # Previous red
            (close > open_) &  # Current green
            (body_size > prev_body)  # Current body larger
        ).astype(int)

        return df

    def _add_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support/resistance level features."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Recent highs and lows
        for period in self.config["lookback_periods"]:
            df[f"highest_{period}d"] = high.rolling(period).max()
            df[f"lowest_{period}d"] = low.rolling(period).min()

            # Distance from extremes
            df[f"dist_from_high_{period}d"] = (
                (close - df[f"highest_{period}d"]) / df[f"highest_{period}d"]
            )
            df[f"dist_from_low_{period}d"] = (
                (close - df[f"lowest_{period}d"]) / df[f"lowest_{period}d"]
            )

        # At new high/low
        df["at_20d_high"] = (close >= df["highest_20d"]).astype(int)
        df["at_20d_low"] = (close <= df["lowest_20d"]).astype(int)
        df["at_50d_high"] = (close >= df["highest_50d"]).astype(int)
        df["at_50d_low"] = (close <= df["lowest_50d"]).astype(int)

        # Price percentile in range
        for period in [20, 50]:
            range_size = df[f"highest_{period}d"] - df[f"lowest_{period}d"]
            df[f"price_percentile_{period}d"] = (
                (close - df[f"lowest_{period}d"]) / range_size.replace(0, np.nan)
            )

        # Pivot points (simple)
        df["pivot"] = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        df["r1"] = 2 * df["pivot"] - low.shift(1)
        df["s1"] = 2 * df["pivot"] - high.shift(1)

        df["above_pivot"] = (close > df["pivot"]).astype(int)
        df["above_r1"] = (close > df["r1"]).astype(int)
        df["below_s1"] = (close < df["s1"]).astype(int)

        return df

    @staticmethod
    def _calculate_slope(x: np.ndarray) -> float:
        """Calculate linear regression slope."""
        if len(x) < 2:
            return 0
        n = len(x)
        x_vals = np.arange(n)
        slope = np.polyfit(x_vals, x, 1)[0]
        # Normalize by mean price
        return slope / np.mean(x) if np.mean(x) != 0 else 0

    @staticmethod
    def _calculate_r2(x: np.ndarray) -> float:
        """Calculate R-squared of linear trend."""
        if len(x) < 2:
            return 0
        n = len(x)
        x_vals = np.arange(n)

        # Fit line and calculate R-squared
        coeffs = np.polyfit(x_vals, x, 1)
        y_pred = np.polyval(coeffs, x_vals)

        ss_res = np.sum((x - y_pred) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)

        if ss_tot == 0:
            return 1.0
        return 1 - (ss_res / ss_tot)

    def get_feature_names(self) -> list[str]:
        """Get list of feature column names."""
        sample_df = pd.DataFrame({
            "open": [100.0] * 60,
            "high": [101.0] * 60,
            "low": [99.0] * 60,
            "close": [100.5] * 60,
            "volume": [1000000] * 60,
        })

        result = self.calculate_all(sample_df)

        original_cols = {"open", "high", "low", "close", "volume", "timestamp", "symbol"}
        return [col for col in result.columns if col not in original_cols]


class FeaturePipeline:
    """
    Combined feature pipeline that orchestrates all feature calculators.

    This is the main interface for feature engineering in the ML models.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the feature pipeline.

        Args:
            config: Configuration for individual calculators
        """
        # Import here to avoid circular imports
        from .technical import TechnicalFeatures

        self.technical = TechnicalFeatures(config.get("technical") if config else None)
        self.price_action = PriceActionFeatures(config.get("price_action") if config else None)

    def calculate_features(
        self,
        df: pd.DataFrame,
        include_technical: bool = True,
        include_price_action: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate all features.

        Args:
            df: DataFrame with OHLCV data
            include_technical: Include technical indicators
            include_price_action: Include price action features

        Returns:
            DataFrame with all features
        """
        result = df.copy()

        if include_technical:
            result = self.technical.calculate_all(result)

        if include_price_action:
            result = self.price_action.calculate_all(result)

        return result

    def get_all_feature_names(self) -> list[str]:
        """Get names of all features."""
        tech_features = self.technical.get_feature_names()
        pa_features = self.price_action.get_feature_names()
        return list(set(tech_features + pa_features))

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_horizon: int = 1,
        target_type: str = "direction",
        dropna: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.

        Args:
            df: DataFrame with OHLCV data
            target_horizon: Periods ahead for target
            target_type: "direction" (binary) or "return" (continuous)
            dropna: Drop rows with missing values

        Returns:
            Tuple of (features_df, target_series)
        """
        # Calculate all features
        features_df = self.calculate_features(df)

        # Create target
        future_return = features_df["close"].pct_change(target_horizon).shift(-target_horizon)

        if target_type == "direction":
            target = (future_return > 0).astype(int)
        else:
            target = future_return

        # Select feature columns
        exclude_cols = {"open", "high", "low", "close", "volume", "timestamp", "symbol"}
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        X = features_df[feature_cols]
        y = target

        if dropna:
            valid_idx = X.notna().all(axis=1) & y.notna()
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]

        return X, y
