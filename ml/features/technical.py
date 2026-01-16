"""
Technical Indicator Features for ML Models.

This module calculates technical indicators using pandas-ta library.
These features are used as inputs to ML models for signal generation.

Indicators included:
- Trend: SMA, EMA, MACD
- Momentum: RSI, Stochastic, ROC
- Volatility: ATR, Bollinger Bands
- Volume: OBV, VWAP ratios
"""

from typing import Any
import logging

import numpy as np
import pandas as pd

# pandas-ta for technical indicators
try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """
    Calculates technical indicators and prepares features for ML models.

    Features are designed to be:
    - Stationary (or made stationary via differencing/normalization)
    - Scaled appropriately for ML models
    - Free of look-ahead bias
    """

    # Default indicator periods
    DEFAULT_CONFIG = {
        # Trend indicators
        "sma_periods": [5, 10, 20, 50],
        "ema_periods": [5, 10, 20, 50],

        # Momentum indicators
        "rsi_period": 14,
        "stoch_k_period": 14,
        "stoch_d_period": 3,
        "roc_periods": [5, 10, 20],

        # Volatility indicators
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,

        # MACD
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize technical features calculator.

        Args:
            config: Custom indicator configuration (uses defaults if not provided)
        """
        if not TA_AVAILABLE:
            raise ImportError(
                "pandas-ta is not installed. Install with: pip install pandas-ta"
            )

        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicator features.

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

        # Calculate all indicator categories
        df = self._add_trend_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_volume_indicators(df)
        df = self._add_price_features(df)

        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        close = df["close"]

        # Simple Moving Averages
        for period in self.config["sma_periods"]:
            df[f"sma_{period}"] = ta.sma(close, length=period)
            # Distance from SMA (normalized)
            df[f"close_sma_{period}_ratio"] = close / df[f"sma_{period}"]

        # Exponential Moving Averages
        for period in self.config["ema_periods"]:
            df[f"ema_{period}"] = ta.ema(close, length=period)
            df[f"close_ema_{period}_ratio"] = close / df[f"ema_{period}"]

        # MACD
        macd = ta.macd(
            close,
            fast=self.config["macd_fast"],
            slow=self.config["macd_slow"],
            signal=self.config["macd_signal"],
        )
        if macd is not None:
            df["macd"] = macd.iloc[:, 0]  # MACD line
            df["macd_signal"] = macd.iloc[:, 1]  # Signal line
            df["macd_hist"] = macd.iloc[:, 2]  # Histogram

        # Moving average crossovers (binary)
        df["sma_5_20_cross"] = (df["sma_5"] > df["sma_20"]).astype(int)
        df["ema_5_20_cross"] = (df["ema_5"] > df["ema_20"]).astype(int)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        df["rsi"] = ta.rsi(close, length=self.config["rsi_period"])

        # RSI zones (useful for classification)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)

        # Stochastic Oscillator
        stoch = ta.stoch(
            high, low, close,
            k=self.config["stoch_k_period"],
            d=self.config["stoch_d_period"],
        )
        if stoch is not None:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]

        # Rate of Change
        for period in self.config["roc_periods"]:
            df[f"roc_{period}"] = ta.roc(close, length=period)

        # Williams %R
        df["willr"] = ta.willr(high, low, close, length=14)

        # CCI (Commodity Channel Index)
        df["cci"] = ta.cci(high, low, close, length=20)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ATR (Average True Range)
        df["atr"] = ta.atr(high, low, close, length=self.config["atr_period"])

        # ATR as percentage of price
        df["atr_pct"] = df["atr"] / close * 100

        # Bollinger Bands
        bbands = ta.bbands(
            close,
            length=self.config["bb_period"],
            std=self.config["bb_std"],
        )
        if bbands is not None:
            df["bb_lower"] = bbands.iloc[:, 0]
            df["bb_mid"] = bbands.iloc[:, 1]
            df["bb_upper"] = bbands.iloc[:, 2]
            df["bb_bandwidth"] = bbands.iloc[:, 3]
            df["bb_percent"] = bbands.iloc[:, 4]

        # Price position in Bollinger Bands
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_range = df["bb_upper"] - df["bb_lower"]
            df["bb_position"] = (close - df["bb_lower"]) / bb_range

        # Historical volatility (rolling std of returns)
        returns = close.pct_change()
        df["volatility_5"] = returns.rolling(5).std() * np.sqrt(252)
        df["volatility_20"] = returns.rolling(20).std() * np.sqrt(252)

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Volume moving averages
        df["volume_sma_20"] = ta.sma(volume, length=20)
        df["volume_ratio"] = volume / df["volume_sma_20"]

        # On-Balance Volume
        df["obv"] = ta.obv(close, volume)

        # OBV momentum
        if "obv" in df.columns:
            df["obv_sma_10"] = ta.sma(df["obv"], length=10)
            df["obv_trend"] = (df["obv"] > df["obv_sma_10"]).astype(int)

        # Money Flow Index
        df["mfi"] = ta.mfi(high, low, close, volume, length=14)

        # VWAP (if we have intraday data, this would be more meaningful)
        if "vwap" in df.columns:
            df["close_vwap_ratio"] = close / df["vwap"]

        # Accumulation/Distribution
        df["ad"] = ta.ad(high, low, close, volume)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-derived features."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # Candle patterns
        df["body_size"] = abs(close - open_) / open_ * 100
        df["upper_shadow"] = (high - df[["close", "open"]].max(axis=1)) / open_ * 100
        df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - low) / open_ * 100

        # Candle direction
        df["is_green"] = (close > open_).astype(int)

        # Day range
        df["day_range"] = (high - low) / open_ * 100

        # Gap (vs previous close)
        df["gap"] = (open_ - close.shift(1)) / close.shift(1) * 100

        # Distance from highs/lows
        df["high_20d"] = high.rolling(20).max()
        df["low_20d"] = low.rolling(20).min()
        df["dist_from_high_20d"] = (close - df["high_20d"]) / df["high_20d"] * 100
        df["dist_from_low_20d"] = (close - df["low_20d"]) / df["low_20d"] * 100

        return df

    def get_feature_names(self) -> list[str]:
        """
        Get list of all feature column names.

        Returns:
            List of feature column names
        """
        # Create a small sample DataFrame to get feature names
        sample_df = pd.DataFrame({
            "open": [100.0] * 60,
            "high": [101.0] * 60,
            "low": [99.0] * 60,
            "close": [100.5] * 60,
            "volume": [1000000] * 60,
        })

        result = self.calculate_all(sample_df)

        # Exclude original OHLCV columns
        original_cols = {"open", "high", "low", "close", "volume", "timestamp", "symbol"}
        feature_cols = [col for col in result.columns if col not in original_cols]

        return feature_cols

    def calculate_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for making predictions (uses only the latest data point).

        This method is optimized for real-time prediction where we only need
        the most recent feature values.

        Args:
            df: DataFrame with recent OHLCV data (at least 60 bars for indicators)

        Returns:
            DataFrame with single row of features for the latest timestamp
        """
        # Calculate all features
        features_df = self.calculate_all(df)

        # Return only the last row (most recent)
        return features_df.iloc[[-1]].copy()


def prepare_ml_features(
    df: pd.DataFrame,
    target_horizon: int = 1,
    target_type: str = "direction",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for ML training.

    Args:
        df: DataFrame with OHLCV data
        target_horizon: Number of periods ahead for target
        target_type: "direction" for binary, "return" for continuous

    Returns:
        Tuple of (features_df, target_series)
    """
    # Calculate features
    calculator = TechnicalFeatures()
    features_df = calculator.calculate_all(df)

    # Create target variable
    future_return = features_df["close"].pct_change(target_horizon).shift(-target_horizon)

    if target_type == "direction":
        # Binary: 1 if positive return, 0 if negative
        target = (future_return > 0).astype(int)
    else:
        # Continuous return
        target = future_return

    # Remove rows with NaN (from indicator warmup and target creation)
    valid_idx = features_df.notna().all(axis=1) & target.notna()

    # Get feature columns (exclude OHLCV and metadata)
    exclude_cols = {"open", "high", "low", "close", "volume", "timestamp", "symbol"}
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    return features_df.loc[valid_idx, feature_cols], target.loc[valid_idx]
