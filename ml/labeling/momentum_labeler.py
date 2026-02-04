"""
Momentum Labeler for ML Training Data.

This module creates training labels for momentum continuation models.
It detects significant price moves and labels what happens afterwards:
- Continuation: Price continues in the same direction
- Reversal: Price reverses significantly
- Neutral: Price stays within a defined range

Labels are designed for supervised learning models that predict
whether to follow momentum or fade it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MomentumLabelerConfig:
    """Configuration for momentum labeling."""
    move_detection_atr: float = 2.0  # ATR multiplier for significant move
    continuation_threshold: float = 0.5  # 50% of original move for continuation
    reversal_threshold: float = 0.5  # 50% retracement for reversal
    look_forward_bars: int = 10  # How many bars to look ahead
    neutral_zone: float = 0.2  # +-20% of threshold = neutral
    atr_period: int = 14  # Period for ATR calculation


class MomentumLabeler:
    """
    Creates training labels for momentum continuation model.

    Labels:
    - Direction: 0=reversal, 1=neutral, 2=continuation
    - Magnitude: Actual % return over look_forward_bars
    - Duration: Bars until target hit or reversal

    Usage:
        config = MomentumLabelerConfig(look_forward_bars=15)
        labeler = MomentumLabeler(config)
        labels_df = labeler.create_labels(ohlcv_df)
        X, y_dir, y_mag, y_dur = labeler.prepare_labeled_data(ohlcv_df, features_df)
    """

    def __init__(self, config: MomentumLabelerConfig | None = None):
        """
        Initialize momentum labeler.

        Args:
            config: Labeling configuration (uses defaults if not provided)
        """
        self.config = config or MomentumLabelerConfig()

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            df: DataFrame with high, low, close columns

        Returns:
            Series with ATR values
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        # True Range is max of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is EMA of True Range
        atr = true_range.ewm(span=self.config.atr_period, adjust=False).mean()

        return atr

    def _calculate_move_pct(
        self,
        df: pd.DataFrame,
        lookback: int
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate move percentage and direction over lookback period.

        Args:
            df: DataFrame with close prices
            lookback: Number of bars to look back

        Returns:
            Tuple of (move_pct, move_direction)
            move_pct: Absolute percentage change
            move_direction: 1 for up, -1 for down
        """
        close = df["close"]
        past_close = close.shift(lookback)

        move_pct = (close - past_close) / past_close * 100
        move_direction = np.sign(move_pct)

        return move_pct, move_direction

    def detect_moves(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect significant price moves.

        A significant move is when |close - close.shift(N)| > move_detection_atr * ATR
        Checks over multiple lookback periods (5, 10, 20) and takes OR.

        Args:
            df: DataFrame with OHLCV columns (high, low, close required)

        Returns:
            Boolean Series where True = significant move detected
        """
        atr = self._calculate_atr(df)
        close = df["close"]

        # Threshold in price terms
        threshold = self.config.move_detection_atr * atr

        # Check multiple lookback periods
        lookback_periods = [5, 10, 20]
        move_detected = pd.Series(False, index=df.index)

        for lookback in lookback_periods:
            price_change = abs(close - close.shift(lookback))
            significant_move = price_change > threshold
            move_detected = move_detected | significant_move

        return move_detected

    def label_direction(self, df: pd.DataFrame) -> pd.Series:
        """
        Label direction for each bar based on future price action.

        For each detected move, looks forward `look_forward_bars`:
        - If future return >= continuation_threshold * original_move_pct: label = 2 (continuation)
        - If future return <= -reversal_threshold * original_move_pct: label = 0 (reversal)
        - Otherwise: label = 1 (neutral)

        Handles direction: for up moves, continuation=more up; for down moves, continuation=more down.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Series with direction labels (0=reversal, 1=neutral, 2=continuation)
        """
        close = df["close"]
        look_forward = self.config.look_forward_bars

        # Calculate future return (percentage)
        future_close = close.shift(-look_forward)
        future_return_pct = (future_close - close) / close * 100

        # Calculate the original move (use 10-bar lookback as reference)
        move_pct, move_direction = self._calculate_move_pct(df, lookback=10)

        # Initialize labels as neutral
        labels = pd.Series(1, index=df.index, dtype=int)

        # For up moves (positive move_pct)
        up_move_mask = move_direction > 0
        # Continuation: future goes up enough
        continuation_threshold_up = self.config.continuation_threshold * abs(move_pct)
        # Reversal: future goes down enough
        reversal_threshold_up = self.config.reversal_threshold * abs(move_pct)

        # Apply labels for up moves
        labels = labels.where(
            ~(up_move_mask & (future_return_pct >= continuation_threshold_up)),
            2  # Continuation
        )
        labels = labels.where(
            ~(up_move_mask & (future_return_pct <= -reversal_threshold_up)),
            0  # Reversal
        )

        # For down moves (negative move_pct)
        down_move_mask = move_direction < 0
        # Continuation for down move: future goes down more
        continuation_threshold_down = self.config.continuation_threshold * abs(move_pct)
        # Reversal for down move: future goes up
        reversal_threshold_down = self.config.reversal_threshold * abs(move_pct)

        # Apply labels for down moves
        labels = labels.where(
            ~(down_move_mask & (future_return_pct <= -continuation_threshold_down)),
            2  # Continuation (more downside)
        )
        labels = labels.where(
            ~(down_move_mask & (future_return_pct >= reversal_threshold_down)),
            0  # Reversal (bounced up)
        )

        return labels

    def label_magnitude(self, df: pd.DataFrame) -> pd.Series:
        """
        Label the magnitude of future price movement.

        Returns the actual % return over look_forward_bars.

        Args:
            df: DataFrame with close prices

        Returns:
            Series with magnitude values (% return)
        """
        close = df["close"]
        look_forward = self.config.look_forward_bars

        # Calculate forward return percentage
        future_close = close.shift(-look_forward)
        magnitude = (future_close - close) / close * 100

        return magnitude

    def label_duration(self, df: pd.DataFrame) -> pd.Series:
        """
        Label the duration until target is hit.

        For each bar, counts bars until either:
        - Continuation target is hit (continuation_threshold * move achieved)
        - Reversal threshold is hit
        - Max look_forward_bars reached

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Series with duration values (bar count)
        """
        close = df["close"]
        look_forward = self.config.look_forward_bars

        # Get move characteristics
        move_pct, move_direction = self._calculate_move_pct(df, lookback=10)

        # Pre-calculate thresholds for each bar
        continuation_target = self.config.continuation_threshold * abs(move_pct) / 100
        reversal_target = self.config.reversal_threshold * abs(move_pct) / 100

        # Initialize duration array
        duration = pd.Series(look_forward, index=df.index, dtype=int)

        # Iterate through each bar to find when target is hit
        for i in range(len(df)):
            if pd.isna(move_direction.iloc[i]) or move_direction.iloc[i] == 0:
                continue

            current_close = close.iloc[i]
            direction = move_direction.iloc[i]
            cont_thresh = continuation_target.iloc[i]
            rev_thresh = reversal_target.iloc[i]

            if pd.isna(cont_thresh) or pd.isna(rev_thresh):
                continue

            # Look forward bar by bar
            for j in range(1, min(look_forward + 1, len(df) - i)):
                future_idx = i + j
                if future_idx >= len(df):
                    break

                future_close = close.iloc[future_idx]
                pct_change = (future_close - current_close) / current_close

                # Check if continuation target hit
                if direction > 0 and pct_change >= cont_thresh:
                    duration.iloc[i] = j
                    break
                elif direction < 0 and pct_change <= -cont_thresh:
                    duration.iloc[i] = j
                    break

                # Check if reversal target hit
                if direction > 0 and pct_change <= -rev_thresh:
                    duration.iloc[i] = j
                    break
                elif direction < 0 and pct_change >= rev_thresh:
                    duration.iloc[i] = j
                    break

        return duration

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all labels for the dataset.

        Main entry point for labeling. Returns DataFrame with columns:
        - 'move_detected': Boolean, True if significant move detected
        - 'direction': 0=reversal, 1=neutral, 2=continuation
        - 'magnitude': Actual % return over look_forward_bars
        - 'duration': Bars until target hit or max reached

        Only rows where move_detected = True should be used for training.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            DataFrame with label columns
        """
        # Ensure required columns exist
        required = ["high", "low", "close"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        labels_df = pd.DataFrame(index=df.index)

        # Detect significant moves
        labels_df["move_detected"] = self.detect_moves(df)

        # Create direction labels
        labels_df["direction"] = self.label_direction(df)

        # Create magnitude labels
        labels_df["magnitude"] = self.label_magnitude(df)

        # Create duration labels
        labels_df["duration"] = self.label_duration(df)

        return labels_df

    def prepare_labeled_data(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare labeled data for model training.

        Returns features and labels aligned and filtered for valid samples:
        - Only includes rows where move_detected = True
        - Drops rows with NaN features or labels

        Args:
            df: DataFrame with OHLCV columns for labeling
            features_df: DataFrame with feature columns (must have same index as df)

        Returns:
            Tuple of (X, y_direction, y_magnitude, y_duration)
            - X: Feature DataFrame
            - y_direction: Series with direction labels (0, 1, 2)
            - y_magnitude: Series with magnitude values
            - y_duration: Series with duration values
        """
        # Create labels
        labels_df = self.create_labels(df)

        # Filter to rows where move detected
        move_mask = labels_df["move_detected"]

        # Get valid indices (move detected AND no NaN in features AND no NaN in labels)
        valid_features = features_df.notna().all(axis=1)
        valid_labels = (
            labels_df["direction"].notna() &
            labels_df["magnitude"].notna() &
            labels_df["duration"].notna()
        )

        valid_mask = move_mask & valid_features & valid_labels

        # Extract aligned data
        X = features_df.loc[valid_mask].copy()
        y_direction = labels_df.loc[valid_mask, "direction"].copy()
        y_magnitude = labels_df.loc[valid_mask, "magnitude"].copy()
        y_duration = labels_df.loc[valid_mask, "duration"].copy()

        return X, y_direction, y_magnitude, y_duration

    def get_label_statistics(self, labels_df: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate statistics about the generated labels.

        Useful for understanding label distribution and quality.

        Args:
            labels_df: DataFrame from create_labels()

        Returns:
            Dictionary with label statistics
        """
        move_detected = labels_df["move_detected"]
        direction = labels_df.loc[move_detected, "direction"]
        magnitude = labels_df.loc[move_detected, "magnitude"]
        duration = labels_df.loc[move_detected, "duration"]

        stats = {
            "total_bars": len(labels_df),
            "moves_detected": move_detected.sum(),
            "move_detection_rate": move_detected.mean(),
            "direction_distribution": {
                "reversal": (direction == 0).sum(),
                "neutral": (direction == 1).sum(),
                "continuation": (direction == 2).sum(),
            },
            "direction_percentages": {
                "reversal": (direction == 0).mean() * 100,
                "neutral": (direction == 1).mean() * 100,
                "continuation": (direction == 2).mean() * 100,
            },
            "magnitude_stats": {
                "mean": magnitude.mean(),
                "std": magnitude.std(),
                "min": magnitude.min(),
                "max": magnitude.max(),
                "median": magnitude.median(),
            },
            "duration_stats": {
                "mean": duration.mean(),
                "std": duration.std(),
                "min": duration.min(),
                "max": duration.max(),
                "median": duration.median(),
            },
        }

        return stats


def create_momentum_labels(
    df: pd.DataFrame,
    config: MomentumLabelerConfig | None = None,
) -> pd.DataFrame:
    """
    Convenience function to create momentum labels.

    Args:
        df: DataFrame with OHLCV columns
        config: Optional labeler configuration

    Returns:
        DataFrame with columns: direction, magnitude, duration
    """
    labeler = MomentumLabeler(config)
    return labeler.create_labels(df)
