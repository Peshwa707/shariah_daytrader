"""
Market Regime Detector.

Classifies current market conditions into regimes based on VIX levels
and SPY trend direction. Stores results in the database for ML model
adaptation and performance tracking.
"""
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional
import logging

import yfinance as yf

from data.storage import DatabaseManager
from config.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class MarketRegimeResult:
    """Result of market regime detection."""
    regime_type: str  # high_volatility, low_volatility, trending_up, trending_down, sideways
    vix_level: float
    spy_return_20d: float
    confidence: float
    regime_date: date


class RegimeDetector:
    """
    Detects market regime based on VIX and SPY trend.

    Regime Classification Logic:
    - if VIX > high_threshold: "high_volatility"
    - elif VIX < low_threshold: "low_volatility"
    - elif SPY 20d return > 2%: "trending_up"
    - elif SPY 20d return < -2%: "trending_down"
    - else: "sideways"
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        vix_high_threshold: Optional[float] = None,
        vix_low_threshold: Optional[float] = None,
    ):
        """
        Initialize the regime detector.

        Args:
            db_manager: Database manager instance. Creates new one if not provided.
            vix_high_threshold: VIX threshold for high volatility regime.
                               Defaults to settings.regime_vix_high_threshold or 25.0.
            vix_low_threshold: VIX threshold for low volatility regime.
                              Defaults to settings.regime_vix_low_threshold or 15.0.
        """
        self.db_manager = db_manager or DatabaseManager()

        # Get thresholds from settings or use defaults
        self.vix_high_threshold = vix_high_threshold or getattr(
            settings.go_live, 'regime_vix_high_threshold', 25.0
        )
        self.vix_low_threshold = vix_low_threshold or getattr(
            settings.go_live, 'regime_vix_low_threshold', 15.0
        )

        # SPY return thresholds (percentage)
        self.spy_trend_threshold = 2.0  # +/- 2% for trending classification

    def _fetch_vix_data(self) -> Optional[float]:
        """
        Fetch current VIX level from yfinance.

        Returns:
            Current VIX level or None if fetch fails.
        """
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")

            if hist.empty:
                logger.warning("VIX data is empty")
                return None

            return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Failed to fetch VIX data: {e}")
            return None

    def _fetch_spy_return_20d(self) -> Optional[float]:
        """
        Fetch SPY 20-day return for trend detection.

        Returns:
            20-day return as percentage or None if fetch fails.
        """
        try:
            spy = yf.Ticker("SPY")
            # Fetch 25 days to ensure we have at least 20 trading days
            hist = spy.history(period="1mo")

            if len(hist) < 20:
                logger.warning(f"Insufficient SPY data: {len(hist)} days")
                return None

            # Calculate 20-day return
            current_price = float(hist['Close'].iloc[-1])
            price_20d_ago = float(hist['Close'].iloc[-20])

            return_20d = ((current_price - price_20d_ago) / price_20d_ago) * 100
            return return_20d
        except Exception as e:
            logger.error(f"Failed to fetch SPY data: {e}")
            return None

    def _classify_regime(
        self,
        vix_level: float,
        spy_return_20d: float,
    ) -> tuple[str, float]:
        """
        Classify market regime based on VIX and SPY trend.

        Args:
            vix_level: Current VIX level.
            spy_return_20d: 20-day SPY return percentage.

        Returns:
            Tuple of (regime_type, confidence).
        """
        # High volatility takes precedence
        if vix_level > self.vix_high_threshold:
            # Confidence based on how far above threshold
            excess = vix_level - self.vix_high_threshold
            confidence = min(0.5 + (excess / 20), 1.0)
            return "high_volatility", confidence

        # Low volatility
        if vix_level < self.vix_low_threshold:
            # Confidence based on how far below threshold
            deficit = self.vix_low_threshold - vix_level
            confidence = min(0.5 + (deficit / 10), 1.0)
            return "low_volatility", confidence

        # Trend-based classification for normal volatility
        if spy_return_20d > self.spy_trend_threshold:
            # Strong uptrend
            excess = spy_return_20d - self.spy_trend_threshold
            confidence = min(0.5 + (excess / 5), 0.9)
            return "trending_up", confidence

        if spy_return_20d < -self.spy_trend_threshold:
            # Strong downtrend
            deficit = abs(spy_return_20d) - self.spy_trend_threshold
            confidence = min(0.5 + (deficit / 5), 0.9)
            return "trending_down", confidence

        # Sideways market
        # Confidence is higher when return is closer to zero
        confidence = max(0.5, 0.8 - abs(spy_return_20d) / 5)
        return "sideways", confidence

    def detect_regime(self, regime_date: Optional[date] = None) -> Optional[MarketRegimeResult]:
        """
        Detect current market regime and save to database.

        Args:
            regime_date: Date for the regime. Defaults to today.

        Returns:
            MarketRegimeResult or None if detection fails.
        """
        regime_date = regime_date or date.today()

        logger.info(f"Detecting market regime for {regime_date}")

        # Fetch market data
        vix_level = self._fetch_vix_data()
        if vix_level is None:
            logger.error("Cannot detect regime: VIX data unavailable")
            return None

        spy_return_20d = self._fetch_spy_return_20d()
        if spy_return_20d is None:
            logger.error("Cannot detect regime: SPY data unavailable")
            return None

        # Classify regime
        regime_type, confidence = self._classify_regime(vix_level, spy_return_20d)

        logger.info(
            f"Regime detected: {regime_type} "
            f"(VIX={vix_level:.2f}, SPY_20d={spy_return_20d:.2f}%, confidence={confidence:.2f})"
        )

        # Create result
        result = MarketRegimeResult(
            regime_type=regime_type,
            vix_level=vix_level,
            spy_return_20d=spy_return_20d,
            confidence=confidence,
            regime_date=regime_date,
        )

        # Save to database
        try:
            now = datetime.now()
            period_start = now - timedelta(days=20)

            self.db_manager.save_market_regime(
                regime_date=regime_date,
                regime_type=regime_type,
                regime_confidence=confidence,
                period_start=period_start,
                period_end=now,
                vix_level=vix_level,
                sp500_return_pct=spy_return_20d,
                additional_metrics={
                    "vix_high_threshold": self.vix_high_threshold,
                    "vix_low_threshold": self.vix_low_threshold,
                    "spy_trend_threshold": self.spy_trend_threshold,
                },
            )
            logger.info(f"Regime saved to database: {regime_type}")
        except Exception as e:
            logger.error(f"Failed to save regime to database: {e}")
            # Still return the result even if save fails

        return result

    def get_current_regime(self) -> Optional[dict]:
        """
        Get the most recent market regime from database.

        Returns:
            Market regime dict or None.
        """
        return self.db_manager.get_current_regime()

    def get_regime_history(self, days: int = 30) -> list[dict]:
        """
        Get market regime history.

        Args:
            days: Number of days to look back.

        Returns:
            List of regime dicts.
        """
        return self.db_manager.get_regime_history(days=days)
