"""
Signal Generator - ML-based trading signal generation.

This module scans Shariah-compliant stocks and generates
buy/sell signals using trained ML models.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config.ibkr_config import ibkr_config
from data.ibkr_client import IBKRClient
from ml.features.technical import TechnicalFeatures
from ml.features.price_action import PriceActionFeatures
from ml.models.random_forest import RandomForestSignalModel
from ml.models.lightgbm_model import LightGBMSignalModel
from shariah.index_integration import load_shariah_universe, ShariahIndexIntegration
from .execution_engine import Signal


logger = logging.getLogger(__name__)


@dataclass
class SignalGeneratorConfig:
    """Configuration for signal generation."""

    # Universe settings
    max_symbols_to_scan: int = 20  # Limit for API rate limiting

    # Data settings
    lookback_days: int = 180  # Days of historical data for features (need 100+ for training)
    bar_size: str = "1 day"  # Bar size for historical data

    # Model settings
    model_type: str = "lightgbm"  # "random_forest" or "lightgbm"
    min_probability: float = 0.60  # Minimum probability for signal

    # Signal generation
    signal_cooldown_minutes: int = 30  # Don't re-signal same stock within cooldown
    max_signals_per_scan: int = 3  # Maximum signals per scan cycle

    # Feature settings
    use_technical_features: bool = True
    use_price_action_features: bool = True


class SignalGenerator:
    """
    Generates trading signals by scanning Shariah-compliant stocks
    and running ML predictions.
    """

    def __init__(
        self,
        ibkr_client: IBKRClient,
        config: SignalGeneratorConfig | None = None,
    ):
        """
        Initialize the signal generator.

        Args:
            ibkr_client: Connected IBKR client
            config: Signal generation configuration
        """
        self.ibkr_client = ibkr_client
        self.config = config or SignalGeneratorConfig()

        # Feature generators
        self.tech_features = TechnicalFeatures()
        self.price_features = PriceActionFeatures()

        # ML models (will be initialized/trained on first use)
        self._model: RandomForestSignalModel | LightGBMSignalModel | None = None
        self._model_trained = False

        # Shariah universe
        self._universe: ShariahIndexIntegration | None = None
        self._symbols: list[str] = []

        # Signal tracking (prevent repeated signals)
        self._last_signal_time: dict[str, datetime] = {}

        # Cache for historical data
        self._data_cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
        self._cache_ttl_minutes: int = 5

    async def initialize(self) -> None:
        """Initialize the signal generator."""
        logger.info("Initializing signal generator...")

        # Load Shariah universe
        self._universe = await load_shariah_universe()
        all_symbols = list(self._universe.get_all_compliant_symbols())

        # Limit symbols to scan (for API rate limiting)
        # Prioritize by some criteria (could be volume, market cap, etc.)
        self._symbols = all_symbols[:self.config.max_symbols_to_scan]

        logger.info(f"Loaded {len(self._symbols)} symbols to scan")

        # Initialize model
        if self.config.model_type == "lightgbm":
            self._model = LightGBMSignalModel()
        else:
            self._model = RandomForestSignalModel()

        logger.info("Signal generator initialized")

    async def _fetch_historical_data(self, symbol: str) -> pd.DataFrame | None:
        """
        Fetch historical data for a symbol with caching.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Check cache
        if symbol in self._data_cache:
            cached_data, cached_time = self._data_cache[symbol]
            if datetime.now() - cached_time < timedelta(minutes=self._cache_ttl_minutes):
                return cached_data

        try:
            # Fetch from IBKR
            bars = await self.ibkr_client.get_historical_data(
                symbol=symbol,
                duration=f"{self.config.lookback_days} D",
                bar_size=self.config.bar_size,
                what_to_show="ADJUSTED_LAST",
            )

            if bars is None or bars.empty:
                logger.warning(f"No historical data for {symbol}")
                return None

            # Cache the data
            self._data_cache[symbol] = (bars, datetime.now())

            return bars

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from price data.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with features
        """
        features = data.copy()

        if self.config.use_technical_features:
            tech_df = self.tech_features.calculate_all(data)
            # Only add new columns not already in features
            new_cols = [c for c in tech_df.columns if c not in features.columns]
            if new_cols:
                features = pd.concat([features, tech_df[new_cols]], axis=1)

        if self.config.use_price_action_features:
            price_df = self.price_features.calculate_all(data)
            # Only add new columns not already in features
            new_cols = [c for c in price_df.columns if c not in features.columns]
            if new_cols:
                features = pd.concat([features, price_df[new_cols]], axis=1)

        # Drop NaN rows (from indicator calculations)
        features = features.dropna()

        return features

    def _train_model_if_needed(self, features: pd.DataFrame) -> bool:
        """
        Train the model if not already trained.

        Uses the features DataFrame to train on historical data.

        Args:
            features: Features DataFrame with target

        Returns:
            True if model is ready for predictions
        """
        if self._model_trained:
            return True

        if len(features) < 100:
            logger.warning("Not enough data to train model")
            return False

        try:
            # Create target variable (next day return > 0)
            features = features.copy()
            features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
            features = features.dropna()

            # Split features and target
            exclude_cols = ['target', 'open', 'high', 'low', 'close', 'volume', 'date', 'timestamp', 'symbol', 'bar_count', 'vwap']
            feature_cols = [c for c in features.columns if c not in exclude_cols]
            X = features[feature_cols]
            y = features['target']

            # Train model
            self._model.train(X, y)
            self._model_trained = True

            logger.info(f"Model trained on {len(X)} samples")
            return True

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def _should_generate_signal(self, symbol: str) -> bool:
        """Check if we should generate a signal for this symbol."""
        if symbol not in self._last_signal_time:
            return True

        elapsed = datetime.now() - self._last_signal_time[symbol]
        return elapsed > timedelta(minutes=self.config.signal_cooldown_minutes)

    async def _analyze_symbol(self, symbol: str) -> Signal | None:
        """
        Analyze a single symbol and generate signal if appropriate.

        Args:
            symbol: Stock symbol

        Returns:
            Signal if generated, None otherwise
        """
        # Check cooldown
        if not self._should_generate_signal(symbol):
            return None

        # Fetch data
        data = await self._fetch_historical_data(symbol)
        if data is None or len(data) < 30:
            return None

        # Generate features
        features = self._generate_features(data)
        if len(features) < 50:
            return None

        # Train model if needed (uses accumulated data)
        if not self._train_model_if_needed(features):
            return None

        try:
            # Prepare latest features for prediction
            exclude_cols = ['target', 'open', 'high', 'low', 'close', 'volume', 'date', 'timestamp', 'symbol', 'bar_count', 'vwap']
            feature_cols = [c for c in features.columns if c not in exclude_cols]
            X_latest = features[feature_cols].iloc[[-1]]

            # Get prediction
            prediction = self._model.predict(X_latest)
            probability = self._model.predict_proba(X_latest)

            # probability is array of [prob_class_0, prob_class_1]
            prob_up = probability[0][1] if len(probability[0]) > 1 else probability[0][0]
            prob_down = 1 - prob_up

            # Determine signal
            if prob_up >= self.config.min_probability:
                signal_type = "buy"
                prob = prob_up
                confidence = "high" if prob >= 0.70 else "medium"
            elif prob_down >= self.config.min_probability:
                signal_type = "sell"
                prob = prob_down
                confidence = "high" if prob >= 0.70 else "medium"
            else:
                return None  # No strong signal

            # Create signal
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                probability=prob,
                confidence=confidence,
                model_name=self.config.model_type,
                features={
                    "rsi": features['rsi_14'].iloc[-1] if 'rsi_14' in features.columns else None,
                    "macd": features['macd'].iloc[-1] if 'macd' in features.columns else None,
                    "bb_pct": features['bb_pct'].iloc[-1] if 'bb_pct' in features.columns else None,
                },
            )

            # Update last signal time
            self._last_signal_time[symbol] = datetime.now()

            logger.info(f"Generated {signal_type.upper()} signal for {symbol} (prob: {prob:.2%})")

            return signal

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    async def generate_signals(self) -> list[Signal]:
        """
        Scan all symbols and generate trading signals.

        Returns:
            List of generated signals
        """
        if not self._symbols:
            await self.initialize()

        signals = []

        logger.info(f"Scanning {len(self._symbols)} symbols for signals...")

        for symbol in self._symbols:
            if len(signals) >= self.config.max_signals_per_scan:
                break

            try:
                signal = await self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)

                # Small delay to avoid API rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue

        logger.info(f"Generated {len(signals)} signals")

        return signals

    def get_callable(self) -> callable:
        """
        Return a synchronous callable for the execution engine.

        The execution engine expects a sync callable, but we need async.
        This returns a wrapper that runs the async method.

        Returns:
            Callable that returns list of signals
        """
        def _sync_generate() -> list[Signal]:
            # Create new event loop if needed
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We're already in an async context - use a future
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.generate_signals())
                    return future.result()
            else:
                return asyncio.run(self.generate_signals())

        return _sync_generate

    def get_status(self) -> dict[str, Any]:
        """Get signal generator status."""
        return {
            "symbols_loaded": len(self._symbols),
            "model_type": self.config.model_type,
            "model_trained": self._model_trained,
            "cached_symbols": len(self._data_cache),
            "signals_generated": len(self._last_signal_time),
            "min_probability": self.config.min_probability,
        }
