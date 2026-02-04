#!/usr/bin/env python3
"""
Rapid Scalping Strategy - ML-Enhanced Momentum Trading

Exploits short-term price movements using ML-based momentum prediction.
Uses MomentumContinuationModel for intelligent entry/exit decisions
with model-driven stop loss and take profit levels.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.ibkr_client import IBKRClient
from data.storage import DatabaseManager
from ml.features.momentum_features import MomentumContinuationFeatures
from ml.models.momentum_continuation_model import (
    MomentumContinuationModel,
    MomentumPredictionResult,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "momentum_model.pkl"


@dataclass
class ScalperConfig:
    """Scalping strategy configuration."""

    # Position sizing
    position_size_usd: float = 5000.0  # $5K per trade
    max_concurrent_positions: int = 3

    # Profit/Loss targets (defaults - can be overridden by model)
    profit_target_pct: float = 0.005  # 0.5% profit target
    stop_loss_pct: float = 0.003  # 0.3% stop loss

    # Timing
    scan_interval_seconds: int = 10  # Scan every 10 seconds
    max_hold_minutes: int = 5  # Force exit after 5 min

    # Entry criteria (legacy - used when model not available)
    min_momentum_pct: float = 0.002  # 0.2% price move to trigger entry
    min_volume_ratio: float = 1.5  # Volume 1.5x average

    # ML Model settings
    use_ml_model: bool = True  # Use ML model for entry decisions
    model_path: Path | None = None  # Path to trained model
    min_momentum_score: float = 65.0  # Minimum momentum score (0-100) for entry
    use_model_stops: bool = True  # Use model-suggested stop/take-profit
    lookback_bars: int = 100  # Historical bars for feature calculation

    # Symbols to scalp (high liquidity, volatile)
    symbols: list = None

    def __post_init__(self):
        if self.symbols is None:
            # Default to liquid, volatile stocks
            self.symbols = ['AAPL', 'NVDA', 'TSLA', 'AMD', 'META', 'GOOGL', 'AMZN', 'MSFT']
        if self.model_path is None:
            self.model_path = DEFAULT_MODEL_PATH


@dataclass
class Position:
    """Active scalping position."""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    target_price: float
    stop_price: float

    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.entry_time).total_seconds()


class Scalper:
    """
    ML-Enhanced Scalping Strategy.

    Uses MomentumContinuationModel for intelligent entry decisions
    based on momentum score, direction confidence, and expected returns.
    Falls back to basic momentum when model not available.
    """

    def __init__(self, config: ScalperConfig = None):
        self.config = config or ScalperConfig()
        self.client: IBKRClient = None
        self.db: DatabaseManager = DatabaseManager()
        self.positions: dict[str, Position] = {}
        self.last_prices: dict[str, float] = {}
        self.trades_today: int = 0
        self.profits_today: float = 0.0
        self.running: bool = False

        # ML Model components
        self.momentum_model: MomentumContinuationModel | None = None
        self.feature_calculator = MomentumContinuationFeatures()
        self._historical_cache: dict[str, pd.DataFrame] = {}

        # Load ML model if configured
        if self.config.use_ml_model:
            self._load_model()

        logger.info("Scalper initialized (ML model: %s)",
                    "loaded" if self.momentum_model else "not available")

    def _load_model(self) -> bool:
        """Load the trained momentum model."""
        model_path = self.config.model_path
        if model_path and model_path.exists():
            try:
                self.momentum_model = MomentumContinuationModel.load(model_path)
                logger.info(f"Loaded momentum model from {model_path}")
                if self.momentum_model.metrics:
                    logger.info(f"Model metrics: {self.momentum_model.metrics}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load momentum model: {e}")
                self.momentum_model = None
        else:
            logger.warning(f"Model file not found: {model_path}")
        return False

    async def connect(self) -> bool:
        """Connect to IBKR."""
        self.client = IBKRClient()
        await self.client.connect()
        return self.client.is_connected

    async def disconnect(self):
        """Disconnect from IBKR."""
        if self.client:
            await self.client.disconnect()

    async def get_current_price(self, symbol: str) -> float | None:
        """Get current price for a symbol."""
        try:
            quote = await self.client.get_realtime_quote(symbol)
            if quote and quote.last_price:
                return quote.last_price
            elif quote and quote.bid and quote.ask:
                return (quote.bid + quote.ask) / 2
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        return None

    async def _get_historical_bars(self, symbol: str) -> pd.DataFrame | None:
        """
        Get historical bars for feature calculation.

        Uses caching to avoid excessive API calls.
        """
        # Check cache freshness (refresh every 5 minutes for intraday)
        cache_key = symbol
        if cache_key in self._historical_cache:
            cached = self._historical_cache[cache_key]
            if len(cached) > 0:
                # Use cached if less than 5 minutes old
                last_time = cached.index[-1] if hasattr(cached.index, '__getitem__') else None
                if last_time and (datetime.now() - last_time).total_seconds() < 300:
                    return cached

        try:
            # Fetch intraday bars (5-minute bars for scalping)
            df = await self.client.get_historical_data(
                symbol=symbol,
                duration="1 D",  # 1 day of data
                bar_size="5 mins",  # 5-minute bars
                what_to_show="TRADES",
                use_rth=True,
            )

            if df is not None and len(df) >= 20:
                # Set timestamp as index
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                self._historical_cache[cache_key] = df
                return df

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")

        return None

    async def _get_ml_prediction(self, symbol: str) -> MomentumPredictionResult | None:
        """
        Get ML model prediction for a symbol.

        Returns MomentumPredictionResult with direction, magnitude,
        duration, momentum score, and suggested stops.
        """
        if not self.momentum_model or not self.momentum_model.is_trained:
            return None

        # Get historical data
        df = await self._get_historical_bars(symbol)
        if df is None or len(df) < self.config.lookback_bars:
            logger.debug(f"Insufficient data for {symbol} ({len(df) if df is not None else 0} bars)")
            return None

        try:
            # Calculate features for most recent bar
            features_df = self.feature_calculator.calculate_for_prediction(
                df.tail(self.config.lookback_bars)
            )

            # Get prediction
            predictions = self.momentum_model.predict(features_df, symbol=symbol)
            if predictions:
                return predictions[0]

        except Exception as e:
            logger.error(f"ML prediction failed for {symbol}: {e}")

        return None

    def calculate_momentum(self, symbol: str, current_price: float) -> float:
        """Calculate momentum as percentage change from last price."""
        if symbol not in self.last_prices:
            self.last_prices[symbol] = current_price
            return 0.0

        last_price = self.last_prices[symbol]
        if last_price == 0:
            return 0.0

        momentum = (current_price - last_price) / last_price
        self.last_prices[symbol] = current_price
        return momentum

    async def check_entry(self, symbol: str) -> tuple[bool, MomentumPredictionResult | None]:
        """
        Check if we should enter a position.

        Returns:
            Tuple of (should_enter, prediction) where prediction is the
            MomentumPredictionResult if using ML model, None otherwise.
        """
        # Skip if already have position
        if symbol in self.positions:
            return False, None

        # Skip if max positions reached
        if len(self.positions) >= self.config.max_concurrent_positions:
            return False, None

        # Get current price
        price = await self.get_current_price(symbol)
        if not price:
            return False, None

        # Try ML model first if available
        if self.momentum_model and self.momentum_model.is_trained:
            prediction = await self._get_ml_prediction(symbol)
            if prediction:
                # Check momentum score threshold
                if prediction.momentum_score >= self.config.min_momentum_score:
                    # Only enter on continuation signals
                    if prediction.direction == "continuation":
                        logger.info(
                            f"ML ENTRY SIGNAL: {symbol} "
                            f"score={prediction.momentum_score:.1f} "
                            f"direction={prediction.direction} "
                            f"prob={prediction.direction_probability:.2%} "
                            f"magnitude={prediction.expected_magnitude:.2%} "
                            f"price=${price:.2f}"
                        )
                        return True, prediction
                    else:
                        logger.debug(
                            f"ML skip {symbol}: direction={prediction.direction} "
                            f"(score={prediction.momentum_score:.1f})"
                        )
                return False, None

        # Fallback to basic momentum
        momentum = self.calculate_momentum(symbol, price)
        if momentum >= self.config.min_momentum_pct:
            logger.info(f"BASIC ENTRY SIGNAL: {symbol} momentum={momentum:.4%} price=${price:.2f}")
            return True, None

        return False, None

    async def enter_position(
        self,
        symbol: str,
        prediction: MomentumPredictionResult | None = None,
    ) -> bool:
        """
        Enter a new scalping position.

        Args:
            symbol: Stock ticker symbol
            prediction: ML model prediction (optional, for model-driven stops)
        """
        price = await self.get_current_price(symbol)
        if not price:
            return False

        # Calculate quantity based on position size
        quantity = int(self.config.position_size_usd / price)
        if quantity < 1:
            logger.warning(f"Position size too small for {symbol} at ${price:.2f}")
            return False

        # Calculate targets using model suggestions or config defaults
        if prediction and self.config.use_model_stops:
            # Use model-suggested stop/take-profit
            stop_loss_pct = prediction.suggested_stop_loss_pct / 100  # Model returns %
            take_profit_pct = prediction.suggested_take_profit_pct / 100
            target_price = price * (1 + take_profit_pct)
            stop_price = price * (1 - stop_loss_pct)
            logger.info(f"Using ML-suggested stops (SL: {stop_loss_pct:.2%}, TP: {take_profit_pct:.2%})")
        else:
            # Use config defaults
            target_price = price * (1 + self.config.profit_target_pct)
            stop_price = price * (1 - self.config.stop_loss_pct)

        # Place order
        logger.info(f"ENTERING: BUY {quantity} {symbol} @ ${price:.2f}")
        logger.info(f"  Target: ${target_price:.2f} (+{((target_price/price)-1):.2%})")
        logger.info(f"  Stop: ${stop_price:.2f} (-{(1-(stop_price/price)):.2%})")

        result = await self.client.place_order(
            symbol=symbol,
            action='BUY',
            quantity=quantity,
            order_type='MKT'
        )

        if result and result.get('filled', 0) > 0:
            actual_price = result.get('avg_fill_price', price)
            target_price = actual_price * (1 + self.config.profit_target_pct)
            stop_price = actual_price * (1 - self.config.stop_loss_pct)

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=actual_price,
                entry_time=datetime.now(),
                target_price=target_price,
                stop_price=stop_price,
            )

            # Record entry trade to database
            try:
                self.db.save_trade({
                    "symbol": symbol,
                    "trade_date": datetime.now(),
                    "order_type": "market",
                    "side": "buy",
                    "quantity": quantity,
                    "price": actual_price,
                    "total_value": quantity * actual_price,
                    "notes": f"SCALPER ENTRY - Target: ${target_price:.2f}, Stop: ${stop_price:.2f}",
                })
                logger.info(f"Trade recorded to DB: BUY {quantity} {symbol}")
            except Exception as e:
                logger.error(f"Failed to record trade: {e}")

            logger.info(f"FILLED: {quantity} {symbol} @ ${actual_price:.2f}")
            return True
        else:
            logger.error(f"Order failed for {symbol}")
            return False

    async def check_exit(self, position: Position) -> str | None:
        """Check if we should exit a position. Returns exit reason or None."""
        price = await self.get_current_price(position.symbol)
        if not price:
            return None

        # Check profit target
        if price >= position.target_price:
            return 'PROFIT_TARGET'

        # Check stop loss
        if price <= position.stop_price:
            return 'STOP_LOSS'

        # Check max hold time
        if position.age_seconds >= self.config.max_hold_minutes * 60:
            return 'MAX_TIME'

        return None

    async def exit_position(self, symbol: str, reason: str) -> bool:
        """Exit a position."""
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        current_price = await self.get_current_price(symbol)

        logger.info(f"EXITING: SELL {position.quantity} {symbol} @ ${current_price:.2f} ({reason})")

        result = await self.client.place_order(
            symbol=symbol,
            action='SELL',
            quantity=position.quantity,
            order_type='MKT'
        )

        if result:
            exit_price = result.get('avg_fill_price', current_price)
            pnl = (exit_price - position.entry_price) * position.quantity
            pnl_pct = (exit_price - position.entry_price) / position.entry_price

            self.trades_today += 1
            self.profits_today += pnl

            # Record exit trade to database with P&L
            try:
                self.db.save_trade({
                    "symbol": symbol,
                    "trade_date": datetime.now(),
                    "order_type": "market",
                    "side": "sell",
                    "quantity": position.quantity,
                    "price": exit_price,
                    "total_value": position.quantity * exit_price,
                    "realized_pnl": pnl,
                    "notes": f"SCALPER EXIT ({reason}) - Entry: ${position.entry_price:.2f}, P&L: {pnl_pct:.2%}",
                })
                logger.info(f"Trade recorded to DB: SELL {position.quantity} {symbol}, P&L: ${pnl:.2f}")
            except Exception as e:
                logger.error(f"Failed to record trade: {e}")

            logger.info(f"CLOSED: {symbol} P&L: ${pnl:.2f} ({pnl_pct:.2%})")
            logger.info(f"  Daily Stats: {self.trades_today} trades, ${self.profits_today:.2f} total P&L")

            del self.positions[symbol]
            return True

        return False

    async def run_cycle(self):
        """Run one scalping cycle."""
        # Check exits first
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            exit_reason = await self.check_exit(position)
            if exit_reason:
                await self.exit_position(symbol, exit_reason)

        # Check entries
        for symbol in self.config.symbols:
            should_enter, prediction = await self.check_entry(symbol)
            if should_enter:
                await self.enter_position(symbol, prediction)

    async def run(self):
        """Run the scalping strategy."""
        logger.info("=" * 60)
        logger.info("SCALPER STARTING")
        logger.info(f"  Symbols: {self.config.symbols}")
        logger.info(f"  Position Size: ${self.config.position_size_usd:,.0f}")

        # ML Model status
        if self.momentum_model and self.momentum_model.is_trained:
            logger.info("  Mode: ML-ENHANCED")
            logger.info(f"  Min Momentum Score: {self.config.min_momentum_score}")
            logger.info(f"  Model Stops: {'enabled' if self.config.use_model_stops else 'disabled'}")
        else:
            logger.info("  Mode: BASIC (no ML model)")
            logger.info(f"  Profit Target: {self.config.profit_target_pct:.2%}")
            logger.info(f"  Stop Loss: {self.config.stop_loss_pct:.2%}")

        logger.info(f"  Scan Interval: {self.config.scan_interval_seconds}s")
        logger.info("=" * 60)

        if not await self.connect():
            logger.error("Failed to connect to IBKR")
            return

        logger.info("Connected to IBKR - Starting scalping loop...")

        self.running = True
        try:
            while self.running:
                await self.run_cycle()
                await asyncio.sleep(self.config.scan_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Scalper stopping...")

        finally:
            # Exit all positions on shutdown
            for symbol in list(self.positions.keys()):
                await self.exit_position(symbol, 'SHUTDOWN')

            await self.disconnect()

            logger.info("=" * 60)
            logger.info("SCALPER STOPPED")
            logger.info(f"  Total Trades: {self.trades_today}")
            logger.info(f"  Total P&L: ${self.profits_today:.2f}")
            logger.info("=" * 60)


async def train_momentum_model(
    symbols: list[str],
    output_path: Path = DEFAULT_MODEL_PATH,
    duration: str = "6 M",
) -> MomentumContinuationModel | None:
    """
    Train a new momentum model using historical data.

    Args:
        symbols: List of symbols to train on
        output_path: Path to save the trained model
        duration: Historical data duration

    Returns:
        Trained model or None if failed
    """
    from ml.labeling.momentum_labeler import create_momentum_labels

    logger.info(f"Training momentum model on {len(symbols)} symbols...")

    client = IBKRClient()
    if not await client.connect():
        logger.error("Failed to connect to IBKR for training data")
        return None

    try:
        feature_calculator = MomentumContinuationFeatures()
        all_features = []
        all_directions = []
        all_magnitudes = []
        all_durations = []

        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            df = await client.get_historical_data(
                symbol=symbol,
                duration=duration,
                bar_size="5 mins",
                what_to_show="TRADES",
            )

            if df is None or len(df) < 200:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # Calculate features
            features_df = feature_calculator.calculate_all(df)

            # Create labels
            labels_df = create_momentum_labels(df)

            # Align features and labels
            valid_idx = features_df.notna().all(axis=1) & labels_df.notna().all(axis=1)
            features_df = features_df[valid_idx]
            labels_df = labels_df[valid_idx]

            # Get feature columns only
            exclude_cols = {"open", "high", "low", "close", "volume", "timestamp", "symbol", "_atr"}
            feature_cols = [c for c in features_df.columns if c not in exclude_cols]

            all_features.append(features_df[feature_cols])
            all_directions.append(labels_df["direction"])
            all_magnitudes.append(labels_df["magnitude"])
            all_durations.append(labels_df["duration"])

        if not all_features:
            logger.error("No valid training data collected")
            return None

        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y_direction = pd.concat(all_directions, ignore_index=True)
        y_magnitude = pd.concat(all_magnitudes, ignore_index=True)
        y_duration = pd.concat(all_durations, ignore_index=True)

        logger.info(f"Training on {len(X)} samples...")

        # Train model
        model = MomentumContinuationModel()
        metrics = model.train(X, y_direction, y_magnitude, y_duration)

        logger.info(f"Training complete: {metrics}")

        # Save model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
        logger.info(f"Model saved to {output_path}")

        return model

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None
    finally:
        await client.disconnect()


async def main():
    """Run the scalper."""
    import argparse

    parser = argparse.ArgumentParser(description="ML-Enhanced Scalper")
    parser.add_argument("--train", action="store_true", help="Train a new model before running")
    parser.add_argument("--train-only", action="store_true", help="Only train, don't run scalper")
    parser.add_argument("--no-ml", action="store_true", help="Run without ML model")
    args = parser.parse_args()

    # Default symbols
    symbols = ['AAPL', 'NVDA', 'TSLA', 'AMD', 'META', 'GOOGL', 'AMZN', 'MSFT']

    # Train if requested
    if args.train or args.train_only:
        model = await train_momentum_model(symbols)
        if args.train_only:
            return

    config = ScalperConfig(
        position_size_usd=5000.0,
        profit_target_pct=0.005,  # 0.5%
        stop_loss_pct=0.003,  # 0.3%
        scan_interval_seconds=10,
        max_hold_minutes=5,
        max_concurrent_positions=3,
        use_ml_model=not args.no_ml,
        min_momentum_score=65.0,
    )

    scalper = Scalper(config)
    await scalper.run()


if __name__ == "__main__":
    asyncio.run(main())
