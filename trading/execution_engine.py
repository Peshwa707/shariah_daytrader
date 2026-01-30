"""
Execution Engine - Main Trading Loop.

This module orchestrates the entire trading process:
1. Receive signals from ML models
2. Validate Shariah compliance
3. Apply risk management
4. Execute orders via IBKR
5. Track positions and P&L
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Callable
import logging

import pandas as pd

from config.settings import settings
from data.ibkr_client import IBKRClient, MarketData
from shariah.compliance_engine import ComplianceEngine
from .order_manager import OrderManager, Order, OrderSide, OrderStatus
from .risk_manager import RiskManager, RiskLimits


logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal from ML model."""

    symbol: str
    signal_type: str  # "buy", "sell", "hold"
    probability: float
    confidence: str
    model_name: str
    features: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "probability": self.probability,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""

    # Trading hours (Eastern Time)
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)

    # Only trade during regular hours
    regular_hours_only: bool = True

    # Signal filtering
    min_confidence: str = "medium"  # "low", "medium", "high"
    min_probability: float = 0.55

    # Execution settings
    use_limit_orders: bool = False
    limit_offset_pct: float = 0.001  # 0.1% offset for limit orders

    # Auto-exit settings
    auto_exit_at_close: bool = True
    exit_minutes_before_close: int = 15

    # Paper trading mode
    paper_trading: bool = True


class ExecutionEngine:
    """
    Main trading execution engine.

    Coordinates:
    - Signal processing
    - Compliance checking
    - Risk management
    - Order execution
    - Position management
    """

    def __init__(
        self,
        ibkr_client: IBKRClient | None = None,
        compliance_engine: ComplianceEngine | None = None,
        order_manager: OrderManager | None = None,
        risk_manager: RiskManager | None = None,
        config: ExecutionConfig | None = None,
    ):
        """
        Initialize the execution engine.

        Args:
            ibkr_client: IBKR API client
            compliance_engine: Shariah compliance engine
            order_manager: Order management
            risk_manager: Risk management
            config: Execution configuration
        """
        self.config = config or ExecutionConfig()

        # Initialize components
        self.ibkr_client = ibkr_client
        self.compliance_engine = compliance_engine or ComplianceEngine()
        self.order_manager = order_manager or OrderManager(ibkr_client)
        self.risk_manager = risk_manager or RiskManager(100_000)

        # State tracking
        self._running = False
        self._positions: dict[str, dict[str, Any]] = {}
        self._positions_lock = asyncio.Lock()  # Protect positions dict from race conditions
        self._pending_signals: list[Signal] = []
        self._execution_history: list[dict[str, Any]] = []

        # Callbacks
        self._on_fill_callback: Callable[[Order], None] | None = None
        self._on_signal_callback: Callable[[Signal], None] | None = None

    async def start(self) -> None:
        """Start the execution engine."""
        logger.info("Starting execution engine...")

        # Connect to IBKR if available
        if self.ibkr_client:
            connected = await self.ibkr_client.connect()
            if not connected:
                logger.warning("IBKR connection failed, running in simulation mode")

        self._running = True
        logger.info("Execution engine started")

    async def stop(self) -> None:
        """Stop the execution engine."""
        logger.info("Stopping execution engine...")
        self._running = False

        # Cancel all open orders
        await self.order_manager.cancel_all_orders()

        # Disconnect from IBKR
        if self.ibkr_client:
            await self.ibkr_client.disconnect()

        logger.info("Execution engine stopped")

    async def process_signal(self, signal: Signal) -> dict[str, Any]:
        """
        Process a trading signal.

        Args:
            signal: Signal from ML model

        Returns:
            Execution result dictionary
        """
        result = {
            "signal": signal.to_dict(),
            "processed_at": datetime.now().isoformat(),
            "actions_taken": [],
            "errors": [],
        }

        try:
            # Step 1: Validate signal
            if not self._validate_signal(signal):
                result["errors"].append("Signal validation failed")
                return result

            # Step 2: Check trading hours
            if not self._is_trading_time():
                result["errors"].append("Outside trading hours")
                return result

            # Step 3: Check Shariah compliance
            compliance = self.compliance_engine.screen(
                signal.symbol,
                screening_level="INDEX_ONLY",
            )
            if not compliance.is_compliant:
                result["errors"].append(f"Shariah non-compliant: {compliance.reasons}")
                return result
            result["actions_taken"].append("Shariah compliance verified")

            # Step 4: Get current market data
            if self.ibkr_client and self.ibkr_client.is_connected:
                quote = await self.ibkr_client.get_realtime_quote(signal.symbol)
                current_price = quote.last_price if quote else None
            else:
                current_price = 100.0  # Placeholder for simulation
                result["actions_taken"].append("Using simulated price")

            if not current_price:
                result["errors"].append("Could not get current price")
                return result

            # Step 5: Calculate position size
            shares, dollar_amount = self.risk_manager.calculate_position_size(
                signal.symbol,
                current_price,
                atr=current_price * 0.02,  # Estimate ATR as 2% of price
            )

            if shares == 0:
                result["errors"].append("Position size too small")
                return result

            # Step 6: Run risk checks
            can_trade, failures = self.risk_manager.can_trade(
                signal.symbol,
                shares,
                current_price,
            )

            if not can_trade:
                result["errors"].extend(failures)
                return result
            result["actions_taken"].append("Risk checks passed")

            # Step 7: Determine action based on signal and current position
            current_position = self._positions.get(signal.symbol, {})
            current_qty = current_position.get("quantity", 0)

            order = None
            if signal.signal_type == "buy" and current_qty <= 0:
                # Open long position
                order = self.order_manager.create_market_order(
                    signal.symbol,
                    OrderSide.BUY,
                    shares,
                    signal_id=str(signal.timestamp),
                )
                result["actions_taken"].append(f"Created BUY order for {shares} shares")

            elif signal.signal_type == "sell" and current_qty > 0:
                # Close long position
                order = self.order_manager.create_market_order(
                    signal.symbol,
                    OrderSide.SELL,
                    current_qty,
                    signal_id=str(signal.timestamp),
                )
                result["actions_taken"].append(f"Created SELL order for {current_qty} shares")

            elif signal.signal_type == "hold":
                result["actions_taken"].append("Hold signal - no action taken")

            # Step 8: Submit order
            if order:
                submitted = await self.order_manager.submit_order(order)
                if submitted:
                    result["order_id"] = order.order_id
                    result["order_status"] = order.status.value
                    result["actions_taken"].append(f"Order submitted: {order.order_id}")
                else:
                    result["errors"].append("Order submission failed")

            # Record execution
            self._execution_history.append(result)

        except Exception as e:
            logger.exception(f"Error processing signal: {e}")
            result["errors"].append(str(e))

        return result

    def _validate_signal(self, signal: Signal) -> bool:
        """Validate signal meets criteria."""
        # Check confidence level
        confidence_levels = {"low": 1, "medium": 2, "high": 3}
        min_level = confidence_levels.get(self.config.min_confidence, 2)
        signal_level = confidence_levels.get(signal.confidence, 0)

        if signal_level < min_level:
            logger.info(f"Signal confidence too low: {signal.confidence}")
            return False

        # Check probability
        if signal.probability < self.config.min_probability:
            logger.info(f"Signal probability too low: {signal.probability}")
            return False

        # Check signal type
        if signal.signal_type not in ["buy", "sell", "hold"]:
            logger.warning(f"Invalid signal type: {signal.signal_type}")
            return False

        return True

    def _is_trading_time(self) -> bool:
        """Check if within trading hours."""
        if not self.config.regular_hours_only:
            return True

        now = datetime.now().time()

        # Adjust for exit before close
        close_time = self.config.market_close
        if self.config.auto_exit_at_close:
            close_minutes = close_time.hour * 60 + close_time.minute
            close_minutes -= self.config.exit_minutes_before_close
            close_time = time(close_minutes // 60, close_minutes % 60)

        return self.config.market_open <= now <= close_time

    async def run_trading_loop(
        self,
        signal_generator: Callable[[], list[Signal]],
        interval_seconds: int = 60,
    ) -> None:
        """
        Run the main trading loop.

        Args:
            signal_generator: Callable that returns list of signals
            interval_seconds: Seconds between signal checks
        """
        logger.info("Starting trading loop...")

        while self._running:
            try:
                # Get new signals
                signals = signal_generator()

                # Process each signal
                for signal in signals:
                    if self._on_signal_callback:
                        self._on_signal_callback(signal)

                    result = await self.process_signal(signal)

                    if result.get("errors"):
                        logger.warning(f"Signal processing errors: {result['errors']}")

                # Check for auto-exit near close
                if self.config.auto_exit_at_close:
                    await self._check_auto_exit()

                # Wait for next iteration
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.exception(f"Error in trading loop: {e}")
                await asyncio.sleep(5)

        logger.info("Trading loop ended")

    async def _check_auto_exit(self) -> None:
        """Check if we need to exit positions before market close."""
        now = datetime.now().time()
        close_time = self.config.market_close

        # Calculate exit time
        close_minutes = close_time.hour * 60 + close_time.minute
        exit_minutes = close_minutes - self.config.exit_minutes_before_close
        exit_time = time(exit_minutes // 60, exit_minutes % 60)

        if now >= exit_time:
            logger.info("Auto-exit triggered - closing all positions")
            await self.close_all_positions()

    async def close_all_positions(self) -> list[dict[str, Any]]:
        """Close all open positions."""
        results = []

        async with self._positions_lock:
            positions_snapshot = list(self._positions.items())

        for symbol, position in positions_snapshot:
            if position.get("quantity", 0) > 0:
                order = self.order_manager.create_market_order(
                    symbol,
                    OrderSide.SELL,
                    position["quantity"],
                )
                submitted = await self.order_manager.submit_order(order)
                results.append({
                    "symbol": symbol,
                    "order_id": order.order_id,
                    "submitted": submitted,
                })

        return results

    async def update_position(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        current_price: float | None = None,
    ) -> None:
        """Update position tracking (async for thread safety)."""
        async with self._positions_lock:
            if quantity == 0:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": avg_price,
                    "current_price": current_price,
                    "updated_at": datetime.now(),
                }

        # Update risk manager
        self.risk_manager.update_position(
            symbol,
            quantity,
            avg_price,
            current_price or avg_price,
        )

    def get_positions(self) -> dict[str, dict[str, Any]]:
        """Get all current positions."""
        return self._positions.copy()

    def get_execution_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent execution history."""
        return self._execution_history[-limit:]

    def get_status(self) -> dict[str, Any]:
        """Get engine status."""
        return {
            "running": self._running,
            "paper_trading": self.config.paper_trading,
            "ibkr_connected": self.ibkr_client.is_connected if self.ibkr_client else False,
            "positions": len(self._positions),
            "pending_signals": len(self._pending_signals),
            "executions_today": len([
                e for e in self._execution_history
                if e.get("processed_at", "").startswith(datetime.now().date().isoformat())
            ]),
            "risk_summary": self.risk_manager.get_portfolio_risk_summary(),
            "order_stats": self.order_manager.get_statistics(),
        }

    def set_on_fill_callback(self, callback: Callable[[Order], None]) -> None:
        """Set callback for order fills."""
        self._on_fill_callback = callback

    def set_on_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """Set callback for received signals."""
        self._on_signal_callback = callback


# Example usage
async def main():
    """Example usage of execution engine."""
    # Create engine
    engine = ExecutionEngine(
        config=ExecutionConfig(paper_trading=True),
    )

    # Start engine
    await engine.start()

    # Create a test signal
    signal = Signal(
        symbol="AAPL",
        signal_type="buy",
        probability=0.65,
        confidence="medium",
        model_name="random_forest",
    )

    # Process signal
    result = await engine.process_signal(signal)
    print(f"Execution result: {result}")

    # Get status
    status = engine.get_status()
    print(f"Engine status: {status}")

    # Stop engine
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
