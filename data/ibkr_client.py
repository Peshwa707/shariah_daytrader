"""
Interactive Brokers API Client.

This module provides a wrapper around the ib_async library for
connecting to Interactive Brokers TWS/Gateway and fetching market data.

Setup Requirements:
1. Install TWS or IB Gateway
2. Enable "Socket Clients" in API settings
3. Add 127.0.0.1 to trusted IPs
4. Use port 7497 for paper trading, 7496 for live

Note: ib_async is the successor to ib_insync and provides
asyncio-native IBKR API access.

Features:
- Automatic reconnection with exponential backoff
- Connection health monitoring
- Graceful degradation during reconnection
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta
from typing import Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

import pandas as pd

# ib_async imports (these will fail if not installed - that's expected)
try:
    from ib_async import IB, Stock, Contract, BarData, Ticker
    from ib_async import MarketOrder, LimitOrder, StopOrder
    from ib_async import util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    IB = None
    Stock = None
    Contract = None
    MarketOrder = None
    LimitOrder = None
    StopOrder = None

from config.ibkr_config import ibkr_config


logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state for IBKR client."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    FAILED = auto()  # Circuit breaker tripped


# Type alias for connection state callback
ConnectionCallback = Callable[["ConnectionState", str | None], None]


@dataclass
class MarketData:
    """Container for market data."""

    symbol: str
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    volume: int | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "volume": self.volume,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Position:
    """Container for position information."""

    symbol: str
    quantity: float
    avg_cost: float
    market_value: float | None = None
    unrealized_pnl: float | None = None
    realized_pnl: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
        }


class IBKRClient:
    """
    Interactive Brokers API client wrapper.

    Provides methods for:
    - Connecting to TWS/Gateway
    - Fetching historical data
    - Streaming real-time quotes
    - Retrieving account positions
    - Placing orders (paper trading)
    - Automatic reconnection with exponential backoff
    """

    def __init__(self, config: ibkr_config.__class__ | None = None):
        """
        Initialize the IBKR client.

        Args:
            config: IBKR configuration (uses global config if not provided)
        """
        if not IB_AVAILABLE:
            raise ImportError(
                "ib_async is not installed. Install with: pip install ib-async"
            )

        self.config = config or ibkr_config
        self._ib: IB | None = None
        self._connected = False
        self._contracts_cache: dict[str, Contract] = {}

        # Reconnection state
        self._connection_state = ConnectionState.DISCONNECTED
        self._reconnect_task: asyncio.Task | None = None
        self._reconnect_attempts = 0
        self._last_disconnect_time: datetime | None = None
        self._shutdown_requested = False

        # Callbacks for connection state changes
        self._connection_callbacks: list[ConnectionCallback] = []

        # Lock for connection operations
        self._connection_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self._connected and self._ib is not None and self._ib.isConnected()

    @property
    def connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self._connection_state

    @property
    def is_reconnecting(self) -> bool:
        """Check if currently attempting to reconnect."""
        return self._connection_state == ConnectionState.RECONNECTING

    def add_connection_callback(self, callback: ConnectionCallback) -> None:
        """
        Register a callback for connection state changes.

        Args:
            callback: Function called with (new_state, error_message)
        """
        self._connection_callbacks.append(callback)

    def remove_connection_callback(self, callback: ConnectionCallback) -> None:
        """Remove a connection callback."""
        if callback in self._connection_callbacks:
            self._connection_callbacks.remove(callback)

    def _notify_state_change(self, state: ConnectionState, message: str | None = None) -> None:
        """Notify all callbacks of state change."""
        self._connection_state = state
        for callback in self._connection_callbacks:
            try:
                callback(state, message)
            except Exception as e:
                logger.error(f"Connection callback error: {e}")

    async def connect(self) -> bool:
        """
        Connect to TWS/Gateway.

        Returns:
            True if connection successful
        """
        async with self._connection_lock:
            if self.is_connected:
                logger.info("Already connected to IBKR")
                return True

            self._shutdown_requested = False
            self._notify_state_change(ConnectionState.CONNECTING)

            self._ib = IB()

            try:
                await self._ib.connectAsync(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=self.config.client_id,
                    timeout=self.config.timeout,
                    readonly=self.config.readonly,
                )

                self._connected = True
                self._reconnect_attempts = 0
                self._notify_state_change(ConnectionState.CONNECTED)
                logger.info(
                    f"Connected to IBKR at {self.config.host}:{self.config.port} "
                    f"(mode: {self.config.mode})"
                )

                # Set market data type
                self._ib.reqMarketDataType(self.config.market_data_type)

                # Register disconnect handler for auto-reconnect
                if self.config.auto_reconnect:
                    self._ib.disconnectedEvent += self._on_disconnect

                return True

            except Exception as e:
                logger.error(f"Failed to connect to IBKR: {e}")
                self._connected = False
                self._notify_state_change(ConnectionState.DISCONNECTED, str(e))
                return False

    def _on_disconnect(self) -> None:
        """Handle unexpected disconnection - trigger reconnection."""
        self._connected = False
        self._last_disconnect_time = datetime.now()

        if self._shutdown_requested:
            logger.info("Disconnect after shutdown request - not reconnecting")
            self._notify_state_change(ConnectionState.DISCONNECTED)
            return

        if not self.config.auto_reconnect:
            logger.warning("IBKR disconnected - auto-reconnect disabled")
            self._notify_state_change(ConnectionState.DISCONNECTED, "Disconnected (auto-reconnect disabled)")
            return

        logger.warning("IBKR disconnected unexpectedly - starting reconnection")
        self._notify_state_change(ConnectionState.RECONNECTING, "Unexpected disconnect")

        # Start reconnection in background
        if self._reconnect_task is None or self._reconnect_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._reconnect_task = loop.create_task(self._reconnect_loop())
            except RuntimeError:
                # No running event loop - will need manual reconnection
                logger.error("No event loop for reconnection - call connect() manually")
                self._notify_state_change(ConnectionState.DISCONNECTED)

    async def _reconnect_loop(self) -> None:
        """
        Attempt to reconnect with exponential backoff.

        Implements:
        - Exponential backoff with jitter
        - Maximum retry limit (circuit breaker)
        - Configurable delays
        """
        delay = self.config.reconnect_base_delay

        while not self._shutdown_requested:
            self._reconnect_attempts += 1

            # Check circuit breaker
            max_attempts = self.config.reconnect_max_attempts
            if max_attempts > 0 and self._reconnect_attempts > max_attempts:
                logger.error(
                    f"Max reconnection attempts ({max_attempts}) exceeded - circuit breaker tripped"
                )
                self._notify_state_change(
                    ConnectionState.FAILED,
                    f"Max attempts ({max_attempts}) exceeded"
                )
                return

            logger.info(
                f"Reconnection attempt {self._reconnect_attempts}"
                + (f"/{max_attempts}" if max_attempts > 0 else "")
                + f" in {delay:.1f}s"
            )

            # Wait with jitter
            jitter = delay * self.config.reconnect_jitter * random.random()
            await asyncio.sleep(delay + jitter)

            if self._shutdown_requested:
                break

            # Attempt connection
            self._ib = IB()
            try:
                await self._ib.connectAsync(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=self.config.client_id,
                    timeout=self.config.timeout,
                    readonly=self.config.readonly,
                )

                self._connected = True
                self._reconnect_attempts = 0
                logger.info(f"Reconnected to IBKR after {self._reconnect_attempts} attempts")

                # Set market data type
                self._ib.reqMarketDataType(self.config.market_data_type)

                # Re-register disconnect handler
                self._ib.disconnectedEvent += self._on_disconnect

                self._notify_state_change(ConnectionState.CONNECTED, "Reconnected")
                return

            except Exception as e:
                logger.warning(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")

                # Exponential backoff
                delay = min(
                    delay * self.config.reconnect_multiplier,
                    self.config.reconnect_max_delay
                )

        logger.info("Reconnection loop stopped (shutdown requested)")

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        async with self._connection_lock:
            self._shutdown_requested = True

            # Cancel any pending reconnection
            if self._reconnect_task and not self._reconnect_task.done():
                self._reconnect_task.cancel()
                try:
                    await self._reconnect_task
                except asyncio.CancelledError:
                    pass
                self._reconnect_task = None

            if self._ib:
                # Remove disconnect handler to prevent reconnection
                if self.config.auto_reconnect:
                    try:
                        self._ib.disconnectedEvent -= self._on_disconnect
                    except Exception:
                        pass

                self._ib.disconnect()
                self._connected = False
                self._notify_state_change(ConnectionState.DISCONNECTED)
                logger.info("Disconnected from IBKR")

    async def ensure_connected(self) -> bool:
        """
        Ensure connection is active, waiting for reconnection if in progress.

        Returns:
            True if connected, False if connection failed
        """
        if self.is_connected:
            return True

        if self._connection_state == ConnectionState.RECONNECTING:
            # Wait for reconnection (with timeout)
            timeout = self.config.reconnect_max_delay * 2
            start = datetime.now()
            while self._connection_state == ConnectionState.RECONNECTING:
                if (datetime.now() - start).total_seconds() > timeout:
                    logger.warning("Timeout waiting for reconnection")
                    return False
                await asyncio.sleep(0.5)
            return self.is_connected

        if self._connection_state == ConnectionState.FAILED:
            logger.error("Connection in FAILED state - reset required")
            return False

        # Not connected and not reconnecting - try to connect
        return await self.connect()

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker to allow new connection attempts."""
        self._reconnect_attempts = 0
        if self._connection_state == ConnectionState.FAILED:
            self._notify_state_change(ConnectionState.DISCONNECTED)
            logger.info("Circuit breaker reset")

    def _get_stock_contract(self, symbol: str, exchange: str = "SMART") -> Contract:
        """
        Get or create a stock contract.

        Args:
            symbol: Stock ticker symbol
            exchange: Exchange (default SMART for best execution)

        Returns:
            Stock Contract object
        """
        cache_key = f"{symbol}_{exchange}"
        if cache_key not in self._contracts_cache:
            self._contracts_cache[cache_key] = Stock(
                symbol=symbol.upper(),
                exchange=exchange,
                currency="USD",
            )
        return self._contracts_cache[cache_key]

    async def qualify_contract(self, symbol: str) -> Contract | None:
        """
        Qualify a contract to get full details from IBKR.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Qualified Contract or None if not found
        """
        if not self.is_connected:
            logger.error("Not connected to IBKR")
            return None

        contract = self._get_stock_contract(symbol)
        try:
            qualified = await self._ib.qualifyContractsAsync(contract)
            if qualified:
                return qualified[0]
        except Exception as e:
            logger.error(f"Failed to qualify contract for {symbol}: {e}")

        return None

    async def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "ADJUSTED_LAST",
        use_rth: bool = True,
    ) -> pd.DataFrame | None:
        """
        Fetch historical bar data.

        Args:
            symbol: Stock ticker symbol
            duration: Duration string (e.g., "1 Y", "6 M", "5 D")
            bar_size: Bar size (e.g., "1 day", "1 hour", "5 mins")
            what_to_show: Data type (TRADES, MIDPOINT, BID, ASK, ADJUSTED_LAST)
            use_rth: Use regular trading hours only

        Returns:
            DataFrame with OHLCV data or None if error
        """
        if not self.is_connected:
            logger.error("Not connected to IBKR")
            return None

        contract = self._get_stock_contract(symbol)

        try:
            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )

            if not bars:
                logger.warning(f"No historical data returned for {symbol}")
                return None

            # Convert to DataFrame
            df = util.df(bars)
            df["symbol"] = symbol

            # Rename columns to standard format
            df = df.rename(columns={
                "date": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "barCount": "bar_count",
                "average": "vwap",
            })

            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return None

    async def get_realtime_quote(self, symbol: str) -> MarketData | None:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            MarketData object or None
        """
        if not self.is_connected:
            logger.error("Not connected to IBKR")
            return None

        # Qualify contract to get conId (required for hashing)
        contract = await self.qualify_contract(symbol)
        if not contract:
            logger.error(f"Could not qualify contract for {symbol}")
            return None

        try:
            ticker = self._ib.reqMktData(contract, "", False, False)

            # Wait for data
            await asyncio.sleep(2)

            self._ib.cancelMktData(contract)

            return MarketData(
                symbol=symbol,
                last_price=ticker.last if ticker.last == ticker.last else None,  # NaN check (NaN != NaN)
                bid=ticker.bid if ticker.bid == ticker.bid else None,
                ask=ticker.ask if ticker.ask == ticker.ask else None,
                bid_size=ticker.bidSize if ticker.bidSize else None,
                ask_size=ticker.askSize if ticker.askSize else None,
                volume=ticker.volume if ticker.volume else None,
                open=ticker.open if ticker.open == ticker.open else None,
                high=ticker.high if ticker.high == ticker.high else None,
                low=ticker.low if ticker.low == ticker.low else None,
                close=ticker.close if ticker.close == ticker.close else None,
            )

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    async def get_quotes_batch(self, symbols: list[str]) -> dict[str, MarketData]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbols to MarketData
        """
        results = {}

        # Request market data for all symbols
        tickers: dict[str, Ticker] = {}
        for symbol in symbols:
            contract = self._get_stock_contract(symbol)
            ticker = self._ib.reqMktData(contract, "", False, False)
            tickers[symbol] = ticker

        # Wait for data
        await asyncio.sleep(3)

        # Collect results and cancel
        for symbol, ticker in tickers.items():
            contract = self._get_stock_contract(symbol)
            self._ib.cancelMktData(contract)

            results[symbol] = MarketData(
                symbol=symbol,
                last_price=ticker.last if ticker.last == ticker.last else None,
                bid=ticker.bid if ticker.bid == ticker.bid else None,
                ask=ticker.ask if ticker.ask == ticker.ask else None,
                volume=ticker.volume if ticker.volume else None,
            )

        return results

    async def get_positions(self) -> list[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects
        """
        if not self.is_connected:
            logger.error("Not connected to IBKR")
            return []

        try:
            positions = self._ib.positions()

            return [
                Position(
                    symbol=pos.contract.symbol,
                    quantity=pos.position,
                    avg_cost=pos.avgCost,
                )
                for pos in positions
            ]

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_account_summary(self) -> dict[str, Any]:
        """
        Get account summary information.

        Returns:
            Dict with account metrics
        """
        if not self.is_connected:
            logger.error("Not connected to IBKR")
            return {}

        try:
            account_values = self._ib.accountValues()

            summary = {}
            for av in account_values:
                if av.tag in [
                    "NetLiquidation",
                    "TotalCashValue",
                    "BuyingPower",
                    "GrossPositionValue",
                    "UnrealizedPnL",
                    "RealizedPnL",
                ]:
                    summary[av.tag] = float(av.value) if av.value else 0.0

            return summary

        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}

    async def get_historical_data_batch(
        self,
        symbols: list[str],
        duration: str = "1 Y",
        bar_size: str = "1 day",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.

        Note: IBKR has rate limits, so this processes sequentially
        with small delays between requests.

        Args:
            symbols: List of ticker symbols
            duration: Duration string
            bar_size: Bar size

        Returns:
            Dict mapping symbols to DataFrames
        """
        results = {}

        for symbol in symbols:
            df = await self.get_historical_data(symbol, duration, bar_size)
            if df is not None:
                results[symbol] = df

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        return results

    async def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Place an order via IBKR.

        Args:
            symbol: Stock ticker symbol
            action: "BUY" or "SELL"
            quantity: Number of shares
            order_type: "MKT" for market, "LMT" for limit
            limit_price: Limit price (required for limit orders)

        Returns:
            Trade object info or None if failed
        """
        if not self.is_connected:
            logger.error("Not connected to IBKR")
            return None

        try:
            # Qualify the contract first
            contract = await self.qualify_contract(symbol)
            if not contract:
                logger.error(f"Could not qualify contract for {symbol}")
                return None

            # Create the order
            if order_type == "MKT":
                order = MarketOrder(action, quantity)
            elif order_type == "LMT" and limit_price:
                order = LimitOrder(action, quantity, limit_price)
            else:
                logger.error(f"Invalid order type: {order_type}")
                return None

            # Place the order
            trade = self._ib.placeOrder(contract, order)

            # Wait briefly for order to be acknowledged
            await asyncio.sleep(1)

            logger.info(f"Order placed: {action} {quantity} {symbol} @ {order_type}")

            return {
                "order_id": trade.order.orderId,
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avg_fill_price": trade.orderStatus.avgFillPrice,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
            }

        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            return None


# Context manager support
class IBKRConnection:
    """Context manager for IBKR connections."""

    def __init__(self, config: ibkr_config.__class__ | None = None):
        self.client = IBKRClient(config)

    async def __aenter__(self) -> IBKRClient:
        await self.client.connect()
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.disconnect()


def connection_required(func):
    """
    Decorator that ensures connection before executing method.

    Use on IBKRClient methods that require an active connection.
    Will wait for reconnection if in progress.
    """
    async def wrapper(self: IBKRClient, *args, **kwargs):
        if not await self.ensure_connected():
            logger.error(f"Cannot execute {func.__name__}: not connected")
            return None
        return await func(self, *args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# Example usage
async def main():
    """Example usage of IBKRClient with reconnection handling."""

    # Define a connection state callback
    def on_connection_change(state: ConnectionState, message: str | None):
        print(f"Connection state: {state.name}" + (f" - {message}" if message else ""))

    async with IBKRConnection() as client:
        # Register callback for connection state changes
        client.add_connection_callback(on_connection_change)

        # Get historical data
        df = await client.get_historical_data("AAPL", duration="3 M")
        if df is not None:
            print(f"Got {len(df)} bars for AAPL")
            print(df.tail())

        # Get quote
        quote = await client.get_realtime_quote("AAPL")
        if quote:
            print(f"AAPL last price: {quote.last_price}")

        # Get positions
        positions = await client.get_positions()
        print(f"Positions: {len(positions)}")

        # Example: Check connection state
        print(f"Current state: {client.connection_state.name}")
        print(f"Is reconnecting: {client.is_reconnecting}")


if __name__ == "__main__":
    asyncio.run(main())
