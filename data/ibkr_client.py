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
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from dataclasses import dataclass, field
import logging

import pandas as pd

# ib_async imports (these will fail if not installed - that's expected)
try:
    from ib_async import IB, Stock, Contract, BarData, Ticker
    from ib_async import util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    IB = None
    Stock = None
    Contract = None

from config.ibkr_config import ibkr_config


logger = logging.getLogger(__name__)


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

    @property
    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self._connected and self._ib is not None and self._ib.isConnected()

    async def connect(self) -> bool:
        """
        Connect to TWS/Gateway.

        Returns:
            True if connection successful
        """
        if self.is_connected:
            logger.info("Already connected to IBKR")
            return True

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
            logger.info(
                f"Connected to IBKR at {self.config.host}:{self.config.port} "
                f"(mode: {self.config.mode})"
            )

            # Set market data type
            self._ib.reqMarketDataType(self.config.market_data_type)

            return True

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

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

        contract = self._get_stock_contract(symbol)

        try:
            ticker = self._ib.reqMktData(contract, "", False, False)

            # Wait for data
            await asyncio.sleep(2)

            self._ib.cancelMktData(contract)

            return MarketData(
                symbol=symbol,
                last_price=ticker.last if ticker.last != ticker.last else None,  # NaN check
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


# Example usage
async def main():
    """Example usage of IBKRClient."""
    async with IBKRConnection() as client:
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


if __name__ == "__main__":
    asyncio.run(main())
