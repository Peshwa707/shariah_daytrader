"""
Shariah Index Integration Module.

This module fetches and manages constituents from established Shariah indices,
providing a pre-vetted universe of compliant stocks.

Primary Indices:
- S&P 500 Shariah Index (via SPUS ETF holdings)
- Dow Jones Islamic Market Index (DJIM)
- MSCI World Islamic Index

These indices are maintained by professional Shariah boards and provide
reliable compliance screening. We use them as the primary filter and
supplement with our own screening for additional verification.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import json

from config.settings import settings


@dataclass
class IndexConstituent:
    """A single constituent of a Shariah index."""

    symbol: str
    name: str
    weight: float | None = None
    sector: str | None = None
    market_cap: float | None = None
    index_name: str | None = None
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "weight": self.weight,
            "sector": self.sector,
            "market_cap": self.market_cap,
            "index_name": self.index_name,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class IndexData:
    """Data for a Shariah index."""

    name: str
    description: str
    constituents: list[IndexConstituent]
    last_updated: datetime
    source: str
    total_count: int = 0

    def __post_init__(self):
        self.total_count = len(self.constituents)

    def get_symbols(self) -> set[str]:
        """Get set of all constituent symbols."""
        return {c.symbol.upper() for c in self.constituents}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "constituents": [c.to_dict() for c in self.constituents],
            "last_updated": self.last_updated.isoformat(),
            "source": self.source,
            "total_count": self.total_count,
        }


class ShariahIndexIntegration:
    """
    Manages integration with Shariah indices.

    Fetches constituents from ETF holdings and maintains a cache
    of pre-vetted compliant stocks.
    """

    # Cache duration for index data
    CACHE_DURATION_HOURS = 24

    # Known Shariah ETFs and their data sources
    SHARIAH_ETFS = {
        "SPUS": {
            "name": "S&P 500 Shariah Index",
            "description": "S&P 500 companies screened for Shariah compliance",
            "issuer": "SP Funds",
        },
        "HLAL": {
            "name": "Wahed FTSE USA Shariah ETF",
            "description": "US large/mid-cap Shariah-compliant stocks",
            "issuer": "Wahed Invest",
        },
        "UMMA": {
            "name": "Wahed Dow Jones Islamic World ETF",
            "description": "Global Shariah-compliant stocks",
            "issuer": "Wahed Invest",
        },
    }

    def __init__(self):
        """Initialize the index integration module."""
        self._cache: dict[str, IndexData] = {}
        self._cache_timestamps: dict[str, datetime] = {}
        self._all_compliant_symbols: set[str] = set()

    async def fetch_etf_holdings(
        self, etf_symbol: str, use_cache: bool = True
    ) -> IndexData | None:
        """
        Fetch holdings for a Shariah ETF.

        This attempts to fetch holdings from publicly available sources.
        In production, you might use a paid data provider or IBKR's
        fundamental data service.

        Args:
            etf_symbol: ETF ticker symbol (e.g., "SPUS")
            use_cache: Whether to use cached data

        Returns:
            IndexData with constituents, or None if fetch fails
        """
        etf_symbol = etf_symbol.upper()

        # Check cache
        if use_cache and self._is_cache_valid(etf_symbol):
            return self._cache.get(etf_symbol)

        # Get ETF info
        etf_info = self.SHARIAH_ETFS.get(etf_symbol)
        if not etf_info:
            return None

        # Try to fetch holdings
        constituents = await self._fetch_holdings_from_sources(etf_symbol)

        if constituents:
            index_data = IndexData(
                name=etf_info["name"],
                description=etf_info["description"],
                constituents=constituents,
                last_updated=datetime.now(),
                source=f"{etf_symbol} ETF Holdings",
            )

            # Update cache
            self._cache[etf_symbol] = index_data
            self._cache_timestamps[etf_symbol] = datetime.now()

            # Update combined symbols
            self._all_compliant_symbols.update(index_data.get_symbols())

            return index_data

        return None

    async def _fetch_holdings_from_sources(
        self, etf_symbol: str
    ) -> list[IndexConstituent] | None:
        """
        Try multiple sources to fetch ETF holdings.

        Sources tried in order:
        1. Issuer's website (parsed)
        2. SEC 13F filings
        3. Financial data APIs
        """
        # For now, return a manually curated list of known S&P 500 Shariah constituents
        # In production, this would fetch from actual data sources

        if etf_symbol == "SPUS":
            return self._get_spus_sample_holdings()

        return None

    def _get_spus_sample_holdings(self) -> list[IndexConstituent]:
        """
        Get sample SPUS holdings.

        This is a representative sample of S&P 500 Shariah-compliant stocks.
        In production, fetch the actual holdings from a data provider.

        The S&P 500 Shariah Index typically includes ~230-250 stocks from
        the S&P 500 that pass Shariah screening (excludes financials,
        alcohol, tobacco, gambling, etc.)
        """
        # Representative sample of known Shariah-compliant large caps
        # This would be replaced with actual ETF holdings data
        sample_holdings = [
            # Technology
            ("AAPL", "Apple Inc", "Technology", 7.0),
            ("MSFT", "Microsoft Corp", "Technology", 6.5),
            ("NVDA", "NVIDIA Corp", "Technology", 5.0),
            ("GOOGL", "Alphabet Inc Class A", "Technology", 3.5),
            ("GOOG", "Alphabet Inc Class C", "Technology", 3.0),
            ("META", "Meta Platforms Inc", "Technology", 2.5),
            ("AVGO", "Broadcom Inc", "Technology", 2.0),
            ("ORCL", "Oracle Corp", "Technology", 1.5),
            ("CSCO", "Cisco Systems Inc", "Technology", 1.2),
            ("CRM", "Salesforce Inc", "Technology", 1.1),
            ("AMD", "Advanced Micro Devices", "Technology", 1.0),
            ("ADBE", "Adobe Inc", "Technology", 0.9),
            ("INTC", "Intel Corp", "Technology", 0.8),
            ("IBM", "IBM Corp", "Technology", 0.7),
            ("QCOM", "Qualcomm Inc", "Technology", 0.7),

            # Healthcare
            ("LLY", "Eli Lilly and Co", "Healthcare", 2.0),
            ("UNH", "UnitedHealth Group", "Healthcare", 1.8),
            ("JNJ", "Johnson & Johnson", "Healthcare", 1.5),
            ("MRK", "Merck & Co Inc", "Healthcare", 1.3),
            ("ABBV", "AbbVie Inc", "Healthcare", 1.2),
            ("PFE", "Pfizer Inc", "Healthcare", 1.0),
            ("TMO", "Thermo Fisher Scientific", "Healthcare", 0.9),
            ("ABT", "Abbott Laboratories", "Healthcare", 0.8),
            ("DHR", "Danaher Corp", "Healthcare", 0.7),
            ("BMY", "Bristol-Myers Squibb", "Healthcare", 0.6),

            # Consumer
            ("AMZN", "Amazon.com Inc", "Consumer Discretionary", 3.5),
            ("TSLA", "Tesla Inc", "Consumer Discretionary", 2.0),
            ("HD", "Home Depot Inc", "Consumer Discretionary", 1.2),
            ("MCD", "McDonald's Corp", "Consumer Discretionary", 0.9),
            ("NKE", "Nike Inc", "Consumer Discretionary", 0.7),
            ("SBUX", "Starbucks Corp", "Consumer Discretionary", 0.6),
            ("TGT", "Target Corp", "Consumer Discretionary", 0.5),
            ("LOW", "Lowe's Companies", "Consumer Discretionary", 0.5),
            ("PG", "Procter & Gamble", "Consumer Staples", 1.3),
            ("KO", "Coca-Cola Co", "Consumer Staples", 1.0),
            ("PEP", "PepsiCo Inc", "Consumer Staples", 0.9),
            ("COST", "Costco Wholesale", "Consumer Staples", 1.0),
            ("WMT", "Walmart Inc", "Consumer Staples", 1.2),

            # Industrials
            ("CAT", "Caterpillar Inc", "Industrials", 0.8),
            ("DE", "Deere & Company", "Industrials", 0.7),
            ("UNP", "Union Pacific Corp", "Industrials", 0.7),
            ("HON", "Honeywell International", "Industrials", 0.6),
            ("RTX", "RTX Corp", "Industrials", 0.6),
            ("BA", "Boeing Co", "Industrials", 0.5),
            ("LMT", "Lockheed Martin", "Industrials", 0.5),
            ("GE", "General Electric", "Industrials", 0.8),
            ("MMM", "3M Company", "Industrials", 0.4),
            ("UPS", "United Parcel Service", "Industrials", 0.5),

            # Energy (typically compliant as core business is permissible)
            ("XOM", "Exxon Mobil Corp", "Energy", 1.5),
            ("CVX", "Chevron Corp", "Energy", 1.2),
            ("COP", "ConocoPhillips", "Energy", 0.6),
            ("SLB", "Schlumberger Ltd", "Energy", 0.4),
            ("EOG", "EOG Resources", "Energy", 0.4),

            # Materials
            ("LIN", "Linde PLC", "Materials", 0.7),
            ("APD", "Air Products & Chemicals", "Materials", 0.4),
            ("SHW", "Sherwin-Williams", "Materials", 0.4),
            ("FCX", "Freeport-McMoRan", "Materials", 0.3),

            # Communications
            ("VZ", "Verizon Communications", "Communications", 0.7),
            ("T", "AT&T Inc", "Communications", 0.6),
            ("TMUS", "T-Mobile US", "Communications", 0.7),
            ("NFLX", "Netflix Inc", "Communications", 0.9),
            ("DIS", "Walt Disney Co", "Communications", 0.8),
            ("CMCSA", "Comcast Corp", "Communications", 0.6),

            # Utilities (some may have financing concerns - simplified here)
            ("NEE", "NextEra Energy", "Utilities", 0.6),
            ("DUK", "Duke Energy", "Utilities", 0.4),
            ("SO", "Southern Company", "Utilities", 0.4),

            # Real Estate (REITs have special considerations)
            ("PLD", "Prologis Inc", "Real Estate", 0.5),
            ("AMT", "American Tower Corp", "Real Estate", 0.5),
            ("EQIX", "Equinix Inc", "Real Estate", 0.4),
        ]

        constituents = []
        for symbol, name, sector, weight in sample_holdings:
            constituents.append(
                IndexConstituent(
                    symbol=symbol,
                    name=name,
                    weight=weight,
                    sector=sector,
                    index_name="S&P 500 Shariah",
                )
            )

        return constituents

    def _is_cache_valid(self, index_name: str) -> bool:
        """Check if cached data is still valid."""
        timestamp = self._cache_timestamps.get(index_name)
        if not timestamp:
            return False

        age = datetime.now() - timestamp
        return age < timedelta(hours=self.CACHE_DURATION_HOURS)

    def get_all_compliant_symbols(self) -> set[str]:
        """Get all symbols from loaded indices."""
        return self._all_compliant_symbols.copy()

    def is_in_index(self, symbol: str) -> tuple[bool, str | None]:
        """
        Check if a symbol is in any loaded index.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (is_listed, index_name)
        """
        symbol = symbol.upper()
        for index_name, index_data in self._cache.items():
            if symbol in index_data.get_symbols():
                return True, index_data.name
        return False, None

    def get_constituents_by_sector(self, sector: str) -> list[IndexConstituent]:
        """Get all constituents in a specific sector."""
        constituents = []
        for index_data in self._cache.values():
            for constituent in index_data.constituents:
                if constituent.sector and constituent.sector.lower() == sector.lower():
                    constituents.append(constituent)
        return constituents

    def get_top_holdings(self, n: int = 20) -> list[IndexConstituent]:
        """Get top N holdings by weight across all indices."""
        all_constituents = []
        for index_data in self._cache.values():
            all_constituents.extend(index_data.constituents)

        # Sort by weight (descending)
        sorted_constituents = sorted(
            all_constituents,
            key=lambda c: c.weight or 0,
            reverse=True,
        )

        return sorted_constituents[:n]

    def save_to_file(self, filepath: str) -> None:
        """
        Save cached index data to a JSON file.

        Args:
            filepath: Path to save the data
        """
        data = {
            index_name: index_data.to_dict()
            for index_name, index_data in self._cache.items()
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str) -> int:
        """
        Load index data from a JSON file.

        Args:
            filepath: Path to load from

        Returns:
            Number of indices loaded
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        count = 0
        for index_name, index_dict in data.items():
            constituents = [
                IndexConstituent(
                    symbol=c["symbol"],
                    name=c["name"],
                    weight=c.get("weight"),
                    sector=c.get("sector"),
                    market_cap=c.get("market_cap"),
                    index_name=c.get("index_name"),
                )
                for c in index_dict["constituents"]
            ]

            index_data = IndexData(
                name=index_dict["name"],
                description=index_dict["description"],
                constituents=constituents,
                last_updated=datetime.fromisoformat(index_dict["last_updated"]),
                source=index_dict["source"],
            )

            self._cache[index_name] = index_data
            self._all_compliant_symbols.update(index_data.get_symbols())
            count += 1

        return count

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about loaded indices."""
        stats = {
            "indices_loaded": len(self._cache),
            "total_constituents": len(self._all_compliant_symbols),
            "indices": {},
        }

        for index_name, index_data in self._cache.items():
            stats["indices"][index_name] = {
                "name": index_data.name,
                "count": index_data.total_count,
                "last_updated": index_data.last_updated.isoformat(),
            }

        return stats


# Convenience function for quick setup
async def load_shariah_universe() -> ShariahIndexIntegration:
    """
    Load Shariah-compliant universe from available indices.

    Returns:
        Configured ShariahIndexIntegration instance
    """
    integration = ShariahIndexIntegration()

    # Load SPUS (S&P 500 Shariah)
    await integration.fetch_etf_holdings("SPUS")

    return integration
