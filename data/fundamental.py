"""
Fundamental Data Fetcher.

This module fetches fundamental financial data needed for Shariah screening,
using free/low-cost data sources:

1. Alpha Vantage (free tier: 25 requests/day)
2. IBKR fundamental data (requires subscription)
3. SEC EDGAR filings (free)

The data is used for:
- AAOIFI financial ratio screening
- Industry/sector classification
- Revenue breakdown analysis
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import logging

import httpx

from config.settings import settings
from shariah.financial_screener import FinancialData


logger = logging.getLogger(__name__)


@dataclass
class CompanyProfile:
    """Basic company profile information."""

    symbol: str
    name: str
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
    description: str | None = None
    market_cap: float | None = None
    employees: int | None = None
    website: str | None = None
    country: str | None = None

    # Industry codes for screening
    naics_code: str | None = None
    sic_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "sector": self.sector,
            "industry": self.industry,
            "description": self.description,
            "market_cap": self.market_cap,
            "employees": self.employees,
            "website": self.website,
            "country": self.country,
            "naics_code": self.naics_code,
            "sic_code": self.sic_code,
        }


class FundamentalDataFetcher:
    """
    Fetches fundamental data from various sources.

    Primary source: Alpha Vantage (free tier)
    Fallback: Mock data for testing

    Note: Alpha Vantage free tier is limited to 25 requests/day.
    For production use, consider upgrading or using IBKR fundamental data.
    """

    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None):
        """
        Initialize the fundamental data fetcher.

        Args:
            api_key: Alpha Vantage API key (uses settings if not provided)
        """
        self.api_key = api_key or settings.alpha_vantage_api_key
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_duration_hours = 24

    async def get_company_overview(self, symbol: str) -> CompanyProfile | None:
        """
        Fetch company overview/profile.

        Args:
            symbol: Stock ticker symbol

        Returns:
            CompanyProfile or None if not found
        """
        # Check cache
        cached = self._get_cached(f"overview_{symbol}")
        if cached:
            return cached

        if not self.api_key:
            logger.warning("No Alpha Vantage API key configured, using mock data")
            return self._get_mock_profile(symbol)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.ALPHA_VANTAGE_BASE_URL,
                    params={
                        "function": "OVERVIEW",
                        "symbol": symbol,
                        "apikey": self.api_key,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                if "Symbol" not in data:
                    logger.warning(f"No overview data for {symbol}")
                    return None

                profile = CompanyProfile(
                    symbol=data.get("Symbol", symbol),
                    name=data.get("Name", ""),
                    exchange=data.get("Exchange"),
                    sector=data.get("Sector"),
                    industry=data.get("Industry"),
                    description=data.get("Description"),
                    market_cap=self._parse_float(data.get("MarketCapitalization")),
                    employees=self._parse_int(data.get("FullTimeEmployees")),
                    country=data.get("Country"),
                )

                self._set_cached(f"overview_{symbol}", profile)
                return profile

        except Exception as e:
            logger.error(f"Error fetching overview for {symbol}: {e}")
            return None

    async def get_financial_data(self, symbol: str) -> FinancialData | None:
        """
        Fetch financial data for Shariah screening.

        Args:
            symbol: Stock ticker symbol

        Returns:
            FinancialData for screening or None
        """
        # Check cache
        cached = self._get_cached(f"financial_{symbol}")
        if cached:
            return cached

        if not self.api_key:
            logger.warning("No Alpha Vantage API key configured, using mock data")
            return self._get_mock_financial_data(symbol)

        # Fetch balance sheet and income statement
        balance_sheet = await self._fetch_balance_sheet(symbol)
        income_statement = await self._fetch_income_statement(symbol)
        overview = await self.get_company_overview(symbol)

        if not balance_sheet or not income_statement:
            return None

        # Extract latest annual data
        bs_data = balance_sheet.get("annualReports", [{}])[0] if balance_sheet else {}
        is_data = income_statement.get("annualReports", [{}])[0] if income_statement else {}

        # Use explicit None checks to avoid treating 0 as falsy
        total_debt_val = self._parse_float(bs_data.get("totalDebt"))
        if total_debt_val is None:
            total_debt_val = self._parse_float(bs_data.get("shortLongTermDebtTotal"))

        cash_val = self._parse_float(bs_data.get("cashAndCashEquivalentsAtCarryingValue"))
        if cash_val is None:
            cash_val = self._parse_float(bs_data.get("cashAndShortTermInvestments"))

        financial_data = FinancialData(
            symbol=symbol,
            market_cap=overview.market_cap if overview else None,
            total_debt=total_debt_val,
            short_term_debt=self._parse_float(bs_data.get("shortTermDebt")),
            long_term_debt=self._parse_float(bs_data.get("longTermDebt")),
            cash_and_equivalents=cash_val,
            accounts_receivable=self._parse_float(bs_data.get("currentNetReceivables")),
            total_assets=self._parse_float(bs_data.get("totalAssets")),
            total_revenue=self._parse_float(is_data.get("totalRevenue")),
            interest_income=self._parse_float(is_data.get("interestIncome")),
            data_source="alpha_vantage",
        )

        self._set_cached(f"financial_{symbol}", financial_data)
        return financial_data

    async def _fetch_balance_sheet(self, symbol: str) -> dict | None:
        """Fetch balance sheet data from Alpha Vantage."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.ALPHA_VANTAGE_BASE_URL,
                    params={
                        "function": "BALANCE_SHEET",
                        "symbol": symbol,
                        "apikey": self.api_key,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {symbol}: {e}")
            return None

    async def _fetch_income_statement(self, symbol: str) -> dict | None:
        """Fetch income statement data from Alpha Vantage."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.ALPHA_VANTAGE_BASE_URL,
                    params={
                        "function": "INCOME_STATEMENT",
                        "symbol": symbol,
                        "apikey": self.api_key,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching income statement for {symbol}: {e}")
            return None

    async def get_financial_data_batch(
        self, symbols: list[str], delay: float = 12.0
    ) -> dict[str, FinancialData]:
        """
        Fetch financial data for multiple symbols.

        Note: Alpha Vantage free tier is rate limited (5 calls/minute).
        This method adds delays between requests.

        Args:
            symbols: List of ticker symbols
            delay: Delay between requests in seconds

        Returns:
            Dict mapping symbols to FinancialData
        """
        results = {}

        for i, symbol in enumerate(symbols):
            data = await self.get_financial_data(symbol)
            if data:
                results[symbol] = data

            # Rate limiting delay (except for last request)
            if i < len(symbols) - 1:
                await asyncio.sleep(delay)

        return results

    def _get_cached(self, key: str) -> Any | None:
        """Get cached data if valid."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            if age_hours < self._cache_duration_hours:
                return data
        return None

    def _set_cached(self, key: str, data: Any) -> None:
        """Cache data with timestamp."""
        self._cache[key] = (data, datetime.now())

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        """Parse string to float, handling None and 'None'."""
        if value is None or value == "None" or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_int(value: str | None) -> int | None:
        """Parse string to int, handling None and 'None'."""
        if value is None or value == "None" or value == "":
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _get_mock_profile(self, symbol: str) -> CompanyProfile:
        """Return mock profile data for testing without API."""
        mock_data = {
            "AAPL": CompanyProfile(
                symbol="AAPL",
                name="Apple Inc",
                exchange="NASDAQ",
                sector="Technology",
                industry="Consumer Electronics",
                market_cap=3_000_000_000_000,
                country="USA",
            ),
            "MSFT": CompanyProfile(
                symbol="MSFT",
                name="Microsoft Corporation",
                exchange="NASDAQ",
                sector="Technology",
                industry="Software - Infrastructure",
                market_cap=2_800_000_000_000,
                country="USA",
            ),
            "GOOGL": CompanyProfile(
                symbol="GOOGL",
                name="Alphabet Inc",
                exchange="NASDAQ",
                sector="Technology",
                industry="Internet Content & Information",
                market_cap=1_700_000_000_000,
                country="USA",
            ),
        }

        if symbol in mock_data:
            return mock_data[symbol]

        # Generic mock
        return CompanyProfile(
            symbol=symbol,
            name=f"{symbol} Corp",
            sector="Unknown",
            industry="Unknown",
            market_cap=10_000_000_000,
        )

    def _get_mock_financial_data(self, symbol: str) -> FinancialData:
        """Return mock financial data for testing without API."""
        # Create reasonable mock data for common stocks
        mock_financials = {
            "AAPL": FinancialData(
                symbol="AAPL",
                market_cap=3_000_000_000_000,
                total_debt=120_000_000_000,
                interest_bearing_debt=100_000_000_000,
                cash_and_equivalents=60_000_000_000,
                accounts_receivable=50_000_000_000,
                total_assets=350_000_000_000,
                total_revenue=380_000_000_000,
                interest_income=3_000_000_000,
                data_source="mock",
            ),
            "MSFT": FinancialData(
                symbol="MSFT",
                market_cap=2_800_000_000_000,
                total_debt=80_000_000_000,
                interest_bearing_debt=70_000_000_000,
                cash_and_equivalents=100_000_000_000,
                accounts_receivable=45_000_000_000,
                total_assets=400_000_000_000,
                total_revenue=210_000_000_000,
                interest_income=2_000_000_000,
                data_source="mock",
            ),
        }

        if symbol in mock_financials:
            return mock_financials[symbol]

        # Generic mock with reasonable ratios
        return FinancialData(
            symbol=symbol,
            market_cap=50_000_000_000,
            total_debt=10_000_000_000,  # 20% debt ratio
            cash_and_equivalents=5_000_000_000,  # 10% cash ratio
            accounts_receivable=8_000_000_000,  # 16% receivables
            total_assets=60_000_000_000,
            total_revenue=30_000_000_000,
            interest_income=500_000_000,  # ~1.7% impermissible
            data_source="mock",
        )


# Convenience function
async def fetch_screening_data(symbol: str) -> tuple[CompanyProfile | None, FinancialData | None]:
    """
    Fetch all data needed for Shariah screening.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Tuple of (CompanyProfile, FinancialData)
    """
    fetcher = FundamentalDataFetcher()
    profile = await fetcher.get_company_overview(symbol)
    financials = await fetcher.get_financial_data(symbol)
    return profile, financials
