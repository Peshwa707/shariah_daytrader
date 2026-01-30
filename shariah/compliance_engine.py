"""
Shariah Compliance Engine - Main Orchestrator.

This module combines multiple screening approaches:
1. Index Integration - Pre-vetted stocks from Shariah indices (SPUS, etc.)
2. Business Activity Screening - NAICS/SIC code and keyword analysis
3. Financial Ratio Screening - AAOIFI standard ratio checks

The engine provides a unified interface for compliance checking and
maintains a cache of screening results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from enum import Enum

from config.shariah_config import shariah_config, ScreeningResult
from .business_screener import BusinessScreener, BusinessScreeningResult
from .financial_screener import FinancialScreener, FinancialScreeningResult, FinancialData


class ComplianceSource(Enum):
    """Source of compliance determination."""

    INDEX = "shariah_index"  # Pre-vetted by Shariah index
    CUSTOM_SCREEN = "custom_screening"  # Our own screening
    MANUAL = "manual_override"  # Manual verification
    CACHED = "cached_result"  # From cache


@dataclass
class ComplianceResult:
    """
    Unified compliance result combining all screening methods.

    This is the primary output of the compliance engine, containing
    the final compliance determination and supporting evidence.
    """

    symbol: str
    is_compliant: bool
    status: str  # ScreeningResult value
    source: str  # ComplianceSource value

    # Component results
    index_listed: bool = False
    index_name: str | None = None
    business_result: BusinessScreeningResult | None = None
    financial_result: FinancialScreeningResult | None = None

    # Summary
    reasons: list[str] = field(default_factory=list)
    confidence: str = "high"  # high, medium, low
    needs_review: bool = False

    # Metadata
    screening_date: datetime = field(default_factory=datetime.now)
    data_freshness: str = "current"  # current, stale, unknown
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "symbol": self.symbol,
            "is_compliant": self.is_compliant,
            "status": self.status,
            "source": self.source,
            "index_listed": self.index_listed,
            "index_name": self.index_name,
            "business_result": self.business_result.to_dict() if self.business_result else None,
            "financial_result": self.financial_result.to_dict() if self.financial_result else None,
            "reasons": self.reasons,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "screening_date": self.screening_date.isoformat(),
            "data_freshness": self.data_freshness,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @property
    def summary(self) -> str:
        """Human-readable summary of compliance status."""
        status_emoji = "✓" if self.is_compliant else "✗"
        source_info = f"({self.source})"

        if self.is_compliant:
            return f"{status_emoji} {self.symbol}: Shariah Compliant {source_info}"
        else:
            reason_summary = "; ".join(self.reasons[:2])
            return f"{status_emoji} {self.symbol}: Non-Compliant - {reason_summary}"


class ComplianceEngine:
    """
    Main Shariah compliance screening engine.

    Combines multiple screening approaches with caching and
    configurable screening depth levels.

    Screening Levels:
    - INDEX_ONLY: Only check if in Shariah index (fastest)
    - QUICK: Index check + business screening
    - FULL: Index + business + financial screening (most thorough)
    """

    # Cache duration for compliance results
    DEFAULT_CACHE_HOURS = 24

    def __init__(
        self,
        business_screener: BusinessScreener | None = None,
        financial_screener: FinancialScreener | None = None,
        config: shariah_config.__class__ | None = None,
    ):
        """
        Initialize the compliance engine.

        Args:
            business_screener: Business activity screener (created if not provided)
            financial_screener: Financial ratio screener (created if not provided)
            config: Shariah configuration
        """
        self.config = config or shariah_config
        self.business_screener = business_screener or BusinessScreener(self.config)
        self.financial_screener = financial_screener or FinancialScreener(self.config)

        # In-memory cache for compliance results
        self._cache: dict[str, ComplianceResult] = {}

        # Index constituents (populated by index integration)
        self._index_constituents: dict[str, set[str]] = {}

    def set_index_constituents(self, index_name: str, symbols: set[str]) -> None:
        """
        Set the constituents of a Shariah index.

        Args:
            index_name: Name of the index (e.g., "SPUS", "DJIM")
            symbols: Set of ticker symbols in the index
        """
        self._index_constituents[index_name] = {s.upper() for s in symbols}

    def is_index_listed(self, symbol: str) -> tuple[bool, str | None]:
        """
        Check if a symbol is listed in any loaded Shariah index.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (is_listed, index_name)
        """
        symbol_upper = symbol.upper()
        for index_name, constituents in self._index_constituents.items():
            if symbol_upper in constituents:
                return True, index_name
        return False, None

    def screen(
        self,
        symbol: str,
        business_data: dict[str, Any] | None = None,
        financial_data: FinancialData | dict[str, Any] | None = None,
        screening_level: str = "FULL",
        use_cache: bool = True,
    ) -> ComplianceResult:
        """
        Screen a symbol for Shariah compliance.

        Args:
            symbol: Stock ticker symbol
            business_data: Optional business data for screening
            financial_data: Optional financial data for screening
            screening_level: "INDEX_ONLY", "QUICK", or "FULL"
            use_cache: Whether to use cached results

        Returns:
            ComplianceResult with compliance determination
        """
        symbol = symbol.upper()

        # Check cache first
        if use_cache:
            cached = self._get_cached_result(symbol)
            if cached:
                return cached

        # Check index listing
        is_listed, index_name = self.is_index_listed(symbol)

        # INDEX_ONLY level - just check index listing
        if screening_level == "INDEX_ONLY":
            if is_listed:
                result = self._create_compliant_result(
                    symbol, ComplianceSource.INDEX, index_name
                )
            else:
                result = ComplianceResult(
                    symbol=symbol,
                    is_compliant=False,
                    status=ScreeningResult.INSUFFICIENT_DATA,
                    source=ComplianceSource.INDEX.value,
                    index_listed=False,
                    reasons=["Not found in any loaded Shariah index"],
                    confidence="low",
                    needs_review=True,
                )
            self._cache_result(result)
            return result

        # QUICK level - index + business screening
        business_result = None
        if business_data or screening_level in ["QUICK", "FULL"]:
            business_result = self._run_business_screening(symbol, business_data)

        # FULL level - add financial screening
        financial_result = None
        if screening_level == "FULL" and financial_data:
            financial_result = self._run_financial_screening(symbol, financial_data)

        # Combine results
        result = self._combine_results(
            symbol=symbol,
            is_index_listed=is_listed,
            index_name=index_name,
            business_result=business_result,
            financial_result=financial_result,
        )

        self._cache_result(result)
        return result

    def _run_business_screening(
        self, symbol: str, data: dict[str, Any] | None
    ) -> BusinessScreeningResult:
        """Run business activity screening."""
        if data is None:
            data = {}

        return self.business_screener.screen(
            symbol=symbol,
            naics_code=data.get("naics_code"),
            sic_code=data.get("sic_code"),
            industry_name=data.get("industry_name") or data.get("industry"),
            sector=data.get("sector"),
            revenue_breakdown=data.get("revenue_breakdown"),
        )

    def _run_financial_screening(
        self, symbol: str, data: FinancialData | dict[str, Any]
    ) -> FinancialScreeningResult:
        """Run financial ratio screening."""
        if isinstance(data, dict):
            data["symbol"] = symbol
            return self.financial_screener.screen_from_dict(data)
        return self.financial_screener.screen(data)

    def _combine_results(
        self,
        symbol: str,
        is_index_listed: bool,
        index_name: str | None,
        business_result: BusinessScreeningResult | None,
        financial_result: FinancialScreeningResult | None,
    ) -> ComplianceResult:
        """
        Combine screening results into a unified compliance determination.

        Decision logic:
        1. If in Shariah index and passes our screens -> Compliant (high confidence)
        2. If in Shariah index but fails our screens -> Needs review
        3. If not in index but passes all screens -> Compliant (medium confidence)
        4. If fails any screen -> Non-compliant
        """
        reasons = []
        is_compliant = True
        confidence = "high"
        needs_review = False
        status = ScreeningResult.COMPLIANT

        # Check business screening result
        if business_result and not business_result.is_compliant:
            is_compliant = False
            status = business_result.status
            reasons.extend(business_result.reasons)

        # Check financial screening result
        if financial_result and not financial_result.is_compliant:
            # If status was already non-compliant, keep it
            if is_compliant:
                is_compliant = False
                status = financial_result.status
            reasons.extend(financial_result.reasons)

        # Adjust confidence based on data availability
        if financial_result and financial_result.data_quality == "partial":
            confidence = "medium"
        if financial_result and financial_result.data_quality == "insufficient":
            confidence = "low"
            needs_review = True

        # Index listing adjustments
        if is_index_listed:
            if is_compliant:
                # Index + our screening agree -> high confidence
                confidence = "high"
                reasons.insert(0, f"Listed in {index_name} Shariah index")
            else:
                # Discrepancy - needs review
                needs_review = True
                reasons.insert(0, f"Listed in {index_name} but failed custom screening")
        else:
            if is_compliant:
                # Not in index but passes our screening
                confidence = "medium"
                reasons.insert(0, "Passed custom screening (not in loaded indices)")

        # Determine source
        if is_index_listed and is_compliant:
            source = ComplianceSource.INDEX.value
        else:
            source = ComplianceSource.CUSTOM_SCREEN.value

        # Handle questionable status
        if business_result and business_result.status == ScreeningResult.QUESTIONABLE:
            if is_compliant:
                status = ScreeningResult.QUESTIONABLE
                needs_review = True

        if financial_result and financial_result.status == ScreeningResult.QUESTIONABLE:
            if is_compliant:
                status = ScreeningResult.QUESTIONABLE
                needs_review = True

        return ComplianceResult(
            symbol=symbol,
            is_compliant=is_compliant,
            status=status,
            source=source,
            index_listed=is_index_listed,
            index_name=index_name,
            business_result=business_result,
            financial_result=financial_result,
            reasons=reasons,
            confidence=confidence,
            needs_review=needs_review,
            expires_at=datetime.now() + timedelta(hours=self.DEFAULT_CACHE_HOURS),
        )

    def _create_compliant_result(
        self, symbol: str, source: ComplianceSource, index_name: str | None
    ) -> ComplianceResult:
        """Create a compliant result for index-listed stocks."""
        return ComplianceResult(
            symbol=symbol,
            is_compliant=True,
            status=ScreeningResult.COMPLIANT,
            source=source.value,
            index_listed=True,
            index_name=index_name,
            reasons=[f"Listed in {index_name} Shariah index"],
            confidence="high",
            expires_at=datetime.now() + timedelta(hours=self.DEFAULT_CACHE_HOURS),
        )

    def _get_cached_result(self, symbol: str) -> ComplianceResult | None:
        """Get cached result if valid."""
        cached = self._cache.get(symbol)
        if cached and cached.expires_at and cached.expires_at > datetime.now():
            # Update source to indicate cached
            cached.data_freshness = "cached"
            return cached
        return None

    def _cache_result(self, result: ComplianceResult) -> None:
        """Cache a compliance result."""
        self._cache[result.symbol] = result

    def clear_cache(self, symbol: str | None = None) -> None:
        """
        Clear cached results.

        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            self._cache.pop(symbol.upper(), None)
        else:
            self._cache.clear()

    def batch_screen(
        self,
        symbols: list[str],
        business_data: dict[str, dict[str, Any]] | None = None,
        financial_data: dict[str, FinancialData | dict[str, Any]] | None = None,
        screening_level: str = "FULL",
    ) -> dict[str, ComplianceResult]:
        """
        Screen multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            business_data: Dict mapping symbols to business data
            financial_data: Dict mapping symbols to financial data
            screening_level: Screening depth level

        Returns:
            Dict mapping symbols to ComplianceResult
        """
        business_data = business_data or {}
        financial_data = financial_data or {}

        results = {}
        for symbol in symbols:
            results[symbol] = self.screen(
                symbol=symbol,
                business_data=business_data.get(symbol),
                financial_data=financial_data.get(symbol),
                screening_level=screening_level,
            )
        return results

    def get_compliant_symbols(self, symbols: list[str]) -> list[str]:
        """
        Filter a list of symbols to only compliant ones.

        Quick method using cached results where available.

        Args:
            symbols: List of symbols to filter

        Returns:
            List of compliant symbols
        """
        compliant = []
        for symbol in symbols:
            result = self.screen(symbol, screening_level="INDEX_ONLY")
            if result.is_compliant:
                compliant.append(symbol)
        return compliant

    def get_statistics(self) -> dict[str, Any]:
        """Get screening statistics from cache."""
        if not self._cache:
            return {"total": 0}

        compliant = sum(1 for r in self._cache.values() if r.is_compliant)
        non_compliant = sum(1 for r in self._cache.values() if not r.is_compliant)
        needs_review = sum(1 for r in self._cache.values() if r.needs_review)

        return {
            "total": len(self._cache),
            "compliant": compliant,
            "non_compliant": non_compliant,
            "needs_review": needs_review,
            "compliance_rate": compliant / len(self._cache) if self._cache else 0,
            "index_count": sum(len(c) for c in self._index_constituents.values()),
            "indices_loaded": list(self._index_constituents.keys()),
        }
