"""
AAOIFI Financial Ratio Screener for Shariah Compliance.

This module screens companies based on financial ratios as defined by
AAOIFI Standard No. 21 (Investment in Shares, Units, and Similar Instruments).

Key Financial Screens:
1. Debt Ratio: Interest-bearing debt / Market Cap < 30%
2. Deposit Ratio: Interest-bearing deposits / Market Cap < 30%
3. Impermissible Income: Non-permissible income / Total Revenue < 5%
4. Receivables Ratio: Accounts receivable / Market Cap < 49% (optional)

These thresholds ensure minimal exposure to interest (riba) and
non-permissible income sources.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config.shariah_config import shariah_config, ScreeningResult, NonComplianceReason


@dataclass
class FinancialData:
    """Financial data required for Shariah screening."""

    symbol: str

    # Market data
    market_cap: float | None = None
    share_price: float | None = None
    shares_outstanding: float | None = None

    # Balance sheet items
    total_debt: float | None = None
    interest_bearing_debt: float | None = None
    short_term_debt: float | None = None
    long_term_debt: float | None = None
    cash_and_equivalents: float | None = None
    interest_bearing_deposits: float | None = None
    accounts_receivable: float | None = None
    total_assets: float | None = None

    # Income statement items
    total_revenue: float | None = None
    interest_income: float | None = None
    other_non_operating_income: float | None = None
    non_permissible_income: float | None = None

    # Metadata
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    data_date: datetime | None = None
    data_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "market_cap": self.market_cap,
            "share_price": self.share_price,
            "shares_outstanding": self.shares_outstanding,
            "total_debt": self.total_debt,
            "interest_bearing_debt": self.interest_bearing_debt,
            "short_term_debt": self.short_term_debt,
            "long_term_debt": self.long_term_debt,
            "cash_and_equivalents": self.cash_and_equivalents,
            "interest_bearing_deposits": self.interest_bearing_deposits,
            "accounts_receivable": self.accounts_receivable,
            "total_assets": self.total_assets,
            "total_revenue": self.total_revenue,
            "interest_income": self.interest_income,
            "other_non_operating_income": self.other_non_operating_income,
            "non_permissible_income": self.non_permissible_income,
            "fiscal_year": self.fiscal_year,
            "fiscal_quarter": self.fiscal_quarter,
            "data_date": self.data_date.isoformat() if self.data_date else None,
            "data_source": self.data_source,
        }


@dataclass
class FinancialRatios:
    """Calculated financial ratios for Shariah screening."""

    debt_to_market_cap: float | None = None
    deposits_to_market_cap: float | None = None
    impermissible_income_ratio: float | None = None
    receivables_to_market_cap: float | None = None

    # For reference/reporting
    debt_to_assets: float | None = None
    cash_to_market_cap: float | None = None


@dataclass
class FinancialScreeningResult:
    """Result of financial ratio screening."""

    symbol: str
    status: str  # ScreeningResult value
    is_compliant: bool
    ratios: FinancialRatios
    thresholds_exceeded: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    data_quality: str = "complete"  # complete, partial, insufficient
    missing_data: list[str] = field(default_factory=list)
    screening_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "symbol": self.symbol,
            "status": self.status,
            "is_compliant": self.is_compliant,
            "ratios": {
                "debt_to_market_cap": self.ratios.debt_to_market_cap,
                "deposits_to_market_cap": self.ratios.deposits_to_market_cap,
                "impermissible_income_ratio": self.ratios.impermissible_income_ratio,
                "receivables_to_market_cap": self.ratios.receivables_to_market_cap,
                "debt_to_assets": self.ratios.debt_to_assets,
                "cash_to_market_cap": self.ratios.cash_to_market_cap,
            },
            "thresholds_exceeded": self.thresholds_exceeded,
            "reasons": self.reasons,
            "data_quality": self.data_quality,
            "missing_data": self.missing_data,
            "screening_date": self.screening_date.isoformat(),
        }


class FinancialScreener:
    """
    Screens companies for Shariah compliance based on AAOIFI financial ratios.

    The AAOIFI standard uses market capitalization as the denominator for most
    ratios. This creates some challenges:
    - Market cap fluctuates daily, potentially changing compliance status
    - A stock could become non-compliant due to price drops
    - Need to use trailing averages or periodic snapshots

    This implementation:
    - Uses point-in-time market cap by default
    - Supports average market cap over a period
    - Flags stocks close to thresholds as "questionable"
    """

    # Buffer zone - stocks within this % of threshold are flagged as questionable
    THRESHOLD_BUFFER = 0.05  # 5% buffer

    def __init__(self, config: shariah_config.__class__ | None = None):
        """
        Initialize the financial screener.

        Args:
            config: Shariah configuration (uses global config if not provided)
        """
        self.config = config or shariah_config

    def screen(self, financial_data: FinancialData) -> FinancialScreeningResult:
        """
        Screen a company based on financial ratios.

        Args:
            financial_data: FinancialData object with company financials

        Returns:
            FinancialScreeningResult with compliance status and ratios
        """
        # Check data completeness
        missing_data = self._check_data_completeness(financial_data)
        data_quality = self._assess_data_quality(missing_data)

        if data_quality == "insufficient":
            return FinancialScreeningResult(
                symbol=financial_data.symbol,
                status=ScreeningResult.INSUFFICIENT_DATA,
                is_compliant=False,
                ratios=FinancialRatios(),
                data_quality=data_quality,
                missing_data=missing_data,
                reasons=["Insufficient financial data for screening"],
            )

        # Calculate ratios
        ratios = self._calculate_ratios(financial_data)

        # Check thresholds
        thresholds_exceeded, reasons = self._check_thresholds(ratios)

        # Determine compliance
        is_compliant, status = self._determine_compliance(
            thresholds_exceeded, ratios, data_quality
        )

        return FinancialScreeningResult(
            symbol=financial_data.symbol,
            status=status,
            is_compliant=is_compliant,
            ratios=ratios,
            thresholds_exceeded=thresholds_exceeded,
            reasons=reasons,
            data_quality=data_quality,
            missing_data=missing_data,
        )

    def _check_data_completeness(self, data: FinancialData) -> list[str]:
        """Check which required data fields are missing."""
        missing = []

        # Critical fields
        if data.market_cap is None:
            missing.append("market_cap")

        # Debt screening
        if data.interest_bearing_debt is None and data.total_debt is None:
            missing.append("interest_bearing_debt or total_debt")

        # Income screening
        if data.total_revenue is None:
            missing.append("total_revenue")

        # Optional but important
        if data.interest_bearing_deposits is None:
            missing.append("interest_bearing_deposits")
        if data.interest_income is None:
            missing.append("interest_income")
        if data.accounts_receivable is None:
            missing.append("accounts_receivable")

        return missing

    def _assess_data_quality(self, missing_data: list[str]) -> str:
        """Assess overall data quality based on missing fields."""
        critical_missing = ["market_cap", "interest_bearing_debt or total_debt", "total_revenue"]

        critical_count = sum(1 for m in missing_data if m in critical_missing)

        if critical_count >= 2:
            return "insufficient"
        elif missing_data:
            return "partial"
        return "complete"

    def _calculate_ratios(self, data: FinancialData) -> FinancialRatios:
        """Calculate all financial ratios."""
        ratios = FinancialRatios()

        market_cap = data.market_cap or 0

        if market_cap > 0:
            # Debt ratio - use interest-bearing debt if available, else total debt
            debt = data.interest_bearing_debt
            if debt is None:
                debt = data.total_debt or 0
            ratios.debt_to_market_cap = debt / market_cap

            # Deposits ratio
            if data.interest_bearing_deposits is not None:
                ratios.deposits_to_market_cap = data.interest_bearing_deposits / market_cap
            elif data.cash_and_equivalents is not None:
                # Conservative: treat all cash as potentially interest-bearing
                # In practice, would need to verify if in Islamic accounts
                ratios.deposits_to_market_cap = data.cash_and_equivalents / market_cap

            # Receivables ratio
            if data.accounts_receivable is not None:
                ratios.receivables_to_market_cap = data.accounts_receivable / market_cap

            # Cash ratio (for reference)
            if data.cash_and_equivalents is not None:
                ratios.cash_to_market_cap = data.cash_and_equivalents / market_cap

        # Impermissible income ratio
        if data.total_revenue and data.total_revenue > 0:
            # Use explicitly defined non-permissible income if available
            if data.non_permissible_income is not None:
                ratios.impermissible_income_ratio = data.non_permissible_income / data.total_revenue
            elif data.interest_income is not None:
                # Conservative: treat interest income as non-permissible
                ratios.impermissible_income_ratio = data.interest_income / data.total_revenue
            else:
                # Cannot calculate - will be flagged as missing data
                pass

        # Debt to assets (supplementary ratio)
        if data.total_assets and data.total_assets > 0:
            debt = data.interest_bearing_debt or data.total_debt or 0
            ratios.debt_to_assets = debt / data.total_assets

        return ratios

    def _check_thresholds(
        self, ratios: FinancialRatios
    ) -> tuple[list[str], list[str]]:
        """
        Check if any ratios exceed AAOIFI thresholds.

        Returns:
            Tuple of (exceeded_thresholds, reasons)
        """
        exceeded = []
        reasons = []

        # Debt ratio check
        if ratios.debt_to_market_cap is not None:
            if ratios.debt_to_market_cap > self.config.max_debt_to_market_cap:
                exceeded.append(NonComplianceReason.DEBT_RATIO)
                reasons.append(
                    f"Debt/Market Cap ratio ({ratios.debt_to_market_cap:.2%}) exceeds "
                    f"threshold ({self.config.max_debt_to_market_cap:.2%})"
                )

        # Deposits ratio check
        if ratios.deposits_to_market_cap is not None:
            if ratios.deposits_to_market_cap > self.config.max_deposits_to_market_cap:
                exceeded.append(NonComplianceReason.DEPOSIT_RATIO)
                reasons.append(
                    f"Deposits/Market Cap ratio ({ratios.deposits_to_market_cap:.2%}) exceeds "
                    f"threshold ({self.config.max_deposits_to_market_cap:.2%})"
                )

        # Impermissible income check
        if ratios.impermissible_income_ratio is not None:
            if ratios.impermissible_income_ratio > self.config.max_impermissible_income_ratio:
                exceeded.append(NonComplianceReason.IMPERMISSIBLE_INCOME)
                reasons.append(
                    f"Impermissible income ratio ({ratios.impermissible_income_ratio:.2%}) exceeds "
                    f"threshold ({self.config.max_impermissible_income_ratio:.2%})"
                )

        # Receivables ratio check (optional, stricter screening)
        if ratios.receivables_to_market_cap is not None:
            if ratios.receivables_to_market_cap > self.config.max_receivables_to_market_cap:
                exceeded.append(NonComplianceReason.RECEIVABLES_RATIO)
                reasons.append(
                    f"Receivables/Market Cap ratio ({ratios.receivables_to_market_cap:.2%}) exceeds "
                    f"threshold ({self.config.max_receivables_to_market_cap:.2%})"
                )

        return exceeded, reasons

    def _determine_compliance(
        self,
        thresholds_exceeded: list[str],
        ratios: FinancialRatios,
        data_quality: str,
    ) -> tuple[bool, str]:
        """
        Determine final compliance status.

        Args:
            thresholds_exceeded: List of exceeded threshold names
            ratios: Calculated financial ratios
            data_quality: Quality assessment of input data

        Returns:
            Tuple of (is_compliant, status_string)
        """
        # Any threshold exceeded = non-compliant
        if thresholds_exceeded:
            return False, ScreeningResult.NON_COMPLIANT

        # Check if close to thresholds (questionable zone)
        if self._is_near_threshold(ratios):
            return True, ScreeningResult.QUESTIONABLE

        # Partial data quality
        if data_quality == "partial":
            return True, ScreeningResult.QUESTIONABLE

        return True, ScreeningResult.COMPLIANT

    def _is_near_threshold(self, ratios: FinancialRatios) -> bool:
        """Check if any ratio is within buffer zone of threshold."""
        checks = [
            (ratios.debt_to_market_cap, self.config.max_debt_to_market_cap),
            (ratios.deposits_to_market_cap, self.config.max_deposits_to_market_cap),
            (ratios.impermissible_income_ratio, self.config.max_impermissible_income_ratio),
        ]

        for ratio, threshold in checks:
            if ratio is not None:
                buffer_threshold = threshold * (1 - self.THRESHOLD_BUFFER)
                if ratio >= buffer_threshold:
                    return True

        return False

    def batch_screen(
        self, financial_data_list: list[FinancialData]
    ) -> list[FinancialScreeningResult]:
        """
        Screen multiple companies.

        Args:
            financial_data_list: List of FinancialData objects

        Returns:
            List of FinancialScreeningResult objects
        """
        return [self.screen(data) for data in financial_data_list]

    def screen_from_dict(self, data: dict[str, Any]) -> FinancialScreeningResult:
        """
        Screen a company from a dictionary of financial data.

        Convenience method for processing data from APIs or databases.

        Args:
            data: Dictionary with financial data fields

        Returns:
            FinancialScreeningResult
        """
        financial_data = FinancialData(
            symbol=data.get("symbol", "UNKNOWN"),
            market_cap=data.get("market_cap") or data.get("marketCap"),
            share_price=data.get("share_price") or data.get("price"),
            shares_outstanding=data.get("shares_outstanding") or data.get("sharesOutstanding"),
            total_debt=data.get("total_debt") or data.get("totalDebt"),
            interest_bearing_debt=data.get("interest_bearing_debt"),
            short_term_debt=data.get("short_term_debt") or data.get("shortTermDebt"),
            long_term_debt=data.get("long_term_debt") or data.get("longTermDebt"),
            cash_and_equivalents=data.get("cash_and_equivalents") or data.get("cash"),
            interest_bearing_deposits=data.get("interest_bearing_deposits"),
            accounts_receivable=data.get("accounts_receivable") or data.get("receivables"),
            total_assets=data.get("total_assets") or data.get("totalAssets"),
            total_revenue=data.get("total_revenue") or data.get("revenue"),
            interest_income=data.get("interest_income") or data.get("interestIncome"),
            non_permissible_income=data.get("non_permissible_income"),
            fiscal_year=data.get("fiscal_year"),
            fiscal_quarter=data.get("fiscal_quarter"),
            data_source=data.get("data_source"),
        )
        return self.screen(financial_data)
