"""
Shariah Purification Calculator.

In Islamic finance, when a Shariah-compliant company derives a small portion
of its income from non-permissible sources (within the 5% threshold), investors
are required to "purify" their returns by donating the proportional amount
to charity.

This module calculates and tracks purification obligations for:
1. Dividend income - most common
2. Capital gains - some scholars require this
3. Total returns - comprehensive approach
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from enum import Enum

from config.shariah_config import shariah_config


class PurificationMethod(Enum):
    """Methods for calculating purification amount."""

    # Only purify dividend income
    DIVIDEND_ONLY = "dividend_only"

    # Purify dividends + realized capital gains
    DIVIDEND_AND_GAINS = "dividend_and_gains"

    # Purify total returns (most conservative)
    TOTAL_RETURNS = "total_returns"


@dataclass
class PurificationRecord:
    """Record of purification calculation for a single transaction/period."""

    symbol: str
    record_date: date
    record_type: str  # "dividend", "capital_gain", "period_end"

    # Income/gain amounts
    gross_amount: Decimal
    purification_ratio: Decimal  # Non-compliant income / Total income
    purification_amount: Decimal

    # Context
    shares_held: Decimal | None = None
    cost_basis: Decimal | None = None
    sale_price: Decimal | None = None

    # Status tracking
    is_donated: bool = False
    donation_date: date | None = None
    donation_reference: str | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "symbol": self.symbol,
            "record_date": self.record_date.isoformat(),
            "record_type": self.record_type,
            "gross_amount": str(self.gross_amount),
            "purification_ratio": str(self.purification_ratio),
            "purification_amount": str(self.purification_amount),
            "shares_held": str(self.shares_held) if self.shares_held else None,
            "cost_basis": str(self.cost_basis) if self.cost_basis else None,
            "sale_price": str(self.sale_price) if self.sale_price else None,
            "is_donated": self.is_donated,
            "donation_date": self.donation_date.isoformat() if self.donation_date else None,
            "donation_reference": self.donation_reference,
            "created_at": self.created_at.isoformat(),
            "notes": self.notes,
        }


@dataclass
class PurificationSummary:
    """Summary of purification obligations over a period."""

    start_date: date
    end_date: date
    total_gross_income: Decimal
    total_purification_due: Decimal
    total_donated: Decimal
    outstanding_balance: Decimal
    records: list[PurificationRecord]
    by_symbol: dict[str, Decimal]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_gross_income": str(self.total_gross_income),
            "total_purification_due": str(self.total_purification_due),
            "total_donated": str(self.total_donated),
            "outstanding_balance": str(self.outstanding_balance),
            "records_count": len(self.records),
            "by_symbol": {k: str(v) for k, v in self.by_symbol.items()},
        }


class PurificationCalculator:
    """
    Calculates and tracks Shariah purification obligations.

    Purification Formula:
        Purification Amount = Gross Income × (Non-Compliant Income / Total Income)

    Example:
        - Company ABC has 3% non-compliant income
        - You receive $100 dividend
        - Purification = $100 × 0.03 = $3 to donate to charity

    The calculator maintains a history of all purification calculations
    and tracks donation status.
    """

    def __init__(
        self,
        method: PurificationMethod = PurificationMethod.DIVIDEND_ONLY,
        config: shariah_config.__class__ | None = None,
    ):
        """
        Initialize the purification calculator.

        Args:
            method: Calculation method to use
            config: Shariah configuration
        """
        self.method = method
        self.config = config or shariah_config
        self._records: list[PurificationRecord] = []
        self._purification_ratios: dict[str, Decimal] = {}

    def set_purification_ratio(self, symbol: str, ratio: float | Decimal) -> None:
        """
        Set the purification ratio for a symbol.

        This ratio represents the proportion of non-compliant income
        to total income for the company.

        Args:
            symbol: Stock ticker symbol
            ratio: Non-compliant income ratio (e.g., 0.03 for 3%)
        """
        self._purification_ratios[symbol.upper()] = Decimal(str(ratio))

    def get_purification_ratio(self, symbol: str) -> Decimal | None:
        """Get the purification ratio for a symbol."""
        return self._purification_ratios.get(symbol.upper())

    def calculate_dividend_purification(
        self,
        symbol: str,
        dividend_amount: float | Decimal,
        shares_held: float | Decimal | None = None,
        ex_date: date | None = None,
        purification_ratio: float | Decimal | None = None,
    ) -> PurificationRecord:
        """
        Calculate purification amount for dividend income.

        Args:
            symbol: Stock ticker symbol
            dividend_amount: Total dividend amount received
            shares_held: Number of shares held (for record keeping)
            ex_date: Ex-dividend date
            purification_ratio: Override ratio (uses stored ratio if not provided)

        Returns:
            PurificationRecord with calculated purification amount
        """
        symbol = symbol.upper()
        gross_amount = Decimal(str(dividend_amount))

        # Get purification ratio
        if purification_ratio is not None:
            ratio = Decimal(str(purification_ratio))
        else:
            ratio = self._purification_ratios.get(symbol, Decimal("0"))

        # Calculate purification amount
        purification_amount = (gross_amount * ratio).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        record = PurificationRecord(
            symbol=symbol,
            record_date=ex_date or date.today(),
            record_type="dividend",
            gross_amount=gross_amount,
            purification_ratio=ratio,
            purification_amount=purification_amount,
            shares_held=Decimal(str(shares_held)) if shares_held else None,
        )

        self._records.append(record)
        return record

    def calculate_capital_gain_purification(
        self,
        symbol: str,
        sale_price: float | Decimal,
        cost_basis: float | Decimal,
        shares_sold: float | Decimal,
        sale_date: date | None = None,
        purification_ratio: float | Decimal | None = None,
    ) -> PurificationRecord:
        """
        Calculate purification amount for capital gains.

        Some scholars require purification of capital gains, others don't.
        This method calculates based on the configured method.

        Args:
            symbol: Stock ticker symbol
            sale_price: Price per share at sale
            cost_basis: Cost per share
            shares_sold: Number of shares sold
            sale_date: Date of sale
            purification_ratio: Override ratio

        Returns:
            PurificationRecord with calculated purification amount
        """
        symbol = symbol.upper()
        sale_total = Decimal(str(sale_price)) * Decimal(str(shares_sold))
        cost_total = Decimal(str(cost_basis)) * Decimal(str(shares_sold))
        gain = max(sale_total - cost_total, Decimal("0"))

        # Get purification ratio
        if purification_ratio is not None:
            ratio = Decimal(str(purification_ratio))
        else:
            ratio = self._purification_ratios.get(symbol, Decimal("0"))

        # Calculate purification amount
        purification_amount = (gain * ratio).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        record = PurificationRecord(
            symbol=symbol,
            record_date=sale_date or date.today(),
            record_type="capital_gain",
            gross_amount=gain,
            purification_ratio=ratio,
            purification_amount=purification_amount,
            shares_held=Decimal(str(shares_sold)),
            cost_basis=Decimal(str(cost_basis)),
            sale_price=Decimal(str(sale_price)),
        )

        self._records.append(record)
        return record

    def mark_as_donated(
        self,
        record: PurificationRecord,
        donation_date: date | None = None,
        reference: str | None = None,
    ) -> None:
        """
        Mark a purification record as donated.

        Args:
            record: The purification record to update
            donation_date: Date of donation
            reference: Reference number or notes
        """
        record.is_donated = True
        record.donation_date = donation_date or date.today()
        record.donation_reference = reference

    def get_outstanding_balance(self) -> Decimal:
        """Get total outstanding purification balance (not yet donated)."""
        return sum(
            (r.purification_amount for r in self._records if not r.is_donated),
            Decimal("0"),
        )

    def get_summary(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        symbol: str | None = None,
    ) -> PurificationSummary:
        """
        Get a summary of purification obligations.

        Args:
            start_date: Filter records from this date
            end_date: Filter records until this date
            symbol: Filter for specific symbol

        Returns:
            PurificationSummary with aggregated data
        """
        # Filter records
        filtered = self._records

        if start_date:
            filtered = [r for r in filtered if r.record_date >= start_date]
        if end_date:
            filtered = [r for r in filtered if r.record_date <= end_date]
        if symbol:
            symbol = symbol.upper()
            filtered = [r for r in filtered if r.symbol == symbol]

        # Calculate totals
        total_gross = sum((r.gross_amount for r in filtered), Decimal("0"))
        total_due = sum((r.purification_amount for r in filtered), Decimal("0"))
        total_donated = sum(
            (r.purification_amount for r in filtered if r.is_donated),
            Decimal("0"),
        )

        # By symbol breakdown
        by_symbol: dict[str, Decimal] = {}
        for record in filtered:
            if record.symbol not in by_symbol:
                by_symbol[record.symbol] = Decimal("0")
            by_symbol[record.symbol] += record.purification_amount

        # Determine date range
        if filtered:
            actual_start = min(r.record_date for r in filtered)
            actual_end = max(r.record_date for r in filtered)
        else:
            actual_start = start_date or date.today()
            actual_end = end_date or date.today()

        return PurificationSummary(
            start_date=actual_start,
            end_date=actual_end,
            total_gross_income=total_gross,
            total_purification_due=total_due,
            total_donated=total_donated,
            outstanding_balance=total_due - total_donated,
            records=filtered,
            by_symbol=by_symbol,
        )

    def get_records(
        self,
        symbol: str | None = None,
        include_donated: bool = True,
    ) -> list[PurificationRecord]:
        """
        Get purification records.

        Args:
            symbol: Filter by symbol
            include_donated: Include already donated records

        Returns:
            List of PurificationRecord objects
        """
        records = self._records

        if symbol:
            symbol = symbol.upper()
            records = [r for r in records if r.symbol == symbol]

        if not include_donated:
            records = [r for r in records if not r.is_donated]

        return sorted(records, key=lambda r: r.record_date, reverse=True)

    def export_records(self, filepath: str | None = None) -> list[dict[str, Any]]:
        """
        Export all records as a list of dictionaries.

        Args:
            filepath: Optional path to save as JSON

        Returns:
            List of record dictionaries
        """
        records_data = [r.to_dict() for r in self._records]

        if filepath:
            import json
            with open(filepath, "w") as f:
                json.dump(records_data, f, indent=2)

        return records_data

    def import_records(self, records_data: list[dict[str, Any]]) -> int:
        """
        Import records from a list of dictionaries.

        Args:
            records_data: List of record dictionaries

        Returns:
            Number of records imported
        """
        count = 0
        for data in records_data:
            record = PurificationRecord(
                symbol=data["symbol"],
                record_date=date.fromisoformat(data["record_date"]),
                record_type=data["record_type"],
                gross_amount=Decimal(data["gross_amount"]),
                purification_ratio=Decimal(data["purification_ratio"]),
                purification_amount=Decimal(data["purification_amount"]),
                shares_held=Decimal(data["shares_held"]) if data.get("shares_held") else None,
                cost_basis=Decimal(data["cost_basis"]) if data.get("cost_basis") else None,
                sale_price=Decimal(data["sale_price"]) if data.get("sale_price") else None,
                is_donated=data.get("is_donated", False),
                donation_date=date.fromisoformat(data["donation_date"]) if data.get("donation_date") else None,
                donation_reference=data.get("donation_reference"),
                notes=data.get("notes"),
            )
            self._records.append(record)
            count += 1

        return count

    def generate_report(self, year: int | None = None) -> str:
        """
        Generate a human-readable purification report.

        Args:
            year: Filter for specific year (defaults to current year)

        Returns:
            Formatted report string
        """
        if year is None:
            year = date.today().year

        start = date(year, 1, 1)
        end = date(year, 12, 31)
        summary = self.get_summary(start_date=start, end_date=end)

        lines = [
            f"=" * 60,
            f"SHARIAH PURIFICATION REPORT - {year}",
            f"=" * 60,
            "",
            f"Period: {summary.start_date} to {summary.end_date}",
            f"Total Gross Income:      ${summary.total_gross_income:>12,.2f}",
            f"Total Purification Due:  ${summary.total_purification_due:>12,.2f}",
            f"Amount Donated:          ${summary.total_donated:>12,.2f}",
            f"Outstanding Balance:     ${summary.outstanding_balance:>12,.2f}",
            "",
            "-" * 60,
            "BY SYMBOL:",
            "-" * 60,
        ]

        for symbol, amount in sorted(summary.by_symbol.items()):
            lines.append(f"  {symbol:<10} ${amount:>12,.2f}")

        lines.extend([
            "",
            "-" * 60,
            "RECORDS:",
            "-" * 60,
        ])

        for record in sorted(summary.records, key=lambda r: r.record_date):
            status = "✓ Donated" if record.is_donated else "◯ Pending"
            lines.append(
                f"  {record.record_date} | {record.symbol:<6} | "
                f"{record.record_type:<12} | ${record.purification_amount:>8,.2f} | {status}"
            )

        lines.extend([
            "",
            "=" * 60,
            "Note: Purification amounts should be donated to charity.",
            "=" * 60,
        ])

        return "\n".join(lines)
