#!/usr/bin/env python3
"""
Standalone Demo - No External Dependencies Required

This demonstrates the core Shariah screening logic without
requiring pydantic, pandas, or other external packages.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ============================================================
# AAOIFI SCREENING THRESHOLDS
# ============================================================

AAOIFI_THRESHOLDS = {
    "max_debt_to_market_cap": 0.30,       # 30%
    "max_deposits_to_market_cap": 0.30,   # 30%
    "max_impermissible_income": 0.05,     # 5%
    "max_receivables_to_market_cap": 0.49, # 49%
}

PROHIBITED_INDUSTRIES = {
    "alcoholic_beverages",
    "tobacco",
    "pork_products",
    "gambling",
    "adult_entertainment",
    "conventional_banking",
    "conventional_insurance",
    "weapons_defense",
}

# NAICS codes for prohibited industries
PROHIBITED_NAICS = {
    "312120": "alcoholic_beverages",  # Breweries
    "312130": "alcoholic_beverages",  # Wineries
    "312140": "alcoholic_beverages",  # Distilleries
    "312230": "tobacco",              # Tobacco manufacturing
    "713210": "gambling",             # Casinos
    "713290": "gambling",             # Other gambling
    "522110": "conventional_banking", # Commercial banking
    "522120": "conventional_banking", # Savings institutions
    "524113": "conventional_insurance", # Life insurance
    "524126": "conventional_insurance", # Property insurance
}


# ============================================================
# SCREENING FUNCTIONS
# ============================================================

def screen_business_activity(
    symbol: str,
    naics_code: str | None = None,
    industry_name: str | None = None,
) -> dict[str, Any]:
    """Screen company based on business activity."""

    result = {
        "symbol": symbol,
        "is_compliant": True,
        "prohibited_industry": None,
        "reason": None,
    }

    # Check NAICS code
    if naics_code and naics_code in PROHIBITED_NAICS:
        result["is_compliant"] = False
        result["prohibited_industry"] = PROHIBITED_NAICS[naics_code]
        result["reason"] = f"NAICS code {naics_code} is prohibited ({result['prohibited_industry']})"
        return result

    # Check industry name keywords
    if industry_name:
        industry_lower = industry_name.lower()
        prohibited_keywords = {
            "bank": "conventional_banking",
            "insurance": "conventional_insurance",
            "casino": "gambling",
            "brewery": "alcoholic_beverages",
            "tobacco": "tobacco",
            "alcohol": "alcoholic_beverages",
        }
        for keyword, industry in prohibited_keywords.items():
            if keyword in industry_lower:
                result["is_compliant"] = False
                result["prohibited_industry"] = industry
                result["reason"] = f"Industry name contains '{keyword}'"
                return result

    result["reason"] = "No prohibited business activity detected"
    return result


def screen_financial_ratios(
    symbol: str,
    market_cap: float,
    total_debt: float,
    interest_bearing_deposits: float = 0,
    non_permissible_income: float = 0,
    total_revenue: float = 1,
    accounts_receivable: float = 0,
) -> dict[str, Any]:
    """Screen company based on AAOIFI financial ratios."""

    result = {
        "symbol": symbol,
        "is_compliant": True,
        "ratios": {},
        "exceeded": [],
        "reasons": [],
    }

    # Calculate ratios
    debt_ratio = total_debt / market_cap if market_cap > 0 else 0
    deposit_ratio = interest_bearing_deposits / market_cap if market_cap > 0 else 0
    income_ratio = non_permissible_income / total_revenue if total_revenue > 0 else 0
    receivables_ratio = accounts_receivable / market_cap if market_cap > 0 else 0

    result["ratios"] = {
        "debt_to_market_cap": debt_ratio,
        "deposits_to_market_cap": deposit_ratio,
        "impermissible_income": income_ratio,
        "receivables_to_market_cap": receivables_ratio,
    }

    # Check thresholds
    if debt_ratio > AAOIFI_THRESHOLDS["max_debt_to_market_cap"]:
        result["is_compliant"] = False
        result["exceeded"].append("debt_ratio")
        result["reasons"].append(
            f"Debt/Market Cap {debt_ratio:.1%} exceeds {AAOIFI_THRESHOLDS['max_debt_to_market_cap']:.0%}"
        )

    if deposit_ratio > AAOIFI_THRESHOLDS["max_deposits_to_market_cap"]:
        result["is_compliant"] = False
        result["exceeded"].append("deposit_ratio")
        result["reasons"].append(
            f"Deposits/Market Cap {deposit_ratio:.1%} exceeds {AAOIFI_THRESHOLDS['max_deposits_to_market_cap']:.0%}"
        )

    if income_ratio > AAOIFI_THRESHOLDS["max_impermissible_income"]:
        result["is_compliant"] = False
        result["exceeded"].append("income_ratio")
        result["reasons"].append(
            f"Impermissible income {income_ratio:.1%} exceeds {AAOIFI_THRESHOLDS['max_impermissible_income']:.0%}"
        )

    if not result["reasons"]:
        result["reasons"].append("All financial ratios within AAOIFI limits")

    return result


def calculate_purification(
    dividend_amount: float,
    impermissible_income_ratio: float,
) -> dict[str, float]:
    """Calculate purification amount for dividends."""
    purification = dividend_amount * impermissible_income_ratio
    return {
        "dividend_amount": dividend_amount,
        "impermissible_ratio": impermissible_income_ratio,
        "purification_amount": round(purification, 2),
        "net_halal_income": round(dividend_amount - purification, 2),
    }


def calculate_position_size(
    portfolio_value: float,
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade: float = 0.02,  # 2%
    max_position_pct: float = 0.05,  # 5%
) -> dict[str, Any]:
    """Calculate position size based on risk parameters."""

    risk_amount = portfolio_value * risk_per_trade
    stop_distance = abs(entry_price - stop_loss_price)

    if stop_distance > 0:
        shares_by_risk = int(risk_amount / stop_distance)
    else:
        shares_by_risk = 0

    max_position_value = portfolio_value * max_position_pct
    shares_by_position = int(max_position_value / entry_price) if entry_price > 0 else 0

    shares = min(shares_by_risk, shares_by_position)
    position_value = shares * entry_price

    return {
        "shares": shares,
        "position_value": round(position_value, 2),
        "risk_amount": round(shares * stop_distance, 2),
        "position_pct": round(position_value / portfolio_value * 100, 2) if portfolio_value > 0 else 0,
    }


# ============================================================
# DEMONSTRATION
# ============================================================

def print_header(text: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_result(label: str, value: Any, indent: int = 2):
    """Print a result line."""
    print(f"{' ' * indent}{label}: {value}")


def main():
    """Run the demonstration."""

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║        SHARIAH-COMPLIANT TRADING BOT DEMO                 ║
    ║                                                           ║
    ║   Demonstrating AAOIFI Screening Standards                ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # --------------------------------------------------------
    # 1. Business Activity Screening
    # --------------------------------------------------------
    print_header("1. BUSINESS ACTIVITY SCREENING")

    test_companies = [
        {"symbol": "AAPL", "industry": "Consumer Electronics"},
        {"symbol": "MSFT", "industry": "Software - Infrastructure"},
        {"symbol": "JPM", "naics": "522110", "industry": "Commercial Banking"},
        {"symbol": "BUD", "naics": "312120", "industry": "Breweries"},
        {"symbol": "NVDA", "industry": "Semiconductors"},
    ]

    print("\nScreening companies by business activity:\n")
    for company in test_companies:
        result = screen_business_activity(
            symbol=company["symbol"],
            naics_code=company.get("naics"),
            industry_name=company.get("industry"),
        )
        status = "✓ COMPLIANT" if result["is_compliant"] else "✗ NON-COMPLIANT"
        print(f"  {company['symbol']:6} ({company.get('industry', 'N/A'):25}) -> {status}")
        if not result["is_compliant"]:
            print(f"         Reason: {result['reason']}")

    # --------------------------------------------------------
    # 2. Financial Ratio Screening
    # --------------------------------------------------------
    print_header("2. FINANCIAL RATIO SCREENING (AAOIFI)")

    print(f"\nAAOIFI Thresholds:")
    print(f"  • Debt/Market Cap:           < {AAOIFI_THRESHOLDS['max_debt_to_market_cap']:.0%}")
    print(f"  • Deposits/Market Cap:       < {AAOIFI_THRESHOLDS['max_deposits_to_market_cap']:.0%}")
    print(f"  • Impermissible Income:      < {AAOIFI_THRESHOLDS['max_impermissible_income']:.0%}")
    print(f"  • Receivables/Market Cap:    < {AAOIFI_THRESHOLDS['max_receivables_to_market_cap']:.0%}")

    # Example: Compliant company
    print(f"\n--- Example 1: TECH_CO (Low Debt Company) ---")
    result = screen_financial_ratios(
        symbol="TECH_CO",
        market_cap=100_000_000_000,    # $100B
        total_debt=15_000_000_000,     # $15B (15%)
        non_permissible_income=500_000_000,  # $500M
        total_revenue=50_000_000_000,  # $50B (1% impermissible)
    )
    status = "✓ COMPLIANT" if result["is_compliant"] else "✗ NON-COMPLIANT"
    print(f"  Status: {status}")
    print(f"  Debt/Market Cap:        {result['ratios']['debt_to_market_cap']:.1%} (limit: 30%)")
    print(f"  Impermissible Income:   {result['ratios']['impermissible_income']:.1%} (limit: 5%)")

    # Example: Non-compliant company (high debt)
    print(f"\n--- Example 2: LEVERED_CO (High Debt Company) ---")
    result = screen_financial_ratios(
        symbol="LEVERED_CO",
        market_cap=50_000_000_000,     # $50B
        total_debt=25_000_000_000,     # $25B (50%!)
        non_permissible_income=300_000_000,
        total_revenue=30_000_000_000,
    )
    status = "✓ COMPLIANT" if result["is_compliant"] else "✗ NON-COMPLIANT"
    print(f"  Status: {status}")
    print(f"  Debt/Market Cap:        {result['ratios']['debt_to_market_cap']:.1%} (limit: 30%)")
    print(f"  Reason: {result['reasons'][0]}")

    # --------------------------------------------------------
    # 3. Purification Calculation
    # --------------------------------------------------------
    print_header("3. DIVIDEND PURIFICATION")

    print("""
When a compliant company has <5% impermissible income, that
portion of dividends must be donated to charity (purified).

Formula: Purification = Dividend × Impermissible Income Ratio
    """)

    # Example purification
    purification = calculate_purification(
        dividend_amount=1000.00,
        impermissible_income_ratio=0.03,  # 3%
    )

    print(f"Example: Company with 3% impermissible income")
    print(f"  Dividend received:    ${purification['dividend_amount']:.2f}")
    print(f"  Impermissible ratio:  {purification['impermissible_ratio']:.1%}")
    print(f"  Purification amount:  ${purification['purification_amount']:.2f} (donate to charity)")
    print(f"  Net halal income:     ${purification['net_halal_income']:.2f}")

    # --------------------------------------------------------
    # 4. Risk Management
    # --------------------------------------------------------
    print_header("4. POSITION SIZING (RISK MANAGEMENT)")

    print("""
Position sizing based on:
  • Max 2% portfolio risk per trade
  • Max 5% portfolio in single position
    """)

    position = calculate_position_size(
        portfolio_value=100_000,
        entry_price=150.00,
        stop_loss_price=145.00,  # $5 stop distance
        risk_per_trade=0.02,
        max_position_pct=0.05,
    )

    print(f"Example: $100,000 portfolio, $150 stock, $145 stop loss")
    print(f"  Shares to buy:      {position['shares']}")
    print(f"  Position value:     ${position['position_value']:,.2f}")
    print(f"  Risk amount:        ${position['risk_amount']:,.2f}")
    print(f"  Position % of portfolio: {position['position_pct']}%")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print_header("SUMMARY")

    print("""
Shariah-Compliant Investing requires:

1. BUSINESS SCREENING
   - Exclude alcohol, tobacco, gambling, conventional finance
   - Use NAICS/SIC codes for classification

2. FINANCIAL SCREENING (AAOIFI)
   - Debt/Market Cap < 30%
   - Interest-bearing deposits/Market Cap < 30%
   - Impermissible income/Revenue < 5%

3. PURIFICATION
   - Donate impermissible portion of dividends to charity

4. RISK MANAGEMENT
   - Apply sound risk principles (not Shariah-specific)
   - 2% risk per trade, 5% max position

Note: Always consult qualified Shariah scholars for guidance.
This is an educational demonstration only.
    """)

    print(f"\n{'=' * 60}")
    print(f"  Demo completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
