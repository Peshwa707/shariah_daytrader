"""
Business Activity Screener for Shariah Compliance.

This module screens companies based on their primary business activities
using NAICS and SIC industry codes, as well as revenue breakdown analysis.

Islamic finance principles prohibit investment in companies primarily engaged in:
- Alcohol production/distribution
- Tobacco products
- Pork-related products
- Gambling and gaming
- Adult entertainment
- Conventional banking/insurance (interest-based)
- Weapons manufacturing (controversial - some scholars allow defense)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config.shariah_config import shariah_config, ScreeningResult, NonComplianceReason


class ProhibitedIndustry(Enum):
    """Categories of prohibited business activities."""

    ALCOHOL = "alcoholic_beverages"
    TOBACCO = "tobacco"
    PORK = "pork_products"
    GAMBLING = "gambling"
    ADULT_ENTERTAINMENT = "adult_entertainment"
    CONVENTIONAL_BANKING = "conventional_banking"
    CONVENTIONAL_INSURANCE = "conventional_insurance"
    WEAPONS = "weapons_defense"


@dataclass
class BusinessScreeningResult:
    """Result of business activity screening."""

    symbol: str
    status: str  # ScreeningResult value
    is_compliant: bool
    prohibited_industries: list[str] = field(default_factory=list)
    prohibited_revenue_ratio: float = 0.0
    naics_code: str | None = None
    sic_code: str | None = None
    industry_name: str | None = None
    sector: str | None = None
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "symbol": self.symbol,
            "status": self.status,
            "is_compliant": self.is_compliant,
            "prohibited_industries": self.prohibited_industries,
            "prohibited_revenue_ratio": self.prohibited_revenue_ratio,
            "naics_code": self.naics_code,
            "sic_code": self.sic_code,
            "industry_name": self.industry_name,
            "sector": self.sector,
            "reasons": self.reasons,
            "details": self.details,
        }


class BusinessScreener:
    """
    Screens companies for Shariah compliance based on business activities.

    Uses a multi-layered approach:
    1. NAICS/SIC code matching for quick industry identification
    2. Industry name/description keyword matching
    3. Revenue breakdown analysis (if available)
    """

    def __init__(self, config: shariah_config.__class__ | None = None):
        """
        Initialize the business screener.

        Args:
            config: Shariah configuration (uses global config if not provided)
        """
        self.config = config or shariah_config
        self._build_code_lookup()
        self._build_keyword_patterns()

    def _build_code_lookup(self) -> None:
        """Build reverse lookup from codes to prohibited industries."""
        self.naics_to_industry: dict[str, str] = {}
        self.sic_to_industry: dict[str, str] = {}

        for industry, codes in self.config.prohibited_naics_codes.items():
            for code in codes:
                self.naics_to_industry[code] = industry

        for industry, codes in self.config.prohibited_sic_codes.items():
            for code in codes:
                self.sic_to_industry[code] = industry

    def _build_keyword_patterns(self) -> None:
        """Build keyword patterns for industry name matching."""
        self.prohibited_keywords: dict[str, list[str]] = {
            ProhibitedIndustry.ALCOHOL.value: [
                "brewery", "breweries", "winery", "wineries", "distillery",
                "distilleries", "alcoholic", "liquor", "beer", "wine", "spirits",
                "vodka", "whiskey", "whisky", "bourbon", "rum", "gin", "tequila",
            ],
            ProhibitedIndustry.TOBACCO.value: [
                "tobacco", "cigarette", "cigar", "smoking", "nicotine", "vape",
                "vaping", "e-cigarette",
            ],
            ProhibitedIndustry.GAMBLING.value: [
                "casino", "gambling", "gaming", "lottery", "betting", "wager",
                "slot machine", "poker", "blackjack", "sportsbook",
            ],
            ProhibitedIndustry.ADULT_ENTERTAINMENT.value: [
                "adult entertainment", "adult content", "pornograph",
            ],
            ProhibitedIndustry.CONVENTIONAL_BANKING.value: [
                "commercial bank", "savings bank", "credit union",
                "consumer lending", "mortgage lend", "credit card issuer",
            ],
            ProhibitedIndustry.CONVENTIONAL_INSURANCE.value: [
                "life insurance", "health insurance", "property insurance",
                "casualty insurance", "insurance carrier", "reinsurance",
            ],
            ProhibitedIndustry.WEAPONS.value: [
                "weapons", "ammunition", "firearms", "defense contractor",
                "military equipment", "ordnance", "missile", "munitions",
            ],
            ProhibitedIndustry.PORK.value: [
                "pork", "swine", "hog farm", "pig farm", "bacon", "ham producer",
            ],
        }

    def screen(
        self,
        symbol: str,
        naics_code: str | None = None,
        sic_code: str | None = None,
        industry_name: str | None = None,
        sector: str | None = None,
        revenue_breakdown: dict[str, float] | None = None,
    ) -> BusinessScreeningResult:
        """
        Screen a company for business activity compliance.

        Args:
            symbol: Stock ticker symbol
            naics_code: NAICS industry code
            sic_code: SIC industry code
            industry_name: Industry description/name
            sector: Sector classification
            revenue_breakdown: Optional dict mapping revenue sources to percentages

        Returns:
            BusinessScreeningResult with compliance status and details
        """
        prohibited_industries: list[str] = []
        reasons: list[str] = []
        details: dict[str, Any] = {}

        # Check NAICS code
        if naics_code:
            naics_match = self._check_naics_code(naics_code)
            if naics_match:
                prohibited_industries.append(naics_match)
                reasons.append(f"NAICS code {naics_code} matches prohibited industry: {naics_match}")
                details["naics_match"] = naics_match

        # Check SIC code
        if sic_code:
            sic_match = self._check_sic_code(sic_code)
            if sic_match:
                if sic_match not in prohibited_industries:
                    prohibited_industries.append(sic_match)
                reasons.append(f"SIC code {sic_code} matches prohibited industry: {sic_match}")
                details["sic_match"] = sic_match

        # Check industry name keywords
        if industry_name:
            keyword_matches = self._check_industry_keywords(industry_name)
            for match in keyword_matches:
                if match not in prohibited_industries:
                    prohibited_industries.append(match)
                    reasons.append(f"Industry name contains prohibited keyword: {match}")
            details["keyword_matches"] = keyword_matches

        # Check revenue breakdown if available
        prohibited_revenue_ratio = 0.0
        if revenue_breakdown:
            prohibited_revenue_ratio, revenue_details = self._analyze_revenue_breakdown(
                revenue_breakdown
            )
            details["revenue_analysis"] = revenue_details
            if prohibited_revenue_ratio > self.config.max_prohibited_revenue_ratio:
                reasons.append(
                    f"Prohibited revenue ratio {prohibited_revenue_ratio:.2%} "
                    f"exceeds threshold {self.config.max_prohibited_revenue_ratio:.2%}"
                )

        # Determine compliance status
        is_compliant, status = self._determine_compliance(
            prohibited_industries, prohibited_revenue_ratio
        )

        return BusinessScreeningResult(
            symbol=symbol,
            status=status,
            is_compliant=is_compliant,
            prohibited_industries=prohibited_industries,
            prohibited_revenue_ratio=prohibited_revenue_ratio,
            naics_code=naics_code,
            sic_code=sic_code,
            industry_name=industry_name,
            sector=sector,
            reasons=reasons,
            details=details,
        )

    def _check_naics_code(self, naics_code: str) -> str | None:
        """Check if NAICS code matches a prohibited industry."""
        # Direct match
        if naics_code in self.naics_to_industry:
            return self.naics_to_industry[naics_code]

        # Check parent codes (NAICS is hierarchical)
        # 6-digit -> 5-digit -> 4-digit -> 3-digit -> 2-digit
        for length in [5, 4, 3, 2]:
            parent_code = naics_code[:length]
            if parent_code in self.naics_to_industry:
                return self.naics_to_industry[parent_code]

        return None

    def _check_sic_code(self, sic_code: str) -> str | None:
        """Check if SIC code matches a prohibited industry."""
        # Direct match
        if sic_code in self.sic_to_industry:
            return self.sic_to_industry[sic_code]

        # Check parent codes (SIC is hierarchical)
        for length in [3, 2]:
            parent_code = sic_code[:length]
            if parent_code in self.sic_to_industry:
                return self.sic_to_industry[parent_code]

        return None

    def _check_industry_keywords(self, industry_name: str) -> list[str]:
        """Check industry name for prohibited keywords."""
        matches = []
        industry_lower = industry_name.lower()

        for industry, keywords in self.prohibited_keywords.items():
            for keyword in keywords:
                if keyword.lower() in industry_lower:
                    if industry not in matches:
                        matches.append(industry)
                    break

        return matches

    def _analyze_revenue_breakdown(
        self, revenue_breakdown: dict[str, float]
    ) -> tuple[float, dict[str, Any]]:
        """
        Analyze revenue breakdown for prohibited income sources.

        Args:
            revenue_breakdown: Dict mapping revenue source names to percentages

        Returns:
            Tuple of (prohibited_revenue_ratio, analysis_details)
        """
        prohibited_revenue = 0.0
        analysis = {
            "sources_checked": len(revenue_breakdown),
            "prohibited_sources": [],
        }

        for source, percentage in revenue_breakdown.items():
            source_lower = source.lower()

            # Check each prohibited category
            for industry, keywords in self.prohibited_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in source_lower:
                        prohibited_revenue += percentage
                        analysis["prohibited_sources"].append({
                            "source": source,
                            "percentage": percentage,
                            "matched_industry": industry,
                        })
                        break

        analysis["total_prohibited_ratio"] = prohibited_revenue
        return prohibited_revenue, analysis

    def _determine_compliance(
        self, prohibited_industries: list[str], prohibited_revenue_ratio: float
    ) -> tuple[bool, str]:
        """
        Determine final compliance status.

        Args:
            prohibited_industries: List of matched prohibited industries
            prohibited_revenue_ratio: Ratio of revenue from prohibited sources

        Returns:
            Tuple of (is_compliant, status_string)
        """
        # Primary prohibited industry - automatic non-compliance
        primary_prohibited = set(self.config.primary_prohibited_industries)
        for industry in prohibited_industries:
            if industry in primary_prohibited:
                return False, ScreeningResult.NON_COMPLIANT

        # Check revenue threshold for secondary activities
        if prohibited_revenue_ratio > self.config.max_prohibited_revenue_ratio:
            return False, ScreeningResult.NON_COMPLIANT

        # If we found some prohibited indicators but within threshold
        if prohibited_industries and prohibited_revenue_ratio <= self.config.max_prohibited_revenue_ratio:
            return True, ScreeningResult.QUESTIONABLE

        return True, ScreeningResult.COMPLIANT

    def batch_screen(
        self, companies: list[dict[str, Any]]
    ) -> list[BusinessScreeningResult]:
        """
        Screen multiple companies.

        Args:
            companies: List of company data dicts with keys:
                - symbol (required)
                - naics_code, sic_code, industry_name, sector, revenue_breakdown (optional)

        Returns:
            List of BusinessScreeningResult objects
        """
        results = []
        for company in companies:
            result = self.screen(
                symbol=company["symbol"],
                naics_code=company.get("naics_code"),
                sic_code=company.get("sic_code"),
                industry_name=company.get("industry_name"),
                sector=company.get("sector"),
                revenue_breakdown=company.get("revenue_breakdown"),
            )
            results.append(result)
        return results
