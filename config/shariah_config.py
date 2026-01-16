"""
Shariah compliance configuration based on AAOIFI standards.

AAOIFI = Accounting and Auditing Organization for Islamic Financial Institutions
These standards are widely accepted for Shariah-compliant investing.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ShariahConfig(BaseSettings):
    """
    Shariah screening thresholds based on AAOIFI Standard No. 21.

    Two main screening approaches exist:
    1. S&P Shariah Index methodology
    2. AAOIFI standards

    This implementation uses AAOIFI as the primary standard with options
    to configure thresholds based on different scholarly opinions.
    """

    model_config = SettingsConfigDict(
        env_prefix="SHARIAH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ========== Financial Ratio Thresholds (AAOIFI) ==========

    # Interest-bearing debt to market capitalization
    # AAOIFI: < 30% (some scholars use 33%)
    max_debt_to_market_cap: float = Field(
        default=0.30,
        description="Maximum interest-bearing debt / market cap ratio"
    )

    # Interest-bearing deposits to market capitalization
    # Cash in conventional interest-bearing accounts
    max_deposits_to_market_cap: float = Field(
        default=0.30,
        description="Maximum interest-bearing deposits / market cap ratio"
    )

    # Non-permissible income to total revenue
    # Income from prohibited activities (interest income, etc.)
    max_impermissible_income_ratio: float = Field(
        default=0.05,
        description="Maximum non-permissible income / total revenue ratio"
    )

    # Accounts receivable to market cap (optional, stricter screening)
    # Some methodologies screen for excessive receivables
    max_receivables_to_market_cap: float = Field(
        default=0.49,
        description="Maximum accounts receivable / market cap ratio"
    )

    # ========== Business Activity Thresholds ==========

    # Revenue threshold for prohibited business activities
    max_prohibited_revenue_ratio: float = Field(
        default=0.05,
        description="Maximum revenue from prohibited activities"
    )

    # ========== Prohibited Business Activities ==========
    # Based on AAOIFI and major Islamic index methodologies

    # Primary prohibited industries (0% tolerance typically)
    primary_prohibited_industries: list[str] = Field(
        default=[
            "alcoholic_beverages",
            "tobacco",
            "pork_products",
            "gambling",
            "adult_entertainment",
            "conventional_banking",
            "conventional_insurance",
            "weapons_defense",  # Controversial - some allow defense
        ],
        description="Industries with primary prohibition"
    )

    # Secondary prohibited activities (5% revenue threshold)
    secondary_prohibited_activities: list[str] = Field(
        default=[
            "interest_income",
            "alcohol_related_services",
            "gambling_related_services",
            "entertainment_restricted",
        ],
        description="Activities prohibited above threshold"
    )

    # ========== NAICS Codes for Prohibited Industries ==========
    # North American Industry Classification System codes

    prohibited_naics_codes: dict[str, list[str]] = Field(
        default={
            "alcoholic_beverages": [
                "312120",  # Breweries
                "312130",  # Wineries
                "312140",  # Distilleries
                "424810",  # Beer and ale merchant wholesalers
                "424820",  # Wine and distilled alcoholic beverage wholesalers
                "445310",  # Beer, wine, and liquor stores
                "722410",  # Drinking places (alcoholic beverages)
            ],
            "tobacco": [
                "312230",  # Tobacco manufacturing
                "424940",  # Tobacco product merchant wholesalers
                "453991",  # Tobacco stores
            ],
            "gambling": [
                "713210",  # Casinos (except casino hotels)
                "713290",  # Other gambling industries
                "721120",  # Casino hotels
            ],
            "conventional_banking": [
                "522110",  # Commercial banking
                "522120",  # Savings institutions
                "522130",  # Credit unions
                "522190",  # Other depository credit intermediation
                "522210",  # Credit card issuing
                "522220",  # Sales financing
                "522291",  # Consumer lending
                "522292",  # Real estate credit
                "522293",  # International trade financing
                "522294",  # Secondary market financing
                "522298",  # All other nondepository credit intermediation
            ],
            "conventional_insurance": [
                "524113",  # Direct life insurance carriers
                "524114",  # Direct health and medical insurance carriers
                "524126",  # Direct property and casualty insurance carriers
                "524127",  # Direct title insurance carriers
                "524128",  # Other direct insurance carriers
                "524130",  # Reinsurance carriers
            ],
            "weapons_defense": [
                "332992",  # Small arms ammunition manufacturing
                "332993",  # Ammunition (except small arms) manufacturing
                "332994",  # Small arms, ordnance, and accessories manufacturing
                "336411",  # Aircraft manufacturing (military specific)
                "336414",  # Guided missile and space vehicle manufacturing
                "336415",  # Guided missile and space vehicle parts manufacturing
            ],
            "adult_entertainment": [
                "512131",  # Motion picture theaters (adult specific)
                "711110",  # Theater companies (adult specific)
            ],
            "pork_products": [
                "311611",  # Animal (except poultry) slaughtering - requires revenue check
                "311612",  # Meat processed from carcasses - requires revenue check
            ],
        },
        description="NAICS codes mapped to prohibited industries"
    )

    # ========== SIC Codes for Prohibited Industries ==========
    # Standard Industrial Classification (legacy, still used by some data providers)

    prohibited_sic_codes: dict[str, list[str]] = Field(
        default={
            "alcoholic_beverages": [
                "2082",  # Malt beverages
                "2083",  # Malt
                "2084",  # Wines, brandy, and brandy spirits
                "2085",  # Distilled and blended liquors
                "5181",  # Beer and ale
                "5182",  # Wine and distilled alcoholic beverages
                "5813",  # Drinking places
                "5921",  # Liquor stores
            ],
            "tobacco": [
                "2111",  # Cigarettes
                "2121",  # Cigars
                "2131",  # Chewing and smoking tobacco
                "2141",  # Tobacco stemming and redrying
                "5194",  # Tobacco and tobacco products
                "5993",  # Tobacco stores and stands
            ],
            "gambling": [
                "7011",  # Hotels and motels (casino hotels)
                "7993",  # Coin-operated amusement devices
                "7999",  # Amusement and recreation services (gambling)
            ],
            "conventional_banking": [
                "6020",  # Commercial banks
                "6022",  # State commercial banks
                "6029",  # Commercial banks, NEC
                "6035",  # Savings institutions
                "6036",  # Savings institutions, federally chartered
                "6061",  # Credit unions
                "6141",  # Personal credit institutions
                "6153",  # Short-term business credit
                "6159",  # Miscellaneous business credit
            ],
            "conventional_insurance": [
                "6311",  # Life insurance
                "6321",  # Accident and health insurance
                "6324",  # Hospital and medical service plans
                "6331",  # Fire, marine, and casualty insurance
                "6351",  # Surety insurance
                "6361",  # Title insurance
                "6399",  # Insurance carriers, NEC
            ],
        },
        description="SIC codes mapped to prohibited industries"
    )

    # ========== Purification Settings ==========

    # Method for calculating purification amount
    purification_method: str = Field(
        default="dividend_based",
        description="Method: 'dividend_based' or 'capital_gains_based'"
    )

    # Whether to track purification obligations
    track_purification: bool = Field(
        default=True,
        description="Track and report purification obligations"
    )


# Global Shariah config instance
shariah_config = ShariahConfig()


# ========== Screening Result Types ==========

class ScreeningResult:
    """Result of a Shariah compliance screening."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    QUESTIONABLE = "questionable"  # Needs manual review
    INSUFFICIENT_DATA = "insufficient_data"


class NonComplianceReason:
    """Reasons for non-compliance."""

    PROHIBITED_BUSINESS = "prohibited_business_activity"
    DEBT_RATIO = "debt_ratio_exceeded"
    DEPOSIT_RATIO = "deposit_ratio_exceeded"
    IMPERMISSIBLE_INCOME = "impermissible_income_exceeded"
    RECEIVABLES_RATIO = "receivables_ratio_exceeded"
    MIXED_BUSINESS = "mixed_prohibited_business"
