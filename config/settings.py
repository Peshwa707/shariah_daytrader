"""
Environment configuration for Shariah Daytrading Bot.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Shariah Daytrader"
    environment: Literal["development", "paper", "live"] = "paper"
    debug: bool = False

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data_store")
    db_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data_store" / "shariah_trader.db")

    # API Keys (optional - for enhanced data)
    alpha_vantage_api_key: str | None = None
    zoya_api_key: str | None = None  # Optional Shariah verification

    # Logging
    log_level: str = "INFO"
    log_file: Path | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
