"""
Interactive Brokers API configuration.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IBKRConfig(BaseSettings):
    """IBKR connection configuration."""

    model_config = SettingsConfigDict(
        env_prefix="IBKR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Connection mode
    mode: Literal["paper", "live"] = "paper"

    # TWS/Gateway host
    host: str = "127.0.0.1"

    # Port configuration
    # TWS Paper: 7497, TWS Live: 7496
    # Gateway Paper: 4002, Gateway Live: 4001
    tws_paper_port: int = 7497
    tws_live_port: int = 7496
    gateway_paper_port: int = 4002
    gateway_live_port: int = 4001

    # Connection settings
    client_id: int = Field(default=1, description="Unique client ID for this connection")
    timeout: int = Field(default=60, description="Connection timeout in seconds")
    readonly: bool = Field(default=False, description="Read-only mode (no order submission)")

    # Reconnection settings
    auto_reconnect: bool = Field(default=True, description="Enable automatic reconnection")
    reconnect_base_delay: float = Field(default=1.0, description="Initial reconnect delay in seconds")
    reconnect_max_delay: float = Field(default=60.0, description="Maximum reconnect delay in seconds")
    reconnect_multiplier: float = Field(default=2.0, description="Exponential backoff multiplier")
    reconnect_max_attempts: int = Field(default=10, description="Max reconnection attempts (0=unlimited)")
    reconnect_jitter: float = Field(default=0.1, description="Random jitter factor (0-1) to prevent thundering herd")

    # Use TWS or Gateway
    use_gateway: bool = Field(default=True, description="Use IB Gateway instead of TWS")

    # Market data
    market_data_type: int = Field(
        default=3,
        description="1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen"
    )

    # Keepalive/heartbeat settings
    keepalive_enabled: bool = Field(default=True, description="Enable periodic connection health checks")
    keepalive_interval: int = Field(default=60, description="Seconds between keepalive pings")
    keepalive_timeout: int = Field(default=10, description="Timeout for keepalive ping in seconds")

    @property
    def port(self) -> int:
        """Get the appropriate port based on mode and gateway preference."""
        if self.use_gateway:
            return self.gateway_paper_port if self.mode == "paper" else self.gateway_live_port
        return self.tws_paper_port if self.mode == "paper" else self.tws_live_port

    @property
    def is_paper(self) -> bool:
        """Check if running in paper trading mode."""
        return self.mode == "paper"


# Global IBKR config instance
ibkr_config = IBKRConfig()
