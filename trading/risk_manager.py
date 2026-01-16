"""
Risk Manager Module.

Implements risk management rules:
- Position sizing based on ATR
- Maximum position size limits
- Per-trade risk limits
- Daily loss limits
- Portfolio-level risk monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """Result of a risk check."""

    passed: bool
    message: str
    risk_value: float | None = None
    limit_value: float | None = None
    check_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "message": self.message,
            "risk_value": self.risk_value,
            "limit_value": self.limit_value,
            "check_name": self.check_name,
        }


@dataclass
class RiskLimits:
    """Risk management limits."""

    # Position limits
    max_position_size_pct: float = 0.05  # 5% of portfolio per position
    max_positions: int = 10  # Maximum concurrent positions

    # Per-trade risk
    max_risk_per_trade_pct: float = 0.02  # 2% of portfolio per trade
    stop_loss_atr_multiplier: float = 2.0  # Stop loss at 2x ATR

    # Daily limits
    max_daily_loss_pct: float = 0.03  # 3% daily loss limit
    max_daily_trades: int = 20  # Maximum trades per day

    # Portfolio limits
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    max_correlation: float = 0.7  # Max correlation between positions

    # Order limits
    min_order_size: float = 100.0  # Minimum order value
    max_order_size: float = 50000.0  # Maximum single order value


class RiskManager:
    """
    Risk management for trading operations.

    Implements various risk checks and position sizing algorithms
    to protect the portfolio from excessive losses.
    """

    def __init__(
        self,
        portfolio_value: float,
        limits: RiskLimits | None = None,
    ):
        """
        Initialize the risk manager.

        Args:
            portfolio_value: Current portfolio value
            limits: Risk limits configuration
        """
        self.portfolio_value = portfolio_value
        self.limits = limits or RiskLimits()

        # Daily tracking
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._current_date: date = date.today()

        # Position tracking
        self._positions: dict[str, dict[str, Any]] = {}
        self._peak_value: float = portfolio_value

    def update_portfolio_value(self, value: float) -> None:
        """Update current portfolio value."""
        self.portfolio_value = value
        if value > self._peak_value:
            self._peak_value = value

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        atr: float | None = None,
        risk_per_trade: float | None = None,
    ) -> tuple[int, float]:
        """
        Calculate position size based on risk parameters.

        Uses ATR-based position sizing:
        Position Size = (Portfolio * Risk%) / (ATR * Multiplier)

        Args:
            symbol: Stock ticker
            entry_price: Entry price
            atr: Average True Range (if None, uses price-based estimate)
            risk_per_trade: Override risk percentage

        Returns:
            Tuple of (shares, dollar_amount)
        """
        risk_pct = risk_per_trade or self.limits.max_risk_per_trade_pct
        risk_amount = self.portfolio_value * risk_pct

        # Calculate stop distance
        if atr:
            stop_distance = atr * self.limits.stop_loss_atr_multiplier
        else:
            # Estimate ATR as 2% of price if not provided
            stop_distance = entry_price * 0.02 * self.limits.stop_loss_atr_multiplier

        # Calculate position size based on risk
        if stop_distance > 0:
            shares = int(risk_amount / stop_distance)
        else:
            shares = 0

        dollar_amount = shares * entry_price

        # Apply position size limits
        max_position_value = self.portfolio_value * self.limits.max_position_size_pct
        if dollar_amount > max_position_value:
            shares = int(max_position_value / entry_price)
            dollar_amount = shares * entry_price

        # Apply order size limits
        if dollar_amount < self.limits.min_order_size:
            logger.warning(f"Position too small for {symbol}: ${dollar_amount:.2f}")
            shares = 0
            dollar_amount = 0
        elif dollar_amount > self.limits.max_order_size:
            shares = int(self.limits.max_order_size / entry_price)
            dollar_amount = shares * entry_price

        logger.info(
            f"Position size for {symbol}: {shares} shares (${dollar_amount:.2f}), "
            f"risk: ${risk_amount:.2f}, stop distance: ${stop_distance:.2f}"
        )

        return shares, dollar_amount

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        side: str = "buy",
    ) -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            entry_price: Entry price
            atr: Average True Range
            side: "buy" or "sell"

        Returns:
            Stop loss price
        """
        stop_distance = atr * self.limits.stop_loss_atr_multiplier

        if side == "buy":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        atr: float,
        risk_reward_ratio: float = 2.0,
        side: str = "buy",
    ) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            atr: Average True Range
            risk_reward_ratio: Reward multiple of risk
            side: "buy" or "sell"

        Returns:
            Take profit price
        """
        stop_distance = atr * self.limits.stop_loss_atr_multiplier
        profit_distance = stop_distance * risk_reward_ratio

        if side == "buy":
            return entry_price + profit_distance
        else:
            return entry_price - profit_distance

    def check_can_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
    ) -> list[RiskCheck]:
        """
        Run all risk checks before placing a trade.

        Args:
            symbol: Stock ticker
            quantity: Number of shares
            price: Current price

        Returns:
            List of RiskCheck results
        """
        checks = []
        trade_value = quantity * price

        # Reset daily counters if new day
        self._check_daily_reset()

        # Check 1: Daily loss limit
        checks.append(self._check_daily_loss_limit())

        # Check 2: Daily trade limit
        checks.append(self._check_daily_trade_limit())

        # Check 3: Position size limit
        checks.append(self._check_position_size_limit(trade_value))

        # Check 4: Max positions
        checks.append(self._check_max_positions(symbol))

        # Check 5: Maximum drawdown
        checks.append(self._check_max_drawdown())

        # Check 6: Order size limits
        checks.append(self._check_order_size(trade_value))

        return checks

    def can_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
    ) -> tuple[bool, list[str]]:
        """
        Simple check if trade is allowed.

        Args:
            symbol: Stock ticker
            quantity: Number of shares
            price: Current price

        Returns:
            Tuple of (allowed, list of failure reasons)
        """
        checks = self.check_can_trade(symbol, quantity, price)
        failures = [c.message for c in checks if not c.passed]
        return len(failures) == 0, failures

    def _check_daily_reset(self) -> None:
        """Reset daily counters if new day."""
        today = date.today()
        if today != self._current_date:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._current_date = today
            logger.info("Daily risk counters reset")

    def _check_daily_loss_limit(self) -> RiskCheck:
        """Check if daily loss limit has been hit."""
        max_loss = self.portfolio_value * self.limits.max_daily_loss_pct
        current_loss = -self._daily_pnl if self._daily_pnl < 0 else 0

        if current_loss >= max_loss:
            return RiskCheck(
                passed=False,
                message=f"Daily loss limit hit: ${current_loss:.2f} >= ${max_loss:.2f}",
                risk_value=current_loss,
                limit_value=max_loss,
                check_name="daily_loss_limit",
            )

        return RiskCheck(
            passed=True,
            message=f"Daily loss OK: ${current_loss:.2f} < ${max_loss:.2f}",
            risk_value=current_loss,
            limit_value=max_loss,
            check_name="daily_loss_limit",
        )

    def _check_daily_trade_limit(self) -> RiskCheck:
        """Check if daily trade limit has been hit."""
        if self._daily_trades >= self.limits.max_daily_trades:
            return RiskCheck(
                passed=False,
                message=f"Daily trade limit hit: {self._daily_trades} >= {self.limits.max_daily_trades}",
                risk_value=self._daily_trades,
                limit_value=self.limits.max_daily_trades,
                check_name="daily_trade_limit",
            )

        return RiskCheck(
            passed=True,
            message=f"Daily trades OK: {self._daily_trades} < {self.limits.max_daily_trades}",
            risk_value=self._daily_trades,
            limit_value=self.limits.max_daily_trades,
            check_name="daily_trade_limit",
        )

    def _check_position_size_limit(self, trade_value: float) -> RiskCheck:
        """Check if trade exceeds position size limit."""
        max_position = self.portfolio_value * self.limits.max_position_size_pct

        if trade_value > max_position:
            return RiskCheck(
                passed=False,
                message=f"Position too large: ${trade_value:.2f} > ${max_position:.2f} ({self.limits.max_position_size_pct:.1%})",
                risk_value=trade_value,
                limit_value=max_position,
                check_name="position_size_limit",
            )

        return RiskCheck(
            passed=True,
            message=f"Position size OK: ${trade_value:.2f} <= ${max_position:.2f}",
            risk_value=trade_value,
            limit_value=max_position,
            check_name="position_size_limit",
        )

    def _check_max_positions(self, symbol: str) -> RiskCheck:
        """Check if maximum positions would be exceeded."""
        current_positions = len(self._positions)
        is_new_position = symbol not in self._positions

        if is_new_position and current_positions >= self.limits.max_positions:
            return RiskCheck(
                passed=False,
                message=f"Max positions reached: {current_positions} >= {self.limits.max_positions}",
                risk_value=current_positions,
                limit_value=self.limits.max_positions,
                check_name="max_positions",
            )

        return RiskCheck(
            passed=True,
            message=f"Positions OK: {current_positions} < {self.limits.max_positions}",
            risk_value=current_positions,
            limit_value=self.limits.max_positions,
            check_name="max_positions",
        )

    def _check_max_drawdown(self) -> RiskCheck:
        """Check if maximum drawdown has been exceeded."""
        if self._peak_value > 0:
            current_drawdown = (self._peak_value - self.portfolio_value) / self._peak_value
        else:
            current_drawdown = 0

        if current_drawdown >= self.limits.max_drawdown_pct:
            return RiskCheck(
                passed=False,
                message=f"Max drawdown exceeded: {current_drawdown:.2%} >= {self.limits.max_drawdown_pct:.2%}",
                risk_value=current_drawdown,
                limit_value=self.limits.max_drawdown_pct,
                check_name="max_drawdown",
            )

        return RiskCheck(
            passed=True,
            message=f"Drawdown OK: {current_drawdown:.2%} < {self.limits.max_drawdown_pct:.2%}",
            risk_value=current_drawdown,
            limit_value=self.limits.max_drawdown_pct,
            check_name="max_drawdown",
        )

    def _check_order_size(self, trade_value: float) -> RiskCheck:
        """Check if order size is within limits."""
        if trade_value < self.limits.min_order_size:
            return RiskCheck(
                passed=False,
                message=f"Order too small: ${trade_value:.2f} < ${self.limits.min_order_size:.2f}",
                risk_value=trade_value,
                limit_value=self.limits.min_order_size,
                check_name="min_order_size",
            )

        if trade_value > self.limits.max_order_size:
            return RiskCheck(
                passed=False,
                message=f"Order too large: ${trade_value:.2f} > ${self.limits.max_order_size:.2f}",
                risk_value=trade_value,
                limit_value=self.limits.max_order_size,
                check_name="max_order_size",
            )

        return RiskCheck(
            passed=True,
            message=f"Order size OK: ${trade_value:.2f}",
            risk_value=trade_value,
            limit_value=self.limits.max_order_size,
            check_name="order_size",
        )

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade for daily tracking."""
        self._daily_pnl += pnl
        self._daily_trades += 1
        logger.info(f"Trade recorded: PnL=${pnl:.2f}, Daily PnL=${self._daily_pnl:.2f}")

    def update_position(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        current_price: float,
    ) -> None:
        """Update position tracking."""
        if quantity == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = {
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "market_value": quantity * current_price,
                "pnl": quantity * (current_price - avg_price),
            }

    def get_portfolio_risk_summary(self) -> dict[str, Any]:
        """Get portfolio risk summary."""
        total_exposure = sum(p["market_value"] for p in self._positions.values())
        total_pnl = sum(p["pnl"] for p in self._positions.values())

        drawdown = (self._peak_value - self.portfolio_value) / self._peak_value if self._peak_value > 0 else 0

        return {
            "portfolio_value": self.portfolio_value,
            "peak_value": self._peak_value,
            "current_drawdown": drawdown,
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0,
            "total_unrealized_pnl": total_pnl,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "position_count": len(self._positions),
            "positions": list(self._positions.keys()),
        }
