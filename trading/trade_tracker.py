"""
Trade Tracker Module.

Provides trade lifecycle tracking with entry/exit logging,
holding period calculation, and comprehensive statistics.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import logging

import numpy as np

from data.storage import DatabaseManager, TradeRecord


logger = logging.getLogger(__name__)


@dataclass
class TradeStatistics:
    """Comprehensive trade statistics for a given period."""

    # Summary counts
    total_trades: int
    winning_trades: int
    losing_trades: int

    # P&L metrics
    total_pnl: float
    gross_profit: float
    gross_loss: float

    # Performance ratios
    win_rate: float
    profit_factor: float
    sharpe_ratio: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    avg_win: float
    avg_loss: float

    # Holding period analysis
    avg_holding_minutes: float
    max_holding_minutes: float
    min_holding_minutes: float

    # Additional context
    period_days: int
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class TradeTracker:
    """
    Tracks trade lifecycle from entry to exit.

    Provides methods for logging trade entries and exits,
    calculating holding periods, realized P&L, and
    comprehensive trading statistics.
    """

    def __init__(self, db: DatabaseManager):
        """
        Initialize the TradeTracker.

        Args:
            db: DatabaseManager instance for persistence
        """
        self.db = db
        self._logger = logging.getLogger(f"{__name__}.TradeTracker")

    def log_trade_entry(
        self,
        symbol: str,
        price: float,
        quantity: float,
        signal_id: Optional[int] = None,
    ) -> int:
        """
        Log a trade entry.

        Creates a trade record with entry_time set to now.
        The trade remains open until log_trade_exit is called.

        Args:
            symbol: Stock ticker symbol
            price: Entry price per share
            quantity: Number of shares (positive for long, negative for short)
            signal_id: Optional ID of the signal that triggered this trade

        Returns:
            trade_id: The ID of the created trade record
        """
        side = "buy" if quantity > 0 else "sell"
        abs_quantity = abs(quantity)
        total_value = price * abs_quantity

        trade_data = {
            "symbol": symbol,
            "trade_date": datetime.now(),  # entry_time
            "side": side,
            "quantity": abs_quantity,
            "price": price,
            "total_value": total_value,
            "notes": f"signal_id:{signal_id}" if signal_id else None,
        }

        trade_id = self.db.save_trade(trade_data)

        self._logger.info(
            f"Trade entry logged: {symbol} {side} {abs_quantity} @ {price:.2f} "
            f"(trade_id={trade_id}, signal_id={signal_id})"
        )

        return trade_id

    def log_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
    ) -> dict:
        """
        Log a trade exit and calculate P&L.

        Updates the trade record with exit details and calculates
        the holding period and realized P&L.

        Args:
            trade_id: ID of the trade to close
            exit_price: Exit price per share
            exit_reason: Reason for exit (e.g., "take_profit", "stop_loss", "signal")

        Returns:
            Summary dict with:
                - trade_id
                - symbol
                - side
                - entry_price
                - exit_price
                - quantity
                - holding_period_minutes
                - realized_pnl
                - exit_reason

        Raises:
            ValueError: If trade_id not found or trade already closed
        """
        session = self.db.get_session()

        try:
            trade = session.query(TradeRecord).filter(
                TradeRecord.id == trade_id
            ).first()

            if not trade:
                raise ValueError(f"Trade not found: {trade_id}")

            # Check if already closed (has realized_pnl set)
            if trade.realized_pnl is not None:
                raise ValueError(f"Trade already closed: {trade_id}")

            entry_time = trade.trade_date
            exit_time = datetime.now()

            # Calculate holding period in minutes
            holding_delta = exit_time - entry_time
            holding_period_minutes = holding_delta.total_seconds() / 60

            # Calculate realized P&L
            # For buys: profit = (exit_price - entry_price) * quantity
            # For sells (shorts): profit = (entry_price - exit_price) * quantity
            if trade.side == "buy":
                realized_pnl = (exit_price - trade.price) * trade.quantity
            else:
                realized_pnl = (trade.price - exit_price) * trade.quantity

            # Update the trade record
            existing_notes = trade.notes or ""
            new_notes = f"{existing_notes}|exit_reason:{exit_reason}|exit_time:{exit_time.isoformat()}|holding_min:{holding_period_minutes:.1f}"

            trade.realized_pnl = realized_pnl
            trade.notes = new_notes

            session.commit()

            summary = {
                "trade_id": trade_id,
                "symbol": trade.symbol,
                "side": trade.side,
                "entry_price": trade.price,
                "exit_price": exit_price,
                "quantity": trade.quantity,
                "holding_period_minutes": round(holding_period_minutes, 1),
                "realized_pnl": round(realized_pnl, 2),
                "exit_reason": exit_reason,
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
            }

            self._logger.info(
                f"Trade exit logged: {trade.symbol} {trade.side} "
                f"entry={trade.price:.2f} exit={exit_price:.2f} "
                f"pnl={realized_pnl:.2f} holding={holding_period_minutes:.1f}min "
                f"reason={exit_reason}"
            )

            return summary

        finally:
            session.close()

    def calculate_statistics(self, days: int = 30) -> TradeStatistics:
        """
        Calculate comprehensive trading statistics for closed trades.

        Args:
            days: Number of days to look back (default 30)

        Returns:
            TradeStatistics dataclass with all calculated metrics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get closed trades in date range
        session = self.db.get_session()

        try:
            trades = session.query(TradeRecord).filter(
                TradeRecord.trade_date >= start_date,
                TradeRecord.trade_date <= end_date,
                TradeRecord.realized_pnl != None,  # noqa: E711 - SQLAlchemy requires this syntax
            ).order_by(TradeRecord.trade_date).all()

            # Handle edge case: no trades
            if not trades:
                return TradeStatistics(
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    total_pnl=0.0,
                    gross_profit=0.0,
                    gross_loss=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    max_drawdown_pct=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    avg_holding_minutes=0.0,
                    max_holding_minutes=0.0,
                    min_holding_minutes=0.0,
                    period_days=days,
                    start_date=start_date,
                    end_date=end_date,
                )

            # Extract P&L values
            pnl_values = [t.realized_pnl for t in trades]

            # Separate wins and losses
            wins = [p for p in pnl_values if p > 0]
            losses = [p for p in pnl_values if p < 0]

            total_trades = len(trades)
            winning_trades = len(wins)
            losing_trades = len(losses)

            # P&L calculations
            total_pnl = sum(pnl_values)
            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0

            # Win rate (handle division by zero)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            # Profit factor (handle division by zero)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
                float('inf') if gross_profit > 0 else 0.0
            )

            # Average win/loss
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

            # Sharpe ratio calculation
            # Using daily returns approximation
            sharpe_ratio = self._calculate_sharpe_ratio(pnl_values)

            # Drawdown calculation
            max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(pnl_values)

            # Holding period analysis
            holding_periods = self._extract_holding_periods(trades)

            if holding_periods:
                avg_holding_minutes = sum(holding_periods) / len(holding_periods)
                max_holding_minutes = max(holding_periods)
                min_holding_minutes = min(holding_periods)
            else:
                avg_holding_minutes = 0.0
                max_holding_minutes = 0.0
                min_holding_minutes = 0.0

            return TradeStatistics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                total_pnl=round(total_pnl, 2),
                gross_profit=round(gross_profit, 2),
                gross_loss=round(gross_loss, 2),
                win_rate=round(win_rate, 4),
                profit_factor=round(profit_factor, 4) if profit_factor != float('inf') else float('inf'),
                sharpe_ratio=round(sharpe_ratio, 4),
                max_drawdown=round(max_drawdown, 2),
                max_drawdown_pct=round(max_drawdown_pct, 4),
                avg_win=round(avg_win, 2),
                avg_loss=round(avg_loss, 2),
                avg_holding_minutes=round(avg_holding_minutes, 1),
                max_holding_minutes=round(max_holding_minutes, 1),
                min_holding_minutes=round(min_holding_minutes, 1),
                period_days=days,
                start_date=start_date,
                end_date=end_date,
            )

        finally:
            session.close()

    def _calculate_sharpe_ratio(
        self,
        pnl_values: list[float],
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0,
    ) -> float:
        """
        Calculate the Sharpe ratio from P&L values.

        Args:
            pnl_values: List of realized P&L values
            risk_free_rate: Annual risk-free rate (default 0)
            annualization_factor: Trading days per year (default 252)

        Returns:
            Annualized Sharpe ratio
        """
        if len(pnl_values) < 2:
            return 0.0

        pnl_array = np.array(pnl_values)

        mean_pnl = np.mean(pnl_array)
        std_pnl = np.std(pnl_array, ddof=1)

        # Handle zero standard deviation
        if std_pnl == 0 or np.isnan(std_pnl):
            return 0.0

        # Daily risk-free rate
        daily_rf = risk_free_rate / annualization_factor

        # Sharpe ratio (annualized)
        sharpe = ((mean_pnl - daily_rf) / std_pnl) * np.sqrt(annualization_factor)

        return float(sharpe) if not np.isnan(sharpe) else 0.0

    def _calculate_max_drawdown(
        self,
        pnl_values: list[float],
    ) -> tuple[float, float]:
        """
        Calculate maximum drawdown from P&L values.

        Args:
            pnl_values: List of realized P&L values

        Returns:
            Tuple of (max_drawdown_absolute, max_drawdown_percentage)
        """
        if not pnl_values:
            return 0.0, 0.0

        # Calculate cumulative P&L (equity curve)
        cumulative = np.cumsum(pnl_values)

        # Track running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Calculate drawdowns
        drawdowns = running_max - cumulative

        # Maximum drawdown
        max_drawdown = float(np.max(drawdowns))

        # Maximum drawdown percentage (relative to peak)
        # Handle case where running_max could be zero or negative
        with np.errstate(divide='ignore', invalid='ignore'):
            dd_pct = np.where(
                running_max > 0,
                drawdowns / running_max,
                0.0
            )

        max_drawdown_pct = float(np.max(dd_pct)) if len(dd_pct) > 0 else 0.0

        return max_drawdown, max_drawdown_pct

    def _extract_holding_periods(
        self,
        trades: list[TradeRecord],
    ) -> list[float]:
        """
        Extract holding periods from trade records.

        Parses the notes field to find holding period data.

        Args:
            trades: List of TradeRecord objects

        Returns:
            List of holding periods in minutes
        """
        holding_periods = []

        for trade in trades:
            if not trade.notes:
                continue

            # Parse notes for holding_min field
            # Format: "...|holding_min:123.4|..."
            for part in trade.notes.split("|"):
                if part.startswith("holding_min:"):
                    try:
                        minutes = float(part.split(":")[1])
                        holding_periods.append(minutes)
                    except (ValueError, IndexError):
                        continue

        return holding_periods

    def get_open_trades(self, symbol: Optional[str] = None) -> list[dict]:
        """
        Get all open (unclosed) trades.

        Args:
            symbol: Optional filter by symbol

        Returns:
            List of trade dictionaries
        """
        session = self.db.get_session()

        try:
            query = session.query(TradeRecord).filter(
                TradeRecord.realized_pnl == None,  # noqa: E711
            )

            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)

            trades = query.order_by(TradeRecord.trade_date.desc()).all()

            return [t.to_dict() for t in trades]

        finally:
            session.close()

    def get_trade_by_id(self, trade_id: int) -> Optional[dict]:
        """
        Get a specific trade by ID.

        Args:
            trade_id: Trade record ID

        Returns:
            Trade dictionary or None if not found
        """
        session = self.db.get_session()

        try:
            trade = session.query(TradeRecord).filter(
                TradeRecord.id == trade_id
            ).first()

            return trade.to_dict() if trade else None

        finally:
            session.close()
