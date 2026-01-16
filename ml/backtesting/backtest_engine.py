"""
Backtesting Engine using VectorBT.

VectorBT is a vectorized backtesting library that enables:
- Fast backtesting via NumPy operations
- Walk-forward optimization
- Comprehensive portfolio analytics
- Monte Carlo simulations

This module provides a unified interface for backtesting
our ML-based trading strategies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
import logging

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""

    # Core metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Time metrics
    start_date: datetime
    end_date: datetime
    total_days: int
    exposure_time: float  # % of time in market

    # Additional data
    equity_curve: pd.Series | None = None
    trades: pd.DataFrame | None = None
    monthly_returns: pd.Series | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "total_days": self.total_days,
            "exposure_time": self.exposure_time,
        }

    def __str__(self) -> str:
        """String summary."""
        return (
            f"=== Backtest Results ===\n"
            f"Period: {self.start_date.date()} to {self.end_date.date()} ({self.total_days} days)\n"
            f"\nPerformance:\n"
            f"  Total Return:   {self.total_return:>10.2%}\n"
            f"  Annual Return:  {self.annual_return:>10.2%}\n"
            f"  Sharpe Ratio:   {self.sharpe_ratio:>10.2f}\n"
            f"  Sortino Ratio:  {self.sortino_ratio:>10.2f}\n"
            f"  Max Drawdown:   {self.max_drawdown:>10.2%}\n"
            f"  Calmar Ratio:   {self.calmar_ratio:>10.2f}\n"
            f"\nTrade Statistics:\n"
            f"  Total Trades:   {self.total_trades:>10}\n"
            f"  Win Rate:       {self.win_rate:>10.2%}\n"
            f"  Profit Factor:  {self.profit_factor:>10.2f}\n"
            f"  Avg Win:        {self.avg_win:>10.2%}\n"
            f"  Avg Loss:       {self.avg_loss:>10.2%}\n"
            f"  Exposure Time:  {self.exposure_time:>10.2%}\n"
        )


class BacktestEngine:
    """
    Vectorized backtesting engine using VectorBT.

    Supports multiple backtesting approaches:
    1. Simple signal-based backtesting
    2. Walk-forward analysis
    3. Strategy comparison
    4. Portfolio backtesting
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.001,  # 0.1%
        size_type: str = "percent",  # "percent" or "fixed"
        size: float = 1.0,  # 100% of capital per trade if percent
    ):
        """
        Initialize the backtest engine.

        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as decimal)
            slippage: Slippage per trade (as decimal)
            size_type: Position sizing method
            size: Position size (interpretation depends on size_type)
        """
        if not VBT_AVAILABLE:
            raise ImportError(
                "VectorBT is not installed. Install with: pip install vectorbt"
            )

        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.size_type = size_type
        self.size = size

    def run_backtest(
        self,
        price: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
    ) -> BacktestResult:
        """
        Run a simple backtest with entry/exit signals.

        Args:
            price: Price series (close prices)
            entries: Boolean series where True = enter position
            exits: Boolean series where True = exit position

        Returns:
            BacktestResult with performance metrics
        """
        # Ensure boolean
        entries = entries.astype(bool)
        exits = exits.astype(bool)

        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=price,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.commission,
            slippage=self.slippage,
            freq="D",
        )

        return self._extract_results(portfolio)

    def run_from_predictions(
        self,
        price: pd.Series,
        predictions: pd.Series,
        hold_period: int = 1,
        long_only: bool = True,
    ) -> BacktestResult:
        """
        Run backtest from ML model predictions.

        Args:
            price: Price series
            predictions: Prediction series (1 = buy, 0 = sell/hold)
            hold_period: Bars to hold each position
            long_only: Only take long positions

        Returns:
            BacktestResult
        """
        # Generate entries from predictions
        entries = predictions == 1

        # Generate exits (after hold_period or on opposite signal)
        if hold_period == 1:
            exits = entries.shift(1).fillna(False)
        else:
            # Exit after N bars
            exits = entries.shift(hold_period).fillna(False)

        if not long_only:
            # TODO: Implement short selling
            pass

        return self.run_backtest(price, entries, exits)

    def walk_forward_backtest(
        self,
        price: pd.Series,
        features: pd.DataFrame,
        model_class: Any,
        model_params: dict[str, Any],
        n_splits: int = 5,
        train_ratio: float = 0.7,
    ) -> list[BacktestResult]:
        """
        Perform walk-forward backtesting.

        This method:
        1. Splits data into multiple train/test windows
        2. Trains model on each training window
        3. Generates predictions on test window
        4. Runs backtest on test window
        5. Combines results

        Args:
            price: Price series
            features: Feature DataFrame (aligned with price)
            model_class: Model class with train() and predict() methods
            model_params: Parameters for model instantiation
            n_splits: Number of walk-forward splits
            train_ratio: Ratio of each split used for training

        Returns:
            List of BacktestResult for each split
        """
        results = []
        n_samples = len(price)
        split_size = n_samples // n_splits

        for i in range(n_splits):
            # Define split boundaries
            split_start = i * split_size
            split_end = min((i + 2) * split_size, n_samples) if i < n_splits - 1 else n_samples

            split_data = features.iloc[split_start:split_end]
            split_price = price.iloc[split_start:split_end]

            # Further split into train/test
            train_size = int(len(split_data) * train_ratio)
            train_features = split_data.iloc[:train_size]
            test_features = split_data.iloc[train_size:]
            test_price = split_price.iloc[train_size:]

            # Create target (next day return > 0)
            train_target = (split_price.pct_change().shift(-1).iloc[:train_size] > 0).astype(int)

            # Train model
            model = model_class(**model_params)
            model.train(train_features, train_target)

            # Generate predictions
            predictions = model.predict(test_features)
            pred_series = pd.Series(
                [p["signal"] == "buy" for p in predictions] if isinstance(predictions[0], dict) else predictions,
                index=test_features.index,
            ).astype(int)

            # Run backtest
            result = self.run_from_predictions(test_price, pred_series)
            results.append(result)

            logger.info(f"Split {i + 1}: Return = {result.total_return:.2%}, Sharpe = {result.sharpe_ratio:.2f}")

        return results

    def compare_strategies(
        self,
        price: pd.Series,
        strategies: dict[str, tuple[pd.Series, pd.Series]],
    ) -> dict[str, BacktestResult]:
        """
        Compare multiple strategies.

        Args:
            price: Price series
            strategies: Dict mapping strategy names to (entries, exits) tuples

        Returns:
            Dict mapping strategy names to BacktestResult
        """
        results = {}

        for name, (entries, exits) in strategies.items():
            result = self.run_backtest(price, entries, exits)
            results[name] = result
            logger.info(f"{name}: Return = {result.total_return:.2%}, Sharpe = {result.sharpe_ratio:.2f}")

        return results

    def _extract_results(self, portfolio) -> BacktestResult:
        """Extract metrics from VectorBT portfolio."""
        stats = portfolio.stats()

        # Get trade records
        trades_df = portfolio.trades.records_readable

        # Calculate trade statistics
        if len(trades_df) > 0:
            wins = trades_df[trades_df["PnL"] > 0]
            losses = trades_df[trades_df["PnL"] < 0]

            total_trades = len(trades_df)
            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            gross_profit = wins["PnL"].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses["PnL"].sum()) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            avg_win = wins["Return"].mean() if len(wins) > 0 else 0
            avg_loss = losses["Return"].mean() if len(losses) > 0 else 0
            largest_win = wins["Return"].max() if len(wins) > 0 else 0
            largest_loss = losses["Return"].min() if len(losses) > 0 else 0
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = profit_factor = 0
            avg_win = avg_loss = largest_win = largest_loss = 0

        # Get equity curve
        equity_curve = portfolio.value()

        # Get monthly returns
        monthly_returns = portfolio.returns().resample("ME").sum()

        # Calculate exposure time
        try:
            positions = portfolio.positions.records_readable
            if len(positions) > 0:
                total_bars = len(portfolio.close)
                # Handle different vectorbt versions with different column names
                exit_col = "Exit Index" if "Exit Index" in positions.columns else "exit_idx"
                entry_col = "Entry Index" if "Entry Index" in positions.columns else "entry_idx"
                if exit_col in positions.columns and entry_col in positions.columns:
                    bars_in_market = positions[exit_col].sum() - positions[entry_col].sum()
                    exposure_time = bars_in_market / total_bars if total_bars > 0 else 0
                else:
                    exposure_time = 0
            else:
                exposure_time = 0
        except Exception:
            exposure_time = 0

        return BacktestResult(
            total_return=stats.get("Total Return [%]", 0) / 100,
            annual_return=stats.get("Annualized Return [%]", 0) / 100,
            sharpe_ratio=stats.get("Sharpe Ratio", 0),
            sortino_ratio=stats.get("Sortino Ratio", 0),
            max_drawdown=stats.get("Max Drawdown [%]", 0) / 100,
            calmar_ratio=stats.get("Calmar Ratio", 0),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            start_date=portfolio.wrapper.index[0],
            end_date=portfolio.wrapper.index[-1],
            total_days=len(portfolio.close),
            exposure_time=exposure_time,
            equity_curve=equity_curve,
            trades=trades_df,
            monthly_returns=monthly_returns,
        )

    def generate_report(
        self,
        result: BacktestResult,
        benchmark: pd.Series | None = None,
    ) -> str:
        """
        Generate a detailed backtest report.

        Args:
            result: BacktestResult to report on
            benchmark: Optional benchmark returns for comparison

        Returns:
            Report string
        """
        report = [
            "=" * 60,
            "           BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"Period: {result.start_date.date()} to {result.end_date.date()}",
            f"Total Trading Days: {result.total_days}",
            "",
            "-" * 60,
            "RETURNS",
            "-" * 60,
            f"  Total Return:      {result.total_return:>12.2%}",
            f"  Annualized Return: {result.annual_return:>12.2%}",
            "",
            "-" * 60,
            "RISK METRICS",
            "-" * 60,
            f"  Sharpe Ratio:      {result.sharpe_ratio:>12.2f}",
            f"  Sortino Ratio:     {result.sortino_ratio:>12.2f}",
            f"  Max Drawdown:      {result.max_drawdown:>12.2%}",
            f"  Calmar Ratio:      {result.calmar_ratio:>12.2f}",
            "",
            "-" * 60,
            "TRADE STATISTICS",
            "-" * 60,
            f"  Total Trades:      {result.total_trades:>12}",
            f"  Winning Trades:    {result.winning_trades:>12}",
            f"  Losing Trades:     {result.losing_trades:>12}",
            f"  Win Rate:          {result.win_rate:>12.2%}",
            f"  Profit Factor:     {result.profit_factor:>12.2f}",
            "",
            f"  Average Win:       {result.avg_win:>12.2%}",
            f"  Average Loss:      {result.avg_loss:>12.2%}",
            f"  Largest Win:       {result.largest_win:>12.2%}",
            f"  Largest Loss:      {result.largest_loss:>12.2%}",
            "",
            f"  Exposure Time:     {result.exposure_time:>12.2%}",
            "",
        ]

        if benchmark is not None:
            bench_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
            report.extend([
                "-" * 60,
                "VS BENCHMARK",
                "-" * 60,
                f"  Benchmark Return:  {bench_return:>12.2%}",
                f"  Alpha:             {result.total_return - bench_return:>12.2%}",
            ])

        report.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(report)


# Standalone backtesting functions for quick analysis

def quick_backtest(
    price: pd.Series,
    signals: pd.Series,
    initial_capital: float = 100_000,
) -> dict[str, float]:
    """
    Quick backtest for rapid strategy evaluation.

    Args:
        price: Price series
        signals: Signal series (1 = long, 0 = flat)
        initial_capital: Starting capital

    Returns:
        Dict with basic metrics
    """
    engine = BacktestEngine(initial_capital=initial_capital)

    # Convert signals to entries/exits
    entries = (signals == 1) & (signals.shift(1) != 1)
    exits = (signals == 0) & (signals.shift(1) == 1)

    result = engine.run_backtest(price, entries, exits)

    return {
        "total_return": result.total_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
    }


def calculate_benchmark_metrics(price: pd.Series) -> dict[str, float]:
    """
    Calculate buy-and-hold benchmark metrics.

    Args:
        price: Price series

    Returns:
        Dict with benchmark metrics
    """
    returns = price.pct_change().dropna()

    total_return = (price.iloc[-1] / price.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(price)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0

    # Max drawdown
    rolling_max = price.cummax()
    drawdowns = (price - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }
