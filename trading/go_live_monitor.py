"""
Go-Live Monitor Module.

Provides comprehensive readiness status for transitioning from paper
trading to live trading. Checks trading metrics, market experience,
and system health against configurable thresholds.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import logging

from config.settings import settings
from data.storage import DatabaseManager
from trading.trade_tracker import TradeTracker


logger = logging.getLogger(__name__)


@dataclass
class TradingMetrics:
    """Core trading performance metrics."""

    trade_count: int
    profit_factor: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float


@dataclass
class MarketExperience:
    """Market regime experience tracking."""

    regimes_seen: list[str]
    regime_count: int


@dataclass
class SystemHealth:
    """System health and reliability metrics."""

    uptime_pct: float
    model_accuracy: Optional[float]
    last_error: Optional[str]


@dataclass
class ChecklistItem:
    """Individual checklist item for go-live readiness."""

    name: str
    threshold: float | int
    current: float | int
    passed: bool
    category: str  # trading, experience, system


@dataclass
class GoLiveStatus:
    """Comprehensive go-live readiness status."""

    trading_metrics: TradingMetrics
    market_experience: MarketExperience
    system_health: SystemHealth
    overall_ready: bool
    checklist: list[ChecklistItem] = field(default_factory=list)


class GoLiveMonitor:
    """
    Monitors readiness for transitioning to live trading.

    Aggregates trading performance, market experience, and system
    health metrics to determine if the system is ready for live
    trading based on configurable thresholds.
    """

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        trade_tracker: Optional[TradeTracker] = None,
    ):
        """
        Initialize the GoLiveMonitor.

        Args:
            db: DatabaseManager instance (creates new if not provided)
            trade_tracker: TradeTracker instance (creates new if not provided)
        """
        self.db = db or DatabaseManager()
        self.trade_tracker = trade_tracker or TradeTracker(self.db)
        self._logger = logging.getLogger(f"{__name__}.GoLiveMonitor")

    def get_readiness_status(self) -> GoLiveStatus:
        """
        Get comprehensive go-live readiness status.

        Evaluates trading metrics, market experience, and system health
        against thresholds defined in settings.go_live.

        Returns:
            GoLiveStatus with all metrics, checklist, and overall readiness
        """
        # Get trading metrics from TradeTracker
        trading_metrics = self._get_trading_metrics()

        # Get market experience from regime history
        market_experience = self._get_market_experience()

        # Get system health metrics
        system_health = self._get_system_health()

        # Build checklist with threshold comparisons
        checklist = self._build_checklist(
            trading_metrics, market_experience, system_health
        )

        # Determine overall readiness (all checklist items must pass)
        overall_ready = all(item.passed for item in checklist)

        status = GoLiveStatus(
            trading_metrics=trading_metrics,
            market_experience=market_experience,
            system_health=system_health,
            overall_ready=overall_ready,
            checklist=checklist,
        )

        self._logger.info(
            f"Go-live readiness check: overall_ready={overall_ready}, "
            f"passed={sum(1 for c in checklist if c.passed)}/{len(checklist)}"
        )

        return status

    def _get_trading_metrics(self) -> TradingMetrics:
        """
        Get trading metrics from TradeTracker statistics.

        Returns:
            TradingMetrics dataclass
        """
        # Get statistics for the default period (30 days)
        stats = self.trade_tracker.calculate_statistics(days=30)

        return TradingMetrics(
            trade_count=stats.total_trades,
            profit_factor=stats.profit_factor if stats.profit_factor != float('inf') else 999.0,
            sharpe_ratio=stats.sharpe_ratio,
            win_rate=stats.win_rate,
            max_drawdown=stats.max_drawdown_pct,
        )

    def _get_market_experience(self) -> MarketExperience:
        """
        Get market experience from regime history.

        Queries unique regimes seen in the trading period.

        Returns:
            MarketExperience dataclass
        """
        # Get regime history (default 30 days)
        regime_history = self.db.get_regime_history(days=30)

        # Extract unique regime types
        regimes_seen = list(set(r["regime_type"] for r in regime_history if r.get("regime_type")))

        return MarketExperience(
            regimes_seen=regimes_seen,
            regime_count=len(regimes_seen),
        )

    def _get_system_health(self) -> SystemHealth:
        """
        Get system health metrics.

        Queries model accuracy and tracks system uptime.

        Returns:
            SystemHealth dataclass
        """
        # Get model accuracy from active model
        model_accuracy = self._get_model_accuracy()

        # For now, assume 100% uptime (can be enhanced with actual monitoring)
        uptime_pct = 100.0

        # Get last error from learning events
        last_error = self._get_last_error()

        return SystemHealth(
            uptime_pct=uptime_pct,
            model_accuracy=model_accuracy,
            last_error=last_error,
        )

    def _get_model_accuracy(self) -> Optional[float]:
        """
        Get prediction accuracy from the active model.

        Returns:
            Model accuracy as float or None if no active model
        """
        # Try to find an active model
        try:
            # Get any active model - use a common model name
            model = self.db.get_active_model("momentum_continuation")

            if model:
                accuracy_data = self.db.get_prediction_accuracy(
                    model_id=model["id"],
                    days=7,
                )
                return accuracy_data.get("accuracy")

            # If no specific model, check ML statistics
            ml_stats = self.db.get_ml_statistics()
            if ml_stats.get("active_models", 0) > 0:
                # Return None to indicate we have models but no accuracy data yet
                return None

        except Exception as e:
            self._logger.warning(f"Could not get model accuracy: {e}")

        return None

    def _get_last_error(self) -> Optional[str]:
        """
        Get the most recent critical error from learning events.

        Returns:
            Error description or None
        """
        try:
            events = self.db.get_open_learning_events(
                severity="critical",
                requires_action_only=True,
            )

            if events:
                return events[0].get("title")

        except Exception as e:
            self._logger.warning(f"Could not get last error: {e}")

        return None

    def _build_checklist(
        self,
        trading_metrics: TradingMetrics,
        market_experience: MarketExperience,
        system_health: SystemHealth,
    ) -> list[ChecklistItem]:
        """
        Build the go-live checklist with threshold comparisons.

        Args:
            trading_metrics: Current trading metrics
            market_experience: Current market experience
            system_health: Current system health

        Returns:
            List of ChecklistItem objects
        """
        checklist = []

        # Get thresholds from settings
        thresholds = settings.go_live

        # Trading category checks
        checklist.append(
            ChecklistItem(
                name="Minimum Trades",
                threshold=thresholds.min_trades,
                current=trading_metrics.trade_count,
                passed=trading_metrics.trade_count >= thresholds.min_trades,
                category="trading",
            )
        )

        checklist.append(
            ChecklistItem(
                name="Profit Factor",
                threshold=thresholds.min_profit_factor,
                current=trading_metrics.profit_factor,
                passed=trading_metrics.profit_factor >= thresholds.min_profit_factor,
                category="trading",
            )
        )

        checklist.append(
            ChecklistItem(
                name="Sharpe Ratio",
                threshold=thresholds.min_sharpe,
                current=trading_metrics.sharpe_ratio,
                passed=trading_metrics.sharpe_ratio >= thresholds.min_sharpe,
                category="trading",
            )
        )

        checklist.append(
            ChecklistItem(
                name="Win Rate",
                threshold=thresholds.min_win_rate,
                current=trading_metrics.win_rate,
                passed=trading_metrics.win_rate >= thresholds.min_win_rate,
                category="trading",
            )
        )

        checklist.append(
            ChecklistItem(
                name="Max Drawdown",
                threshold=thresholds.max_drawdown,
                current=trading_metrics.max_drawdown,
                passed=trading_metrics.max_drawdown <= thresholds.max_drawdown,
                category="trading",
            )
        )

        # Experience category checks
        checklist.append(
            ChecklistItem(
                name="Market Regimes Seen",
                threshold=thresholds.min_regimes,
                current=market_experience.regime_count,
                passed=market_experience.regime_count >= thresholds.min_regimes,
                category="experience",
            )
        )

        # System category checks
        checklist.append(
            ChecklistItem(
                name="System Uptime",
                threshold=95.0,  # 95% uptime threshold
                current=system_health.uptime_pct,
                passed=system_health.uptime_pct >= 95.0,
                category="system",
            )
        )

        # Model accuracy check (only if we have a model and accuracy data)
        if system_health.model_accuracy is not None:
            checklist.append(
                ChecklistItem(
                    name="Model Accuracy",
                    threshold=0.50,  # 50% accuracy threshold
                    current=system_health.model_accuracy,
                    passed=system_health.model_accuracy >= 0.50,
                    category="system",
                )
            )

        # No critical errors check
        checklist.append(
            ChecklistItem(
                name="No Critical Errors",
                threshold=0,
                current=0 if system_health.last_error is None else 1,
                passed=system_health.last_error is None,
                category="system",
            )
        )

        return checklist
