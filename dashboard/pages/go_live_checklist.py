"""
Go-Live Checklist Dashboard Page.

Displays readiness status for transitioning from paper trading to live trading.
Shows trading metrics, market experience, and system health with visual indicators.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading.go_live_monitor import GoLiveMonitor, GoLiveStatus
from data.storage import DatabaseManager


def get_progress_color(percentage: float) -> str:
    """
    Get color based on progress percentage.

    Args:
        percentage: Progress as decimal (0.0 to 1.0+)

    Returns:
        Color string for display
    """
    if percentage >= 1.0:
        return "green"
    elif percentage >= 0.5:
        return "orange"
    else:
        return "red"


def render_metric_card(
    label: str,
    current_value: float | str,
    threshold: float | str,
    percentage: float,
    unit: str = "",
    inverse: bool = False,
) -> None:
    """
    Render a single metric card with progress bar.

    Args:
        label: Metric name
        current_value: Current value
        threshold: Required threshold
        percentage: Progress percentage (0.0 to 1.0+)
        unit: Unit string to display
        inverse: If True, lower is better (e.g., drawdown)
    """
    color = get_progress_color(percentage)
    passed = percentage >= 1.0

    # Format values for display
    if isinstance(current_value, float):
        if unit == "%":
            display_value = f"{current_value:.1f}%"
        else:
            display_value = f"{current_value:.2f}{unit}"
    else:
        display_value = f"{current_value}{unit}"

    if isinstance(threshold, float):
        if unit == "%":
            display_threshold = f"{threshold:.1f}%"
        else:
            display_threshold = f"{threshold:.2f}{unit}"
    else:
        display_threshold = f"{threshold}{unit}"

    # Container for the metric
    with st.container():
        # Pass/fail indicator
        if passed:
            st.markdown(f"**{label}** :green_circle:")
        else:
            st.markdown(f"**{label}** :red_circle:")

        # Current vs threshold
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Current")
            st.markdown(f"**{display_value}**")
        with col2:
            st.caption("Required")
            comparison = "<" if inverse else ">="
            st.markdown(f"{comparison} {display_threshold}")

        # Progress bar with color
        progress_display = min(percentage, 1.0)  # Cap at 100% for display

        if color == "green":
            st.progress(progress_display, text=f"{percentage*100:.0f}%")
        elif color == "orange":
            st.progress(progress_display, text=f"{percentage*100:.0f}%")
        else:
            st.progress(progress_display, text=f"{percentage*100:.0f}%")

        st.divider()


def render_overall_readiness_gauge(status: GoLiveStatus) -> None:
    """
    Render the overall readiness gauge at the top of the page.

    Args:
        status: GoLiveStatus object with readiness data
    """
    # Calculate overall percentage
    overall_pct = status.overall_readiness_pct

    # Determine status
    if overall_pct >= 100:
        status_text = "READY FOR LIVE TRADING"
        status_color = "green"
        icon = ":white_check_mark:"
    elif overall_pct >= 75:
        status_text = "ALMOST READY"
        status_color = "orange"
        icon = ":hourglass_flowing_sand:"
    elif overall_pct >= 50:
        status_text = "IN PROGRESS"
        status_color = "orange"
        icon = ":chart_with_upwards_trend:"
    else:
        status_text = "NOT READY"
        status_color = "red"
        icon = ":x:"

    # Create centered header
    st.markdown(f"## {icon} Go-Live Readiness")

    # Big percentage display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Circular gauge effect using metric
        st.metric(
            label="Overall Readiness",
            value=f"{overall_pct:.0f}%",
            delta=status_text,
        )

        # Progress bar for visual effect
        st.progress(min(overall_pct / 100, 1.0))

        # Passed requirements count
        passed = status.requirements_passed
        total = status.requirements_total
        st.caption(f"Requirements: {passed}/{total} passed")

    st.divider()


def render_trading_metrics(status: GoLiveStatus) -> None:
    """
    Render the Trading Metrics column.

    Args:
        status: GoLiveStatus object
    """
    st.subheader("Trading Metrics")

    metrics = status.trading_metrics

    # Trade Count
    render_metric_card(
        label="Trade Count",
        current_value=metrics.trade_count,
        threshold=metrics.trade_count_goal,
        percentage=metrics.trade_count / metrics.trade_count_goal if metrics.trade_count_goal > 0 else 0,
        unit="",
    )

    # Profit Factor
    render_metric_card(
        label="Profit Factor",
        current_value=metrics.profit_factor,
        threshold=metrics.profit_factor_goal,
        percentage=metrics.profit_factor / metrics.profit_factor_goal if metrics.profit_factor_goal > 0 else 0,
        unit="",
    )

    # Sharpe Ratio
    render_metric_card(
        label="Sharpe Ratio",
        current_value=metrics.sharpe_ratio,
        threshold=metrics.sharpe_ratio_goal,
        percentage=metrics.sharpe_ratio / metrics.sharpe_ratio_goal if metrics.sharpe_ratio_goal > 0 else 0,
        unit="",
    )

    # Win Rate
    render_metric_card(
        label="Win Rate",
        current_value=metrics.win_rate * 100,
        threshold=metrics.win_rate_goal * 100,
        percentage=metrics.win_rate / metrics.win_rate_goal if metrics.win_rate_goal > 0 else 0,
        unit="%",
    )

    # Max Drawdown (inverse - lower is better)
    # For drawdown, we invert the logic: passing means staying UNDER the threshold
    dd_passed = metrics.max_drawdown <= metrics.max_drawdown_goal
    dd_percentage = 1.0 if dd_passed else (metrics.max_drawdown_goal / metrics.max_drawdown if metrics.max_drawdown > 0 else 0)

    render_metric_card(
        label="Max Drawdown",
        current_value=metrics.max_drawdown * 100,
        threshold=metrics.max_drawdown_goal * 100,
        percentage=dd_percentage,
        unit="%",
        inverse=True,
    )


def render_market_experience(status: GoLiveStatus) -> None:
    """
    Render the Market Experience column.

    Args:
        status: GoLiveStatus object
    """
    st.subheader("Market Experience")

    experience = status.market_experience

    # Regimes Seen
    render_metric_card(
        label="Regimes Seen",
        current_value=experience.regimes_seen,
        threshold=experience.regimes_required,
        percentage=experience.regimes_seen / experience.regimes_required if experience.regimes_required > 0 else 0,
        unit="",
    )

    # List actual regimes experienced
    st.markdown("**Regimes Experienced:**")
    if experience.regimes_list:
        for regime in experience.regimes_list:
            regime_emoji = {
                "trending_up": ":chart_with_upwards_trend:",
                "trending_down": ":chart_with_downwards_trend:",
                "volatile": ":ocean:",
                "quiet": ":zzz:",
                "mixed": ":twisted_rightwards_arrows:",
            }.get(regime.lower(), ":question:")

            st.markdown(f"- {regime_emoji} {regime.replace('_', ' ').title()}")
    else:
        st.info("No market regimes recorded yet")

    st.divider()

    # Days Trading
    st.markdown("**Trading Days:**")
    st.metric(
        label="Days Active",
        value=experience.days_trading,
        delta=f"Goal: {experience.days_required}" if experience.days_trading < experience.days_required else "Complete",
    )


def render_system_health(status: GoLiveStatus) -> None:
    """
    Render the System Health column.

    Args:
        status: GoLiveStatus object
    """
    st.subheader("System Health")

    health = status.system_health

    # System Uptime
    render_metric_card(
        label="System Uptime",
        current_value=health.uptime_pct * 100,
        threshold=health.uptime_goal * 100,
        percentage=health.uptime_pct / health.uptime_goal if health.uptime_goal > 0 else 0,
        unit="%",
    )

    # Model Accuracy
    st.markdown("**Model Accuracy (Recent):**")
    if health.model_accuracy is not None:
        accuracy_color = "green" if health.model_accuracy >= 0.55 else "orange" if health.model_accuracy >= 0.50 else "red"
        st.metric(
            label="7-Day Accuracy",
            value=f"{health.model_accuracy * 100:.1f}%",
        )
    else:
        st.info("No accuracy data available")

    st.divider()

    # Last Error
    st.markdown("**Last Error:**")
    if health.last_error_time:
        time_since = datetime.now() - health.last_error_time
        hours_since = time_since.total_seconds() / 3600

        if hours_since < 1:
            st.error(f":warning: {time_since.seconds // 60} minutes ago")
        elif hours_since < 24:
            st.warning(f":clock1: {hours_since:.1f} hours ago")
        else:
            st.success(f":white_check_mark: {time_since.days} days ago")

        if health.last_error_message:
            with st.expander("Error Details"):
                st.code(health.last_error_message)
    else:
        st.success(":white_check_mark: No errors recorded")

    st.divider()

    # Connection Status
    st.markdown("**Connection Status:**")
    if health.ibkr_connected:
        st.success(":green_circle: IBKR Connected")
    else:
        st.error(":red_circle: IBKR Disconnected")

    if health.data_feed_active:
        st.success(":green_circle: Data Feed Active")
    else:
        st.warning(":orange_circle: Data Feed Inactive")


def render_action_items(status: GoLiveStatus) -> None:
    """
    Render action items for things that need attention.

    Args:
        status: GoLiveStatus object
    """
    if not status.action_items:
        return

    st.divider()
    st.subheader("Action Items")

    for item in status.action_items:
        severity = item.get("severity", "info")
        message = item.get("message", "")

        if severity == "critical":
            st.error(f":rotating_light: {message}")
        elif severity == "warning":
            st.warning(f":warning: {message}")
        else:
            st.info(f":information_source: {message}")


def render_go_live_checklist() -> None:
    """
    Main function to render the Go-Live Checklist page.

    This function can be called from dashboard/app.py to display the page.
    """
    st.header(":clipboard: Go-Live Checklist")
    st.caption("Paper trading readiness assessment for live trading transition")

    # Initialize database and monitor
    try:
        db = DatabaseManager()
        monitor = GoLiveMonitor(db_manager=db)
        status = monitor.get_readiness_status()
    except ImportError as e:
        st.error(f"GoLiveMonitor not yet implemented: {e}")
        st.info("This page requires the trading.go_live_monitor module to be created.")

        # Show placeholder with expected structure
        st.divider()
        st.subheader("Expected Implementation")
        st.code("""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class TradingMetrics:
    trade_count: int
    trade_count_goal: int = 100
    profit_factor: float = 0.0
    profit_factor_goal: float = 1.2
    sharpe_ratio: float = 0.0
    sharpe_ratio_goal: float = 0.8
    win_rate: float = 0.0
    win_rate_goal: float = 0.45
    max_drawdown: float = 0.0
    max_drawdown_goal: float = 0.15

@dataclass
class MarketExperience:
    regimes_seen: int
    regimes_required: int = 2
    regimes_list: List[str] = None
    days_trading: int = 0
    days_required: int = 20

@dataclass
class SystemHealth:
    uptime_pct: float
    uptime_goal: float = 0.99
    model_accuracy: Optional[float] = None
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None
    ibkr_connected: bool = False
    data_feed_active: bool = False

@dataclass
class GoLiveStatus:
    overall_readiness_pct: float
    requirements_passed: int
    requirements_total: int
    trading_metrics: TradingMetrics
    market_experience: MarketExperience
    system_health: SystemHealth
    action_items: List[dict] = None

class GoLiveMonitor:
    def __init__(self, db_manager):
        self.db = db_manager

    def get_readiness_status(self) -> GoLiveStatus:
        # Implementation here
        pass
        """, language="python")
        return
    except Exception as e:
        st.error(f"Error loading readiness status: {e}")
        return

    # Render overall readiness gauge
    render_overall_readiness_gauge(status)

    # Three column layout
    col1, col2, col3 = st.columns(3)

    with col1:
        render_trading_metrics(status)

    with col2:
        render_market_experience(status)

    with col3:
        render_system_health(status)

    # Action items at the bottom
    render_action_items(status)

    # Refresh button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Refresh Status", use_container_width=True):
            st.rerun()

    # Last updated timestamp
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Allow running as standalone for testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="Go-Live Checklist - Shariah Daytrader",
        page_icon=":clipboard:",
        layout="wide",
    )
    render_go_live_checklist()
