#!/usr/bin/env python3
"""
ML Insights CLI - Command-line tool for monitoring ML model health.

Commands:
    status      Model health summary with traffic light indicators
    report      Weekly performance report
    alerts      View recent learning events and alerts
    drift       Feature importance drift summary

Usage:
    python insights.py status
    python insights.py report --weekly
    python insights.py alerts
    python insights.py drift
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from typing import Any

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def color(text: str, *codes: str) -> str:
    """Apply color codes to text."""
    return "".join(codes) + str(text) + Colors.RESET


def status_indicator(is_good: bool | None) -> str:
    """Return colored status indicator."""
    if is_good is None:
        return color("○", Colors.DIM)
    elif is_good:
        return color("●", Colors.BRIGHT_GREEN)
    else:
        return color("●", Colors.BRIGHT_RED)


def severity_color(severity: str) -> str:
    """Get color for severity level."""
    if severity == "critical":
        return Colors.BRIGHT_RED
    elif severity == "warning":
        return Colors.BRIGHT_YELLOW
    else:
        return Colors.CYAN


def print_header(title: str) -> None:
    """Print a styled header."""
    print()
    print(color("═" * 60, Colors.BLUE))
    print(color(f"  {title}", Colors.BOLD, Colors.BRIGHT_CYAN))
    print(color("═" * 60, Colors.BLUE))
    print()


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print(color(f"── {title} ", Colors.BOLD) + color("─" * (50 - len(title)), Colors.DIM))
    print()


def print_metric(label: str, value: Any, good: bool | None = None, suffix: str = "") -> None:
    """Print a metric with optional status indicator."""
    indicator = status_indicator(good) if good is not None else "  "
    value_str = f"{value}{suffix}" if value is not None else color("N/A", Colors.DIM)
    print(f"  {indicator} {label:.<30} {value_str}")


def cmd_status(args: argparse.Namespace) -> int:
    """Show model health status summary."""
    from data.storage import DatabaseManager
    from ml.insights.monitor import MLInsightsMonitor

    db = DatabaseManager()
    monitor = MLInsightsMonitor(db_manager=db)

    model_name = args.model or "momentum_continuation"

    print_header(f"ML Model Health Status: {model_name}")

    # Get health summary
    summary = monitor.get_health_summary(model_name)

    # Overall status
    overall = summary.get("overall_status", "unknown")
    status_colors = {
        "healthy": Colors.BRIGHT_GREEN,
        "attention": Colors.BRIGHT_YELLOW,
        "warning": Colors.BRIGHT_YELLOW,
        "critical": Colors.BRIGHT_RED,
    }
    print(f"  Overall Status: {color(overall.upper(), Colors.BOLD, status_colors.get(overall, Colors.WHITE))}")
    health_score = summary.get('health_score', 0)
    print(f"  Health Score:   {color(f'{health_score:.0f}%', Colors.BOLD)}")
    print(f"  Checks Passed:  {summary.get('checks_passed', 0)}/{summary.get('checks_total', 0)}")
    print()

    # Individual checks
    print_section("Check Results")

    for check in summary.get("checks", []):
        passed = check.get("passed", False)
        severity = check.get("severity", "info")
        name = check.get("name", "").replace("_", " ").title()
        message = check.get("message", "")

        indicator = status_indicator(passed)
        sev_color = severity_color(severity)

        print(f"  {indicator} {name:.<25} {color(message[:40], sev_color)}")

    # Model details
    health = db.get_model_health_metrics(model_name)

    if health.get("status") != "no_active_model":
        print_section("Model Details")
        print_metric("Model Version", health.get("model_version"))
        print_metric("Days Since Training", health.get("days_since_training"), health.get("days_since_training", 999) < 14, " days")
        print_metric("7-Day Accuracy", f"{health.get('accuracy_7d', 0):.1%}" if health.get('accuracy_7d') else None, (health.get('accuracy_7d') or 0) >= 0.55)
        print_metric("Predictions (7d)", health.get("predictions_7d"))
        print_metric("Calibration ECE", f"{health.get('calibration_ece', 0):.4f}" if health.get('calibration_ece') else None, (health.get('calibration_ece') or 1) < 0.10)
        print_metric("Drift Alerts", health.get("drift_alerts"), health.get("drift_alerts", 999) == 0)
        print_metric("Open Events", health.get("open_events"), health.get("open_events", 999) == 0)

    print()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate a performance report."""
    from data.storage import DatabaseManager
    from ml.insights.monitor import MLInsightsMonitor

    db = DatabaseManager()
    monitor = MLInsightsMonitor(db_manager=db)

    model_name = args.model or "momentum_continuation"
    days = 7 if args.weekly else 30

    print_header(f"ML Performance Report ({days} Days)")

    # Get active model
    active_model = db.get_active_model(model_name)
    if not active_model:
        print(color(f"  No active model found for '{model_name}'", Colors.YELLOW))
        return 1

    model_id = active_model.get("id")

    # Model info
    print_section("Model Information")
    print_metric("Model Name", model_name)
    print_metric("Model Version", active_model.get("model_version"))
    print_metric("Status", active_model.get("status"))
    print_metric("Created", active_model.get("created_at", "")[:10] if active_model.get("created_at") else None)

    # Performance metrics
    print_section("Prediction Performance")
    accuracy_data = db.get_prediction_accuracy(model_id, days=days)
    print_metric("Total Predictions", accuracy_data.get("total"))
    print_metric("Correct Predictions", accuracy_data.get("correct"))
    print_metric("Accuracy", f"{accuracy_data.get('accuracy', 0):.2%}" if accuracy_data.get('accuracy') else None, (accuracy_data.get('accuracy') or 0) >= 0.55)
    print_metric("Avg Error %", f"{accuracy_data.get('avg_error_pct', 0):.2f}%" if accuracy_data.get('avg_error_pct') else None)

    # Calibration
    print_section("Calibration Metrics")
    calibration = db.get_latest_calibration(model_id)
    if calibration:
        print_metric("ECE", f"{calibration.get('expected_calibration_error', 0):.4f}", calibration.get('is_well_calibrated'))
        print_metric("Brier Score", f"{calibration.get('brier_score', 0):.4f}" if calibration.get('brier_score') else None)
        print_metric("Sample Count", calibration.get("sample_count"))
        print_metric("Period", f"{calibration.get('period_start', '')[:10]} to {calibration.get('period_end', '')[:10]}")
    else:
        print(color("  No calibration data available", Colors.DIM))

    # Training metrics
    print_section("Training Metrics")
    metrics = active_model.get("metrics", {})
    print_metric("Direction Accuracy", f"{metrics.get('direction_accuracy', 0):.2%}" if metrics.get('direction_accuracy') else None)
    print_metric("Direction F1", f"{metrics.get('direction_f1', 0):.2%}" if metrics.get('direction_f1') else None)
    print_metric("Magnitude MAE", f"{metrics.get('magnitude_mae', 0):.4f}" if metrics.get('magnitude_mae') else None)
    print_metric("Duration MAE", f"{metrics.get('duration_mae', 0):.2f} bars" if metrics.get('duration_mae') else None)

    # Recent events
    print_section("Recent Events")
    events = db.get_learning_events_history(days=days, limit=5)
    if events:
        for event in events:
            sev = event.get("severity", "info")
            sev_color = severity_color(sev)
            print(f"  {color(sev.upper()[:4], sev_color)} {event.get('event_time', '')[:10]} {event.get('title', 'Event')}")
    else:
        print(color("  No events in this period", Colors.DIM))

    # Summary
    print_section("Summary")
    summary = monitor.get_health_summary(model_name)
    overall = summary.get("overall_status", "unknown")
    print(f"  Overall Health: {color(overall.upper(), Colors.BOLD)}")
    print(f"  Recommendation: ", end="")

    if summary.get("critical_issues", 0) > 0:
        print(color("Immediate attention required - critical issues detected", Colors.BRIGHT_RED))
    elif summary.get("warnings", 0) > 0:
        print(color("Review warnings and consider retraining", Colors.BRIGHT_YELLOW))
    else:
        print(color("Model is healthy, continue monitoring", Colors.BRIGHT_GREEN))

    print()
    return 0


def cmd_alerts(args: argparse.Namespace) -> int:
    """Show recent learning events and alerts."""
    from data.storage import DatabaseManager

    db = DatabaseManager()

    print_header("ML Learning Events & Alerts")

    # Open events requiring attention
    open_events = db.get_open_learning_events()

    if open_events:
        print_section(f"Open Events ({len(open_events)})")

        for event in open_events[:20]:
            sev = event.get("severity", "info")
            sev_color = severity_color(sev)

            status_icon = "⚡" if event.get("requires_action") else "○"
            print(f"  {status_icon} [{color(sev.upper()[:4], sev_color)}] {event.get('title', 'Event')}")
            print(f"       {color(event.get('description', '')[:60], Colors.DIM)}")
            print(f"       Category: {event.get('category', 'N/A')} | ID: {event.get('id')} | {event.get('event_time', '')[:16]}")
            print()
    else:
        print(color("  ✓ No open events requiring attention", Colors.BRIGHT_GREEN))

    # Recent history
    print_section("Recent History (30 Days)")

    history = db.get_learning_events_history(days=30, limit=15)
    if history:
        print(f"  {'TIME':<16} {'SEV':<8} {'TYPE':<12} {'STATUS':<10} TITLE")
        print(color("  " + "-" * 70, Colors.DIM))

        for event in history:
            sev = event.get("severity", "info")
            sev_color = severity_color(sev)

            time_str = event.get("event_time", "")[:16]
            sev_str = color(sev[:7], sev_color)
            type_str = event.get("event_type", "")[:10]
            status_str = event.get("status", "")[:8]
            title_str = event.get("title", "")[:30]

            print(f"  {time_str:<16} {sev_str:<17} {type_str:<12} {status_str:<10} {title_str}")
    else:
        print(color("  No events recorded", Colors.DIM))

    # Stats summary
    print_section("Statistics")
    ml_stats = db.get_ml_statistics()
    print_metric("Total Learning Events", ml_stats.get("learning_events_total"))
    print_metric("Open Events", ml_stats.get("learning_events_open"), ml_stats.get("learning_events_open", 999) == 0)

    print()
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    """Show feature importance drift summary."""
    from data.storage import DatabaseManager

    db = DatabaseManager()

    model_name = args.model or "momentum_continuation"

    print_header(f"Feature Importance Drift: {model_name}")

    # Get active model
    active_model = db.get_active_model(model_name)
    if not active_model:
        print(color(f"  No active model found for '{model_name}'", Colors.YELLOW))
        return 1

    model_id = active_model.get("id")

    # Get drift alerts
    print_section("Drift Alerts")
    alerts = db.get_feature_drift_alerts(model_id, rank_change_threshold=3, top_n=15)

    if alerts:
        print(color(f"  {len(alerts)} feature(s) with significant rank changes:", Colors.BRIGHT_YELLOW))
        print()

        for alert in alerts:
            feature = alert.get("feature", "unknown")
            alert_type = alert.get("type", "")

            if alert_type == "rank_change":
                change = alert.get("change", 0)
                prev = alert.get("previous_rank")
                curr = alert.get("current_rank")

                if change > 0:
                    arrow = color("↑", Colors.BRIGHT_GREEN)
                    change_str = color(f"+{change}", Colors.BRIGHT_GREEN)
                else:
                    arrow = color("↓", Colors.BRIGHT_RED)
                    change_str = color(str(change), Colors.BRIGHT_RED)

                print(f"  {arrow} {feature:.<35} Rank {prev} → {curr} ({change_str})")
            elif alert_type == "new_feature":
                print(f"  {color('★', Colors.BRIGHT_CYAN)} {feature:.<35} {color('NEW in top features', Colors.CYAN)}")
    else:
        print(color("  ✓ No significant feature drift detected", Colors.BRIGHT_GREEN))

    # Current top features
    print_section("Current Top Features")
    history = db.get_feature_importance_history(model_id, limit=1)

    if history:
        # Group by component
        combined = [h for h in history if h.get("model_component") == "combined"]
        combined.sort(key=lambda x: x.get("rank", 999))

        print(f"  {'RANK':<6} {'FEATURE':<30} {'IMPORTANCE':<12}")
        print(color("  " + "-" * 50, Colors.DIM))

        for feat in combined[:15]:
            rank = feat.get("rank", 0)
            name = feat.get("feature_name", "")[:28]
            score = feat.get("importance_score", 0)

            # Color code top features
            if rank <= 3:
                name_color = Colors.BRIGHT_GREEN
            elif rank <= 7:
                name_color = Colors.CYAN
            else:
                name_color = Colors.WHITE

            print(f"  {rank:<6} {color(name, name_color):<39} {score:.6f}")
    else:
        print(color("  No feature importance data available", Colors.DIM))

    print()
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML Insights CLI - Monitor ML model health and performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python insights.py status                   # Show model health status
    python insights.py status -m lightgbm       # Status for specific model
    python insights.py report --weekly          # Weekly performance report
    python insights.py alerts                   # View learning events
    python insights.py drift                    # Feature drift summary
        """
    )

    parser.add_argument(
        "-m", "--model",
        default="momentum_continuation",
        help="Model name to analyze (default: momentum_continuation)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show model health status")
    status_parser.set_defaults(func=cmd_status)

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate performance report")
    report_parser.add_argument("--weekly", action="store_true", help="Generate weekly report (default)")
    report_parser.add_argument("--monthly", action="store_true", help="Generate monthly report")
    report_parser.set_defaults(func=cmd_report)

    # Alerts command
    alerts_parser = subparsers.add_parser("alerts", help="Show learning events and alerts")
    alerts_parser.set_defaults(func=cmd_alerts)

    # Drift command
    drift_parser = subparsers.add_parser("drift", help="Show feature drift summary")
    drift_parser.set_defaults(func=cmd_drift)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except Exception as e:
        print(color(f"Error: {e}", Colors.BRIGHT_RED))
        return 1


if __name__ == "__main__":
    sys.exit(main())
