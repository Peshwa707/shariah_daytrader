"""
LLM Summary Generator - Natural language weekly reports using Claude.

Generates human-readable summaries of ML model performance,
market conditions, and recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig:
    """Configuration for summary generation."""
    # Time period
    lookback_days: int = 7

    # Output settings
    output_dir: str | Path = "reports"
    save_to_file: bool = True
    filename_format: str = "weekly_summary_{date}.md"

    # LLM settings
    model: str = "claude-3-haiku-20240307"  # Use haiku for cost efficiency
    max_tokens: int = 2000
    temperature: float = 0.3  # Low for factual summaries

    # Include sections
    include_performance: bool = True
    include_calibration: bool = True
    include_drift: bool = True
    include_regime: bool = True
    include_events: bool = True
    include_recommendations: bool = True


class WeeklySummaryGenerator:
    """
    Generates natural language weekly summaries using an LLM.

    Gathers data from ML insight tables and formats a prompt
    for Claude to generate a human-readable report.
    """

    SYSTEM_PROMPT = """You are an ML Operations analyst for a Shariah-compliant trading system.
Your role is to generate concise, actionable weekly summaries of model performance.

Guidelines:
- Be factual and data-driven
- Highlight concerning trends clearly
- Provide specific, actionable recommendations
- Use bullet points for clarity
- Keep the tone professional but accessible
- Flag any issues that require immediate attention
- Compare current metrics to thresholds/baselines where available"""

    SUMMARY_TEMPLATE = """Generate a weekly ML model performance summary based on the following data.

## Model Information
{model_info}

## Performance Metrics (Last {days} Days)
{performance_data}

## Calibration Status
{calibration_data}

## Feature Drift Analysis
{drift_data}

## Market Regime Summary
{regime_data}

## Learning Events & Alerts
{events_data}

---

Please generate a structured summary with these sections:
1. **Executive Summary** (2-3 sentences)
2. **Performance Highlights** (bullet points)
3. **Areas of Concern** (if any)
4. **Market Conditions Impact** (brief)
5. **Recommendations** (prioritized action items)

Keep the summary concise but comprehensive. Highlight anything that needs immediate attention."""

    def __init__(
        self,
        config: SummaryConfig | None = None,
        db_manager=None,
    ):
        """
        Initialize the summary generator.

        Args:
            config: Summary configuration
            db_manager: Database manager instance
        """
        self.config = config or SummaryConfig()
        self._db_manager = db_manager

    @property
    def db(self):
        """Get database manager (lazy loaded)."""
        if self._db_manager is None:
            from data.storage import DatabaseManager
            self._db_manager = DatabaseManager()
        return self._db_manager

    def gather_data(self, model_name: str = "momentum_continuation") -> dict[str, Any]:
        """
        Gather all data needed for the summary.

        Args:
            model_name: Name of the model to report on

        Returns:
            Dict with all gathered data
        """
        data = {
            "model_name": model_name,
            "period_start": (datetime.now() - timedelta(days=self.config.lookback_days)).isoformat()[:10],
            "period_end": datetime.now().isoformat()[:10],
            "days": self.config.lookback_days,
        }

        # Get active model
        active_model = self.db.get_active_model(model_name)
        if active_model:
            data["model_info"] = {
                "version": active_model.get("model_version"),
                "status": active_model.get("status"),
                "created_at": active_model.get("created_at"),
                "training_metrics": active_model.get("metrics", {}),
            }

            model_id = active_model.get("id")

            # Performance data
            if self.config.include_performance:
                accuracy = self.db.get_prediction_accuracy(
                    model_id, days=self.config.lookback_days
                )
                data["performance"] = accuracy

            # Calibration data
            if self.config.include_calibration:
                calibration = self.db.get_latest_calibration(model_id)
                data["calibration"] = calibration

            # Drift data
            if self.config.include_drift:
                drift_alerts = self.db.get_feature_drift_alerts(model_id)
                data["drift_alerts"] = drift_alerts

            # Health metrics
            data["health"] = self.db.get_model_health_metrics(model_name)

        # Regime data
        if self.config.include_regime:
            regimes = self.db.get_regime_history(days=self.config.lookback_days)
            data["regimes"] = regimes

        # Learning events
        if self.config.include_events:
            events = self.db.get_learning_events_history(days=self.config.lookback_days)
            open_events = self.db.get_open_learning_events()
            data["events"] = events
            data["open_events"] = open_events

        # ML statistics
        data["statistics"] = self.db.get_ml_statistics()

        return data

    def format_prompt_data(self, data: dict[str, Any]) -> dict[str, str]:
        """
        Format gathered data into prompt sections.

        Args:
            data: Raw gathered data

        Returns:
            Dict with formatted string sections
        """
        sections = {}

        # Model info
        model_info = data.get("model_info", {})
        if model_info:
            training_metrics = model_info.get("training_metrics", {})
            sections["model_info"] = f"""- Model: {data.get('model_name')}
- Version: {model_info.get('version', 'N/A')}
- Status: {model_info.get('status', 'N/A')}
- Created: {model_info.get('created_at', 'N/A')[:10] if model_info.get('created_at') else 'N/A'}
- Training Accuracy: {training_metrics.get('direction_accuracy', 'N/A')}
- Training F1: {training_metrics.get('direction_f1', 'N/A')}"""
        else:
            sections["model_info"] = "No active model found."

        # Performance
        perf = data.get("performance", {})
        if perf:
            sections["performance_data"] = f"""- Total Predictions: {perf.get('total', 0)}
- Correct Predictions: {perf.get('correct', 0)}
- Accuracy: {perf.get('accuracy', 0):.2%} if perf.get('accuracy') else 'N/A'
- Average Error %: {perf.get('avg_error_pct', 'N/A')}
- Threshold: 55% (minimum acceptable)"""
        else:
            sections["performance_data"] = "No prediction data available."

        # Calibration
        cal = data.get("calibration", {})
        if cal:
            sections["calibration_data"] = f"""- ECE (Expected Calibration Error): {cal.get('expected_calibration_error', 'N/A')}
- Brier Score: {cal.get('brier_score', 'N/A')}
- Is Well Calibrated: {'Yes' if cal.get('is_well_calibrated') else 'No'}
- Sample Count: {cal.get('sample_count', 0)}
- ECE Threshold: 0.10 (maximum acceptable)"""
        else:
            sections["calibration_data"] = "No calibration data available."

        # Drift
        drift = data.get("drift_alerts", [])
        if drift:
            drift_lines = [f"- {len(drift)} feature(s) with significant rank changes:"]
            for d in drift[:5]:
                if d.get("type") == "rank_change":
                    drift_lines.append(f"  - {d['feature']}: rank {d.get('previous_rank')} → {d.get('current_rank')} (change: {d.get('change', 0):+d})")
            sections["drift_data"] = "\n".join(drift_lines)
        else:
            sections["drift_data"] = "No significant feature drift detected."

        # Regime
        regimes = data.get("regimes", [])
        if regimes:
            # Count regime types
            regime_counts = {}
            for r in regimes:
                rt = r.get("regime_type", "unknown")
                regime_counts[rt] = regime_counts.get(rt, 0) + 1

            current = regimes[0] if regimes else {}
            regime_lines = [
                f"- Current Regime: {current.get('regime_type', 'N/A')} (confidence: {current.get('regime_confidence', 0):.1%})",
                f"- VIX Level: {current.get('vix_level', 'N/A')}",
                f"- Regime Distribution ({data['days']} days):",
            ]
            for rt, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
                regime_lines.append(f"  - {rt}: {count} days")
            sections["regime_data"] = "\n".join(regime_lines)
        else:
            sections["regime_data"] = "No regime data available."

        # Events
        events = data.get("events", [])
        open_events = data.get("open_events", [])
        if events or open_events:
            event_lines = [f"- Open Events Requiring Action: {len(open_events)}"]
            if open_events:
                for e in open_events[:3]:
                    event_lines.append(f"  - [{e.get('severity', 'info').upper()}] {e.get('title', 'Event')}")

            # Count by severity
            severity_counts = {}
            for e in events:
                sev = e.get("severity", "info")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            event_lines.append(f"- Total Events ({data['days']} days): {len(events)}")
            for sev, count in sorted(severity_counts.items()):
                event_lines.append(f"  - {sev}: {count}")

            sections["events_data"] = "\n".join(event_lines)
        else:
            sections["events_data"] = "No learning events recorded."

        return sections

    def build_prompt(self, data: dict[str, Any]) -> str:
        """
        Build the complete prompt for the LLM.

        Args:
            data: Raw gathered data

        Returns:
            Formatted prompt string
        """
        sections = self.format_prompt_data(data)

        prompt = self.SUMMARY_TEMPLATE.format(
            model_info=sections.get("model_info", "N/A"),
            days=data.get("days", 7),
            performance_data=sections.get("performance_data", "N/A"),
            calibration_data=sections.get("calibration_data", "N/A"),
            drift_data=sections.get("drift_data", "N/A"),
            regime_data=sections.get("regime_data", "N/A"),
            events_data=sections.get("events_data", "N/A"),
        )

        return prompt

    def generate_summary(
        self,
        model_name: str = "momentum_continuation",
        use_llm: bool = True,
    ) -> str:
        """
        Generate the weekly summary.

        Args:
            model_name: Name of the model to report on
            use_llm: Whether to use LLM for generation (False for template-only)

        Returns:
            Generated summary text
        """
        # Gather data
        data = self.gather_data(model_name)

        # Build prompt
        prompt = self.build_prompt(data)

        if not use_llm:
            # Return structured data without LLM generation
            return self._generate_template_summary(data)

        # Call LLM
        try:
            summary = self._call_llm(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            logger.info("Falling back to template summary")
            summary = self._generate_template_summary(data)

        # Save to file if configured
        if self.config.save_to_file:
            self._save_summary(summary, data)

        return summary

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM API to generate summary.

        Args:
            prompt: The formatted prompt

        Returns:
            Generated summary text
        """
        try:
            import anthropic

            client = anthropic.Anthropic()

            message = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return message.content[0].text

        except ImportError:
            logger.warning("anthropic package not installed, using template summary")
            raise
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            raise

    def _generate_template_summary(self, data: dict[str, Any]) -> str:
        """
        Generate a template-based summary without LLM.

        Args:
            data: Gathered data

        Returns:
            Template-based summary
        """
        lines = [
            f"# Weekly ML Performance Summary",
            f"**Period:** {data.get('period_start')} to {data.get('period_end')}",
            f"**Model:** {data.get('model_name')}",
            "",
            "## Executive Summary",
        ]

        # Determine overall status
        health = data.get("health", {})
        accuracy = data.get("performance", {}).get("accuracy")
        calibration = data.get("calibration", {})
        open_events = len(data.get("open_events", []))

        issues = []
        if accuracy and accuracy < 0.55:
            issues.append("accuracy below threshold")
        if calibration and not calibration.get("is_well_calibrated"):
            issues.append("calibration drift detected")
        if data.get("drift_alerts"):
            issues.append("feature drift detected")
        if open_events > 0:
            issues.append(f"{open_events} open alert(s)")

        if not issues:
            lines.append("Model is performing within acceptable parameters. No immediate action required.")
        else:
            lines.append(f"Attention needed: {', '.join(issues)}.")

        # Performance
        lines.extend([
            "",
            "## Performance Highlights",
        ])

        perf = data.get("performance", {})
        if perf.get("total", 0) > 0:
            lines.append(f"- Predictions made: {perf.get('total')}")
            lines.append(f"- Accuracy: {perf.get('accuracy', 0):.1%}" if perf.get('accuracy') else "- Accuracy: N/A")
        else:
            lines.append("- No predictions in this period")

        # Issues
        if issues:
            lines.extend([
                "",
                "## Areas of Concern",
            ])
            for issue in issues:
                lines.append(f"- {issue.capitalize()}")

        # Recommendations
        lines.extend([
            "",
            "## Recommendations",
        ])

        if accuracy and accuracy < 0.55:
            lines.append("1. **High Priority:** Investigate low accuracy and consider model retraining")
        if calibration and not calibration.get("is_well_calibrated"):
            lines.append("2. Recalibrate model probabilities")
        if data.get("drift_alerts"):
            lines.append("3. Review feature importance changes and investigate root cause")
        if open_events > 0:
            lines.append(f"4. Address {open_events} open learning event(s)")

        if not issues:
            lines.append("- Continue regular monitoring")
            lines.append("- Schedule next model refresh within 14 days of last training")

        lines.extend([
            "",
            f"---",
            f"*Generated: {datetime.now().isoformat()[:19]}*",
        ])

        return "\n".join(lines)

    def _save_summary(self, summary: str, data: dict[str, Any]) -> Path:
        """
        Save summary to file.

        Args:
            summary: Generated summary text
            data: Source data

        Returns:
            Path to saved file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = self.config.filename_format.format(
            date=data.get("period_end", datetime.now().isoformat()[:10])
        )
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            f.write(summary)

        logger.info(f"Summary saved to {filepath}")
        return filepath

    def get_email_stub(
        self,
        summary: str,
        recipient: str = "user@example.com",
    ) -> dict[str, str]:
        """
        Generate an email stub for the summary.

        Args:
            summary: Generated summary text
            recipient: Email recipient

        Returns:
            Dict with email fields
        """
        # Extract first paragraph as preview
        lines = summary.split("\n")
        preview = ""
        for line in lines:
            if line.strip() and not line.startswith("#"):
                preview = line.strip()[:100]
                break

        return {
            "to": recipient,
            "subject": f"Weekly ML Performance Summary - {datetime.now().strftime('%Y-%m-%d')}",
            "body": summary,
            "preview": preview,
            "content_type": "text/markdown",
        }


# CLI entry point
def main():
    """Command-line entry point for summary generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate weekly ML summary")
    parser.add_argument("--model", default="momentum_continuation", help="Model name")
    parser.add_argument("--days", type=int, default=7, help="Lookback days")
    parser.add_argument("--no-llm", action="store_true", help="Use template only (no LLM)")
    parser.add_argument("--output", default="reports", help="Output directory")

    args = parser.parse_args()

    config = SummaryConfig(
        lookback_days=args.days,
        output_dir=args.output,
    )

    generator = WeeklySummaryGenerator(config=config)
    summary = generator.generate_summary(
        model_name=args.model,
        use_llm=not args.no_llm,
    )

    print(summary)


if __name__ == "__main__":
    main()
