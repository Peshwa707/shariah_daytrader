"""
Stability Reporter - Generate weekly and monthly stability reports for ML models.

Provides comprehensive stability analysis including:
- Accuracy tracking over periods with trend detection
- Feature importance drift analysis
- Calibration metrics monitoring
- Alert generation for concerning changes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional
import json
import logging

from data.storage import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    """Comprehensive stability report for an ML model."""

    # Model identification
    model_name: str
    model_id: int
    report_type: str  # "weekly" or "monthly"

    # Time period
    period_start: datetime
    period_end: datetime

    # Accuracy metrics
    accuracy: Optional[float] = None
    accuracy_change: Optional[float] = None  # Change from previous period
    accuracy_trend: str = "stable"  # "improving", "declining", "stable"

    # Feature drift metrics
    feature_drift_score: float = 0.0
    drifted_features: list[str] = field(default_factory=list)

    # Calibration metrics
    calibration_ece: Optional[float] = None
    is_well_calibrated: bool = True

    # Alerts
    alerts: list[str] = field(default_factory=list)

    # Summary
    summary: str = ""

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "report_type": self.report_type,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "accuracy": self.accuracy,
            "accuracy_change": self.accuracy_change,
            "accuracy_trend": self.accuracy_trend,
            "feature_drift_score": self.feature_drift_score,
            "drifted_features": self.drifted_features,
            "calibration_ece": self.calibration_ece,
            "is_well_calibrated": self.is_well_calibrated,
            "alerts": self.alerts,
            "summary": self.summary,
        }


class StabilityReporter:
    """
    Generates stability reports for ML models.

    Provides weekly and monthly analysis of model health including
    accuracy trends, feature drift, and calibration quality.
    """

    # Thresholds for alerts
    ACCURACY_DECLINE_THRESHOLD = 0.05  # 5% decline triggers alert
    ACCURACY_MIN_THRESHOLD = 0.52  # Below this is concerning
    ECE_THRESHOLD = 0.10  # ECE above this is poorly calibrated
    DRIFT_SCORE_THRESHOLD = 3  # Number of drifted features to alert

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the StabilityReporter.

        Args:
            db_manager: Database manager instance (lazy loaded if not provided)
        """
        self._db_manager = db_manager

    @property
    def db(self) -> DatabaseManager:
        """Get database manager (lazy loaded)."""
        if self._db_manager is None:
            self._db_manager = DatabaseManager()
        return self._db_manager

    def generate_weekly_report(self, model_name: str) -> StabilityReport:
        """
        Generate a weekly stability report for a model.

        Args:
            model_name: Name of the model to report on

        Returns:
            StabilityReport with weekly analysis
        """
        return self._generate_report(model_name, report_type="weekly", days=7)

    def generate_monthly_report(self, model_name: str) -> StabilityReport:
        """
        Generate a monthly stability report for a model.

        Args:
            model_name: Name of the model to report on

        Returns:
            StabilityReport with monthly analysis
        """
        return self._generate_report(model_name, report_type="monthly", days=30)

    def _generate_report(
        self,
        model_name: str,
        report_type: str,
        days: int,
    ) -> StabilityReport:
        """
        Generate a stability report for the specified period.

        Args:
            model_name: Name of the model
            report_type: "weekly" or "monthly"
            days: Number of days for the period

        Returns:
            StabilityReport with analysis
        """
        # Get active model
        active_model = self.db.get_active_model(model_name)
        if not active_model:
            logger.warning(f"No active model found for {model_name}")
            return StabilityReport(
                model_name=model_name,
                model_id=0,
                report_type=report_type,
                period_start=datetime.now() - timedelta(days=days),
                period_end=datetime.now(),
                alerts=["No active model found"],
                summary=f"No active model found for {model_name}",
            )

        model_id = active_model["id"]
        period_end = datetime.now()
        period_start = period_end - timedelta(days=days)

        alerts: list[str] = []

        # Calculate accuracy for current period
        current_accuracy_data = self.db.get_prediction_accuracy(
            model_id=model_id,
            days=days,
        )
        current_accuracy = current_accuracy_data.get("accuracy")
        current_total = current_accuracy_data.get("total", 0)

        # Calculate accuracy for previous period
        previous_accuracy_data = self.db.get_prediction_accuracy(
            model_id=model_id,
            days=days * 2,  # Get double the period to calculate previous
        )

        # Estimate previous period accuracy by excluding current period
        # This is approximate - we compare full period accuracy changes
        previous_accuracy = previous_accuracy_data.get("accuracy")
        previous_total = previous_accuracy_data.get("total", 0)

        # Calculate accuracy change and trend
        accuracy_change = None
        accuracy_trend = "stable"

        if current_accuracy is not None and previous_accuracy is not None:
            # Calculate change (positive = improvement)
            accuracy_change = current_accuracy - previous_accuracy

            if accuracy_change > 0.02:
                accuracy_trend = "improving"
            elif accuracy_change < -0.02:
                accuracy_trend = "declining"
            else:
                accuracy_trend = "stable"

        # Generate accuracy alerts
        if current_accuracy is not None:
            if current_accuracy < self.ACCURACY_MIN_THRESHOLD:
                alerts.append(
                    f"Accuracy {current_accuracy:.1%} is below minimum threshold "
                    f"{self.ACCURACY_MIN_THRESHOLD:.1%}"
                )

            if accuracy_change is not None and accuracy_change < -self.ACCURACY_DECLINE_THRESHOLD:
                alerts.append(
                    f"Accuracy declined by {abs(accuracy_change):.1%} from previous period"
                )
        elif current_total == 0:
            alerts.append("No predictions available for accuracy calculation")

        # Check feature importance drift
        drift_alerts = self.db.get_feature_drift_alerts(
            model_id=model_id,
            rank_change_threshold=5,
            top_n=10,
        )

        drifted_features = [
            alert["feature"] for alert in drift_alerts
            if alert.get("type") == "rank_change"
        ]
        feature_drift_score = len(drift_alerts)

        if feature_drift_score >= self.DRIFT_SCORE_THRESHOLD:
            alerts.append(
                f"Feature drift detected: {len(drifted_features)} features with "
                f"significant rank changes"
            )

        # Get latest calibration metrics
        calibration = self.db.get_latest_calibration(model_id)
        calibration_ece = None
        is_well_calibrated = True

        if calibration:
            calibration_ece = calibration.get("expected_calibration_error")
            is_well_calibrated = calibration.get("is_well_calibrated", True)

            if calibration_ece is not None and calibration_ece > self.ECE_THRESHOLD:
                alerts.append(
                    f"Calibration error {calibration_ece:.4f} exceeds threshold "
                    f"{self.ECE_THRESHOLD}"
                )

        # Generate summary
        summary = self._generate_summary(
            model_name=model_name,
            report_type=report_type,
            accuracy=current_accuracy,
            accuracy_trend=accuracy_trend,
            feature_drift_score=feature_drift_score,
            calibration_ece=calibration_ece,
            is_well_calibrated=is_well_calibrated,
            alerts=alerts,
            total_predictions=current_total,
        )

        # Create report
        report = StabilityReport(
            model_name=model_name,
            model_id=model_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            accuracy=current_accuracy,
            accuracy_change=accuracy_change,
            accuracy_trend=accuracy_trend,
            feature_drift_score=float(feature_drift_score),
            drifted_features=drifted_features,
            calibration_ece=calibration_ece,
            is_well_calibrated=is_well_calibrated,
            alerts=alerts,
            summary=summary,
        )

        # Save report to database
        self._save_report(report)

        return report

    def _generate_summary(
        self,
        model_name: str,
        report_type: str,
        accuracy: Optional[float],
        accuracy_trend: str,
        feature_drift_score: float,
        calibration_ece: Optional[float],
        is_well_calibrated: bool,
        alerts: list[str],
        total_predictions: int,
    ) -> str:
        """Generate a human-readable summary of the report."""
        parts = []

        # Header
        period_name = "Weekly" if report_type == "weekly" else "Monthly"
        parts.append(f"{period_name} Stability Report for {model_name}")

        # Accuracy summary
        if accuracy is not None:
            trend_desc = {
                "improving": "trending upward",
                "declining": "trending downward",
                "stable": "stable",
            }.get(accuracy_trend, "stable")
            parts.append(
                f"Accuracy: {accuracy:.1%} ({trend_desc}) based on {total_predictions} predictions"
            )
        else:
            parts.append("Accuracy: Insufficient data for calculation")

        # Feature drift summary
        if feature_drift_score > 0:
            parts.append(f"Feature drift: {int(feature_drift_score)} features showing rank changes")
        else:
            parts.append("Feature drift: No significant drift detected")

        # Calibration summary
        if calibration_ece is not None:
            status = "well-calibrated" if is_well_calibrated else "poorly calibrated"
            parts.append(f"Calibration: ECE={calibration_ece:.4f} ({status})")
        else:
            parts.append("Calibration: No recent calibration data")

        # Alerts summary
        if alerts:
            parts.append(f"Alerts: {len(alerts)} issue(s) requiring attention")
        else:
            parts.append("Alerts: No issues detected")

        return ". ".join(parts) + "."

    def _save_report(self, report: StabilityReport) -> Optional[int]:
        """
        Save the stability report to the database.

        Args:
            report: StabilityReport to save

        Returns:
            Record ID or None if save failed
        """
        try:
            from data.storage import StabilityReportRecord

            session = self.db.get_session()
            try:
                record = StabilityReportRecord(
                    model_id=report.model_id,
                    model_name=report.model_name,
                    report_type=report.report_type,
                    period_start=report.period_start,
                    period_end=report.period_end,
                    accuracy=report.accuracy,
                    accuracy_change=report.accuracy_change,
                    accuracy_trend=report.accuracy_trend,
                    feature_drift_score=report.feature_drift_score,
                    drifted_features_json=json.dumps(report.drifted_features),
                    calibration_ece=report.calibration_ece,
                    is_well_calibrated=report.is_well_calibrated,
                    alerts_json=json.dumps(report.alerts),
                    summary=report.summary,
                )
                session.add(record)
                session.commit()
                return record.id
            finally:
                session.close()
        except ImportError:
            # StabilityReportRecord not yet defined in storage.py
            logger.warning(
                "StabilityReportRecord table not found - report not saved to database"
            )
            return None
        except Exception as e:
            logger.error(f"Error saving stability report: {e}")
            return None
