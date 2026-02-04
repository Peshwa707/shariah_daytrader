"""
ML Insights Monitor - Performance monitoring and feedback loops.

Monitors model performance, detects drift, and triggers alerts/retraining.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for ML monitoring thresholds."""

    # Performance thresholds
    accuracy_threshold: float = 0.55  # Minimum acceptable accuracy
    accuracy_drop_alert: float = 0.05  # Alert if accuracy drops by this much
    min_predictions_for_eval: int = 20  # Minimum predictions to evaluate

    # Calibration thresholds
    calibration_error_threshold: float = 0.10  # Max acceptable ECE
    calibration_check_days: int = 7  # Days between calibration checks

    # Feature drift thresholds
    drift_rank_change_alert: int = 5  # Alert if top feature rank changes by this
    drift_top_n_features: int = 10  # Number of top features to track

    # Retraining triggers
    retrain_cooldown_days: int = 7  # Minimum days between retrains
    max_days_since_train: int = 14  # Maximum days before scheduled retrain
    max_consecutive_poor_days: int = 3  # Days of poor performance before retrain

    # Regime detection
    regime_change_alert_days: int = 3  # Alert if new regime persists this long

    # Alert settings
    max_open_alerts: int = 50  # Maximum open alerts before cleanup


@dataclass
class MonitorResult:
    """Result from a monitoring check."""
    check_name: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, critical
    details: dict = field(default_factory=dict)
    requires_action: bool = False
    suggested_action: str | None = None


class MLInsightsMonitor:
    """
    Monitors ML model performance and triggers alerts/actions.

    Provides automated feedback loops for:
    - Performance degradation detection
    - Calibration drift monitoring
    - Feature importance drift
    - Market regime change detection
    - Retraining triggers
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        db_manager=None,
    ):
        """
        Initialize the ML Insights Monitor.

        Args:
            config: Monitoring configuration
            db_manager: Database manager instance (lazy loaded if not provided)
        """
        self.config = config or MonitorConfig()
        self._db_manager = db_manager
        self._last_checks: dict[str, datetime] = {}

    @property
    def db(self):
        """Get database manager (lazy loaded)."""
        if self._db_manager is None:
            from data.storage import DatabaseManager
            self._db_manager = DatabaseManager()
        return self._db_manager

    # =========================================================================
    # PERFORMANCE MONITORING
    # =========================================================================

    def check_performance_degradation(
        self,
        model_name: str = "momentum_continuation",
    ) -> MonitorResult:
        """
        Check if model performance has degraded below threshold.

        Daily check comparing recent accuracy to training accuracy.

        Args:
            model_name: Name of the model to check

        Returns:
            MonitorResult with check status
        """
        try:
            # Get active model info
            active_model = self.db.get_active_model(model_name)
            if not active_model:
                return MonitorResult(
                    check_name="performance_degradation",
                    passed=True,
                    message=f"No active model found for {model_name}",
                    severity="info",
                )

            model_id = active_model["id"]
            training_metrics = active_model.get("metrics", {})
            training_accuracy = training_metrics.get("direction_accuracy")

            # Get recent prediction accuracy
            accuracy_data = self.db.get_prediction_accuracy(
                model_id=model_id,
                days=7,
                prediction_type="direction",
            )

            total = accuracy_data.get("total", 0)
            if total < self.config.min_predictions_for_eval:
                return MonitorResult(
                    check_name="performance_degradation",
                    passed=True,
                    message=f"Insufficient predictions ({total}) for evaluation",
                    severity="info",
                    details={"predictions_count": total},
                )

            recent_accuracy = accuracy_data.get("accuracy")
            if recent_accuracy is None:
                return MonitorResult(
                    check_name="performance_degradation",
                    passed=True,
                    message="No accuracy data available",
                    severity="info",
                )

            # Check absolute threshold
            if recent_accuracy < self.config.accuracy_threshold:
                return MonitorResult(
                    check_name="performance_degradation",
                    passed=False,
                    message=f"Accuracy {recent_accuracy:.2%} below threshold {self.config.accuracy_threshold:.2%}",
                    severity="warning",
                    details={
                        "recent_accuracy": recent_accuracy,
                        "threshold": self.config.accuracy_threshold,
                        "training_accuracy": training_accuracy,
                        "predictions_count": total,
                    },
                    requires_action=True,
                    suggested_action="Consider retraining the model",
                )

            # Check relative drop from training
            if training_accuracy and (training_accuracy - recent_accuracy) > self.config.accuracy_drop_alert:
                return MonitorResult(
                    check_name="performance_degradation",
                    passed=False,
                    message=f"Accuracy dropped {(training_accuracy - recent_accuracy):.2%} from training",
                    severity="warning",
                    details={
                        "recent_accuracy": recent_accuracy,
                        "training_accuracy": training_accuracy,
                        "drop": training_accuracy - recent_accuracy,
                        "predictions_count": total,
                    },
                    requires_action=True,
                    suggested_action="Investigate recent predictions and consider retraining",
                )

            return MonitorResult(
                check_name="performance_degradation",
                passed=True,
                message=f"Accuracy {recent_accuracy:.2%} is healthy",
                severity="info",
                details={
                    "recent_accuracy": recent_accuracy,
                    "training_accuracy": training_accuracy,
                    "predictions_count": total,
                },
            )

        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
            return MonitorResult(
                check_name="performance_degradation",
                passed=False,
                message=f"Error during check: {e}",
                severity="critical",
            )

    # =========================================================================
    # CALIBRATION MONITORING
    # =========================================================================

    def check_calibration_drift(
        self,
        model_name: str = "momentum_continuation",
    ) -> MonitorResult:
        """
        Check if model calibration has drifted (ECE exceeded threshold).

        Weekly check of Expected Calibration Error.

        Args:
            model_name: Name of the model to check

        Returns:
            MonitorResult with check status
        """
        try:
            active_model = self.db.get_active_model(model_name)
            if not active_model:
                return MonitorResult(
                    check_name="calibration_drift",
                    passed=True,
                    message=f"No active model found for {model_name}",
                    severity="info",
                )

            model_id = active_model["id"]

            # Get latest calibration report
            calibration = self.db.get_latest_calibration(model_id)
            if not calibration:
                return MonitorResult(
                    check_name="calibration_drift",
                    passed=True,
                    message="No calibration data available - run calibration check",
                    severity="info",
                    requires_action=True,
                    suggested_action="Run calibration report for model",
                )

            ece = calibration.get("expected_calibration_error")
            if ece is None:
                return MonitorResult(
                    check_name="calibration_drift",
                    passed=True,
                    message="No ECE data in calibration report",
                    severity="info",
                )

            is_well_calibrated = calibration.get("is_well_calibrated", True)

            if not is_well_calibrated or ece > self.config.calibration_error_threshold:
                return MonitorResult(
                    check_name="calibration_drift",
                    passed=False,
                    message=f"ECE {ece:.4f} exceeds threshold {self.config.calibration_error_threshold}",
                    severity="warning",
                    details={
                        "ece": ece,
                        "threshold": self.config.calibration_error_threshold,
                        "brier_score": calibration.get("brier_score"),
                        "sample_count": calibration.get("sample_count"),
                    },
                    requires_action=True,
                    suggested_action="Recalibrate model probabilities or retrain",
                )

            return MonitorResult(
                check_name="calibration_drift",
                passed=True,
                message=f"ECE {ece:.4f} is within acceptable range",
                severity="info",
                details={
                    "ece": ece,
                    "threshold": self.config.calibration_error_threshold,
                    "brier_score": calibration.get("brier_score"),
                },
            )

        except Exception as e:
            logger.error(f"Error checking calibration drift: {e}")
            return MonitorResult(
                check_name="calibration_drift",
                passed=False,
                message=f"Error during check: {e}",
                severity="critical",
            )

    def compute_calibration_metrics(
        self,
        model_name: str = "momentum_continuation",
        days: int = 7,
        n_bins: int = 10,
    ) -> dict[str, Any] | None:
        """
        Compute calibration metrics from recent predictions.

        Args:
            model_name: Name of the model
            days: Number of days to look back
            n_bins: Number of bins for calibration curve

        Returns:
            Dict with calibration metrics or None
        """
        try:
            active_model = self.db.get_active_model(model_name)
            if not active_model:
                return None

            model_id = active_model["id"]

            # Get predictions with outcomes
            session = self.db.get_session()
            try:
                from data.storage import PredictionOutcome

                cutoff = datetime.now() - timedelta(days=days)
                predictions = (
                    session.query(PredictionOutcome)
                    .filter(
                        PredictionOutcome.model_id == model_id,
                        PredictionOutcome.prediction_time >= cutoff,
                        PredictionOutcome.is_correct != None,  # noqa: E711
                    )
                    .all()
                )

                if len(predictions) < 10:
                    return None

                # Extract probabilities and outcomes
                probs = [p.predicted_probability for p in predictions if p.predicted_probability is not None]
                outcomes = [1.0 if p.is_correct else 0.0 for p in predictions if p.predicted_probability is not None]

                if not probs:
                    return None

                probs = np.array(probs)
                outcomes = np.array(outcomes)

                # Compute ECE (Expected Calibration Error)
                bin_edges = np.linspace(0, 1, n_bins + 1)
                ece = 0.0
                reliability_diagram = []

                for i in range(n_bins):
                    mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
                    if mask.sum() > 0:
                        bin_acc = outcomes[mask].mean()
                        bin_conf = probs[mask].mean()
                        bin_count = mask.sum()
                        ece += (bin_count / len(probs)) * abs(bin_acc - bin_conf)
                        reliability_diagram.append({
                            "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                            "accuracy": float(bin_acc),
                            "confidence": float(bin_conf),
                            "count": int(bin_count),
                        })

                # Compute Brier score
                brier_score = ((probs - outcomes) ** 2).mean()

                # Compute log loss (avoiding log(0))
                eps = 1e-15
                probs_clipped = np.clip(probs, eps, 1 - eps)
                log_loss = -(outcomes * np.log(probs_clipped) + (1 - outcomes) * np.log(1 - probs_clipped)).mean()

                # Compute MCE (Maximum Calibration Error)
                mce = 0.0
                for item in reliability_diagram:
                    mce = max(mce, abs(item["accuracy"] - item["confidence"]))

                return {
                    "ece": float(ece),
                    "mce": float(mce),
                    "brier_score": float(brier_score),
                    "log_loss": float(log_loss),
                    "sample_count": len(predictions),
                    "reliability_diagram": reliability_diagram,
                }

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error computing calibration metrics: {e}")
            return None

    # =========================================================================
    # FEATURE DRIFT MONITORING
    # =========================================================================

    def check_feature_drift(
        self,
        model_name: str = "momentum_continuation",
    ) -> MonitorResult:
        """
        Check for feature importance drift between training sessions.

        Compares current feature rankings to historical rankings.

        Args:
            model_name: Name of the model to check

        Returns:
            MonitorResult with check status
        """
        try:
            active_model = self.db.get_active_model(model_name)
            if not active_model:
                return MonitorResult(
                    check_name="feature_drift",
                    passed=True,
                    message=f"No active model found for {model_name}",
                    severity="info",
                )

            model_id = active_model["id"]

            # Get drift alerts
            alerts = self.db.get_feature_drift_alerts(
                model_id=model_id,
                rank_change_threshold=self.config.drift_rank_change_alert,
                top_n=self.config.drift_top_n_features,
            )

            if not alerts:
                return MonitorResult(
                    check_name="feature_drift",
                    passed=True,
                    message="No significant feature drift detected",
                    severity="info",
                )

            # Count significant changes
            rank_changes = [a for a in alerts if a["type"] == "rank_change"]
            new_features = [a for a in alerts if a["type"] == "new_feature"]

            if rank_changes:
                # Get worst change
                worst = max(rank_changes, key=lambda x: abs(x["change"] or 0))
                return MonitorResult(
                    check_name="feature_drift",
                    passed=False,
                    message=f"Feature '{worst['feature']}' rank changed by {worst['change']} positions",
                    severity="warning",
                    details={
                        "rank_changes": rank_changes,
                        "new_features": new_features,
                        "total_alerts": len(alerts),
                    },
                    requires_action=True,
                    suggested_action="Investigate feature drift and consider retraining",
                )

            return MonitorResult(
                check_name="feature_drift",
                passed=True,
                message=f"Minor feature changes detected ({len(alerts)} alerts)",
                severity="info",
                details={"alerts": alerts},
            )

        except Exception as e:
            logger.error(f"Error checking feature drift: {e}")
            return MonitorResult(
                check_name="feature_drift",
                passed=False,
                message=f"Error during check: {e}",
                severity="critical",
            )

    # =========================================================================
    # REGIME MONITORING
    # =========================================================================

    def check_regime_change(self) -> MonitorResult:
        """
        Check for market regime changes and model adaptation needs.

        Detects when market regime has changed significantly.

        Returns:
            MonitorResult with check status
        """
        try:
            # Get recent regime history
            regimes = self.db.get_regime_history(days=self.config.regime_change_alert_days + 7)

            if not regimes:
                return MonitorResult(
                    check_name="regime_change",
                    passed=True,
                    message="No regime data available",
                    severity="info",
                )

            # Check for regime persistence
            recent_regimes = regimes[:self.config.regime_change_alert_days]
            if not recent_regimes:
                return MonitorResult(
                    check_name="regime_change",
                    passed=True,
                    message="Insufficient recent regime data",
                    severity="info",
                )

            current_regime = recent_regimes[0]["regime_type"]
            regime_types = [r["regime_type"] for r in recent_regimes]

            # Check if same regime persisted
            same_regime_count = sum(1 for r in regime_types if r == current_regime)

            # Check if this is different from historical average
            older_regimes = regimes[self.config.regime_change_alert_days:]
            if older_regimes:
                historical_regime_counts = {}
                for r in older_regimes:
                    rt = r["regime_type"]
                    historical_regime_counts[rt] = historical_regime_counts.get(rt, 0) + 1

                most_common_historical = max(
                    historical_regime_counts.items(),
                    key=lambda x: x[1],
                )[0] if historical_regime_counts else None

                # Regime shift detection
                if most_common_historical and current_regime != most_common_historical:
                    if same_regime_count >= self.config.regime_change_alert_days:
                        return MonitorResult(
                            check_name="regime_change",
                            passed=False,
                            message=f"Regime shifted from {most_common_historical} to {current_regime}",
                            severity="warning",
                            details={
                                "current_regime": current_regime,
                                "previous_regime": most_common_historical,
                                "days_in_current": same_regime_count,
                            },
                            requires_action=True,
                            suggested_action="Evaluate model performance in new regime",
                        )

            return MonitorResult(
                check_name="regime_change",
                passed=True,
                message=f"Current regime: {current_regime}",
                severity="info",
                details={
                    "current_regime": current_regime,
                    "days_in_regime": same_regime_count,
                },
            )

        except Exception as e:
            logger.error(f"Error checking regime change: {e}")
            return MonitorResult(
                check_name="regime_change",
                passed=False,
                message=f"Error during check: {e}",
                severity="critical",
            )

    # =========================================================================
    # RETRAINING TRIGGERS
    # =========================================================================

    def check_retrain_needed(
        self,
        model_name: str = "momentum_continuation",
    ) -> MonitorResult:
        """
        Check if model retraining is needed.

        Triggers based on:
        - Time since last training
        - Performance degradation
        - Calibration drift
        - Feature drift

        Args:
            model_name: Name of the model to check

        Returns:
            MonitorResult with retrain recommendation
        """
        try:
            active_model = self.db.get_active_model(model_name)
            if not active_model:
                return MonitorResult(
                    check_name="retrain_needed",
                    passed=True,
                    message=f"No active model found for {model_name}",
                    severity="info",
                )

            reasons = []
            details = {}

            # Check time since training
            created_at = active_model.get("created_at")
            if created_at:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                days_since = (datetime.now() - created_at).days
                details["days_since_training"] = days_since

                if days_since >= self.config.max_days_since_train:
                    reasons.append(f"Scheduled: {days_since} days since training")

            # Check performance
            perf_result = self.check_performance_degradation(model_name)
            if not perf_result.passed:
                reasons.append(f"Performance: {perf_result.message}")
                details["performance"] = perf_result.details

            # Check calibration
            cal_result = self.check_calibration_drift(model_name)
            if not cal_result.passed:
                reasons.append(f"Calibration: {cal_result.message}")
                details["calibration"] = cal_result.details

            # Check feature drift
            drift_result = self.check_feature_drift(model_name)
            if not drift_result.passed:
                reasons.append(f"Drift: {drift_result.message}")
                details["drift"] = drift_result.details

            if reasons:
                return MonitorResult(
                    check_name="retrain_needed",
                    passed=False,
                    message=f"Retraining recommended: {len(reasons)} trigger(s)",
                    severity="warning",
                    details={
                        "reasons": reasons,
                        **details,
                    },
                    requires_action=True,
                    suggested_action="Initiate model retraining",
                )

            return MonitorResult(
                check_name="retrain_needed",
                passed=True,
                message="No retraining needed",
                severity="info",
                details=details,
            )

        except Exception as e:
            logger.error(f"Error checking retrain needed: {e}")
            return MonitorResult(
                check_name="retrain_needed",
                passed=False,
                message=f"Error during check: {e}",
                severity="critical",
            )

    # =========================================================================
    # ALERT MANAGEMENT
    # =========================================================================

    def trigger_alert(
        self,
        result: MonitorResult,
        model_name: str = "momentum_continuation",
    ) -> int | None:
        """
        Log a monitoring result as a learning event.

        Args:
            result: MonitorResult from a check
            model_name: Model name for context

        Returns:
            Learning event ID or None
        """
        if result.passed and result.severity == "info":
            # Don't log passing info checks
            return None

        try:
            active_model = self.db.get_active_model(model_name)
            model_id = active_model.get("id") if active_model else None

            # Map check name to category
            category_map = {
                "performance_degradation": "performance",
                "calibration_drift": "calibration",
                "feature_drift": "drift",
                "regime_change": "regime",
                "retrain_needed": "system",
            }

            event_id = self.db.log_learning_event(
                event_type="alert" if not result.passed else "insight",
                severity=result.severity,
                category=category_map.get(result.check_name, "system"),
                title=result.check_name.replace("_", " ").title(),
                description=result.message,
                details=result.details,
                source="monitor",
                model_id=model_id,
                requires_action=result.requires_action,
            )

            logger.info(f"Alert logged: {result.check_name} - {result.message}")
            return event_id

        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
            return None

    # =========================================================================
    # COMPREHENSIVE MONITORING
    # =========================================================================

    def run_all_checks(
        self,
        model_name: str = "momentum_continuation",
        log_alerts: bool = True,
    ) -> list[MonitorResult]:
        """
        Run all monitoring checks.

        Args:
            model_name: Name of the model to check
            log_alerts: Whether to log failures as learning events

        Returns:
            List of MonitorResults
        """
        results = []

        # Run all checks
        checks = [
            ("performance", self.check_performance_degradation),
            ("calibration", self.check_calibration_drift),
            ("feature_drift", self.check_feature_drift),
            ("regime", self.check_regime_change),
            ("retrain", self.check_retrain_needed),
        ]

        for check_name, check_func in checks:
            try:
                if check_name == "regime":
                    result = check_func()
                else:
                    result = check_func(model_name)

                results.append(result)

                # Log alerts for failures
                if log_alerts and not result.passed:
                    self.trigger_alert(result, model_name)

            except Exception as e:
                logger.error(f"Error running check {check_name}: {e}")
                results.append(MonitorResult(
                    check_name=check_name,
                    passed=False,
                    message=f"Check failed: {e}",
                    severity="critical",
                ))

        return results

    def get_health_summary(
        self,
        model_name: str = "momentum_continuation",
    ) -> dict[str, Any]:
        """
        Get comprehensive health summary for a model.

        Args:
            model_name: Name of the model

        Returns:
            Dict with health summary
        """
        results = self.run_all_checks(model_name, log_alerts=False)

        # Compute overall health score
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        health_score = (passed_count / total_count * 100) if total_count > 0 else 0

        # Determine overall status
        critical_count = sum(1 for r in results if r.severity == "critical")
        warning_count = sum(1 for r in results if not r.passed and r.severity == "warning")

        if critical_count > 0:
            overall_status = "critical"
        elif warning_count >= 2:
            overall_status = "warning"
        elif warning_count == 1:
            overall_status = "attention"
        else:
            overall_status = "healthy"

        return {
            "model_name": model_name,
            "overall_status": overall_status,
            "health_score": health_score,
            "checks_passed": passed_count,
            "checks_total": total_count,
            "critical_issues": critical_count,
            "warnings": warning_count,
            "checks": [
                {
                    "name": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "message": r.message,
                }
                for r in results
            ],
            "timestamp": datetime.now().isoformat(),
        }
