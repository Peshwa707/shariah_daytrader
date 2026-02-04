"""
Database Storage Module.

SQLite-based storage for:
- Historical price data
- Compliance screening results
- Trade records
- Purification tracking

Uses SQLAlchemy for ORM and easy migration to PostgreSQL if needed.
"""
from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Any
import json

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Date,
    Text,
    Index,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from config.settings import settings


Base = declarative_base()


class StockData(Base):
    """Historical OHLCV data for stocks."""

    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer)
    vwap = Column(Float)
    bar_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_stock_data_symbol_timestamp", "symbol", "timestamp", unique=True),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
        }


class ComplianceRecord(Base):
    """Shariah compliance screening results."""

    __tablename__ = "compliance_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    screening_date = Column(DateTime, nullable=False, default=datetime.now)

    # Overall result
    is_compliant = Column(Boolean, nullable=False)
    status = Column(String(50), nullable=False)
    source = Column(String(50))  # index, custom_screening, etc.
    confidence = Column(String(20))

    # Index listing
    index_listed = Column(Boolean, default=False)
    index_name = Column(String(100))

    # Business screening
    prohibited_industries = Column(Text)  # JSON list
    prohibited_revenue_ratio = Column(Float)

    # Financial screening
    debt_to_market_cap = Column(Float)
    deposits_to_market_cap = Column(Float)
    impermissible_income_ratio = Column(Float)
    receivables_to_market_cap = Column(Float)

    # Details
    reasons = Column(Text)  # JSON list
    needs_review = Column(Boolean, default=False)

    # Metadata
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_compliance_symbol_date", "symbol", "screening_date"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "screening_date": self.screening_date.isoformat() if self.screening_date else None,
            "is_compliant": self.is_compliant,
            "status": self.status,
            "source": self.source,
            "confidence": self.confidence,
            "index_listed": self.index_listed,
            "index_name": self.index_name,
            "prohibited_industries": json.loads(self.prohibited_industries) if self.prohibited_industries else [],
            "debt_to_market_cap": self.debt_to_market_cap,
            "impermissible_income_ratio": self.impermissible_income_ratio,
            "reasons": json.loads(self.reasons) if self.reasons else [],
            "needs_review": self.needs_review,
        }


class TradeRecord(Base):
    """Trade execution records."""

    __tablename__ = "trade_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    trade_date = Column(DateTime, nullable=False)

    # Order details
    order_id = Column(String(50))
    order_type = Column(String(20))  # market, limit, stop
    side = Column(String(10), nullable=False)  # buy, sell

    # Execution
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0)
    total_value = Column(Float)

    # Position tracking
    position_after = Column(Float)
    avg_cost_after = Column(Float)

    # P&L (for closing trades)
    realized_pnl = Column(Float)

    # Compliance
    was_compliant_at_trade = Column(Boolean, default=True)
    compliance_record_id = Column(Integer, ForeignKey("compliance_records.id"))

    # Metadata
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    # Trade timing
    entry_time = Column(DateTime)  # When position was opened
    exit_time = Column(DateTime)  # When position was closed (nullable)
    holding_period_minutes = Column(Integer)  # Duration of trade (nullable)
    exit_reason = Column(String(50))  # take_profit, stop_loss, manual, auto_exit (nullable)

    __table_args__ = (
        Index("ix_trade_symbol_date", "symbol", "trade_date"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "trade_date": self.trade_date.isoformat() if self.trade_date else None,
            "order_id": self.order_id,
            "order_type": self.order_type,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "total_value": self.total_value,
            "realized_pnl": self.realized_pnl,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "holding_period_minutes": self.holding_period_minutes,
            "exit_reason": self.exit_reason,
        }


class PurificationRecord(Base):
    """Purification tracking records."""

    __tablename__ = "purification_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    record_date = Column(Date, nullable=False)
    record_type = Column(String(20), nullable=False)  # dividend, capital_gain

    # Amounts
    gross_amount = Column(Float, nullable=False)
    purification_ratio = Column(Float, nullable=False)
    purification_amount = Column(Float, nullable=False)

    # Context
    shares_held = Column(Float)
    cost_basis = Column(Float)
    sale_price = Column(Float)

    # Donation tracking
    is_donated = Column(Boolean, default=False)
    donation_date = Column(Date)
    donation_reference = Column(String(100))

    # Metadata
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_purification_symbol_date", "symbol", "record_date"),
    )


class SignalRecord(Base):
    """ML signal/prediction records."""

    __tablename__ = "signal_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    signal_date = Column(DateTime, nullable=False)

    # Signal details
    model_name = Column(String(50))
    signal_type = Column(String(20))  # buy, sell, hold
    confidence = Column(Float)
    probability = Column(Float)

    # Features used (for debugging/analysis)
    features_json = Column(Text)

    # Outcome tracking
    actual_return = Column(Float)  # Filled in later
    was_correct = Column(Boolean)

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_signal_symbol_date", "symbol", "signal_date"),
    )


# =============================================================================
# ML INSIGHTS TABLES
# =============================================================================


class ModelRegistry(Base):
    """Track model versions with lineage, hyperparameters, and metrics."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # momentum_continuation, lightgbm, etc.

    # Status: training, validated, active, deprecated
    status = Column(String(20), default="training")
    is_active = Column(Boolean, default=False, index=True)

    # Configuration
    hyperparameters_json = Column(Text)  # JSON serialized hyperparameters
    feature_names_json = Column(Text)  # JSON list of feature names

    # Metrics
    metrics_json = Column(Text)  # JSON serialized training metrics

    # Storage
    model_path = Column(String(500))  # Path to saved model file

    # Lineage
    parent_model_id = Column(Integer, ForeignKey("model_registry.id"))

    # Training data range
    training_data_start = Column(DateTime)
    training_data_end = Column(DateTime)
    training_samples = Column(Integer)

    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    validated_at = Column(DateTime)
    activated_at = Column(DateTime)
    deprecated_at = Column(DateTime)

    __table_args__ = (
        Index("ix_model_name_version", "model_name", "model_version", unique=True),
        Index("ix_model_active", "model_name", "is_active"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "status": self.status,
            "is_active": self.is_active,
            "hyperparameters": json.loads(self.hyperparameters_json) if self.hyperparameters_json else {},
            "feature_names": json.loads(self.feature_names_json) if self.feature_names_json else [],
            "metrics": json.loads(self.metrics_json) if self.metrics_json else {},
            "model_path": self.model_path,
            "parent_model_id": self.parent_model_id,
            "training_data_start": self.training_data_start.isoformat() if self.training_data_start else None,
            "training_data_end": self.training_data_end.isoformat() if self.training_data_end else None,
            "training_samples": self.training_samples,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
        }


class PredictionOutcome(Base):
    """Link predictions to actual results for accuracy tracking."""

    __tablename__ = "prediction_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("model_registry.id"), nullable=False, index=True)
    signal_id = Column(Integer, ForeignKey("signal_records.id"), index=True)

    # Prediction details
    symbol = Column(String(10), nullable=False, index=True)
    prediction_time = Column(DateTime, nullable=False)
    prediction_type = Column(String(20), nullable=False)  # direction, magnitude, duration
    predicted_value = Column(Float, nullable=False)
    predicted_probability = Column(Float)
    predicted_class = Column(String(50))  # For classification: continuation, reversal, neutral

    # Actual outcome
    outcome_time = Column(DateTime)
    actual_value = Column(Float)
    actual_class = Column(String(50))

    # Evaluation
    is_correct = Column(Boolean)
    error = Column(Float)  # For regression: actual - predicted
    error_pct = Column(Float)  # Percentage error

    # Trade linkage
    trade_id = Column(Integer, ForeignKey("trade_records.id"))
    realized_pnl = Column(Float)

    created_at = Column(DateTime, default=datetime.now)
    evaluated_at = Column(DateTime)

    __table_args__ = (
        Index("ix_prediction_model_time", "model_id", "prediction_time"),
        Index("ix_prediction_symbol_time", "symbol", "prediction_time"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "prediction_time": self.prediction_time.isoformat() if self.prediction_time else None,
            "prediction_type": self.prediction_type,
            "predicted_value": self.predicted_value,
            "predicted_probability": self.predicted_probability,
            "predicted_class": self.predicted_class,
            "outcome_time": self.outcome_time.isoformat() if self.outcome_time else None,
            "actual_value": self.actual_value,
            "actual_class": self.actual_class,
            "is_correct": self.is_correct,
            "error": self.error,
            "error_pct": self.error_pct,
            "realized_pnl": self.realized_pnl,
        }


class FeatureImportance(Base):
    """Track feature importance scores over time for drift detection."""

    __tablename__ = "feature_importance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("model_registry.id"), nullable=False, index=True)

    # Feature info
    feature_name = Column(String(100), nullable=False, index=True)
    importance_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)  # 1 = most important

    # Context
    importance_type = Column(String(50))  # gain, split, permutation
    model_component = Column(String(50))  # direction, magnitude, duration (for multi-model)

    recorded_at = Column(DateTime, default=datetime.now, nullable=False)

    __table_args__ = (
        Index("ix_feature_model_recorded", "model_id", "recorded_at"),
        Index("ix_feature_name_recorded", "feature_name", "recorded_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "feature_name": self.feature_name,
            "importance_score": self.importance_score,
            "rank": self.rank,
            "importance_type": self.importance_type,
            "model_component": self.model_component,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
        }


class ConfidenceCalibration(Base):
    """Weekly calibration metrics (ECE, Brier score, reliability)."""

    __tablename__ = "confidence_calibration"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("model_registry.id"), nullable=False, index=True)

    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    sample_count = Column(Integer, nullable=False)

    # Calibration metrics
    expected_calibration_error = Column(Float)  # ECE - lower is better
    maximum_calibration_error = Column(Float)  # MCE
    brier_score = Column(Float)  # Brier score - lower is better
    log_loss = Column(Float)

    # Reliability diagram data (JSON: list of {bin_center, accuracy, confidence, count})
    reliability_diagram_json = Column(Text)

    # By-class metrics (JSON: {class_name: {accuracy, count, avg_confidence}})
    class_metrics_json = Column(Text)

    # Thresholds and alerts
    is_well_calibrated = Column(Boolean)  # ECE < threshold
    calibration_alert = Column(String(200))  # Alert message if poorly calibrated

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_calibration_model_period", "model_id", "period_start"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "sample_count": self.sample_count,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "reliability_diagram": json.loads(self.reliability_diagram_json) if self.reliability_diagram_json else [],
            "class_metrics": json.loads(self.class_metrics_json) if self.class_metrics_json else {},
            "is_well_calibrated": self.is_well_calibrated,
            "calibration_alert": self.calibration_alert,
        }


class MarketRegime(Base):
    """Regime detection results (trending, volatile, quiet, etc.)."""

    __tablename__ = "market_regimes"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Time period
    regime_date = Column(Date, nullable=False, index=True)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)

    # Regime classification
    regime_type = Column(String(50), nullable=False)  # trending_up, trending_down, volatile, quiet, mixed
    regime_confidence = Column(Float)  # Confidence in classification

    # Market metrics used for classification
    vix_level = Column(Float)
    vix_change_pct = Column(Float)
    sp500_return_pct = Column(Float)
    sp500_volatility = Column(Float)
    breadth_advance_decline = Column(Float)
    sector_correlation = Column(Float)

    # Additional metrics (JSON for extensibility)
    metrics_json = Column(Text)

    # Model performance in this regime (filled in later)
    model_accuracy_in_regime = Column(Float)
    model_pnl_in_regime = Column(Float)

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_regime_date", "regime_date", unique=True),
        Index("ix_regime_type_date", "regime_type", "regime_date"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "regime_date": self.regime_date.isoformat() if self.regime_date else None,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "regime_type": self.regime_type,
            "regime_confidence": self.regime_confidence,
            "vix_level": self.vix_level,
            "sp500_return_pct": self.sp500_return_pct,
            "sp500_volatility": self.sp500_volatility,
            "metrics": json.loads(self.metrics_json) if self.metrics_json else {},
            "model_accuracy_in_regime": self.model_accuracy_in_regime,
            "model_pnl_in_regime": self.model_pnl_in_regime,
        }


class StabilityReportRecord(Base):
    """Stability reports for ML models (weekly/monthly)."""

    __tablename__ = "stability_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("model_registry.id"), nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    report_type = Column(String(20), nullable=False)  # weekly, monthly

    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)

    # Accuracy metrics
    accuracy = Column(Float)
    accuracy_change = Column(Float)
    accuracy_trend = Column(String(20))  # improving, declining, stable

    # Feature drift metrics
    feature_drift_score = Column(Float, default=0.0)
    drifted_features_json = Column(Text)  # JSON list of feature names

    # Calibration metrics
    calibration_ece = Column(Float)
    is_well_calibrated = Column(Boolean, default=True)

    # Alerts and summary
    alerts_json = Column(Text)  # JSON list of alert strings
    summary = Column(Text)

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_stability_model_period", "model_id", "period_end"),
        Index("ix_stability_type_date", "report_type", "period_end"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "report_type": self.report_type,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "accuracy": self.accuracy,
            "accuracy_change": self.accuracy_change,
            "accuracy_trend": self.accuracy_trend,
            "feature_drift_score": self.feature_drift_score,
            "drifted_features": json.loads(self.drifted_features_json) if self.drifted_features_json else [],
            "calibration_ece": self.calibration_ece,
            "is_well_calibrated": self.is_well_calibrated,
            "alerts": json.loads(self.alerts_json) if self.alerts_json else [],
            "summary": self.summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class LearningEvent(Base):
    """System learnings, alerts, and insights."""

    __tablename__ = "learning_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Event classification
    event_type = Column(String(50), nullable=False, index=True)  # alert, insight, adaptation, retrain_trigger
    severity = Column(String(20), nullable=False)  # info, warning, critical
    category = Column(String(50), nullable=False)  # performance, calibration, drift, regime, system

    # Event details
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    details_json = Column(Text)  # JSON with additional context

    # Source
    source = Column(String(100))  # monitor, manual, model, system
    model_id = Column(Integer, ForeignKey("model_registry.id"), index=True)

    # Status
    status = Column(String(20), default="open")  # open, acknowledged, resolved, dismissed
    requires_action = Column(Boolean, default=False)
    action_taken = Column(Text)

    # Timestamps
    event_time = Column(DateTime, nullable=False, default=datetime.now)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("ix_learning_type_time", "event_type", "event_time"),
        Index("ix_learning_status", "status", "event_time"),
        Index("ix_learning_severity", "severity", "event_time"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "details": json.loads(self.details_json) if self.details_json else {},
            "source": self.source,
            "model_id": self.model_id,
            "status": self.status,
            "requires_action": self.requires_action,
            "action_taken": self.action_taken,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class DatabaseManager:
    """
    Manages database connections and operations.

    Provides a clean interface for storing and retrieving data
    used by the trading system.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the database manager.

        Args:
            db_path: Path to SQLite database (uses settings if not provided)
        """
        self.db_path = db_path or str(settings.db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # Stock Data Operations

    def save_stock_data(self, symbol: str, data: list[dict]) -> int:
        """
        Save historical stock data.

        Args:
            symbol: Stock ticker symbol
            data: List of OHLCV dictionaries

        Returns:
            Number of records saved
        """
        session = self.get_session()
        count = 0

        try:
            for row in data:
                # Check if record exists
                existing = session.query(StockData).filter(
                    StockData.symbol == symbol,
                    StockData.timestamp == row.get("timestamp"),
                ).first()

                if existing:
                    # Update existing
                    existing.open = row.get("open")
                    existing.high = row.get("high")
                    existing.low = row.get("low")
                    existing.close = row.get("close")
                    existing.volume = row.get("volume")
                    existing.vwap = row.get("vwap")
                else:
                    # Insert new
                    record = StockData(
                        symbol=symbol,
                        timestamp=row.get("timestamp"),
                        open=row.get("open"),
                        high=row.get("high"),
                        low=row.get("low"),
                        close=row.get("close"),
                        volume=row.get("volume"),
                        vwap=row.get("vwap"),
                        bar_count=row.get("bar_count"),
                    )
                    session.add(record)
                    count += 1

            session.commit()
        finally:
            session.close()

        return count

    def get_stock_data(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
        """
        Retrieve historical stock data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of OHLCV dictionaries
        """
        session = self.get_session()

        try:
            query = session.query(StockData).filter(StockData.symbol == symbol)

            if start_date:
                query = query.filter(StockData.timestamp >= start_date)
            if end_date:
                query = query.filter(StockData.timestamp <= end_date)

            query = query.order_by(StockData.timestamp)

            return [record.to_dict() for record in query.all()]
        finally:
            session.close()

    # Compliance Operations

    def save_compliance_result(self, result: dict) -> int:
        """
        Save a compliance screening result.

        Args:
            result: ComplianceResult as dictionary

        Returns:
            Record ID
        """
        session = self.get_session()

        try:
            record = ComplianceRecord(
                symbol=result.get("symbol"),
                screening_date=datetime.now(),
                is_compliant=result.get("is_compliant"),
                status=result.get("status"),
                source=result.get("source"),
                confidence=result.get("confidence"),
                index_listed=result.get("index_listed"),
                index_name=result.get("index_name"),
                prohibited_industries=json.dumps(result.get("prohibited_industries", [])),
                prohibited_revenue_ratio=result.get("prohibited_revenue_ratio"),
                debt_to_market_cap=result.get("debt_to_market_cap"),
                deposits_to_market_cap=result.get("deposits_to_market_cap"),
                impermissible_income_ratio=result.get("impermissible_income_ratio"),
                receivables_to_market_cap=result.get("receivables_to_market_cap"),
                reasons=json.dumps(result.get("reasons", [])),
                needs_review=result.get("needs_review", False),
                expires_at=result.get("expires_at"),
            )
            session.add(record)
            session.commit()
            return record.id
        finally:
            session.close()

    def get_latest_compliance(self, symbol: str) -> dict | None:
        """
        Get the most recent compliance result for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ComplianceRecord as dictionary or None
        """
        session = self.get_session()

        try:
            record = (
                session.query(ComplianceRecord)
                .filter(ComplianceRecord.symbol == symbol)
                .order_by(ComplianceRecord.screening_date.desc())
                .first()
            )

            if record:
                return record.to_dict()
            return None
        finally:
            session.close()

    def get_compliant_symbols(self) -> list[str]:
        """
        Get all symbols with current compliant status.

        Returns:
            List of compliant symbols
        """
        session = self.get_session()

        try:
            # Subquery to get latest screening date per symbol
            from sqlalchemy import func

            subquery = (
                session.query(
                    ComplianceRecord.symbol,
                    func.max(ComplianceRecord.screening_date).label("max_date"),
                )
                .group_by(ComplianceRecord.symbol)
                .subquery()
            )

            # Join to get latest records
            records = (
                session.query(ComplianceRecord)
                .join(
                    subquery,
                    (ComplianceRecord.symbol == subquery.c.symbol)
                    & (ComplianceRecord.screening_date == subquery.c.max_date),
                )
                .filter(ComplianceRecord.is_compliant == True)
                .all()
            )

            return [r.symbol for r in records]
        finally:
            session.close()

    # Trade Operations

    def save_trade(self, trade: dict) -> int:
        """
        Save a trade record.

        Args:
            trade: Trade details dictionary

        Returns:
            Record ID
        """
        session = self.get_session()

        try:
            record = TradeRecord(
                symbol=trade.get("symbol"),
                trade_date=trade.get("trade_date", datetime.now()),
                order_id=trade.get("order_id"),
                order_type=trade.get("order_type"),
                side=trade.get("side"),
                quantity=trade.get("quantity"),
                price=trade.get("price"),
                commission=trade.get("commission", 0),
                total_value=trade.get("total_value"),
                position_after=trade.get("position_after"),
                avg_cost_after=trade.get("avg_cost_after"),
                realized_pnl=trade.get("realized_pnl"),
                was_compliant_at_trade=trade.get("was_compliant", True),
                notes=trade.get("notes"),
            )
            session.add(record)
            session.commit()
            return record.id
        finally:
            session.close()

    def get_trades(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
        """
        Retrieve trade records.

        Args:
            symbol: Filter by symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of trade dictionaries
        """
        session = self.get_session()

        try:
            query = session.query(TradeRecord)

            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
            if start_date:
                query = query.filter(TradeRecord.trade_date >= start_date)
            if end_date:
                query = query.filter(TradeRecord.trade_date <= end_date)

            query = query.order_by(TradeRecord.trade_date.desc())

            return [record.to_dict() for record in query.all()]
        finally:
            session.close()

    # Signal Operations

    def save_signal(self, signal: dict) -> int:
        """
        Save a trading signal.

        Args:
            signal: Signal details dictionary

        Returns:
            Record ID
        """
        session = self.get_session()

        try:
            record = SignalRecord(
                symbol=signal.get("symbol"),
                signal_date=signal.get("signal_date", datetime.now()),
                model_name=signal.get("model_name"),
                signal_type=signal.get("signal_type"),
                confidence=signal.get("confidence"),
                probability=signal.get("probability"),
                features_json=json.dumps(signal.get("features", {})),
            )
            session.add(record)
            session.commit()
            return record.id
        finally:
            session.close()

    # Utility Operations

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        session = self.get_session()

        try:
            return {
                "stock_data_count": session.query(StockData).count(),
                "compliance_records": session.query(ComplianceRecord).count(),
                "trade_records": session.query(TradeRecord).count(),
                "signal_records": session.query(SignalRecord).count(),
                "unique_symbols": session.query(StockData.symbol).distinct().count(),
            }
        finally:
            session.close()

    # =========================================================================
    # ML INSIGHTS OPERATIONS
    # =========================================================================

    # --- Model Registry ---

    def register_model(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        hyperparameters: dict | None = None,
        feature_names: list[str] | None = None,
        metrics: dict | None = None,
        model_path: str | None = None,
        parent_model_id: int | None = None,
        training_data_start: datetime | None = None,
        training_data_end: datetime | None = None,
        training_samples: int | None = None,
    ) -> int:
        """
        Register a new model in the registry.

        Args:
            model_name: Name of the model (e.g., "momentum_continuation")
            model_version: Version string (e.g., "1.0.0", "2024-01-15_v1")
            model_type: Type of model (e.g., "lightgbm", "random_forest")
            hyperparameters: Model hyperparameters dict
            feature_names: List of feature names used
            metrics: Training metrics dict
            model_path: Path to saved model file
            parent_model_id: ID of parent model for lineage tracking
            training_data_start: Start date of training data
            training_data_end: End date of training data
            training_samples: Number of training samples

        Returns:
            Model registry ID
        """
        session = self.get_session()

        try:
            record = ModelRegistry(
                model_name=model_name,
                model_version=model_version,
                model_type=model_type,
                status="training",
                hyperparameters_json=json.dumps(hyperparameters) if hyperparameters else None,
                feature_names_json=json.dumps(feature_names) if feature_names else None,
                metrics_json=json.dumps(metrics) if metrics else None,
                model_path=model_path,
                parent_model_id=parent_model_id,
                training_data_start=training_data_start,
                training_data_end=training_data_end,
                training_samples=training_samples,
            )
            session.add(record)
            session.commit()
            return record.id
        finally:
            session.close()

    def update_model_status(
        self,
        model_id: int,
        status: str,
        metrics: dict | None = None,
    ) -> bool:
        """
        Update model status (training -> validated -> active -> deprecated).

        Args:
            model_id: Model registry ID
            status: New status
            metrics: Updated metrics (optional)

        Returns:
            True if updated successfully
        """
        session = self.get_session()

        try:
            record = session.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
            if not record:
                return False

            record.status = status

            if status == "validated":
                record.validated_at = datetime.now()
            elif status == "active":
                record.activated_at = datetime.now()
            elif status == "deprecated":
                record.deprecated_at = datetime.now()

            if metrics:
                record.metrics_json = json.dumps(metrics)

            session.commit()
            return True
        finally:
            session.close()

    def activate_model(self, model_id: int) -> bool:
        """
        Activate a model and deactivate all other versions of the same model.

        Args:
            model_id: Model registry ID to activate

        Returns:
            True if activated successfully
        """
        session = self.get_session()

        try:
            record = session.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
            if not record:
                return False

            # Deactivate all other versions of this model
            session.query(ModelRegistry).filter(
                ModelRegistry.model_name == record.model_name,
                ModelRegistry.id != model_id,
            ).update({"is_active": False})

            # Activate this model
            record.is_active = True
            record.status = "active"
            record.activated_at = datetime.now()

            session.commit()
            return True
        finally:
            session.close()

    def get_active_model(self, model_name: str) -> dict | None:
        """
        Get the currently active model for a given model name.

        Args:
            model_name: Name of the model

        Returns:
            Model registry dict or None
        """
        session = self.get_session()

        try:
            record = session.query(ModelRegistry).filter(
                ModelRegistry.model_name == model_name,
                ModelRegistry.is_active == True,
            ).first()

            return record.to_dict() if record else None
        finally:
            session.close()

    def get_model_history(self, model_name: str, limit: int = 10) -> list[dict]:
        """
        Get version history for a model.

        Args:
            model_name: Name of the model
            limit: Maximum number of records

        Returns:
            List of model registry dicts
        """
        session = self.get_session()

        try:
            records = (
                session.query(ModelRegistry)
                .filter(ModelRegistry.model_name == model_name)
                .order_by(ModelRegistry.created_at.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in records]
        finally:
            session.close()

    # --- Prediction Outcomes ---

    def record_prediction_outcome(
        self,
        model_id: int,
        symbol: str,
        prediction_type: str,
        predicted_value: float,
        predicted_probability: float | None = None,
        predicted_class: str | None = None,
        signal_id: int | None = None,
        prediction_time: datetime | None = None,
    ) -> int:
        """
        Record a prediction for later outcome tracking.

        Args:
            model_id: Model registry ID
            symbol: Stock symbol
            prediction_type: Type of prediction (direction, magnitude, duration)
            predicted_value: Predicted value
            predicted_probability: Probability/confidence
            predicted_class: Predicted class label
            signal_id: Associated signal record ID
            prediction_time: Time of prediction

        Returns:
            Prediction outcome ID
        """
        session = self.get_session()

        try:
            record = PredictionOutcome(
                model_id=model_id,
                signal_id=signal_id,
                symbol=symbol,
                prediction_time=prediction_time or datetime.now(),
                prediction_type=prediction_type,
                predicted_value=predicted_value,
                predicted_probability=predicted_probability,
                predicted_class=predicted_class,
            )
            session.add(record)
            session.commit()
            return record.id
        finally:
            session.close()

    def update_prediction_outcome(
        self,
        prediction_id: int,
        actual_value: float,
        actual_class: str | None = None,
        trade_id: int | None = None,
        realized_pnl: float | None = None,
    ) -> bool:
        """
        Update a prediction with actual outcome.

        Args:
            prediction_id: Prediction outcome ID
            actual_value: Actual value observed
            actual_class: Actual class label
            trade_id: Associated trade record ID
            realized_pnl: Realized P&L if traded

        Returns:
            True if updated successfully
        """
        session = self.get_session()

        try:
            record = session.query(PredictionOutcome).filter(
                PredictionOutcome.id == prediction_id
            ).first()

            if not record:
                return False

            record.actual_value = actual_value
            record.actual_class = actual_class
            record.outcome_time = datetime.now()
            record.evaluated_at = datetime.now()

            # Calculate correctness - use multiple methods for robustness
            # Method 1: Class-based comparison (preferred if classes match semantically)
            if record.predicted_class and actual_class:
                record.is_correct = record.predicted_class == actual_class
            # Method 2: Direction-based comparison using value signs
            elif record.prediction_type == "direction":
                # For direction predictions, same sign = correct
                # Handle zero values: consider 0 as neutral (neither positive nor negative)
                if record.predicted_value == 0 or actual_value == 0:
                    record.is_correct = record.predicted_value == actual_value
                else:
                    # Both positive or both negative = correct prediction
                    record.is_correct = (record.predicted_value > 0) == (actual_value > 0)
            # Method 3: Fallback - exact match for other prediction types
            else:
                record.is_correct = abs(record.predicted_value - actual_value) < 0.01

            # Calculate error
            record.error = actual_value - record.predicted_value
            if record.predicted_value != 0:
                record.error_pct = (record.error / abs(record.predicted_value)) * 100

            if trade_id:
                record.trade_id = trade_id
            if realized_pnl is not None:
                record.realized_pnl = realized_pnl

            session.commit()
            return True
        finally:
            session.close()

    def get_prediction_accuracy(
        self,
        model_id: int,
        days: int = 7,
        prediction_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get prediction accuracy metrics for a model.

        Args:
            model_id: Model registry ID
            days: Number of days to look back
            prediction_type: Filter by prediction type

        Returns:
            Dict with accuracy metrics
        """
        session = self.get_session()

        try:
            from sqlalchemy import func

            cutoff = datetime.now() - timedelta(days=days)

            query = session.query(PredictionOutcome).filter(
                PredictionOutcome.model_id == model_id,
                PredictionOutcome.prediction_time >= cutoff,
                PredictionOutcome.is_correct != None,  # noqa: E711
            )

            if prediction_type:
                query = query.filter(PredictionOutcome.prediction_type == prediction_type)

            records = query.all()

            if not records:
                return {"total": 0, "accuracy": None, "avg_error": None}

            correct = sum(1 for r in records if r.is_correct)
            total = len(records)
            errors = [r.error for r in records if r.error is not None]

            return {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else None,
                "avg_error": sum(errors) / len(errors) if errors else None,
                "avg_error_pct": sum(r.error_pct for r in records if r.error_pct) / len(records) if records else None,
            }
        finally:
            session.close()

    def get_unevaluated_predictions(
        self,
        model_id: int | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get predictions that haven't been evaluated yet.

        Args:
            model_id: Filter by model ID
            limit: Maximum records to return

        Returns:
            List of prediction outcome dicts
        """
        session = self.get_session()

        try:
            query = session.query(PredictionOutcome).filter(
                PredictionOutcome.actual_value == None,  # noqa: E711
            )

            if model_id:
                query = query.filter(PredictionOutcome.model_id == model_id)

            records = query.order_by(PredictionOutcome.prediction_time).limit(limit).all()
            return [r.to_dict() for r in records]
        finally:
            session.close()

    # --- Feature Importance ---

    def save_feature_importance(
        self,
        model_id: int,
        importances: dict[str, float],
        importance_type: str = "gain",
        model_component: str | None = None,
    ) -> int:
        """
        Save feature importance scores for a model.

        Args:
            model_id: Model registry ID
            importances: Dict of {feature_name: importance_score}
            importance_type: Type of importance (gain, split, permutation)
            model_component: Model component (for multi-model systems)

        Returns:
            Number of records saved
        """
        session = self.get_session()
        count = 0

        try:
            # Sort by importance and assign ranks
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            recorded_at = datetime.now()

            for rank, (feature_name, score) in enumerate(sorted_features, 1):
                record = FeatureImportance(
                    model_id=model_id,
                    feature_name=feature_name,
                    importance_score=score,
                    rank=rank,
                    importance_type=importance_type,
                    model_component=model_component,
                    recorded_at=recorded_at,
                )
                session.add(record)
                count += 1

            session.commit()
            return count
        finally:
            session.close()

    def get_feature_drift_alerts(
        self,
        model_id: int,
        rank_change_threshold: int = 5,
        top_n: int = 10,
    ) -> list[dict]:
        """
        Detect feature importance drift by comparing recent vs historical rankings.

        Args:
            model_id: Model registry ID
            rank_change_threshold: Alert if rank changes by this much
            top_n: Only check top N features

        Returns:
            List of drift alerts
        """
        session = self.get_session()

        try:
            from sqlalchemy import func

            # Get most recent importance snapshot
            latest_time = (
                session.query(func.max(FeatureImportance.recorded_at))
                .filter(FeatureImportance.model_id == model_id)
                .scalar()
            )

            if not latest_time:
                return []

            # Get previous snapshot
            prev_time = (
                session.query(func.max(FeatureImportance.recorded_at))
                .filter(
                    FeatureImportance.model_id == model_id,
                    FeatureImportance.recorded_at < latest_time,
                )
                .scalar()
            )

            if not prev_time:
                return []

            # Get current top features
            current = {
                r.feature_name: r.rank
                for r in session.query(FeatureImportance)
                .filter(
                    FeatureImportance.model_id == model_id,
                    FeatureImportance.recorded_at == latest_time,
                    FeatureImportance.rank <= top_n,
                )
                .all()
            }

            # Get previous rankings
            previous = {
                r.feature_name: r.rank
                for r in session.query(FeatureImportance)
                .filter(
                    FeatureImportance.model_id == model_id,
                    FeatureImportance.recorded_at == prev_time,
                )
                .all()
            }

            # Find drift
            alerts = []
            for feature, current_rank in current.items():
                prev_rank = previous.get(feature)
                if prev_rank is None:
                    alerts.append({
                        "feature": feature,
                        "type": "new_feature",
                        "current_rank": current_rank,
                        "previous_rank": None,
                        "change": None,
                    })
                elif abs(current_rank - prev_rank) >= rank_change_threshold:
                    alerts.append({
                        "feature": feature,
                        "type": "rank_change",
                        "current_rank": current_rank,
                        "previous_rank": prev_rank,
                        "change": prev_rank - current_rank,  # Positive = improved
                    })

            return alerts
        finally:
            session.close()

    def get_feature_importance_history(
        self,
        model_id: int,
        feature_name: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get feature importance history over time.

        Args:
            model_id: Model registry ID
            feature_name: Filter by specific feature
            limit: Number of snapshots to return

        Returns:
            List of importance records
        """
        session = self.get_session()

        try:
            from sqlalchemy import func

            # Get distinct snapshot times
            times = (
                session.query(FeatureImportance.recorded_at)
                .filter(FeatureImportance.model_id == model_id)
                .distinct()
                .order_by(FeatureImportance.recorded_at.desc())
                .limit(limit)
                .all()
            )

            query = session.query(FeatureImportance).filter(
                FeatureImportance.model_id == model_id,
                FeatureImportance.recorded_at.in_([t[0] for t in times]),
            )

            if feature_name:
                query = query.filter(FeatureImportance.feature_name == feature_name)

            records = query.order_by(
                FeatureImportance.recorded_at.desc(),
                FeatureImportance.rank,
            ).all()

            return [r.to_dict() for r in records]
        finally:
            session.close()

    # --- Confidence Calibration ---

    def save_calibration_report(
        self,
        model_id: int,
        period_start: datetime,
        period_end: datetime,
        sample_count: int,
        ece: float,
        mce: float | None = None,
        brier_score: float | None = None,
        log_loss: float | None = None,
        reliability_diagram: list[dict] | None = None,
        class_metrics: dict | None = None,
        ece_threshold: float = 0.10,
    ) -> int:
        """
        Save a calibration report.

        Args:
            model_id: Model registry ID
            period_start: Start of evaluation period
            period_end: End of evaluation period
            sample_count: Number of predictions evaluated
            ece: Expected Calibration Error
            mce: Maximum Calibration Error
            brier_score: Brier score
            log_loss: Log loss
            reliability_diagram: Reliability diagram data
            class_metrics: Per-class metrics
            ece_threshold: Threshold for well-calibrated

        Returns:
            Calibration record ID
        """
        session = self.get_session()

        try:
            is_well_calibrated = ece < ece_threshold
            alert = None if is_well_calibrated else f"ECE {ece:.4f} exceeds threshold {ece_threshold}"

            record = ConfidenceCalibration(
                model_id=model_id,
                period_start=period_start,
                period_end=period_end,
                sample_count=sample_count,
                expected_calibration_error=ece,
                maximum_calibration_error=mce,
                brier_score=brier_score,
                log_loss=log_loss,
                reliability_diagram_json=json.dumps(reliability_diagram) if reliability_diagram else None,
                class_metrics_json=json.dumps(class_metrics) if class_metrics else None,
                is_well_calibrated=is_well_calibrated,
                calibration_alert=alert,
            )
            session.add(record)
            session.commit()
            return record.id
        finally:
            session.close()

    def get_calibration_history(
        self,
        model_id: int,
        limit: int = 12,
    ) -> list[dict]:
        """
        Get calibration history for a model.

        Args:
            model_id: Model registry ID
            limit: Number of records

        Returns:
            List of calibration report dicts
        """
        session = self.get_session()

        try:
            records = (
                session.query(ConfidenceCalibration)
                .filter(ConfidenceCalibration.model_id == model_id)
                .order_by(ConfidenceCalibration.period_end.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in records]
        finally:
            session.close()

    def get_latest_calibration(self, model_id: int) -> dict | None:
        """
        Get the most recent calibration report.

        Args:
            model_id: Model registry ID

        Returns:
            Calibration report dict or None
        """
        session = self.get_session()

        try:
            record = (
                session.query(ConfidenceCalibration)
                .filter(ConfidenceCalibration.model_id == model_id)
                .order_by(ConfidenceCalibration.period_end.desc())
                .first()
            )
            return record.to_dict() if record else None
        finally:
            session.close()

    # --- Market Regimes ---

    def save_market_regime(
        self,
        regime_date: date,
        regime_type: str,
        regime_confidence: float,
        period_start: datetime,
        period_end: datetime,
        vix_level: float | None = None,
        vix_change_pct: float | None = None,
        sp500_return_pct: float | None = None,
        sp500_volatility: float | None = None,
        breadth_advance_decline: float | None = None,
        sector_correlation: float | None = None,
        additional_metrics: dict | None = None,
    ) -> int:
        """
        Save a market regime classification.

        Args:
            regime_date: Date of regime
            regime_type: Type (trending_up, trending_down, volatile, quiet, mixed)
            regime_confidence: Confidence score
            period_start: Start of analysis period
            period_end: End of analysis period
            vix_level: VIX level
            vix_change_pct: VIX change percentage
            sp500_return_pct: S&P 500 return percentage
            sp500_volatility: S&P 500 volatility
            breadth_advance_decline: Market breadth
            sector_correlation: Sector correlation
            additional_metrics: Additional metrics dict

        Returns:
            Regime record ID
        """
        session = self.get_session()

        try:
            # Check if regime exists for this date
            existing = session.query(MarketRegime).filter(
                MarketRegime.regime_date == regime_date
            ).first()

            if existing:
                # Update existing
                existing.regime_type = regime_type
                existing.regime_confidence = regime_confidence
                existing.period_start = period_start
                existing.period_end = period_end
                existing.vix_level = vix_level
                existing.vix_change_pct = vix_change_pct
                existing.sp500_return_pct = sp500_return_pct
                existing.sp500_volatility = sp500_volatility
                existing.breadth_advance_decline = breadth_advance_decline
                existing.sector_correlation = sector_correlation
                existing.metrics_json = json.dumps(additional_metrics) if additional_metrics else None
                session.commit()
                return existing.id

            record = MarketRegime(
                regime_date=regime_date,
                regime_type=regime_type,
                regime_confidence=regime_confidence,
                period_start=period_start,
                period_end=period_end,
                vix_level=vix_level,
                vix_change_pct=vix_change_pct,
                sp500_return_pct=sp500_return_pct,
                sp500_volatility=sp500_volatility,
                breadth_advance_decline=breadth_advance_decline,
                sector_correlation=sector_correlation,
                metrics_json=json.dumps(additional_metrics) if additional_metrics else None,
            )
            session.add(record)
            session.commit()
            return record.id
        finally:
            session.close()

    def get_current_regime(self) -> dict | None:
        """
        Get the most recent market regime.

        Returns:
            Market regime dict or None
        """
        session = self.get_session()

        try:
            record = (
                session.query(MarketRegime)
                .order_by(MarketRegime.regime_date.desc())
                .first()
            )
            return record.to_dict() if record else None
        finally:
            session.close()

    def get_regime_history(self, days: int = 30) -> list[dict]:
        """
        Get market regime history.

        Args:
            days: Number of days to look back

        Returns:
            List of regime dicts
        """
        session = self.get_session()

        try:
            cutoff = date.today() - timedelta(days=days)
            records = (
                session.query(MarketRegime)
                .filter(MarketRegime.regime_date >= cutoff)
                .order_by(MarketRegime.regime_date.desc())
                .all()
            )
            return [r.to_dict() for r in records]
        finally:
            session.close()

    def update_regime_model_performance(
        self,
        regime_date: date,
        model_accuracy: float,
        model_pnl: float,
    ) -> bool:
        """
        Update model performance for a regime.

        Args:
            regime_date: Date of regime
            model_accuracy: Model accuracy in this regime
            model_pnl: Model P&L in this regime

        Returns:
            True if updated successfully
        """
        session = self.get_session()

        try:
            record = session.query(MarketRegime).filter(
                MarketRegime.regime_date == regime_date
            ).first()

            if not record:
                return False

            record.model_accuracy_in_regime = model_accuracy
            record.model_pnl_in_regime = model_pnl
            session.commit()
            return True
        finally:
            session.close()

    # --- Learning Events ---

    def log_learning_event(
        self,
        event_type: str,
        severity: str,
        category: str,
        title: str,
        description: str,
        details: dict | None = None,
        source: str | None = None,
        model_id: int | None = None,
        requires_action: bool = False,
    ) -> int:
        """
        Log a learning event (alert, insight, adaptation).

        Args:
            event_type: Type (alert, insight, adaptation, retrain_trigger)
            severity: Severity (info, warning, critical)
            category: Category (performance, calibration, drift, regime, system)
            title: Short title
            description: Detailed description
            details: Additional details dict
            source: Source of event (monitor, manual, model, system)
            model_id: Associated model ID
            requires_action: Whether action is required

        Returns:
            Learning event ID
        """
        session = self.get_session()

        try:
            record = LearningEvent(
                event_type=event_type,
                severity=severity,
                category=category,
                title=title,
                description=description,
                details_json=json.dumps(details) if details else None,
                source=source,
                model_id=model_id,
                requires_action=requires_action,
            )
            session.add(record)
            session.commit()
            return record.id
        finally:
            session.close()

    def get_open_learning_events(
        self,
        severity: str | None = None,
        category: str | None = None,
        requires_action_only: bool = False,
    ) -> list[dict]:
        """
        Get open (unresolved) learning events.

        Args:
            severity: Filter by severity
            category: Filter by category
            requires_action_only: Only show events requiring action

        Returns:
            List of learning event dicts
        """
        session = self.get_session()

        try:
            query = session.query(LearningEvent).filter(
                LearningEvent.status.in_(["open", "acknowledged"])
            )

            if severity:
                query = query.filter(LearningEvent.severity == severity)
            if category:
                query = query.filter(LearningEvent.category == category)
            if requires_action_only:
                query = query.filter(LearningEvent.requires_action == True)

            records = query.order_by(
                LearningEvent.severity.desc(),
                LearningEvent.event_time.desc(),
            ).all()

            return [r.to_dict() for r in records]
        finally:
            session.close()

    def resolve_learning_event(
        self,
        event_id: int,
        action_taken: str | None = None,
        status: str = "resolved",
    ) -> bool:
        """
        Resolve a learning event.

        Args:
            event_id: Learning event ID
            action_taken: Description of action taken
            status: New status (resolved, dismissed)

        Returns:
            True if resolved successfully
        """
        session = self.get_session()

        try:
            record = session.query(LearningEvent).filter(
                LearningEvent.id == event_id
            ).first()

            if not record:
                return False

            record.status = status
            record.resolved_at = datetime.now()
            if action_taken:
                record.action_taken = action_taken

            session.commit()
            return True
        finally:
            session.close()

    def acknowledge_learning_event(self, event_id: int) -> bool:
        """
        Acknowledge a learning event (mark as seen).

        Args:
            event_id: Learning event ID

        Returns:
            True if acknowledged successfully
        """
        session = self.get_session()

        try:
            record = session.query(LearningEvent).filter(
                LearningEvent.id == event_id
            ).first()

            if not record:
                return False

            record.status = "acknowledged"
            record.acknowledged_at = datetime.now()
            session.commit()
            return True
        finally:
            session.close()

    def get_learning_events_history(
        self,
        days: int = 30,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get learning events history.

        Args:
            days: Number of days to look back
            event_type: Filter by event type
            limit: Maximum records

        Returns:
            List of learning event dicts
        """
        session = self.get_session()

        try:
            cutoff = datetime.now() - timedelta(days=days)

            query = session.query(LearningEvent).filter(
                LearningEvent.event_time >= cutoff
            )

            if event_type:
                query = query.filter(LearningEvent.event_type == event_type)

            records = (
                query.order_by(LearningEvent.event_time.desc())
                .limit(limit)
                .all()
            )

            return [r.to_dict() for r in records]
        finally:
            session.close()

    # --- ML Statistics ---

    def get_ml_statistics(self) -> dict[str, Any]:
        """Get ML-related database statistics."""
        session = self.get_session()

        try:
            return {
                "model_registry_count": session.query(ModelRegistry).count(),
                "active_models": session.query(ModelRegistry).filter(ModelRegistry.is_active == True).count(),
                "prediction_outcomes_count": session.query(PredictionOutcome).count(),
                "unevaluated_predictions": session.query(PredictionOutcome).filter(
                    PredictionOutcome.actual_value == None  # noqa: E711
                ).count(),
                "feature_importance_records": session.query(FeatureImportance).count(),
                "calibration_reports": session.query(ConfidenceCalibration).count(),
                "market_regime_records": session.query(MarketRegime).count(),
                "learning_events_total": session.query(LearningEvent).count(),
                "learning_events_open": session.query(LearningEvent).filter(
                    LearningEvent.status == "open"
                ).count(),
            }
        finally:
            session.close()

    def get_model_health_metrics(self, model_name: str) -> dict[str, Any]:
        """
        Get comprehensive health metrics for a model.

        Args:
            model_name: Name of the model

        Returns:
            Dict with health metrics
        """
        session = self.get_session()

        try:
            # Get active model
            model = session.query(ModelRegistry).filter(
                ModelRegistry.model_name == model_name,
                ModelRegistry.is_active == True,
            ).first()

            if not model:
                return {"status": "no_active_model", "model_name": model_name}

            # Get recent accuracy
            accuracy = self.get_prediction_accuracy(model.id, days=7)

            # Get latest calibration
            calibration = self.get_latest_calibration(model.id)

            # Get feature drift alerts
            drift_alerts = self.get_feature_drift_alerts(model.id)

            # Get open events
            events = session.query(LearningEvent).filter(
                LearningEvent.model_id == model.id,
                LearningEvent.status == "open",
            ).count()

            # Calculate days since training
            days_since_training = None
            if model.created_at:
                days_since_training = (datetime.now() - model.created_at).days

            return {
                "model_name": model_name,
                "model_id": model.id,
                "model_version": model.model_version,
                "status": model.status,
                "days_since_training": days_since_training,
                "accuracy_7d": accuracy.get("accuracy"),
                "predictions_7d": accuracy.get("total"),
                "calibration_ece": calibration.get("expected_calibration_error") if calibration else None,
                "is_well_calibrated": calibration.get("is_well_calibrated") if calibration else None,
                "drift_alerts": len(drift_alerts),
                "open_events": events,
                "training_metrics": json.loads(model.metrics_json) if model.metrics_json else None,
            }
        finally:
            session.close()
