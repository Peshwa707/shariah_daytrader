"""
Database Storage Module.

SQLite-based storage for:
- Historical price data
- Compliance screening results
- Trade records
- Purification tracking

Uses SQLAlchemy for ORM and easy migration to PostgreSQL if needed.
"""

from datetime import datetime, date
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
