"""
Order Manager Module.

Handles order lifecycle:
- Order creation and validation
- Order submission to IBKR
- Order status tracking
- Fill handling
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import logging
import uuid

import asyncio


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status states."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """
    Represents a trading order.

    Contains all information needed to submit and track an order.
    """

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET

    # Price fields (for limit/stop orders)
    limit_price: float | None = None
    stop_price: float | None = None

    # Bracket order prices (optional)
    take_profit_price: float | None = None
    stop_loss_price: float | None = None

    # Order identification
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    client_order_id: str | None = None
    ibkr_order_id: int | None = None

    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None
    commission: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: datetime | None = None
    filled_at: datetime | None = None

    # Additional info
    notes: str | None = None
    signal_id: str | None = None  # Link to ML signal

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
        }

    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.ERROR,
        ]

    @property
    def is_active(self) -> bool:
        """Check if order is active (submitted but not complete)."""
        return self.status in [
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILL,
        ]

    @property
    def remaining_quantity(self) -> float:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity


class OrderManager:
    """
    Manages order lifecycle.

    Responsibilities:
    - Order creation and validation
    - Submission to IBKR (via IBKRClient)
    - Status tracking and updates
    - Fill handling
    """

    def __init__(self, ibkr_client=None):
        """
        Initialize the order manager.

        Args:
            ibkr_client: IBKRClient instance for order submission
        """
        self.ibkr_client = ibkr_client
        self._orders: dict[str, Order] = {}
        self._active_orders: dict[str, Order] = {}
        self._filled_orders: list[Order] = []

    def create_market_order(
        self,
        symbol: str,
        side: OrderSide | str,
        quantity: float,
        signal_id: str | None = None,
    ) -> Order:
        """
        Create a market order.

        Args:
            symbol: Stock ticker symbol
            side: Order side (buy/sell)
            quantity: Number of shares
            signal_id: Optional ML signal ID

        Returns:
            Created Order object
        """
        if isinstance(side, str):
            side = OrderSide(side.lower())

        order = Order(
            symbol=symbol.upper(),
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            signal_id=signal_id,
        )

        self._orders[order.order_id] = order
        return order

    def create_limit_order(
        self,
        symbol: str,
        side: OrderSide | str,
        quantity: float,
        limit_price: float,
        signal_id: str | None = None,
    ) -> Order:
        """
        Create a limit order.

        Args:
            symbol: Stock ticker symbol
            side: Order side
            quantity: Number of shares
            limit_price: Limit price

        Returns:
            Created Order object
        """
        if isinstance(side, str):
            side = OrderSide(side.lower())

        order = Order(
            symbol=symbol.upper(),
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            signal_id=signal_id,
        )

        self._orders[order.order_id] = order
        return order

    def create_bracket_order(
        self,
        symbol: str,
        side: OrderSide | str,
        quantity: float,
        entry_price: float | None = None,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
        signal_id: str | None = None,
    ) -> Order:
        """
        Create a bracket order (entry + take profit + stop loss).

        Args:
            symbol: Stock ticker symbol
            side: Order side
            quantity: Number of shares
            entry_price: Entry limit price (None for market)
            take_profit_price: Take profit price
            stop_loss_price: Stop loss price

        Returns:
            Created Order object (parent order)
        """
        if isinstance(side, str):
            side = OrderSide(side.lower())

        order_type = OrderType.LIMIT if entry_price else OrderType.MARKET

        order = Order(
            symbol=symbol.upper(),
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            signal_id=signal_id,
        )

        self._orders[order.order_id] = order
        return order

    async def submit_order(self, order: Order) -> bool:
        """
        Submit an order to IBKR.

        Args:
            order: Order to submit

        Returns:
            True if submission successful
        """
        if self.ibkr_client is None:
            logger.warning("No IBKR client configured - simulating order submission")
            return self._simulate_submit(order)

        if not self.ibkr_client.is_connected:
            logger.error("IBKR not connected")
            order.status = OrderStatus.ERROR
            return False

        try:
            # Build IBKR order
            # This would use ib_async to create and submit the order
            # For now, we'll simulate the submission

            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            self._active_orders[order.order_id] = order

            logger.info(f"Order submitted: {order.order_id} - {order.side.value} {order.quantity} {order.symbol}")
            return True

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            order.status = OrderStatus.ERROR
            return False

    def _simulate_submit(self, order: Order) -> bool:
        """Simulate order submission for paper trading without IBKR."""
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        self._active_orders[order.order_id] = order

        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            self._simulate_fill(order, 100.0)  # Placeholder price

        logger.info(f"Order simulated: {order.order_id}")
        return True

    def _simulate_fill(self, order: Order, fill_price: float) -> None:
        """Simulate an order fill."""
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.filled_at = datetime.now()
        order.commission = order.quantity * fill_price * 0.001  # 0.1% commission

        self._active_orders.pop(order.order_id, None)
        self._filled_orders.append(order)

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        order = self._orders.get(order_id)
        if not order:
            logger.warning(f"Order not found: {order_id}")
            return False

        if not order.is_active:
            logger.warning(f"Order not active: {order_id}")
            return False

        try:
            # Cancel via IBKR
            if self.ibkr_client and self.ibkr_client.is_connected:
                # self.ibkr_client.cancel_order(order.ibkr_order_id)
                pass

            order.status = OrderStatus.CANCELLED
            self._active_orders.pop(order_id, None)

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """
        Cancel all active orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        orders_to_cancel = list(self._active_orders.values())

        if symbol:
            orders_to_cancel = [o for o in orders_to_cancel if o.symbol == symbol.upper()]

        for order in orders_to_cancel:
            if await self.cancel_order(order.order_id):
                cancelled += 1

        return cancelled

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_active_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all active orders, optionally filtered by symbol."""
        orders = list(self._active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        return orders

    def get_filled_orders(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> list[Order]:
        """Get filled orders with optional filters."""
        orders = self._filled_orders

        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]

        if since:
            orders = [o for o in orders if o.filled_at and o.filled_at >= since]

        return orders

    def get_order_history(self, limit: int = 100) -> list[Order]:
        """Get order history."""
        all_orders = sorted(
            self._orders.values(),
            key=lambda o: o.created_at,
            reverse=True,
        )
        return all_orders[:limit]

    def get_statistics(self) -> dict[str, Any]:
        """Get order statistics."""
        total = len(self._orders)
        filled = sum(1 for o in self._orders.values() if o.status == OrderStatus.FILLED)
        cancelled = sum(1 for o in self._orders.values() if o.status == OrderStatus.CANCELLED)
        rejected = sum(1 for o in self._orders.values() if o.status == OrderStatus.REJECTED)

        total_volume = sum(
            o.filled_quantity * (o.avg_fill_price or 0)
            for o in self._orders.values()
            if o.status == OrderStatus.FILLED
        )

        total_commission = sum(
            o.commission
            for o in self._orders.values()
            if o.status == OrderStatus.FILLED
        )

        return {
            "total_orders": total,
            "filled": filled,
            "cancelled": cancelled,
            "rejected": rejected,
            "active": len(self._active_orders),
            "fill_rate": filled / total if total > 0 else 0,
            "total_volume": total_volume,
            "total_commission": total_commission,
        }
