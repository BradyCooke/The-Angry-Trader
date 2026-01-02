"""Order management."""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Trading order."""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None
    filled_price: float | None = None
    filled_quantity: int = 0
    order_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"{self.symbol}_{self.side.value}_{self.created_at.strftime('%Y%m%d%H%M%S%f')}"

    @property
    def is_buy(self) -> bool:
        """Check if order is a buy order."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Check if order is a sell order."""
        return self.side == OrderSide.SELL

    @property
    def is_filled(self) -> bool:
        """Check if order is filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_pending(self) -> bool:
        """Check if order is pending."""
        return self.status == OrderStatus.PENDING

    def fill(self, price: float, quantity: int | None = None, fill_time: datetime | None = None) -> None:
        """Mark order as filled.

        Args:
            price: Fill price.
            quantity: Fill quantity (defaults to full order quantity).
            fill_time: Fill timestamp (defaults to now).
        """
        self.filled_price = price
        self.filled_quantity = quantity if quantity is not None else self.quantity
        self.filled_at = fill_time or datetime.now()
        self.status = OrderStatus.FILLED

    def cancel(self) -> None:
        """Cancel the order."""
        self.status = OrderStatus.CANCELLED

    def reject(self, reason: str = "") -> None:
        """Reject the order.

        Args:
            reason: Rejection reason.
        """
        self.status = OrderStatus.REJECTED
        self.metadata["rejection_reason"] = reason


@dataclass
class OrderBook:
    """Simple order book for tracking orders."""
    orders: list[Order] = field(default_factory=list)

    def add_order(self, order: Order) -> None:
        """Add order to the book."""
        self.orders.append(order)

    def get_pending_orders(self) -> list[Order]:
        """Get all pending orders."""
        return [o for o in self.orders if o.is_pending]

    def get_filled_orders(self) -> list[Order]:
        """Get all filled orders."""
        return [o for o in self.orders if o.is_filled]

    def get_orders_for_symbol(self, symbol: str) -> list[Order]:
        """Get all orders for a symbol."""
        return [o for o in self.orders if o.symbol == symbol]

    def cancel_pending_orders(self, symbol: str | None = None) -> int:
        """Cancel pending orders.

        Args:
            symbol: If provided, only cancel orders for this symbol.

        Returns:
            Number of orders cancelled.
        """
        count = 0
        for order in self.orders:
            if order.is_pending:
                if symbol is None or order.symbol == symbol:
                    order.cancel()
                    count += 1
        return count
