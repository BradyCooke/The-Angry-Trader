"""Base broker interface for execution layer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class BrokerError(Exception):
    """Exception raised for broker-related errors."""
    pass


class ConnectionStatus(Enum):
    """Broker connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class OrderResult:
    """Result of an order execution attempt.

    Attributes:
        order_id: Unique order identifier.
        symbol: Ticker symbol.
        side: BUY or SELL.
        quantity: Number of shares.
        status: Current order status.
        filled_quantity: Number of shares filled.
        filled_price: Average fill price.
        commission: Total commission charged.
        submitted_at: Timestamp when order was submitted.
        filled_at: Timestamp when order was filled (if filled).
        message: Status message or error details.
        metadata: Additional broker-specific data.
    """
    order_id: str
    symbol: str
    side: str
    quantity: int
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_pending(self) -> bool:
        """Check if order is still pending."""
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)

    @property
    def total_cost(self) -> float:
        """Calculate total cost including commission."""
        return (self.filled_quantity * self.filled_price) + self.commission


@dataclass
class Position:
    """Current position held at broker.

    Attributes:
        symbol: Ticker symbol.
        quantity: Number of shares (negative for short).
        avg_cost: Average cost per share.
        market_value: Current market value.
        unrealized_pnl: Unrealized profit/loss.
        realized_pnl: Realized profit/loss.
    """
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0


@dataclass
class AccountInfo:
    """Broker account information.

    Attributes:
        account_id: Account identifier.
        buying_power: Available buying power.
        cash: Cash balance.
        equity: Total account equity.
        margin_used: Margin currently in use.
        positions: Dictionary of current positions.
    """
    account_id: str
    buying_power: float
    cash: float
    equity: float
    margin_used: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)


class BaseBroker(ABC):
    """Abstract base class for broker implementations.

    This defines the interface that all broker implementations must follow.
    Currently serves as a placeholder for future Interactive Brokers integration.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize broker.

        Args:
            config: Broker-specific configuration.
        """
        self.config = config or {}
        self._status = ConnectionStatus.DISCONNECTED
        self._account_info: AccountInfo | None = None

    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._status == ConnectionStatus.CONNECTED

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker.

        Returns:
            True if connection successful.

        Raises:
            BrokerError: If connection fails.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get current account information.

        Returns:
            AccountInfo object with current account state.

        Raises:
            BrokerError: If not connected or request fails.
        """
        pass

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Get current positions.

        Returns:
            Dictionary mapping symbol to Position.

        Raises:
            BrokerError: If not connected or request fails.
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> dict[str, float]:
        """Get current quote for symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Dictionary with bid, ask, last, volume.

        Raises:
            BrokerError: If symbol not found or request fails.
        """
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> OrderResult:
        """Submit an order to the broker.

        Args:
            symbol: Ticker symbol.
            side: BUY or SELL.
            quantity: Number of shares.
            order_type: MARKET, LIMIT, STOP, or STOP_LIMIT.
            limit_price: Limit price for LIMIT and STOP_LIMIT orders.
            stop_price: Stop price for STOP and STOP_LIMIT orders.

        Returns:
            OrderResult with execution status.

        Raises:
            BrokerError: If order submission fails.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancellation successful.

        Raises:
            BrokerError: If cancellation fails.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResult:
        """Get current status of an order.

        Args:
            order_id: Order ID to check.

        Returns:
            OrderResult with current status.

        Raises:
            BrokerError: If order not found.
        """
        pass

    def submit_market_order(self, symbol: str, side: str, quantity: int) -> OrderResult:
        """Convenience method for market orders.

        Args:
            symbol: Ticker symbol.
            side: BUY or SELL.
            quantity: Number of shares.

        Returns:
            OrderResult with execution status.
        """
        return self.submit_order(symbol, side, quantity, order_type="MARKET")

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        limit_price: float,
    ) -> OrderResult:
        """Convenience method for limit orders.

        Args:
            symbol: Ticker symbol.
            side: BUY or SELL.
            quantity: Number of shares.
            limit_price: Limit price.

        Returns:
            OrderResult with execution status.
        """
        return self.submit_order(
            symbol, side, quantity,
            order_type="LIMIT",
            limit_price=limit_price,
        )

    def submit_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        stop_price: float,
    ) -> OrderResult:
        """Convenience method for stop orders.

        Args:
            symbol: Ticker symbol.
            side: BUY or SELL.
            quantity: Number of shares.
            stop_price: Stop trigger price.

        Returns:
            OrderResult with execution status.
        """
        return self.submit_order(
            symbol, side, quantity,
            order_type="STOP",
            stop_price=stop_price,
        )
