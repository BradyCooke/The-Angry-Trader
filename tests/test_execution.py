"""Tests for execution layer module."""

import pytest

from src.execution.broker_base import (
    AccountInfo,
    BrokerError,
    ConnectionStatus,
    OrderResult,
    OrderStatus,
    Position,
)
from src.execution.ib_broker import IBBroker


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_order_result_creation(self) -> None:
        """Test basic order result creation."""
        result = OrderResult(
            order_id="12345",
            symbol="SPY",
            side="BUY",
            quantity=100,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            filled_price=450.0,
            commission=1.0,
        )

        assert result.order_id == "12345"
        assert result.symbol == "SPY"
        assert result.is_filled
        assert not result.is_pending

    def test_order_result_pending(self) -> None:
        """Test pending order status."""
        result = OrderResult(
            order_id="12345",
            symbol="SPY",
            side="BUY",
            quantity=100,
            status=OrderStatus.SUBMITTED,
        )

        assert result.is_pending
        assert not result.is_filled

    def test_order_total_cost(self) -> None:
        """Test total cost calculation."""
        result = OrderResult(
            order_id="12345",
            symbol="SPY",
            side="BUY",
            quantity=100,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            filled_price=450.0,
            commission=1.0,
        )

        # 100 * 450 + 1 = 45001
        assert result.total_cost == 45001.0


class TestPosition:
    """Tests for Position dataclass."""

    def test_long_position(self) -> None:
        """Test long position properties."""
        pos = Position(
            symbol="SPY",
            quantity=100,
            avg_cost=450.0,
            market_value=46000.0,
        )

        assert pos.is_long
        assert not pos.is_short

    def test_short_position(self) -> None:
        """Test short position properties."""
        pos = Position(
            symbol="SPY",
            quantity=-100,
            avg_cost=450.0,
            market_value=44000.0,
        )

        assert pos.is_short
        assert not pos.is_long


class TestAccountInfo:
    """Tests for AccountInfo dataclass."""

    def test_account_info_creation(self) -> None:
        """Test account info creation."""
        account = AccountInfo(
            account_id="DU123456",
            buying_power=200000.0,
            cash=100000.0,
            equity=150000.0,
        )

        assert account.account_id == "DU123456"
        assert account.buying_power == 200000.0
        assert account.cash == 100000.0


class TestIBBroker:
    """Tests for IBBroker stub implementation."""

    def test_broker_initialization(self) -> None:
        """Test broker initialization."""
        broker = IBBroker({
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 1,
        })

        assert broker.host == "127.0.0.1"
        assert broker.port == 7497
        assert broker.client_id == 1
        assert broker.status == ConnectionStatus.DISCONNECTED

    def test_broker_connect(self) -> None:
        """Test broker connection."""
        broker = IBBroker()
        result = broker.connect()

        assert result is True
        assert broker.is_connected
        assert broker.status == ConnectionStatus.CONNECTED

    def test_broker_disconnect(self) -> None:
        """Test broker disconnection."""
        broker = IBBroker()
        broker.connect()
        broker.disconnect()

        assert not broker.is_connected
        assert broker.status == ConnectionStatus.DISCONNECTED

    def test_get_account_info(self) -> None:
        """Test getting account info."""
        broker = IBBroker()
        broker.connect()

        account = broker.get_account_info()

        assert isinstance(account, AccountInfo)
        assert account.cash > 0
        assert account.equity > 0

    def test_get_account_info_disconnected(self) -> None:
        """Test getting account info when disconnected."""
        broker = IBBroker()

        with pytest.raises(BrokerError, match="Not connected"):
            broker.get_account_info()

    def test_get_positions_empty(self) -> None:
        """Test getting positions when none exist."""
        broker = IBBroker()
        broker.connect()

        positions = broker.get_positions()

        assert isinstance(positions, dict)
        assert len(positions) == 0

    def test_get_quote(self) -> None:
        """Test getting a quote."""
        broker = IBBroker()
        broker.connect()

        quote = broker.get_quote("SPY")

        assert "bid" in quote
        assert "ask" in quote
        assert "last" in quote
        assert "volume" in quote

    def test_submit_market_order(self) -> None:
        """Test submitting a market order."""
        broker = IBBroker()
        broker.connect()

        result = broker.submit_market_order("SPY", "BUY", 100)

        assert isinstance(result, OrderResult)
        assert result.symbol == "SPY"
        assert result.side == "BUY"
        assert result.quantity == 100
        assert result.is_filled

    def test_submit_order_creates_position(self) -> None:
        """Test that submitting an order creates a position."""
        broker = IBBroker()
        broker.connect()

        broker.submit_market_order("SPY", "BUY", 100)
        positions = broker.get_positions()

        assert "SPY" in positions
        assert positions["SPY"].quantity == 100

    def test_submit_sell_order_closes_position(self) -> None:
        """Test that selling closes a position."""
        broker = IBBroker()
        broker.connect()

        broker.submit_market_order("SPY", "BUY", 100)
        broker.submit_market_order("SPY", "SELL", 100)

        positions = broker.get_positions()

        assert "SPY" not in positions

    def test_submit_order_invalid_side(self) -> None:
        """Test submitting order with invalid side."""
        broker = IBBroker()
        broker.connect()

        with pytest.raises(BrokerError, match="Invalid order side"):
            broker.submit_order("SPY", "INVALID", 100)

    def test_submit_order_invalid_quantity(self) -> None:
        """Test submitting order with invalid quantity."""
        broker = IBBroker()
        broker.connect()

        with pytest.raises(BrokerError, match="Invalid quantity"):
            broker.submit_order("SPY", "BUY", 0)

    def test_submit_order_readonly_mode(self) -> None:
        """Test that readonly mode prevents orders."""
        broker = IBBroker({"readonly": True})
        broker.connect()

        with pytest.raises(BrokerError, match="readonly mode"):
            broker.submit_market_order("SPY", "BUY", 100)

    def test_submit_order_disconnected(self) -> None:
        """Test submitting order when disconnected."""
        broker = IBBroker()

        with pytest.raises(BrokerError, match="Not connected"):
            broker.submit_market_order("SPY", "BUY", 100)

    def test_get_order_status(self) -> None:
        """Test getting order status."""
        broker = IBBroker()
        broker.connect()

        result = broker.submit_market_order("SPY", "BUY", 100)
        status = broker.get_order_status(result.order_id)

        assert status.order_id == result.order_id
        assert status.is_filled

    def test_get_order_status_not_found(self) -> None:
        """Test getting status of non-existent order."""
        broker = IBBroker()
        broker.connect()

        with pytest.raises(BrokerError, match="Order not found"):
            broker.get_order_status("nonexistent")

    def test_cancel_order_not_found(self) -> None:
        """Test cancelling non-existent order."""
        broker = IBBroker()
        broker.connect()

        with pytest.raises(BrokerError, match="Order not found"):
            broker.cancel_order("nonexistent")

    def test_submit_limit_order(self) -> None:
        """Test submitting a limit order."""
        broker = IBBroker()
        broker.connect()

        result = broker.submit_limit_order("SPY", "BUY", 100, limit_price=450.0)

        assert result.is_filled  # Stub fills immediately

    def test_submit_stop_order(self) -> None:
        """Test submitting a stop order."""
        broker = IBBroker()
        broker.connect()

        result = broker.submit_stop_order("SPY", "SELL", 100, stop_price=440.0)

        assert result.is_filled  # Stub fills immediately

    def test_get_historical_data(self) -> None:
        """Test getting historical data."""
        broker = IBBroker()
        broker.connect()

        data = broker.get_historical_data("SPY", duration="1 Y")

        assert isinstance(data, list)
        # Stub returns empty list

    def test_subscribe_market_data(self) -> None:
        """Test subscribing to market data."""
        broker = IBBroker()
        broker.connect()

        # Should not raise
        broker.subscribe_market_data(["SPY", "QQQ"])

    def test_multiple_positions(self) -> None:
        """Test managing multiple positions."""
        broker = IBBroker()
        broker.connect()

        broker.submit_market_order("SPY", "BUY", 100)
        broker.submit_market_order("QQQ", "BUY", 50)
        broker.submit_market_order("IWM", "SELL", 75)  # Short

        positions = broker.get_positions()

        assert len(positions) == 3
        assert positions["SPY"].quantity == 100
        assert positions["QQQ"].quantity == 50
        assert positions["IWM"].quantity == -75  # Short position

    def test_cash_updates_on_trade(self) -> None:
        """Test that cash balance updates on trades."""
        broker = IBBroker()
        broker.connect()

        initial_cash = broker.get_account_info().cash
        broker.submit_market_order("SPY", "BUY", 100)

        new_cash = broker.get_account_info().cash

        assert new_cash < initial_cash  # Cash reduced after buy
