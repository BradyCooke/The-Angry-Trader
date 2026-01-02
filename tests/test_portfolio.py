"""Tests for portfolio management."""

from datetime import date

import pytest

from src.portfolio.orders import Order, OrderBook, OrderSide, OrderStatus, OrderType
from src.portfolio.portfolio import Portfolio, Position, Trade
from src.signals.base import SignalType


class TestOrder:
    """Tests for Order class."""

    def test_order_creation(self) -> None:
        """Test basic order creation."""
        order = Order(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        assert order.symbol == "SPY"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.is_buy
        assert not order.is_sell
        assert order.is_pending
        assert not order.is_filled

    def test_order_fill(self) -> None:
        """Test order fill."""
        order = Order(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=100,
        )

        order.fill(price=450.0)

        assert order.is_filled
        assert order.filled_price == 450.0
        assert order.filled_quantity == 100
        assert order.filled_at is not None

    def test_order_cancel(self) -> None:
        """Test order cancellation."""
        order = Order(
            symbol="SPY",
            side=OrderSide.SELL,
            quantity=50,
        )

        order.cancel()

        assert order.status == OrderStatus.CANCELLED
        assert not order.is_pending

    def test_order_reject(self) -> None:
        """Test order rejection."""
        order = Order(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=100,
        )

        order.reject("Insufficient funds")

        assert order.status == OrderStatus.REJECTED
        assert order.metadata["rejection_reason"] == "Insufficient funds"


class TestOrderBook:
    """Tests for OrderBook class."""

    def test_add_and_get_orders(self) -> None:
        """Test adding and retrieving orders."""
        book = OrderBook()

        order1 = Order(symbol="SPY", side=OrderSide.BUY, quantity=100)
        order2 = Order(symbol="QQQ", side=OrderSide.SELL, quantity=50)

        book.add_order(order1)
        book.add_order(order2)

        assert len(book.orders) == 2
        assert len(book.get_pending_orders()) == 2

    def test_get_filled_orders(self) -> None:
        """Test getting filled orders."""
        book = OrderBook()

        order1 = Order(symbol="SPY", side=OrderSide.BUY, quantity=100)
        order2 = Order(symbol="QQQ", side=OrderSide.BUY, quantity=50)

        book.add_order(order1)
        book.add_order(order2)

        order1.fill(price=450.0)

        assert len(book.get_filled_orders()) == 1
        assert len(book.get_pending_orders()) == 1

    def test_cancel_pending_orders(self) -> None:
        """Test cancelling pending orders."""
        book = OrderBook()

        order1 = Order(symbol="SPY", side=OrderSide.BUY, quantity=100)
        order2 = Order(symbol="SPY", side=OrderSide.SELL, quantity=50)
        order3 = Order(symbol="QQQ", side=OrderSide.BUY, quantity=75)

        book.add_order(order1)
        book.add_order(order2)
        book.add_order(order3)

        # Cancel only SPY orders
        cancelled = book.cancel_pending_orders("SPY")

        assert cancelled == 2
        assert len(book.get_pending_orders()) == 1


class TestPosition:
    """Tests for Position class."""

    def test_position_creation(self) -> None:
        """Test position creation."""
        position = Position(
            symbol="SPY",
            side=SignalType.LONG,
            shares=100,
            entry_price=450.0,
            entry_date=date(2024, 1, 1),
            stop_price=440.0,
            atr_at_entry=5.0,
        )

        assert position.symbol == "SPY"
        assert position.is_long
        assert not position.is_short
        assert position.market_value == 45000.0

    def test_long_position_pnl(self) -> None:
        """Test P&L calculation for long position."""
        position = Position(
            symbol="SPY",
            side=SignalType.LONG,
            shares=100,
            entry_price=450.0,
            entry_date=date(2024, 1, 1),
            stop_price=440.0,
            atr_at_entry=5.0,
        )

        # Price up $10 = $1000 profit
        assert position.calculate_pnl(460.0) == 1000.0
        assert abs(position.calculate_pnl_pct(460.0) - 2.22) < 0.01

        # Price down $10 = $1000 loss
        assert position.calculate_pnl(440.0) == -1000.0

    def test_short_position_pnl(self) -> None:
        """Test P&L calculation for short position."""
        position = Position(
            symbol="SPY",
            side=SignalType.SHORT,
            shares=100,
            entry_price=450.0,
            entry_date=date(2024, 1, 1),
            stop_price=460.0,
            atr_at_entry=5.0,
        )

        # Price down $10 = $1000 profit (short)
        assert position.calculate_pnl(440.0) == 1000.0

        # Price up $10 = $1000 loss (short)
        assert position.calculate_pnl(460.0) == -1000.0

    def test_trailing_stop_long(self) -> None:
        """Test trailing stop update for long position."""
        position = Position(
            symbol="SPY",
            side=SignalType.LONG,
            shares=100,
            entry_price=100.0,
            entry_date=date(2024, 1, 1),
            stop_price=96.0,  # Initial stop 4 below entry
            atr_at_entry=2.0,
        )

        # Not enough profit yet
        updated = position.update_trailing_stop(
            current_high=101.0,
            current_low=99.0,
            current_close=100.5,
            current_atr=2.0,
            trailing_activation_atr=1.0,  # Need 2 point move
            trailing_stop_atr=2.0,
        )
        assert not updated
        assert not position.trailing_stop_active

        # Now profitable enough to activate
        updated = position.update_trailing_stop(
            current_high=105.0,
            current_low=102.0,
            current_close=104.0,  # 4 points profit > 2 (1 ATR)
            current_atr=2.0,
            trailing_activation_atr=1.0,
            trailing_stop_atr=2.0,
        )

        assert updated
        assert position.trailing_stop_active
        assert position.stop_price == 101.0  # 105 - 2*2 = 101


class TestPortfolio:
    """Tests for Portfolio class."""

    def test_portfolio_initialization(self) -> None:
        """Test portfolio initialization."""
        portfolio = Portfolio(starting_capital=100000)

        assert portfolio.cash == 100000
        assert portfolio.starting_capital == 100000
        assert portfolio.position_count == 0

    def test_open_position(self) -> None:
        """Test opening a position."""
        portfolio = Portfolio(starting_capital=100000)

        position = portfolio.open_position(
            symbol="SPY",
            side=SignalType.LONG,
            shares=100,
            entry_price=450.0,
            entry_date=date(2024, 1, 1),
            stop_price=440.0,
            atr=5.0,
        )

        assert portfolio.has_position("SPY")
        assert portfolio.position_count == 1
        assert portfolio.cash == 55000  # 100000 - 45000

    def test_close_position(self) -> None:
        """Test closing a position."""
        portfolio = Portfolio(starting_capital=100000)

        portfolio.open_position(
            symbol="SPY",
            side=SignalType.LONG,
            shares=100,
            entry_price=450.0,
            entry_date=date(2024, 1, 1),
            stop_price=440.0,
            atr=5.0,
        )

        trade = portfolio.close_position(
            symbol="SPY",
            exit_price=460.0,
            exit_date=date(2024, 1, 15),
            exit_reason="stop",
        )

        assert not portfolio.has_position("SPY")
        assert portfolio.position_count == 0
        assert portfolio.cash == 101000  # 55000 + 46000 proceeds
        assert trade.pnl == 1000.0
        assert len(portfolio.trades) == 1

    def test_calculate_equity(self) -> None:
        """Test equity calculation."""
        portfolio = Portfolio(starting_capital=100000)

        portfolio.open_position(
            symbol="SPY",
            side=SignalType.LONG,
            shares=100,
            entry_price=450.0,
            entry_date=date(2024, 1, 1),
            stop_price=440.0,
            atr=5.0,
        )

        # Price unchanged
        equity = portfolio.calculate_equity({"SPY": 450.0})
        assert equity == 100000

        # Price up
        equity = portfolio.calculate_equity({"SPY": 460.0})
        assert equity == 101000

    def test_calculate_exposure(self) -> None:
        """Test exposure calculations."""
        portfolio = Portfolio(starting_capital=100000)

        portfolio.open_position(
            symbol="SPY",
            side=SignalType.LONG,
            shares=100,
            entry_price=450.0,
            entry_date=date(2024, 1, 1),
            stop_price=440.0,
            atr=5.0,
        )

        portfolio.open_position(
            symbol="QQQ",
            side=SignalType.SHORT,
            shares=50,
            entry_price=400.0,
            entry_date=date(2024, 1, 1),
            stop_price=410.0,
            atr=4.0,
        )

        prices = {"SPY": 450.0, "QQQ": 400.0}

        # Gross = |45000| + |20000| = 65000
        gross = portfolio.calculate_gross_exposure(prices)
        assert gross == 65000.0

        # Net = 45000 - 20000 = 25000
        net = portfolio.calculate_net_exposure(prices)
        assert net == 25000.0

    def test_duplicate_position_raises(self) -> None:
        """Test that opening duplicate position raises error."""
        portfolio = Portfolio(starting_capital=100000)

        portfolio.open_position(
            symbol="SPY",
            side=SignalType.LONG,
            shares=100,
            entry_price=450.0,
            entry_date=date(2024, 1, 1),
            stop_price=440.0,
            atr=5.0,
        )

        with pytest.raises(ValueError, match="already exists"):
            portfolio.open_position(
                symbol="SPY",
                side=SignalType.SHORT,
                shares=50,
                entry_price=450.0,
                entry_date=date(2024, 1, 2),
                stop_price=460.0,
                atr=5.0,
            )

    def test_close_nonexistent_position_raises(self) -> None:
        """Test that closing nonexistent position raises error."""
        portfolio = Portfolio(starting_capital=100000)

        with pytest.raises(ValueError, match="No position"):
            portfolio.close_position(
                symbol="SPY",
                exit_price=450.0,
                exit_date=date(2024, 1, 1),
            )

    def test_position_weights(self) -> None:
        """Test position weight calculation."""
        portfolio = Portfolio(starting_capital=100000)

        portfolio.open_position(
            symbol="SPY",
            side=SignalType.LONG,
            shares=100,
            entry_price=400.0,  # $40,000 position
            entry_date=date(2024, 1, 1),
            stop_price=390.0,
            atr=5.0,
        )

        weights = portfolio.get_position_weights({"SPY": 400.0})

        # $40,000 / $100,000 = 40%
        assert abs(weights["SPY"] - 0.40) < 0.01

    def test_cash_interest_accrual(self) -> None:
        """Test cash interest accrual."""
        portfolio = Portfolio(
            starting_capital=100000,
            risk_free_rate=0.05,  # 5% annual
        )

        interest = portfolio.accrue_cash_interest(days=1)

        # Daily rate = 5% / 252 ≈ 0.0198%
        expected = 100000 * 0.05 / 252
        assert abs(interest - expected) < 0.01
        assert portfolio.cash > 100000
