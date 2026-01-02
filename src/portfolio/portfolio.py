"""Portfolio state tracking."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.signals.base import SignalType
from src.utils.logging import get_logger

logger = get_logger("portfolio")


@dataclass
class Position:
    """Open position in the portfolio."""
    symbol: str
    side: SignalType  # LONG or SHORT
    shares: int
    entry_price: float
    entry_date: date
    stop_price: float
    atr_at_entry: float
    highest_high: float = 0.0  # For trailing stop (longs)
    lowest_low: float = float("inf")  # For trailing stop (shorts)
    trailing_stop_active: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.side == SignalType.LONG:
            self.highest_high = self.entry_price
        else:
            self.lowest_low = self.entry_price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == SignalType.LONG

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == SignalType.SHORT

    @property
    def market_value(self) -> float:
        """Calculate current market value (always positive)."""
        return abs(self.shares * self.entry_price)

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L.

        Args:
            current_price: Current market price.

        Returns:
            P&L in dollars (positive = profit).
        """
        if self.is_long:
            return self.shares * (current_price - self.entry_price)
        else:
            return self.shares * (self.entry_price - current_price)

    def calculate_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage.

        Args:
            current_price: Current market price.

        Returns:
            P&L as percentage of entry value.
        """
        if self.entry_price == 0:
            return 0.0
        pnl = self.calculate_pnl(current_price)
        return (pnl / (self.shares * self.entry_price)) * 100

    def update_trailing_stop(
        self,
        current_high: float,
        current_low: float,
        current_close: float,
        current_atr: float,
        trailing_activation_atr: float,
        trailing_stop_atr: float,
    ) -> bool:
        """Update trailing stop if conditions are met.

        Args:
            current_high: Current bar high.
            current_low: Current bar low.
            current_close: Current bar close.
            current_atr: Current ATR value.
            trailing_activation_atr: ATR multiple for activation.
            trailing_stop_atr: ATR multiple for stop distance.

        Returns:
            True if stop was updated.
        """
        if self.is_long:
            # Update highest high
            self.highest_high = max(self.highest_high, current_high)

            # Check if profit threshold reached for activation
            profit_threshold = self.entry_price + (trailing_activation_atr * current_atr)
            if current_close >= profit_threshold:
                self.trailing_stop_active = True

            if self.trailing_stop_active:
                # Calculate new trailing stop
                new_stop = self.highest_high - (trailing_stop_atr * current_atr)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    return True

        else:  # SHORT
            # Update lowest low
            self.lowest_low = min(self.lowest_low, current_low)

            # Check if profit threshold reached for activation
            profit_threshold = self.entry_price - (trailing_activation_atr * current_atr)
            if current_close <= profit_threshold:
                self.trailing_stop_active = True

            if self.trailing_stop_active:
                # Calculate new trailing stop
                new_stop = self.lowest_low + (trailing_stop_atr * current_atr)
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
                    return True

        return False


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    side: SignalType
    shares: int
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    exit_reason: str
    pnl: float
    pnl_pct: float
    holding_days: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Portfolio:
    """Portfolio state manager."""

    def __init__(
        self,
        starting_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
    ):
        """Initialize portfolio.

        Args:
            starting_capital: Initial cash.
            risk_free_rate: Annual risk-free rate for cash.
        """
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.risk_free_rate = risk_free_rate
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_history: list[tuple[date, float]] = []

    @property
    def position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    @property
    def long_positions(self) -> list[Position]:
        """Get all long positions."""
        return [p for p in self.positions.values() if p.is_long]

    @property
    def short_positions(self) -> list[Position]:
        """Get all short positions."""
        return [p for p in self.positions.values() if p.is_short]

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol."""
        return symbol in self.positions

    def calculate_equity(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio equity.

        Args:
            prices: Current prices for each symbol.

        Returns:
            Total equity (cash + positions).
        """
        positions_value = 0.0
        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.entry_price)
            positions_value += position.calculate_pnl(price) + position.market_value

        return self.cash + positions_value

    def calculate_gross_exposure(self, prices: dict[str, float]) -> float:
        """Calculate gross exposure (sum of absolute position values).

        Args:
            prices: Current prices for each symbol.

        Returns:
            Gross exposure in dollars.
        """
        exposure = 0.0
        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.entry_price)
            exposure += abs(position.shares * price)
        return exposure

    def calculate_gross_exposure_pct(self, prices: dict[str, float]) -> float:
        """Calculate gross exposure as percentage of equity.

        Args:
            prices: Current prices for each symbol.

        Returns:
            Gross exposure percentage.
        """
        equity = self.calculate_equity(prices)
        if equity <= 0:
            return 0.0
        return (self.calculate_gross_exposure(prices) / equity) * 100

    def calculate_net_exposure(self, prices: dict[str, float]) -> float:
        """Calculate net exposure (long - short).

        Args:
            prices: Current prices for each symbol.

        Returns:
            Net exposure in dollars.
        """
        long_exposure = sum(
            p.shares * prices.get(p.symbol, p.entry_price)
            for p in self.long_positions
        )
        short_exposure = sum(
            p.shares * prices.get(p.symbol, p.entry_price)
            for p in self.short_positions
        )
        return long_exposure - short_exposure

    def open_position(
        self,
        symbol: str,
        side: SignalType,
        shares: int,
        entry_price: float,
        entry_date: date,
        stop_price: float,
        atr: float,
        metadata: dict[str, Any] | None = None,
    ) -> Position:
        """Open a new position.

        Args:
            symbol: ETF symbol.
            side: LONG or SHORT.
            shares: Number of shares.
            entry_price: Entry price.
            entry_date: Entry date.
            stop_price: Initial stop loss price.
            atr: ATR at entry.
            metadata: Optional metadata.

        Returns:
            The new Position object.
        """
        if symbol in self.positions:
            raise ValueError(f"Position already exists for {symbol}")

        position = Position(
            symbol=symbol,
            side=side,
            shares=shares,
            entry_price=entry_price,
            entry_date=entry_date,
            stop_price=stop_price,
            atr_at_entry=atr,
            metadata=metadata or {},
        )

        # Adjust cash
        cost = shares * entry_price
        self.cash -= cost

        self.positions[symbol] = position
        logger.info(f"Opened {side.value} position: {shares} {symbol} @ {entry_price:.2f}")

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: date,
        exit_reason: str = "signal",
    ) -> Trade:
        """Close a position.

        Args:
            symbol: ETF symbol.
            exit_price: Exit price.
            exit_date: Exit date.
            exit_reason: Reason for exit.

        Returns:
            Trade record.
        """
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")

        position = self.positions[symbol]

        # Calculate P&L
        pnl = position.calculate_pnl(exit_price)
        pnl_pct = position.calculate_pnl_pct(exit_price)

        # Create trade record
        trade = Trade(
            symbol=symbol,
            side=position.side,
            shares=position.shares,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=(exit_date - position.entry_date).days,
            metadata=position.metadata,
        )

        # Adjust cash
        proceeds = position.shares * exit_price
        self.cash += proceeds

        # Remove position
        del self.positions[symbol]
        self.trades.append(trade)

        logger.info(
            f"Closed {position.side.value} position: {symbol} @ {exit_price:.2f} "
            f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) [{exit_reason}]"
        )

        return trade

    def record_equity(self, current_date: date, prices: dict[str, float]) -> float:
        """Record equity for the current date.

        Args:
            current_date: Current date.
            prices: Current prices.

        Returns:
            Current equity.
        """
        equity = self.calculate_equity(prices)
        self.equity_history.append((current_date, equity))
        return equity

    def accrue_cash_interest(self, days: int = 1) -> float:
        """Accrue interest on cash holdings.

        Args:
            days: Number of days to accrue.

        Returns:
            Interest earned.
        """
        if self.risk_free_rate <= 0 or self.cash <= 0:
            return 0.0

        daily_rate = self.risk_free_rate / 252
        interest = self.cash * daily_rate * days
        self.cash += interest
        return interest

    def get_position_weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Get position weights as fractions of equity.

        Args:
            prices: Current prices.

        Returns:
            Dictionary mapping symbol to weight (0-1).
        """
        equity = self.calculate_equity(prices)
        if equity <= 0:
            return {}

        weights = {}
        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.entry_price)
            weights[symbol] = abs(position.shares * price) / equity

        return weights
