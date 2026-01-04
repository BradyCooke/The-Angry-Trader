#!/usr/bin/env python3
"""
ChatGPT Variant Backtest

Implements ChatGPT's suggestions:
1. Fix short mechanics (cash-secured short model)
2. Make VaR weights signed
3. Next-open execution (remove lookahead)
4. Size positions off stop distance
5. Allow reversals (flip long<->short)
6. Add 200-day trend filter
7. Trailing stop active immediately (activation = 0)
8. Time stop (40 days without progress)
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.data.database import Database
from src.signals.base import Signal, SignalType
from src.signals.indicators import calculate_atr, calculate_ema, calculate_keltner_channels
from src.utils.logging import get_logger

logger = get_logger("chatgpt_variant")


# ============================================================================
# MODIFIED POSITION CLASS
# ============================================================================

@dataclass
class Position:
    """Open position with ChatGPT modifications."""
    symbol: str
    side: SignalType
    shares: int
    entry_price: float
    entry_date: date
    stop_price: float
    atr_at_entry: float
    highest_high: float = 0.0
    lowest_low: float = float("inf")
    trailing_stop_active: bool = True  # CHANGE: Active immediately
    days_since_new_extreme: int = 0  # CHANGE: For time stop
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.side == SignalType.LONG:
            self.highest_high = self.entry_price
        else:
            self.lowest_low = self.entry_price

    @property
    def is_long(self) -> bool:
        return self.side == SignalType.LONG

    @property
    def is_short(self) -> bool:
        return self.side == SignalType.SHORT

    def calculate_pnl(self, current_price: float) -> float:
        if self.is_long:
            return self.shares * (current_price - self.entry_price)
        else:
            return self.shares * (self.entry_price - current_price)

    def update_trailing_stop(
        self,
        current_high: float,
        current_low: float,
        current_atr: float,
        trailing_stop_atr: float,
    ) -> bool:
        """Update trailing stop - active immediately (ChatGPT suggestion #6)."""
        updated = False

        if self.is_long:
            if current_high > self.highest_high:
                self.highest_high = current_high
                self.days_since_new_extreme = 0
            else:
                self.days_since_new_extreme += 1

            new_stop = self.highest_high - (trailing_stop_atr * current_atr)
            if new_stop > self.stop_price:
                self.stop_price = new_stop
                updated = True
        else:
            if current_low < self.lowest_low:
                self.lowest_low = current_low
                self.days_since_new_extreme = 0
            else:
                self.days_since_new_extreme += 1

            new_stop = self.lowest_low + (trailing_stop_atr * current_atr)
            if new_stop < self.stop_price:
                self.stop_price = new_stop
                updated = True

        return updated


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


# ============================================================================
# MODIFIED PORTFOLIO CLASS
# ============================================================================

class Portfolio:
    """Portfolio with ChatGPT modifications."""

    def __init__(self, starting_capital: float = 100000.0):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []

    @property
    def position_count(self) -> int:
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def calculate_equity(self, prices: dict[str, float]) -> float:
        equity = self.cash
        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.entry_price)
            equity += position.calculate_pnl(price)
            # Add back the collateral for the position
            equity += position.shares * position.entry_price
        return equity

    def calculate_gross_exposure_pct(self, prices: dict[str, float]) -> float:
        equity = self.calculate_equity(prices)
        if equity <= 0:
            return 0.0
        exposure = sum(
            abs(pos.shares * prices.get(sym, pos.entry_price))
            for sym, pos in self.positions.items()
        )
        return (exposure / equity) * 100

    def get_position_weights_signed(self, prices: dict[str, float]) -> dict[str, float]:
        """Get SIGNED position weights (ChatGPT suggestion #1)."""
        equity = self.calculate_equity(prices)
        if equity <= 0:
            return {}
        weights = {}
        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.entry_price)
            weight = (position.shares * price) / equity
            if position.is_short:
                weight *= -1  # CHANGE: Signed weights for shorts
            weights[symbol] = weight
        return weights

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

        # Cash-secured: deduct collateral
        cost = shares * entry_price
        self.cash -= cost
        self.positions[symbol] = position
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: date,
        exit_reason: str = "signal",
    ) -> Trade:
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")

        position = self.positions[symbol]
        pnl = position.calculate_pnl(exit_price)
        entry_value = position.shares * position.entry_price
        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

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

        # ChatGPT suggestion #1: Fix short mechanics
        if position.is_long:
            proceeds = position.shares * exit_price
            self.cash += proceeds
        else:
            # Cash-secured short: release collateral + realize PnL
            collateral = position.shares * position.entry_price
            self.cash += collateral + pnl

        del self.positions[symbol]
        self.trades.append(trade)
        return trade


# ============================================================================
# MODIFIED SIGNAL GENERATOR
# ============================================================================

class ChatGPTKeltnerSignalGenerator:
    """Keltner signal generator with ChatGPT modifications."""

    def __init__(
        self,
        ema_period: int = 35,
        atr_period: int = 20,
        atr_multiplier: float = 2.0,
        initial_stop_atr: float = 2.5,
        trailing_stop_atr: float = 2.5,
        use_trend_filter: bool = True,  # ChatGPT #5
        trend_filter_period: int = 200,
    ):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.initial_stop_atr = initial_stop_atr
        self.trailing_stop_atr = trailing_stop_atr
        self.use_trend_filter = use_trend_filter
        self.trend_filter_period = trend_filter_period

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        middle, upper, lower = calculate_keltner_channels(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            ema_period=self.ema_period,
            atr_period=self.atr_period,
            atr_multiplier=self.atr_multiplier,
        )
        df["keltner_middle"] = middle
        df["keltner_upper"] = upper
        df["keltner_lower"] = lower
        df["atr"] = calculate_atr(df["high"], df["low"], df["close"], period=self.atr_period)

        # ChatGPT suggestion #5: Add 200-day trend filter
        if self.use_trend_filter:
            df["ema_200"] = calculate_ema(df["close"], self.trend_filter_period)

        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        """Generate signals with next-open execution (ChatGPT suggestion #2)."""
        if len(df) < max(self.ema_period, self.atr_period, self.trend_filter_period):
            return []

        df = self.calculate_indicators(df)
        signals = []
        inside_band = True

        for i in range(1, len(df) - 1):  # Stop one before end for next-open
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            next_row = df.iloc[i + 1]  # For next-open execution

            if pd.isna(row["keltner_upper"]) or pd.isna(row["atr"]):
                continue

            close = row["close"]
            upper = row["keltner_upper"]
            lower = row["keltner_lower"]
            atr = row["atr"]
            prev_close = prev_row["close"]
            prev_upper = prev_row["keltner_upper"]
            prev_lower = prev_row["keltner_lower"]

            # Check trend filter
            trend_ok_long = True
            trend_ok_short = True
            if self.use_trend_filter and "ema_200" in row.index and pd.notna(row["ema_200"]):
                trend_ok_long = close > row["ema_200"]
                trend_ok_short = close < row["ema_200"]

            if lower <= close <= upper:
                inside_band = True

            # ChatGPT suggestion #2: Execute at next day's open
            next_date = df.index[i + 1]
            if isinstance(next_date, pd.Timestamp):
                next_date = next_date.date()
            next_open = next_row["open"]
            next_atr = next_row["atr"] if pd.notna(next_row["atr"]) else atr

            # Long breakout with trend filter
            if close > upper and prev_close <= prev_upper and inside_band and trend_ok_long:
                strength = (close - upper) / atr if atr > 0 else 0
                entry_price = next_open  # Execute at next open
                stop_price = entry_price - (self.initial_stop_atr * next_atr)

                signal = Signal(
                    date=next_date,
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    price=entry_price,
                    strength=strength,
                    atr=next_atr,
                    stop_price=stop_price,
                    metadata={"breakout_close": close, "breakout_upper": upper}
                )
                signals.append(signal)
                inside_band = False

            # Short breakout with trend filter
            elif close < lower and prev_close >= prev_lower and inside_band and trend_ok_short:
                strength = (lower - close) / atr if atr > 0 else 0
                entry_price = next_open
                stop_price = entry_price + (self.initial_stop_atr * next_atr)

                signal = Signal(
                    date=next_date,
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    price=entry_price,
                    strength=strength,
                    atr=next_atr,
                    stop_price=stop_price,
                    metadata={"breakout_close": close, "breakout_lower": lower}
                )
                signals.append(signal)
                inside_band = False

        return signals


# ============================================================================
# MODIFIED POSITION SIZER
# ============================================================================

class ChatGPTPositionSizer:
    """Position sizer using stop distance (ChatGPT suggestion #3)."""

    def __init__(
        self,
        volatility_target_pct: float = 2.0,
        max_position_pct: float = 15.0,
        max_gross_exposure_pct: float = 100.0,
    ):
        self.volatility_target_pct = volatility_target_pct
        self.max_position_pct = max_position_pct
        self.max_gross_exposure_pct = max_gross_exposure_pct

    def calculate_position_size(
        self,
        equity: float,
        price: float,
        stop_distance: float,  # Use stop distance, not ATR
        current_exposure_pct: float = 0.0,
    ) -> int:
        """Size position based on stop distance (ChatGPT suggestion #3)."""
        if price <= 0 or stop_distance <= 0 or equity <= 0:
            return 0

        # Target dollar risk = equity * volatility_target_pct
        target_dollar_risk = equity * (self.volatility_target_pct / 100)

        # Shares = target_risk / stop_distance
        shares = int(target_dollar_risk / stop_distance)
        dollar_value = shares * price

        # Apply max position limit
        max_position_value = equity * (self.max_position_pct / 100)
        if dollar_value > max_position_value:
            shares = int(max_position_value / price)
            dollar_value = shares * price

        # Check gross exposure
        available_exposure = (self.max_gross_exposure_pct - current_exposure_pct) / 100
        max_available_value = equity * available_exposure
        if dollar_value > max_available_value:
            shares = int(max_available_value / price)

        return max(0, shares)


# ============================================================================
# MODIFIED BACKTEST ENGINE
# ============================================================================

class ChatGPTBacktestEngine:
    """Backtest engine with ChatGPT modifications."""

    def __init__(
        self,
        ema_period: int = 35,
        atr_period: int = 20,
        atr_multiplier: float = 2.0,
        initial_stop_atr: float = 2.5,
        trailing_stop_atr: float = 2.5,
        volatility_target_pct: float = 2.0,
        max_position_pct: float = 15.0,
        max_gross_exposure_pct: float = 100.0,
        use_trend_filter: bool = True,
        allow_reversals: bool = True,  # ChatGPT #4
        time_stop_days: int = 40,  # ChatGPT #7
        starting_capital: float = 100000.0,
    ):
        self.signal_gen = ChatGPTKeltnerSignalGenerator(
            ema_period=ema_period,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            initial_stop_atr=initial_stop_atr,
            trailing_stop_atr=trailing_stop_atr,
            use_trend_filter=use_trend_filter,
        )
        self.position_sizer = ChatGPTPositionSizer(
            volatility_target_pct=volatility_target_pct,
            max_position_pct=max_position_pct,
            max_gross_exposure_pct=max_gross_exposure_pct,
        )
        self.trailing_stop_atr = trailing_stop_atr
        self.allow_reversals = allow_reversals
        self.time_stop_days = time_stop_days
        self.starting_capital = starting_capital

    def run(
        self,
        data: dict[str, pd.DataFrame],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[pd.Series, list[Trade], dict]:
        """Run backtest."""
        portfolio = Portfolio(self.starting_capital)

        # Get all trading dates
        all_dates = set()
        for df in data.values():
            for d in df.index:
                if isinstance(d, pd.Timestamp):
                    d = d.date()
                all_dates.add(d)
        dates = sorted(all_dates)

        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        # Pre-calculate indicators
        indicator_data = {}
        for symbol, df in data.items():
            indicator_data[symbol] = self.signal_gen.calculate_indicators(df)

        # Pre-compute signals
        signals_by_date: dict[date, list[Signal]] = {}
        for symbol, df in indicator_data.items():
            signals = self.signal_gen.generate_signals(df, symbol)
            for signal in signals:
                if signal.date not in signals_by_date:
                    signals_by_date[signal.date] = []
                signals_by_date[signal.date].append(signal)

        for d in signals_by_date:
            signals_by_date[d].sort(key=lambda s: s.strength, reverse=True)

        # Main loop
        equity_history = []

        for current_date in dates:
            prices = self._get_prices(data, current_date)
            if not prices:
                continue

            # Process exits (stops and time stops)
            self._process_exits(portfolio, data, indicator_data, current_date, prices)

            # Get signals
            signals = signals_by_date.get(current_date, [])

            # Process reversals and entries (ChatGPT #4)
            equity = portfolio.calculate_equity(prices)
            exposure_pct = portfolio.calculate_gross_exposure_pct(prices)

            for signal in signals:
                # Check for reversal
                if self.allow_reversals and portfolio.has_position(signal.symbol):
                    pos = portfolio.get_position(signal.symbol)
                    if pos.side != signal.signal_type:
                        # Close and reverse
                        portfolio.close_position(signal.symbol, signal.price, current_date, "reverse")
                        exposure_pct = portfolio.calculate_gross_exposure_pct(prices)

                # Skip if already have position
                if portfolio.has_position(signal.symbol):
                    continue

                # Calculate position size using stop distance
                stop_distance = abs(signal.price - signal.stop_price)
                shares = self.position_sizer.calculate_position_size(
                    equity=equity,
                    price=signal.price,
                    stop_distance=stop_distance,
                    current_exposure_pct=exposure_pct,
                )

                if shares <= 0:
                    continue

                cost = shares * signal.price
                if cost > portfolio.cash:
                    continue

                try:
                    portfolio.open_position(
                        symbol=signal.symbol,
                        side=signal.signal_type,
                        shares=shares,
                        entry_price=signal.price,
                        entry_date=current_date,
                        stop_price=signal.stop_price,
                        atr=signal.atr,
                        metadata=signal.metadata,
                    )
                    exposure_pct = portfolio.calculate_gross_exposure_pct(prices)
                except ValueError:
                    pass

            equity_history.append((current_date, portfolio.calculate_equity(prices)))

        equity_curve = pd.Series({d: e for d, e in equity_history}, name="equity")
        return equity_curve, portfolio.trades, {}

    def _get_prices(self, data: dict[str, pd.DataFrame], current_date: date) -> dict[str, float]:
        prices = {}
        for symbol, df in data.items():
            if current_date in df.index:
                prices[symbol] = df.loc[current_date, "close"]
            elif isinstance(df.index, pd.DatetimeIndex):
                ts = pd.Timestamp(current_date)
                if ts in df.index:
                    prices[symbol] = df.loc[ts, "close"]
        return prices

    def _get_bar(self, data: dict[str, pd.DataFrame], symbol: str, current_date: date) -> pd.Series | None:
        df = data.get(symbol)
        if df is None:
            return None
        if current_date in df.index:
            return df.loc[current_date]
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = pd.Timestamp(current_date)
            if ts in df.index:
                return df.loc[ts]
        return None

    def _process_exits(
        self,
        portfolio: Portfolio,
        data: dict[str, pd.DataFrame],
        indicator_data: dict[str, pd.DataFrame],
        current_date: date,
        prices: dict[str, float],
    ) -> None:
        symbols_to_close = []

        for symbol, position in list(portfolio.positions.items()):
            bar = self._get_bar(data, symbol, current_date)
            if bar is None:
                continue

            high = bar["high"]
            low = bar["low"]
            open_price = bar["open"]

            # Check stop hit
            if position.is_long:
                if low <= position.stop_price:
                    exit_price = min(position.stop_price, open_price)
                    symbols_to_close.append((symbol, exit_price, "stop"))
                    continue
            else:
                if high >= position.stop_price:
                    exit_price = max(position.stop_price, open_price)
                    symbols_to_close.append((symbol, exit_price, "stop"))
                    continue

            # ChatGPT suggestion #7: Time stop
            if position.days_since_new_extreme >= self.time_stop_days:
                ind_bar = self._get_bar(indicator_data, symbol, current_date)
                if ind_bar is not None and "atr" in ind_bar.index:
                    atr = ind_bar["atr"]
                    current_price = prices.get(symbol, position.entry_price)

                    # Check if made at least 1 ATR progress
                    if position.is_long:
                        required = position.entry_price + atr
                        if current_price < required:
                            symbols_to_close.append((symbol, current_price, "time_stop"))
                            continue
                    else:
                        required = position.entry_price - atr
                        if current_price > required:
                            symbols_to_close.append((symbol, current_price, "time_stop"))
                            continue

            # Update trailing stop
            ind_bar = self._get_bar(indicator_data, symbol, current_date)
            if ind_bar is not None and "atr" in ind_bar.index:
                position.update_trailing_stop(
                    current_high=high,
                    current_low=low,
                    current_atr=ind_bar["atr"],
                    trailing_stop_atr=self.trailing_stop_atr,
                )

        for symbol, exit_price, reason in symbols_to_close:
            portfolio.close_position(symbol, exit_price, current_date, reason)


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(equity_curve: pd.Series, trades: list[Trade], start_date: date, end_date: date):
    """Calculate all required metrics."""
    if len(equity_curve) < 2:
        return None

    # Returns
    returns = equity_curve.pct_change().dropna()

    # Total return and CAGR
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Volatility
    volatility = returns.std() * np.sqrt(252)

    # Drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = abs(drawdown.min())

    # Sharpe
    sharpe = (cagr / volatility) if volatility > 0 else 0

    # Sortino
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (cagr / downside_std) if downside_std > 0 else 0

    # Calmar
    calmar = (cagr / max_drawdown) if max_drawdown > 0 else 0

    # Trade statistics
    total_trades = len(trades)
    if total_trades > 0:
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]
        win_rate = len(wins) / total_trades * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    return {
        'cagr': cagr * 100,
        'total_return': total_return * 100,
        'volatility': volatility * 100,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'total_trades': total_trades,
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = load_config("config.yaml")
    db = Database(config.data.database_path)

    # Load data
    data = {}
    for symbol in config.data.etf_symbols:
        df = db.get_price_data(symbol)
        if df is not None and len(df) > 0:
            data[symbol] = df

    # Define periods
    FULL_START = date(2007, 1, 1)
    FULL_END = date(2025, 12, 31)
    IS_END = date(2021, 6, 30)
    OOS_START = date(2021, 7, 1)

    # Create ChatGPT variant engine
    engine = ChatGPTBacktestEngine(
        ema_period=35,
        atr_period=20,
        atr_multiplier=2.0,
        initial_stop_atr=2.5,
        trailing_stop_atr=2.5,
        volatility_target_pct=2.0,
        max_position_pct=15.0,
        max_gross_exposure_pct=100.0,
        use_trend_filter=True,  # ChatGPT #5
        allow_reversals=True,   # ChatGPT #4
        time_stop_days=40,      # ChatGPT #7
        starting_capital=100000.0,
    )

    # Run full period
    eq_full, trades_full, _ = engine.run(data, FULL_START, FULL_END)
    m_full = calculate_metrics(eq_full, trades_full, FULL_START, FULL_END)

    # Run in-sample
    eq_is, trades_is, _ = engine.run(data, FULL_START, IS_END)
    m_is = calculate_metrics(eq_is, trades_is, FULL_START, IS_END)

    # Run out-of-sample
    eq_oos, trades_oos, _ = engine.run(data, OOS_START, FULL_END)
    m_oos = calculate_metrics(eq_oos, trades_oos, OOS_START, FULL_END)

    # Calculate OOS/IS ratio
    oos_is_ratio = m_oos['cagr'] / m_is['cagr'] if m_is['cagr'] > 0 else 0

    print("=" * 80)
    print("CHATGPT VARIANT RESULTS")
    print("=" * 80)
    print("\nKey Changes from Baseline:")
    print("  1. Fixed short mechanics (cash-secured model)")
    print("  2. Next-open execution (removed lookahead)")
    print("  3. Position sizing based on stop distance")
    print("  4. Allow reversals on opposite signals")
    print("  5. Added 200-day trend filter")
    print("  6. Trailing stop active immediately")
    print("  7. Time stop after 40 days without progress")
    print()
    print("RETURN METRICS:")
    print(f"  CAGR (Full):        {m_full['cagr']:.2f}%")
    print(f"  CAGR (In-Sample):   {m_is['cagr']:.2f}%")
    print(f"  CAGR (OOS):         {m_oos['cagr']:.2f}%")
    print()
    print("RISK METRICS:")
    print(f"  Max Drawdown:       {m_full['max_drawdown_pct']:.2f}%")
    print(f"  Volatility:         {m_full['volatility']:.2f}%")
    print()
    print("RISK-ADJUSTED METRICS:")
    print(f"  Sharpe Ratio:       {m_full['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:      {m_full['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:       {m_full['calmar_ratio']:.2f}")
    print()
    print("TRADE STATISTICS:")
    print(f"  Total Trades:       {m_full['total_trades']}")
    print(f"  Win Rate:           {m_full['win_rate_pct']:.1f}%")
    print(f"  Profit Factor:      {m_full['profit_factor']:.2f}")
    print(f"  Avg Win:            ${m_full['avg_win']:,.0f}")
    print(f"  Avg Loss:           ${m_full['avg_loss']:,.0f}")
    print()
    print("ROBUSTNESS:")
    print(f"  OOS/IS CAGR Ratio:  {oos_is_ratio:.2f}")
    print()

    # Check disqualification
    dq_reasons = []
    if m_full['max_drawdown_pct'] > 40:
        dq_reasons.append(f"Max DD {m_full['max_drawdown_pct']:.1f}% > 40%")
    if oos_is_ratio < 0.5:
        dq_reasons.append(f"OOS/IS ratio {oos_is_ratio:.2f} < 0.5 (overfit)")
    if m_full['profit_factor'] < 1.2:
        dq_reasons.append(f"Profit Factor {m_full['profit_factor']:.2f} < 1.2")
    if m_full['total_trades'] < 100:
        dq_reasons.append(f"Trades {m_full['total_trades']} < 100")
    if m_full['win_rate_pct'] > 70:
        dq_reasons.append(f"Win Rate {m_full['win_rate_pct']:.1f}% > 70%")

    if dq_reasons:
        print("DISQUALIFICATION CHECK:")
        for reason in dq_reasons:
            print(f"  ❌ {reason}")
        print("\n  SCORE: DISQUALIFIED")
    else:
        score = (
            (m_full['calmar_ratio'] * 35) +
            (m_full['sharpe_ratio'] * 25) +
            (oos_is_ratio * 20) +
            (m_full['profit_factor'] * 10) +
            (m_full['sortino_ratio'] * 10)
        )
        print("DISQUALIFICATION CHECK: ✓ All criteria passed")
        print(f"\nCOMPOSITE SCORE: {score:.2f}")

    db.close()


if __name__ == "__main__":
    main()
