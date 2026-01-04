#!/usr/bin/env python3
"""
Chutes Variant Backtest

Implements Chutes' suggestions:
1. Tighten Initial Stops (2.5 -> 2.0 ATR)
2. Profit-Taking Rule at 3× ATR
3. Dynamic Volatility Target Scaling (reduce with more positions)
4. Asymmetric Trailing Stops (1.5 for longs, 3.0 for shorts)
5. Reduce VaR parameters (optional - not implementing VaR changes)
"""

from datetime import date
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.data.database import Database
from src.signals.base import Signal, SignalType
from src.signals.indicators import calculate_atr, calculate_keltner_channels
from src.backtest.metrics import calculate_returns


@dataclass
class Position:
    symbol: str
    side: SignalType
    shares: int
    entry_price: float
    entry_date: date
    stop_price: float
    atr_at_entry: float
    highest_high: float = 0.0
    lowest_low: float = float("inf")
    trailing_stop_active: bool = False
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
        current_close: float,
        current_atr: float,
        trailing_activation_atr: float,
        trailing_stop_atr_long: float,  # Chutes: asymmetric
        trailing_stop_atr_short: float,
    ) -> bool:
        """Update trailing stop with asymmetric ATR multipliers."""
        if self.is_long:
            self.highest_high = max(self.highest_high, current_high)
            profit_threshold = self.entry_price + (trailing_activation_atr * current_atr)
            if current_close >= profit_threshold:
                self.trailing_stop_active = True
            if self.trailing_stop_active:
                new_stop = self.highest_high - (trailing_stop_atr_long * current_atr)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    return True
        else:
            self.lowest_low = min(self.lowest_low, current_low)
            profit_threshold = self.entry_price - (trailing_activation_atr * current_atr)
            if current_close <= profit_threshold:
                self.trailing_stop_active = True
            if self.trailing_stop_active:
                new_stop = self.lowest_low + (trailing_stop_atr_short * current_atr)
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
                    return True
        return False

    def check_profit_target(self, current_price: float, current_atr: float, profit_target_atr: float) -> bool:
        """Check if profit target is hit (Chutes suggestion #2)."""
        if self.is_long:
            target = self.entry_price + (profit_target_atr * current_atr)
            return current_price >= target
        else:
            target = self.entry_price - (profit_target_atr * current_atr)
            return current_price <= target


@dataclass
class Trade:
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
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            equity += pos.shares * price
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

    def open_position(self, symbol, side, shares, entry_price, entry_date, stop_price, atr, metadata=None):
        if symbol in self.positions:
            raise ValueError(f"Position already exists for {symbol}")
        pos = Position(
            symbol=symbol, side=side, shares=shares, entry_price=entry_price,
            entry_date=entry_date, stop_price=stop_price, atr_at_entry=atr,
            metadata=metadata or {}
        )
        self.cash -= shares * entry_price
        self.positions[symbol] = pos
        return pos

    def close_position(self, symbol, exit_price, exit_date, exit_reason="signal"):
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")
        pos = self.positions[symbol]
        pnl = pos.calculate_pnl(exit_price)
        entry_value = pos.shares * pos.entry_price
        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

        trade = Trade(
            symbol=symbol, side=pos.side, shares=pos.shares,
            entry_date=pos.entry_date, entry_price=pos.entry_price,
            exit_date=exit_date, exit_price=exit_price, exit_reason=exit_reason,
            pnl=pnl, pnl_pct=pnl_pct,
            holding_days=(exit_date - pos.entry_date).days, metadata=pos.metadata
        )

        self.cash += pos.shares * exit_price
        del self.positions[symbol]
        self.trades.append(trade)
        return trade


class ChutesSignalGenerator:
    def __init__(
        self,
        ema_period: int = 35,
        atr_period: int = 20,
        atr_multiplier: float = 2.0,
        initial_stop_atr: float = 2.0,  # Chutes #1: tighter (was 2.5)
        trailing_stop_atr_long: float = 1.5,  # Chutes #6: asymmetric
        trailing_stop_atr_short: float = 3.0,  # Chutes #6: asymmetric
        trailing_activation_atr: float = 1.0,
        profit_target_atr: float = 3.0,  # Chutes #2: profit taking
    ):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.initial_stop_atr = initial_stop_atr
        self.trailing_stop_atr_long = trailing_stop_atr_long
        self.trailing_stop_atr_short = trailing_stop_atr_short
        self.trailing_activation_atr = trailing_activation_atr
        self.profit_target_atr = profit_target_atr

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        middle, upper, lower = calculate_keltner_channels(
            df["high"], df["low"], df["close"],
            self.ema_period, self.atr_period, self.atr_multiplier
        )
        df["keltner_middle"] = middle
        df["keltner_upper"] = upper
        df["keltner_lower"] = lower
        df["atr"] = calculate_atr(df["high"], df["low"], df["close"], self.atr_period)
        return df

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        if len(df) < max(self.ema_period, self.atr_period):
            return []
        df = self.calculate_indicators(df)
        signals = []
        inside_band = True

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            if pd.isna(row["keltner_upper"]) or pd.isna(row["atr"]):
                continue

            current_date = df.index[i]
            if isinstance(current_date, pd.Timestamp):
                current_date = current_date.date()

            close = row["close"]
            upper = row["keltner_upper"]
            lower = row["keltner_lower"]
            atr = row["atr"]
            prev_close = prev_row["close"]
            prev_upper = prev_row["keltner_upper"]
            prev_lower = prev_row["keltner_lower"]

            if lower <= close <= upper:
                inside_band = True

            if close > upper and prev_close <= prev_upper and inside_band:
                strength = (close - upper) / atr if atr > 0 else 0
                stop_price = close - (self.initial_stop_atr * atr)
                signals.append(Signal(
                    date=current_date, symbol=symbol, signal_type=SignalType.LONG,
                    price=close, strength=strength, atr=atr, stop_price=stop_price,
                    metadata={}
                ))
                inside_band = False

            elif close < lower and prev_close >= prev_lower and inside_band:
                strength = (lower - close) / atr if atr > 0 else 0
                stop_price = close + (self.initial_stop_atr * atr)
                signals.append(Signal(
                    date=current_date, symbol=symbol, signal_type=SignalType.SHORT,
                    price=close, strength=strength, atr=atr, stop_price=stop_price,
                    metadata={}
                ))
                inside_band = False

        return signals


class ChutesBacktestEngine:
    def __init__(
        self,
        signal_gen: ChutesSignalGenerator,
        base_volatility_target_pct: float = 2.0,
        max_position_pct: float = 15.0,
        max_gross_exposure_pct: float = 100.0,
        max_positions_for_scaling: int = 20,  # Chutes #3
        use_profit_target: bool = True,  # Chutes #2
    ):
        self.signal_gen = signal_gen
        self.base_volatility_target_pct = base_volatility_target_pct
        self.max_position_pct = max_position_pct
        self.max_gross_exposure_pct = max_gross_exposure_pct
        self.max_positions_for_scaling = max_positions_for_scaling
        self.use_profit_target = use_profit_target

    def get_adjusted_vol_target(self, position_count: int) -> float:
        """Dynamic volatility scaling (Chutes suggestion #3)."""
        heat = position_count / self.max_positions_for_scaling
        adjusted = self.base_volatility_target_pct * (1.0 - heat * 0.5)
        return max(0.5, adjusted)  # Floor at 0.5%

    def calculate_position_size(
        self,
        equity: float,
        price: float,
        atr: float,
        position_count: int,
        current_exposure_pct: float,
    ) -> int:
        if price <= 0 or atr <= 0 or equity <= 0:
            return 0

        vol_target = self.get_adjusted_vol_target(position_count)
        target_dollar_risk = equity * (vol_target / 100)
        dollar_value = target_dollar_risk * price / atr

        max_pos = equity * (self.max_position_pct / 100)
        if dollar_value > max_pos:
            dollar_value = max_pos

        avail = (self.max_gross_exposure_pct - current_exposure_pct) / 100 * equity
        if dollar_value > avail:
            dollar_value = max(0, avail)

        return int(dollar_value / price)

    def run(self, data, start_date=None, end_date=None):
        portfolio = Portfolio(100000.0)

        all_dates = set()
        for df in data.values():
            for d in df.index:
                all_dates.add(d.date() if isinstance(d, pd.Timestamp) else d)
        dates = sorted(all_dates)
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        indicator_data = {sym: self.signal_gen.calculate_indicators(df) for sym, df in data.items()}

        signals_by_date = {}
        for sym, df in indicator_data.items():
            for sig in self.signal_gen.generate_signals(df, sym):
                if sig.date not in signals_by_date:
                    signals_by_date[sig.date] = []
                signals_by_date[sig.date].append(sig)
        for d in signals_by_date:
            signals_by_date[d].sort(key=lambda s: s.strength, reverse=True)

        equity_history = []

        for current_date in dates:
            prices = {}
            for sym, df in data.items():
                if current_date in df.index:
                    prices[sym] = df.loc[current_date, "close"]
                else:
                    ts = pd.Timestamp(current_date)
                    if ts in df.index:
                        prices[sym] = df.loc[ts, "close"]
            if not prices:
                continue

            # Process exits
            symbols_to_close = []
            for sym, pos in list(portfolio.positions.items()):
                if sym not in data:
                    continue
                df = data[sym]
                bar = None
                if current_date in df.index:
                    bar = df.loc[current_date]
                else:
                    ts = pd.Timestamp(current_date)
                    if ts in df.index:
                        bar = df.loc[ts]
                if bar is None:
                    continue

                high, low, open_price, close = bar["high"], bar["low"], bar["open"], bar["close"]

                # Get current ATR
                idf = indicator_data.get(sym)
                current_atr = pos.atr_at_entry
                if idf is not None:
                    if current_date in idf.index:
                        current_atr = idf.loc[current_date, "atr"]
                    else:
                        ts = pd.Timestamp(current_date)
                        if ts in idf.index:
                            current_atr = idf.loc[ts, "atr"]

                # Check stop hit
                if pos.is_long and low <= pos.stop_price:
                    exit_price = min(pos.stop_price, open_price)
                    symbols_to_close.append((sym, exit_price, "stop"))
                    continue
                elif pos.is_short and high >= pos.stop_price:
                    exit_price = max(pos.stop_price, open_price)
                    symbols_to_close.append((sym, exit_price, "stop"))
                    continue

                # Check profit target (Chutes #2)
                if self.use_profit_target and pos.check_profit_target(close, current_atr, self.signal_gen.profit_target_atr):
                    symbols_to_close.append((sym, close, "profit_target"))
                    continue

                # Update trailing stop with asymmetric multipliers
                pos.update_trailing_stop(
                    high, low, close, current_atr,
                    self.signal_gen.trailing_activation_atr,
                    self.signal_gen.trailing_stop_atr_long,
                    self.signal_gen.trailing_stop_atr_short,
                )

            for sym, exit_price, reason in symbols_to_close:
                portfolio.close_position(sym, exit_price, current_date, reason)

            # Process entries
            signals = signals_by_date.get(current_date, [])
            equity = portfolio.calculate_equity(prices)
            exposure_pct = portfolio.calculate_gross_exposure_pct(prices)

            for sig in signals:
                if portfolio.has_position(sig.symbol):
                    continue

                shares = self.calculate_position_size(
                    equity, sig.price, sig.atr,
                    portfolio.position_count, exposure_pct
                )

                if shares <= 0 or shares * sig.price > portfolio.cash:
                    continue

                try:
                    portfolio.open_position(
                        sig.symbol, sig.signal_type, shares, sig.price,
                        current_date, sig.stop_price, sig.atr
                    )
                    exposure_pct = portfolio.calculate_gross_exposure_pct(prices)
                except ValueError:
                    pass

            equity_history.append((current_date, portfolio.calculate_equity(prices)))

        eq_curve = pd.Series({d: e for d, e in equity_history})
        return eq_curve, portfolio.trades


def calc_metrics(eq, trades):
    if len(eq) < 2:
        return None
    returns = eq.pct_change().dropna()
    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    max_dd = abs(dd)
    sharpe = cagr / vol if vol > 0 else 0
    down_ret = returns[returns < 0]
    down_std = down_ret.std() * np.sqrt(252) if len(down_ret) > 0 else 0
    sortino = cagr / down_std if down_std > 0 else 0
    calmar = cagr / max_dd if max_dd > 0 else 0

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    pf = sum(wins) / abs(sum(losses)) if losses else 0

    return {
        'cagr': cagr * 100, 'max_dd': max_dd * 100, 'sharpe': sharpe,
        'sortino': sortino, 'calmar': calmar, 'trades': len(trades),
        'win_rate': win_rate, 'pf': pf,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
    }


def main():
    config = load_config("config.yaml")
    db = Database(config.data.database_path)

    data = {}
    for sym in config.data.etf_symbols:
        df = db.get_price_data(sym)
        if df is not None and len(df) > 0:
            data[sym] = df

    FULL_START = date(2007, 1, 1)
    FULL_END = date(2025, 12, 31)
    IS_END = date(2021, 6, 30)
    OOS_START = date(2021, 7, 1)

    sig_gen = ChutesSignalGenerator(
        ema_period=35,
        atr_period=20,
        atr_multiplier=2.0,
        initial_stop_atr=2.0,  # Chutes #1: tighter
        trailing_stop_atr_long=1.5,  # Chutes #6: asymmetric
        trailing_stop_atr_short=3.0,  # Chutes #6: asymmetric
        trailing_activation_atr=1.0,
        profit_target_atr=3.0,  # Chutes #2
    )

    engine = ChutesBacktestEngine(
        sig_gen,
        base_volatility_target_pct=2.0,
        max_position_pct=15.0,
        max_gross_exposure_pct=100.0,
        max_positions_for_scaling=20,  # Chutes #3
        use_profit_target=True,  # Chutes #2
    )

    eq_full, trades_full = engine.run(data, FULL_START, FULL_END)
    eq_is, trades_is = engine.run(data, FULL_START, IS_END)
    eq_oos, trades_oos = engine.run(data, OOS_START, FULL_END)

    m_full = calc_metrics(eq_full, trades_full)
    m_is = calc_metrics(eq_is, trades_is)
    m_oos = calc_metrics(eq_oos, trades_oos)

    oos_is = m_oos['cagr'] / m_is['cagr'] if m_is['cagr'] > 0 else 0

    print("=" * 80)
    print("CHUTES VARIANT RESULTS")
    print("=" * 80)
    print("\nKey Changes from Baseline:")
    print("  1. Tighter initial stops (2.5 -> 2.0 ATR)")
    print("  2. Profit-taking at 3× ATR")
    print("  3. Dynamic volatility scaling (reduce with more positions)")
    print("  4. Asymmetric trailing stops (1.5× long, 3.0× short)")
    print()
    print("RETURN METRICS:")
    print(f"  CAGR (Full):        {m_full['cagr']:.2f}%")
    print(f"  CAGR (In-Sample):   {m_is['cagr']:.2f}%")
    print(f"  CAGR (OOS):         {m_oos['cagr']:.2f}%")
    print()
    print("RISK METRICS:")
    print(f"  Max Drawdown:       {m_full['max_dd']:.2f}%")
    print()
    print("RISK-ADJUSTED METRICS:")
    print(f"  Sharpe Ratio:       {m_full['sharpe']:.2f}")
    print(f"  Sortino Ratio:      {m_full['sortino']:.2f}")
    print(f"  Calmar Ratio:       {m_full['calmar']:.2f}")
    print()
    print("TRADE STATISTICS:")
    print(f"  Total Trades:       {m_full['trades']}")
    print(f"  Win Rate:           {m_full['win_rate']:.1f}%")
    print(f"  Profit Factor:      {m_full['pf']:.2f}")
    print(f"  Avg Win:            ${m_full['avg_win']:,.0f}")
    print(f"  Avg Loss:           ${m_full['avg_loss']:,.0f}")
    print()
    print("ROBUSTNESS:")
    print(f"  OOS/IS CAGR Ratio:  {oos_is:.2f}")
    print()

    dq = []
    if m_full['max_dd'] > 40:
        dq.append(f"Max DD {m_full['max_dd']:.1f}% > 40%")
    if oos_is < 0.5:
        dq.append(f"OOS/IS {oos_is:.2f} < 0.5")
    if m_full['pf'] < 1.2:
        dq.append(f"PF {m_full['pf']:.2f} < 1.2")
    if m_full['trades'] < 100:
        dq.append(f"Trades {m_full['trades']} < 100")
    if m_full['win_rate'] > 70:
        dq.append(f"WR {m_full['win_rate']:.1f}% > 70%")

    if dq:
        print("DQ:", ", ".join(dq))
        print("SCORE: DISQUALIFIED")
    else:
        score = (m_full['calmar']*35 + m_full['sharpe']*25 + oos_is*20 + m_full['pf']*10 + m_full['sortino']*10)
        print(f"SCORE: {score:.2f}")

    db.close()


if __name__ == "__main__":
    main()
