#!/usr/bin/env python3
"""Paper trading script for daily signal monitoring.

Run this script daily after market close to:
1. Update market data
2. Check exits for current positions
3. Generate new entry signals
4. Track paper portfolio performance

Usage:
    python paper_trade.py           # Show today's signals
    python paper_trade.py --execute # Execute signals (update portfolio)
    python paper_trade.py --status  # Show portfolio status only
    python paper_trade.py --reset   # Reset paper portfolio
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from src.utils.config import load_config
from src.data.database import Database
from src.data.fetcher import DataFetcher
from src.signals.keltner import KeltnerConfig, KeltnerSignalGenerator
from src.risk.position_sizing import PositionSizer, PositionSizingConfig


# Paper portfolio state file
PORTFOLIO_FILE = Path("paper_portfolio.json")


def load_portfolio() -> dict:
    """Load paper portfolio state from file."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "cash": 100000.0,
        "starting_capital": 100000.0,
        "positions": {},
        "trades": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


def save_portfolio(portfolio: dict) -> None:
    """Save paper portfolio state to file."""
    portfolio["updated_at"] = datetime.now().isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)


def calculate_equity(portfolio: dict, prices: dict[str, float]) -> float:
    """Calculate total portfolio equity."""
    equity = portfolio["cash"]
    for symbol, pos in portfolio["positions"].items():
        if symbol in prices:
            current_price = prices[symbol]
            if pos["side"] == "LONG":
                equity += pos["shares"] * current_price
            else:  # SHORT
                equity += pos["shares"] * (2 * pos["entry_price"] - current_price)
    return equity


def check_stops(portfolio: dict, data: dict, config) -> list[dict]:
    """Check if any positions hit their stops."""
    exits = []

    for symbol, pos in list(portfolio["positions"].items()):
        if symbol not in data:
            continue

        df = data[symbol]
        if len(df) == 0:
            continue

        latest = df.iloc[-1]
        high = latest["high"]
        low = latest["low"]
        stop = pos["stop_price"]

        hit_stop = False
        exit_price = None

        if pos["side"] == "LONG":
            if low <= stop:
                hit_stop = True
                exit_price = stop
        else:  # SHORT
            if high >= stop:
                hit_stop = True
                exit_price = stop

        if hit_stop:
            exits.append({
                "symbol": symbol,
                "side": pos["side"],
                "shares": pos["shares"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "entry_date": pos["entry_date"],
                "stop_price": stop,
            })

    return exits


def generate_signals(data: dict, config, portfolio: dict) -> list[dict]:
    """Generate entry signals for symbols not in portfolio."""
    signal_gen = KeltnerSignalGenerator(
        KeltnerConfig(
            ema_period=config.strategy.keltner.ema_period,
            atr_period=config.strategy.keltner.atr_period,
            atr_multiplier=config.strategy.keltner.atr_multiplier,
            initial_stop_atr=config.strategy.stops.initial_atr_multiple,
            trailing_stop_atr=config.strategy.stops.trailing_atr_multiple,
            trailing_activation_atr=config.strategy.stops.trailing_activation_atr,
        )
    )

    signals = []

    for symbol, df in data.items():
        # Skip if already have position
        if symbol in portfolio["positions"]:
            continue

        if len(df) < config.strategy.keltner.ema_period:
            continue

        # Calculate indicators
        df_with_ind = signal_gen.calculate_indicators(df)

        # Check for signal on latest bar
        symbol_signals = signal_gen.generate_signals(df_with_ind, symbol)

        # Get signal for today only
        today = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
        for sig in symbol_signals:
            if sig.date == today:
                signals.append({
                    "symbol": symbol,
                    "side": sig.signal_type.name,
                    "price": sig.price,
                    "stop_price": sig.stop_price,
                    "atr": sig.atr,
                    "strength": sig.strength,
                })

    # Sort by strength
    signals.sort(key=lambda x: x["strength"], reverse=True)
    return signals


def calculate_position_size(config, equity: float, price: float, atr: float) -> int:
    """Calculate position size based on volatility targeting."""
    sizer = PositionSizer(
        PositionSizingConfig(
            volatility_target_pct=config.risk_management.position_sizing.volatility_target_pct,
            max_position_pct=config.risk_management.position_sizing.max_position_pct,
            max_gross_exposure_pct=config.risk_management.portfolio.max_gross_exposure_pct,
        )
    )

    result = sizer.calculate_position_size(
        equity=equity,
        price=price,
        atr=atr,
        current_exposure_pct=0,  # Simplified
    )
    return result.shares


def print_status(portfolio: dict, prices: dict[str, float]) -> None:
    """Print current portfolio status."""
    equity = calculate_equity(portfolio, prices)
    starting = portfolio["starting_capital"]
    pnl = equity - starting
    pnl_pct = (pnl / starting) * 100

    print("=" * 60)
    print("PAPER PORTFOLIO STATUS")
    print("=" * 60)
    print(f"Started: {portfolio['created_at'][:10]}")
    print(f"Updated: {portfolio['updated_at'][:10]}")
    print()
    print(f"Starting Capital: ${starting:,.2f}")
    print(f"Current Equity:   ${equity:,.2f}")
    print(f"Cash:             ${portfolio['cash']:,.2f}")
    print(f"P&L:              ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    print()

    if portfolio["positions"]:
        print("OPEN POSITIONS")
        print("-" * 60)
        print(f"{'Symbol':<8} {'Side':<6} {'Shares':>8} {'Entry':>10} {'Current':>10} {'P&L':>12}")
        print("-" * 60)

        for symbol, pos in portfolio["positions"].items():
            current = prices.get(symbol, pos["entry_price"])
            if pos["side"] == "LONG":
                pnl = (current - pos["entry_price"]) * pos["shares"]
            else:
                pnl = (pos["entry_price"] - current) * pos["shares"]

            print(f"{symbol:<8} {pos['side']:<6} {pos['shares']:>8} "
                  f"${pos['entry_price']:>9.2f} ${current:>9.2f} ${pnl:>11.2f}")
        print("-" * 60)
    else:
        print("No open positions.")

    print()
    print(f"Total trades: {len(portfolio['trades'])}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Paper trading signal monitor")
    parser.add_argument("--execute", action="store_true", help="Execute signals (update portfolio)")
    parser.add_argument("--status", action="store_true", help="Show portfolio status only")
    parser.add_argument("--reset", action="store_true", help="Reset paper portfolio")
    args = parser.parse_args()

    # Load config
    config = load_config("config.yaml")

    # Reset portfolio if requested
    if args.reset:
        if PORTFOLIO_FILE.exists():
            PORTFOLIO_FILE.unlink()
        print("Paper portfolio reset.")
        return

    # Initialize database
    db = Database(config.data.database_path)
    fetcher = DataFetcher(db)

    # Update data
    print("Updating market data...")
    symbols = config.data.etf_symbols + [config.data.benchmark_symbol]
    fetcher.fetch_multiple(symbols)

    # Load data
    data = {}
    prices = {}
    for symbol in config.data.etf_symbols:
        df = db.get_price_data(symbol)
        if df is not None and len(df) > 0:
            data[symbol] = df
            prices[symbol] = df.iloc[-1]["close"]

    # Load portfolio
    portfolio = load_portfolio()

    # Status only mode
    if args.status:
        print_status(portfolio, prices)
        db.close()
        return

    # Get latest date
    if data:
        sample = list(data.values())[0]
        latest_date = sample.index[-1]
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()
        print(f"Latest data: {latest_date}")
    print()

    # Check stops
    exits = check_stops(portfolio, data, config)

    # Generate new signals
    signals = generate_signals(data, config, portfolio)

    # Calculate current equity
    equity = calculate_equity(portfolio, prices)

    # Print results
    print("=" * 60)
    print(f"PAPER TRADING SIGNALS - {latest_date}")
    print("=" * 60)
    print(f"Portfolio Equity: ${equity:,.2f}")
    print(f"Open Positions: {len(portfolio['positions'])}")
    print()

    # Exits
    if exits:
        print("EXIT SIGNALS (Stop Hit)")
        print("-" * 60)
        for exit in exits:
            pnl = (exit["exit_price"] - exit["entry_price"]) * exit["shares"]
            if exit["side"] == "SHORT":
                pnl = -pnl
            print(f"  {exit['symbol']:<6} {exit['side']:<5} | "
                  f"Entry: ${exit['entry_price']:.2f} -> Exit: ${exit['exit_price']:.2f} | "
                  f"P&L: ${pnl:,.2f}")
        print()
    else:
        print("No exit signals.\n")

    # Entries
    if signals:
        print("ENTRY SIGNALS (Top 5)")
        print("-" * 60)
        for sig in signals[:5]:
            shares = calculate_position_size(config, equity, sig["price"], sig["atr"])
            cost = shares * sig["price"]
            print(f"  {sig['symbol']:<6} {sig['side']:<5} | "
                  f"Price: ${sig['price']:.2f} | Stop: ${sig['stop_price']:.2f} | "
                  f"Size: {shares} shares (${cost:,.0f})")
        print()
    else:
        print("No entry signals.\n")

    # Execute if requested
    if args.execute and (exits or signals):
        print("EXECUTING SIGNALS...")
        print("-" * 60)

        # Process exits
        for exit in exits:
            pos = portfolio["positions"].pop(exit["symbol"])
            proceeds = exit["shares"] * exit["exit_price"]
            if exit["side"] == "SHORT":
                # Return borrowed shares value, keep profit/loss
                pnl = (pos["entry_price"] - exit["exit_price"]) * exit["shares"]
                portfolio["cash"] += pos["shares"] * pos["entry_price"] + pnl
            else:
                portfolio["cash"] += proceeds

            pnl = (exit["exit_price"] - pos["entry_price"]) * exit["shares"]
            if exit["side"] == "SHORT":
                pnl = -pnl

            portfolio["trades"].append({
                "symbol": exit["symbol"],
                "side": exit["side"],
                "shares": exit["shares"],
                "entry_price": pos["entry_price"],
                "exit_price": exit["exit_price"],
                "entry_date": pos["entry_date"],
                "exit_date": str(date.today()),
                "pnl": pnl,
            })
            print(f"  CLOSED: {exit['symbol']} {exit['side']} | P&L: ${pnl:,.2f}")

        # Process entries (limited by cash and risk constraints)
        for sig in signals:
            if sig["symbol"] in portfolio["positions"]:
                continue

            shares = calculate_position_size(config, equity, sig["price"], sig["atr"])
            cost = shares * sig["price"]

            if cost > portfolio["cash"]:
                print(f"  SKIP: {sig['symbol']} - Insufficient cash")
                continue

            portfolio["positions"][sig["symbol"]] = {
                "side": sig["side"],
                "shares": shares,
                "entry_price": sig["price"],
                "entry_date": str(date.today()),
                "stop_price": sig["stop_price"],
                "atr": sig["atr"],
            }

            if sig["side"] == "LONG":
                portfolio["cash"] -= cost
            else:
                # Short: receive proceeds but owe shares
                portfolio["cash"] -= cost  # Margin requirement simplified

            print(f"  OPENED: {sig['symbol']} {sig['side']} | {shares} shares @ ${sig['price']:.2f}")

        save_portfolio(portfolio)
        print()
        print("Portfolio updated and saved.")

    elif not args.execute and (exits or signals):
        print("Run with --execute to apply these signals.")

    print()
    print_status(portfolio, prices)

    db.close()


if __name__ == "__main__":
    main()
