"""Console output formatting for backtest results."""

from dataclasses import dataclass
from typing import Any

from src.backtest.metrics import PerformanceMetrics
from src.portfolio.portfolio import Trade


def format_number(value: float, decimals: int = 2, suffix: str = "") -> str:
    """Format a number with specified decimals and optional suffix."""
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Format a number as percentage."""
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}%"


def format_currency(value: float, decimals: int = 0) -> str:
    """Format a number as currency."""
    if value is None:
        return "N/A"
    return f"${value:,.{decimals}f}"


def format_metrics_table(metrics: PerformanceMetrics) -> str:
    """Format performance metrics as a table string.

    Args:
        metrics: PerformanceMetrics object.

    Returns:
        Formatted string table.
    """
    lines = []
    separator = "=" * 50

    lines.append(separator)
    lines.append("PERFORMANCE SUMMARY")
    lines.append(separator)

    # Time period
    if metrics.start_date and metrics.end_date:
        lines.append(f"Period: {metrics.start_date} to {metrics.end_date}")
        lines.append(f"Trading Days: {metrics.trading_days:,}")
    lines.append("")

    # Return metrics
    lines.append("RETURNS")
    lines.append("-" * 30)
    lines.append(f"Total Return:      {format_percent(metrics.total_return_pct)}")
    lines.append(f"CAGR:              {format_percent(metrics.cagr * 100)}")
    lines.append("")

    # Risk metrics
    lines.append("RISK")
    lines.append("-" * 30)
    lines.append(f"Max Drawdown:      {format_percent(metrics.max_drawdown_pct)}")
    lines.append(f"Volatility (Ann.): {format_percent(metrics.volatility * 100)}")
    lines.append(f"Downside Dev.:     {format_percent(metrics.downside_deviation * 100)}")
    lines.append(f"VaR (95%):         {format_percent(metrics.var_95)}")
    lines.append("")

    # Risk-adjusted returns
    lines.append("RISK-ADJUSTED RETURNS")
    lines.append("-" * 30)
    lines.append(f"Sharpe Ratio:      {format_number(metrics.sharpe_ratio)}")
    lines.append(f"Sortino Ratio:     {format_number(metrics.sortino_ratio)}")
    lines.append(f"Calmar Ratio:      {format_number(metrics.calmar_ratio)}")
    lines.append("")

    # Trade statistics
    lines.append("TRADE STATISTICS")
    lines.append("-" * 30)
    lines.append(f"Total Trades:      {metrics.total_trades}")
    lines.append(f"Winning Trades:    {metrics.winning_trades}")
    lines.append(f"Losing Trades:     {metrics.losing_trades}")
    lines.append(f"Win Rate:          {format_percent(metrics.win_rate)}")
    lines.append(f"Avg Win:           {format_currency(metrics.avg_win, 2)}")
    lines.append(f"Avg Loss:          {format_currency(metrics.avg_loss, 2)}")
    lines.append(f"Profit Factor:     {format_number(metrics.profit_factor)}")
    lines.append(f"Avg Trade Duration:{format_number(metrics.avg_trade_duration, 1)} days")
    lines.append(f"Longest Win Streak:{metrics.longest_win_streak}")
    lines.append(f"Longest Lose Streak:{metrics.longest_lose_streak}")
    lines.append("")

    # Benchmark comparison
    if metrics.alpha != 0 or metrics.beta != 0:
        lines.append("BENCHMARK COMPARISON")
        lines.append("-" * 30)
        lines.append(f"Alpha (Ann.):      {format_percent(metrics.alpha * 100)}")
        lines.append(f"Beta:              {format_number(metrics.beta)}")
        lines.append(f"Correlation:       {format_number(metrics.correlation)}")
        lines.append("")

    lines.append(separator)

    return "\n".join(lines)


def print_metrics(metrics: PerformanceMetrics) -> None:
    """Print performance metrics to console.

    Args:
        metrics: PerformanceMetrics object.
    """
    print(format_metrics_table(metrics))


def format_trade_row(trade: Trade, index: int) -> str:
    """Format a single trade as a table row.

    Args:
        trade: Trade object.
        index: Trade index/number.

    Returns:
        Formatted string row.
    """
    side = "LONG" if trade.side.name == "LONG" else "SHORT"
    pnl_str = format_currency(trade.pnl, 2)
    pnl_pct_str = format_percent(trade.pnl_pct)

    return (
        f"{index:>4} | {trade.symbol:<6} | {side:<5} | "
        f"{trade.entry_date} | {trade.exit_date} | "
        f"{trade.shares:>6} | {trade.entry_price:>8.2f} | {trade.exit_price:>8.2f} | "
        f"{pnl_str:>12} | {pnl_pct_str:>8} | {trade.exit_reason:<8}"
    )


def format_trade_summary(trades: list[Trade], max_trades: int | None = None) -> str:
    """Format trades as a summary table.

    Args:
        trades: List of Trade objects.
        max_trades: Maximum number of trades to display (None for all).

    Returns:
        Formatted string table.
    """
    if not trades:
        return "No trades to display."

    lines = []
    separator = "=" * 120

    lines.append(separator)
    lines.append("TRADE HISTORY")
    lines.append(separator)

    # Header
    header = (
        f"{'#':>4} | {'Symbol':<6} | {'Side':<5} | "
        f"{'Entry Date':<10} | {'Exit Date':<10} | "
        f"{'Shares':>6} | {'Entry':>8} | {'Exit':>8} | "
        f"{'P&L':>12} | {'P&L %':>8} | {'Reason':<8}"
    )
    lines.append(header)
    lines.append("-" * 120)

    # Trades
    display_trades = trades[:max_trades] if max_trades else trades

    for i, trade in enumerate(display_trades, 1):
        lines.append(format_trade_row(trade, i))

    if max_trades and len(trades) > max_trades:
        lines.append(f"... and {len(trades) - max_trades} more trades")

    lines.append(separator)

    # Summary stats
    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    lines.append(f"Total P&L: {format_currency(total_pnl, 2)}")
    lines.append(f"Average P&L: {format_currency(avg_pnl, 2)}")
    lines.append(f"Win/Loss: {len(wins)}/{len(losses)}")
    lines.append(separator)

    return "\n".join(lines)


def print_trade_summary(trades: list[Trade], max_trades: int | None = 20) -> None:
    """Print trade summary to console.

    Args:
        trades: List of Trade objects.
        max_trades: Maximum number of trades to display.
    """
    print(format_trade_summary(trades, max_trades))


def format_monthly_returns_table(monthly_returns: dict[tuple[int, int], float]) -> str:
    """Format monthly returns as a yearly table.

    Args:
        monthly_returns: Dictionary mapping (year, month) to return percentage.

    Returns:
        Formatted string table.
    """
    if not monthly_returns:
        return "No monthly returns data."

    # Get year range
    years = sorted(set(ym[0] for ym in monthly_returns.keys()))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    lines = []
    lines.append("MONTHLY RETURNS (%)")
    lines.append("=" * 100)

    # Header
    header = f"{'Year':>6} | " + " | ".join(f"{m:>6}" for m in months) + " | " + f"{'YTD':>7}"
    lines.append(header)
    lines.append("-" * 100)

    for year in years:
        row_values = []
        ytd = 1.0

        for month in range(1, 13):
            ret = monthly_returns.get((year, month))
            if ret is not None:
                row_values.append(f"{ret * 100:>6.1f}")
                ytd *= (1 + ret)
            else:
                row_values.append(f"{'--':>6}")

        ytd_pct = (ytd - 1) * 100
        row = f"{year:>6} | " + " | ".join(row_values) + f" | {ytd_pct:>6.1f}%"
        lines.append(row)

    lines.append("=" * 100)

    return "\n".join(lines)


def print_config_summary(config: dict[str, Any]) -> None:
    """Print configuration summary.

    Args:
        config: Configuration dictionary.
    """
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)

    if "strategy" in config:
        print("\nStrategy:")
        for key, value in config["strategy"].items():
            print(f"  {key}: {value}")

    if "risk" in config:
        print("\nRisk Management:")
        for key, value in config["risk"].items():
            print(f"  {key}: {value}")

    if "portfolio" in config:
        print("\nPortfolio:")
        for key, value in config["portfolio"].items():
            print(f"  {key}: {value}")

    print("=" * 50)
