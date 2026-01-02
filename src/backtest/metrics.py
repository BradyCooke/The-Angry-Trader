"""Performance metrics calculations."""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    # Return metrics
    total_return_pct: float = 0.0
    cagr: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0
    longest_win_streak: int = 0
    longest_lose_streak: int = 0

    # Benchmark comparison
    alpha: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0

    # Time period
    start_date: date | None = None
    end_date: date | None = None
    trading_days: int = 0

    # Returns series for further analysis
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    monthly_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    yearly_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """Calculate daily returns from equity curve.

    Args:
        equity_curve: Series of portfolio values indexed by date.

    Returns:
        Series of daily returns.
    """
    return equity_curve.pct_change().dropna()


def calculate_cagr(equity_curve: pd.Series) -> float:
    """Calculate Compound Annual Growth Rate.

    Args:
        equity_curve: Series of portfolio values.

    Returns:
        CAGR as decimal (e.g., 0.10 for 10%).
    """
    if len(equity_curve) < 2:
        return 0.0

    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]

    if start_value <= 0:
        return 0.0

    # Calculate years
    if isinstance(equity_curve.index[0], date):
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
    else:
        days = len(equity_curve)

    years = days / 365.25

    if years <= 0:
        return 0.0

    return (end_value / start_value) ** (1 / years) - 1


def calculate_max_drawdown(equity_curve: pd.Series) -> tuple[float, date | None, date | None]:
    """Calculate maximum drawdown.

    Args:
        equity_curve: Series of portfolio values.

    Returns:
        Tuple of (max_drawdown_pct, peak_date, trough_date).
    """
    if len(equity_curve) < 2:
        return 0.0, None, None

    # Calculate running maximum
    running_max = equity_curve.cummax()

    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max

    # Find maximum drawdown
    max_dd = abs(drawdown.min())

    # If no drawdown (monotonically increasing), return 0
    if max_dd == 0:
        return 0.0, None, None

    max_dd_idx = drawdown.idxmin()

    # Find peak (most recent peak before trough)
    subset = equity_curve[:max_dd_idx]
    if len(subset) == 0:
        peak_date = equity_curve.index[0]
    else:
        peak_date = subset.idxmax()

    return max_dd * 100, peak_date, max_dd_idx


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Calculate volatility (standard deviation of returns).

    Args:
        returns: Daily returns series.
        annualize: If True, annualize the volatility.

    Returns:
        Volatility as decimal.
    """
    if len(returns) < 2:
        return 0.0

    vol = returns.std()

    if annualize:
        vol = vol * np.sqrt(252)

    return vol


def calculate_downside_deviation(returns: pd.Series, mar: float = 0.0, annualize: bool = True) -> float:
    """Calculate downside deviation.

    Args:
        returns: Daily returns series.
        mar: Minimum acceptable return (default 0).
        annualize: If True, annualize the result.

    Returns:
        Downside deviation as decimal.
    """
    if len(returns) < 2:
        return 0.0

    downside_returns = returns[returns < mar]

    if len(downside_returns) == 0:
        return 0.0

    dd = np.sqrt(np.mean((downside_returns - mar) ** 2))

    if annualize:
        dd = dd * np.sqrt(252)

    return dd


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe Ratio.

    Args:
        returns: Daily returns series.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Sharpe ratio (annualized).
    """
    if len(returns) < 2:
        return 0.0

    # Daily risk-free rate
    daily_rf = risk_free_rate / 252

    excess_returns = returns - daily_rf

    if excess_returns.std() == 0:
        return 0.0

    # Annualized Sharpe
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino Ratio.

    Args:
        returns: Daily returns series.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Sortino ratio (annualized).
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    downside_dev = calculate_downside_deviation(returns, mar=daily_rf, annualize=False)

    if downside_dev == 0:
        return 0.0

    # Annualized
    annual_excess = excess_returns.mean() * 252
    annual_dd = downside_dev * np.sqrt(252)

    return annual_excess / annual_dd


def calculate_calmar_ratio(cagr: float, max_drawdown_pct: float) -> float:
    """Calculate Calmar Ratio (CAGR / Max Drawdown).

    Args:
        cagr: Compound annual growth rate.
        max_drawdown_pct: Maximum drawdown percentage.

    Returns:
        Calmar ratio.
    """
    if max_drawdown_pct == 0:
        return 0.0

    return (cagr * 100) / max_drawdown_pct


def calculate_trade_statistics(trades: list) -> dict:
    """Calculate trade statistics.

    Args:
        trades: List of Trade objects.

    Returns:
        Dictionary of trade statistics.
    """
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_duration": 0.0,
            "longest_win_streak": 0,
            "longest_lose_streak": 0,
        }

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    total_trades = len(trades)
    winning_trades = len(wins)
    losing_trades = len(losses)

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    avg_duration = np.mean([t.holding_days for t in trades])

    # Calculate streaks
    win_streak = 0
    lose_streak = 0
    max_win_streak = 0
    max_lose_streak = 0

    for trade in trades:
        if trade.pnl > 0:
            win_streak += 1
            lose_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
        else:
            lose_streak += 1
            win_streak = 0
            max_lose_streak = max(max_lose_streak, lose_streak)

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_trade_duration": avg_duration,
        "longest_win_streak": max_win_streak,
        "longest_lose_streak": max_lose_streak,
    }


def calculate_benchmark_comparison(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """Calculate benchmark comparison metrics.

    Args:
        portfolio_returns: Portfolio daily returns.
        benchmark_returns: Benchmark daily returns.

    Returns:
        Dictionary with alpha, beta, correlation.
    """
    # Align returns
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()

    if len(aligned) < 2:
        return {"alpha": 0.0, "beta": 0.0, "correlation": 0.0}

    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]

    # Correlation
    correlation = port_ret.corr(bench_ret)

    # Beta (covariance / variance)
    covariance = np.cov(port_ret, bench_ret)[0, 1]
    variance = np.var(bench_ret)
    beta = covariance / variance if variance > 0 else 0.0

    # Alpha (annualized)
    port_annual = port_ret.mean() * 252
    bench_annual = bench_ret.mean() * 252
    alpha = port_annual - (beta * bench_annual)

    return {"alpha": alpha, "beta": beta, "correlation": correlation}


def calculate_metrics(
    equity_curve: pd.Series,
    trades: list,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Calculate all performance metrics.

    Args:
        equity_curve: Portfolio equity curve indexed by date.
        trades: List of Trade objects.
        benchmark_returns: Optional benchmark returns for comparison.
        risk_free_rate: Annual risk-free rate.

    Returns:
        PerformanceMetrics object with all metrics.
    """
    metrics = PerformanceMetrics()

    if len(equity_curve) < 2:
        return metrics

    # Time period
    metrics.start_date = equity_curve.index[0]
    metrics.end_date = equity_curve.index[-1]
    metrics.trading_days = len(equity_curve)

    # Calculate returns
    daily_returns = calculate_returns(equity_curve)
    metrics.daily_returns = daily_returns

    # Monthly and yearly returns
    if isinstance(equity_curve.index, pd.DatetimeIndex):
        monthly = equity_curve.resample("ME").last()
        metrics.monthly_returns = calculate_returns(monthly)

        yearly = equity_curve.resample("YE").last()
        metrics.yearly_returns = calculate_returns(yearly)

    # Return metrics
    metrics.total_return_pct = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100
    metrics.cagr = calculate_cagr(equity_curve)

    # Risk metrics
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    metrics.max_drawdown_pct = max_dd
    metrics.volatility = calculate_volatility(daily_returns)
    metrics.downside_deviation = calculate_downside_deviation(daily_returns)

    # VaR at 95%
    if len(daily_returns) > 0:
        metrics.var_95 = abs(np.percentile(daily_returns, 5)) * 100

    # Risk-adjusted metrics
    metrics.sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    metrics.sortino_ratio = calculate_sortino_ratio(daily_returns, risk_free_rate)
    metrics.calmar_ratio = calculate_calmar_ratio(metrics.cagr, metrics.max_drawdown_pct)

    # Trade statistics
    trade_stats = calculate_trade_statistics(trades)
    metrics.total_trades = trade_stats["total_trades"]
    metrics.winning_trades = trade_stats["winning_trades"]
    metrics.losing_trades = trade_stats["losing_trades"]
    metrics.win_rate = trade_stats["win_rate"]
    metrics.avg_win = trade_stats["avg_win"]
    metrics.avg_loss = trade_stats["avg_loss"]
    metrics.profit_factor = trade_stats["profit_factor"]
    metrics.avg_trade_duration = trade_stats["avg_trade_duration"]
    metrics.longest_win_streak = trade_stats["longest_win_streak"]
    metrics.longest_lose_streak = trade_stats["longest_lose_streak"]

    # Benchmark comparison
    if benchmark_returns is not None:
        bench_stats = calculate_benchmark_comparison(daily_returns, benchmark_returns)
        metrics.alpha = bench_stats["alpha"]
        metrics.beta = bench_stats["beta"]
        metrics.correlation = bench_stats["correlation"]

    return metrics
