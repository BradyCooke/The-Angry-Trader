"""Matplotlib visualizations for backtest results."""

from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.backtest.metrics import PerformanceMetrics


# Set style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series | None = None,
    title: str = "Portfolio Equity Curve",
    figsize: tuple[int, int] = (12, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot equity curve with optional benchmark overlay.

    Args:
        equity_curve: Portfolio equity series indexed by date.
        benchmark_curve: Optional benchmark equity series.
        title: Chart title.
        figsize: Figure size tuple.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize to starting value of 100
    normalized_equity = (equity_curve / equity_curve.iloc[0]) * 100

    ax.plot(
        normalized_equity.index,
        normalized_equity.values,
        label="Portfolio",
        linewidth=2,
        color="#2196F3",
    )

    if benchmark_curve is not None and len(benchmark_curve) > 0:
        normalized_benchmark = (benchmark_curve / benchmark_curve.iloc[0]) * 100
        ax.plot(
            normalized_benchmark.index,
            normalized_benchmark.values,
            label="Benchmark",
            linewidth=1.5,
            color="#9E9E9E",
            linestyle="--",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value (Normalized to 100)", fontsize=12)
    ax.legend(loc="upper left")

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    figsize: tuple[int, int] = (12, 4),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot drawdown chart.

    Args:
        equity_curve: Portfolio equity series indexed by date.
        title: Chart title.
        figsize: Figure size tuple.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100

    # Fill area
    ax.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        color="#F44336",
        alpha=0.5,
        label="Drawdown",
    )

    ax.plot(
        drawdown.index,
        drawdown.values,
        color="#D32F2F",
        linewidth=1,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    ax.grid(True, alpha=0.3)

    # Set y-axis to always show 0 at top
    ax.set_ylim(top=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_monthly_returns(
    monthly_returns: pd.Series,
    title: str = "Monthly Returns Heatmap",
    figsize: tuple[int, int] = (14, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot monthly returns heatmap.

    Args:
        monthly_returns: Series of monthly returns indexed by date.
        title: Chart title.
        figsize: Figure size tuple.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    if len(monthly_returns) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    # Create pivot table
    returns_df = monthly_returns.to_frame("return")
    returns_df["year"] = returns_df.index.year
    returns_df["month"] = returns_df.index.month

    pivot = returns_df.pivot(index="year", columns="month", values="return")

    # Fill missing months with NaN
    all_months = list(range(1, 13))
    for month in all_months:
        if month not in pivot.columns:
            pivot[month] = np.nan
    pivot = pivot[sorted(pivot.columns)]

    # Convert to percentage
    pivot = pivot * 100

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-10, vmax=10)

    # Set labels
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add values to cells
    for i in range(len(pivot.index)):
        for j in range(12):
            value = pivot.iloc[i, j]
            if not np.isnan(value):
                text_color = "white" if abs(value) > 5 else "black"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                       color=text_color, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Year", fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Return (%)", fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_trade_distribution(
    trades: list,
    title: str = "Trade P&L Distribution",
    figsize: tuple[int, int] = (10, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot histogram of trade P&L.

    Args:
        trades: List of Trade objects.
        title: Chart title.
        figsize: Figure size tuple.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not trades:
        ax.text(0.5, 0.5, "No trades to display", ha="center", va="center")
        return fig

    pnls = [t.pnl for t in trades]

    # Create histogram
    n, bins, patches = ax.hist(pnls, bins=30, edgecolor="white", alpha=0.7)

    # Color positive/negative differently
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor("#F44336")
        else:
            patch.set_facecolor("#4CAF50")

    # Add vertical line at zero
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Add mean line
    mean_pnl = np.mean(pnls)
    ax.axvline(x=mean_pnl, color="#2196F3", linestyle="-", linewidth=2,
               label=f"Mean: ${mean_pnl:,.0f}")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("P&L ($)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_rolling_metrics(
    daily_returns: pd.Series,
    window: int = 60,
    title: str = "Rolling Performance Metrics",
    figsize: tuple[int, int] = (12, 8),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot rolling Sharpe ratio and volatility.

    Args:
        daily_returns: Series of daily returns.
        window: Rolling window size in days.
        title: Chart title.
        figsize: Figure size tuple.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    if len(daily_returns) < window:
        for ax in axes:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig

    # Rolling Sharpe
    rolling_mean = daily_returns.rolling(window=window).mean()
    rolling_std = daily_returns.rolling(window=window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    axes[0].plot(rolling_sharpe.index, rolling_sharpe.values,
                 color="#2196F3", linewidth=1.5)
    axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[0].axhline(y=1, color="#4CAF50", linestyle="--", alpha=0.5, label="Sharpe=1")
    axes[0].axhline(y=-1, color="#F44336", linestyle="--", alpha=0.5, label="Sharpe=-1")
    axes[0].set_ylabel(f"Rolling Sharpe ({window}d)", fontsize=11)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Rolling Volatility
    rolling_vol = rolling_std * np.sqrt(252) * 100
    axes[1].plot(rolling_vol.index, rolling_vol.values,
                 color="#FF9800", linewidth=1.5)
    axes[1].set_ylabel(f"Rolling Volatility ({window}d, %)", fontsize=11)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Format x-axis dates
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_yearly_returns(
    yearly_returns: pd.Series,
    title: str = "Annual Returns",
    figsize: tuple[int, int] = (10, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot bar chart of yearly returns.

    Args:
        yearly_returns: Series of yearly returns indexed by date.
        title: Chart title.
        figsize: Figure size tuple.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if len(yearly_returns) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    years = [d.year for d in yearly_returns.index]
    returns_pct = yearly_returns.values * 100

    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in returns_pct]

    bars = ax.bar(years, returns_pct, color=colors, edgecolor="white", width=0.7)

    # Add value labels
    for bar, val in zip(bars, returns_pct):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 0.5 if height >= 0 else -0.5
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                f"{val:.1f}%", ha="center", va=va, fontsize=10)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Return (%)", fontsize=12)

    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_backtest_report(
    metrics: PerformanceMetrics,
    equity_curve: pd.Series,
    trades: list,
    benchmark_curve: pd.Series | None = None,
    title: str = "Backtest Report",
    figsize: tuple[int, int] = (14, 16),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create comprehensive backtest report with multiple charts.

    Args:
        metrics: PerformanceMetrics object.
        equity_curve: Portfolio equity series.
        trades: List of Trade objects.
        benchmark_curve: Optional benchmark equity series.
        title: Report title.
        figsize: Figure size tuple.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)

    # 1. Equity Curve (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    normalized_equity = (equity_curve / equity_curve.iloc[0]) * 100
    ax1.plot(normalized_equity.index, normalized_equity.values,
             label="Portfolio", linewidth=2, color="#2196F3")

    if benchmark_curve is not None and len(benchmark_curve) > 0:
        normalized_benchmark = (benchmark_curve / benchmark_curve.iloc[0]) * 100
        ax1.plot(normalized_benchmark.index, normalized_benchmark.values,
                 label="Benchmark", linewidth=1.5, color="#9E9E9E", linestyle="--")

    ax1.set_title("Equity Curve", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Value (Normalized)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values, color="#D32F2F", linewidth=1)
    ax2.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(top=0)
    ax2.grid(True, alpha=0.3)

    # 3. Monthly Returns Heatmap
    ax3 = fig.add_subplot(gs[2, 0])
    if len(metrics.monthly_returns) > 0:
        returns_df = metrics.monthly_returns.to_frame("return")
        returns_df["year"] = returns_df.index.year
        returns_df["month"] = returns_df.index.month
        pivot = returns_df.pivot(index="year", columns="month", values="return")

        # Ensure all months present
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan
        pivot = pivot[sorted(pivot.columns)]
        pivot_pct = pivot * 100

        im = ax3.imshow(pivot_pct.values, cmap=plt.cm.RdYlGn, aspect="auto",
                        vmin=-10, vmax=10)
        ax3.set_xticks(np.arange(12))
        ax3.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax3.set_yticks(np.arange(len(pivot.index)))
        ax3.set_yticklabels(pivot.index)
        ax3.set_title("Monthly Returns (%)", fontsize=12, fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center")
        ax3.set_title("Monthly Returns", fontsize=12, fontweight="bold")

    # 4. Trade Distribution
    ax4 = fig.add_subplot(gs[2, 1])
    if trades:
        pnls = [t.pnl for t in trades]
        n, bins, patches = ax4.hist(pnls, bins=20, edgecolor="white", alpha=0.7)
        for patch, left_edge in zip(patches, bins[:-1]):
            patch.set_facecolor("#F44336" if left_edge < 0 else "#4CAF50")
        ax4.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax4.set_title("Trade P&L Distribution", fontsize=12, fontweight="bold")
    ax4.set_xlabel("P&L ($)")
    ax4.grid(True, alpha=0.3)

    # 5. Rolling Sharpe
    ax5 = fig.add_subplot(gs[3, 0])
    if len(metrics.daily_returns) >= 60:
        rolling_mean = metrics.daily_returns.rolling(window=60).mean()
        rolling_std = metrics.daily_returns.rolling(window=60).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        ax5.plot(rolling_sharpe.index, rolling_sharpe.values, color="#2196F3", linewidth=1)
        ax5.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax5.axhline(y=1, color="#4CAF50", linestyle="--", alpha=0.3)
    ax5.set_title("Rolling 60-Day Sharpe", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # 6. Yearly Returns Bar Chart
    ax6 = fig.add_subplot(gs[3, 1])
    if len(metrics.yearly_returns) > 0:
        years = [d.year for d in metrics.yearly_returns.index]
        returns_pct = metrics.yearly_returns.values * 100
        colors = ["#4CAF50" if r >= 0 else "#F44336" for r in returns_pct]
        ax6.bar(years, returns_pct, color=colors, edgecolor="white", width=0.6)
        ax6.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax6.set_title("Annual Returns", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Year")
    ax6.set_ylabel("Return (%)")
    ax6.grid(True, alpha=0.3, axis="y")

    # Main title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
