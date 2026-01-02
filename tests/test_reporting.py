"""Tests for reporting and visualization module."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import PerformanceMetrics, calculate_metrics
from src.portfolio.portfolio import Trade
from src.reporting.console import (
    format_currency,
    format_metrics_table,
    format_number,
    format_percent,
    format_trade_summary,
)
from src.reporting.charts import (
    create_backtest_report,
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_returns,
    plot_trade_distribution,
    plot_rolling_metrics,
    plot_yearly_returns,
)
from src.signals.base import SignalType


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """Create sample equity curve for testing."""
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0003
    equity = 100000 * (1 + returns).cumprod()
    return pd.Series(equity, index=dates)


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Create sample trades for testing."""
    return [
        Trade(
            symbol="SPY", side=SignalType.LONG, shares=100,
            entry_date=date(2024, 1, 1), entry_price=450.0,
            exit_date=date(2024, 1, 15), exit_price=460.0,
            exit_reason="stop", pnl=1000.0, pnl_pct=2.22, holding_days=14
        ),
        Trade(
            symbol="QQQ", side=SignalType.LONG, shares=50,
            entry_date=date(2024, 1, 5), entry_price=400.0,
            exit_date=date(2024, 1, 20), exit_price=395.0,
            exit_reason="stop", pnl=-250.0, pnl_pct=-1.25, holding_days=15
        ),
        Trade(
            symbol="IWM", side=SignalType.SHORT, shares=75,
            entry_date=date(2024, 1, 10), entry_price=200.0,
            exit_date=date(2024, 1, 25), exit_price=190.0,
            exit_reason="stop", pnl=750.0, pnl_pct=5.0, holding_days=15
        ),
    ]


@pytest.fixture
def sample_metrics(sample_equity_curve: pd.Series, sample_trades: list[Trade]) -> PerformanceMetrics:
    """Create sample metrics for testing."""
    return calculate_metrics(
        equity_curve=sample_equity_curve,
        trades=sample_trades,
        risk_free_rate=0.02,
    )


class TestFormatFunctions:
    """Tests for formatting utility functions."""

    def test_format_number(self) -> None:
        """Test number formatting."""
        assert format_number(1234.5678, 2) == "1,234.57"
        assert format_number(1000000, 0) == "1,000,000"
        assert format_number(0.5, 1, "%") == "0.5%"

    def test_format_percent(self) -> None:
        """Test percentage formatting."""
        assert format_percent(50.5) == "50.50%"
        assert format_percent(-10.123, 1) == "-10.1%"

    def test_format_currency(self) -> None:
        """Test currency formatting."""
        assert format_currency(1000) == "$1,000"
        assert format_currency(1234.56, 2) == "$1,234.56"
        assert format_currency(-500, 0) == "$-500"

    def test_format_none_values(self) -> None:
        """Test formatting with None values."""
        assert format_number(None) == "N/A"
        assert format_percent(None) == "N/A"
        assert format_currency(None) == "N/A"


class TestMetricsFormatting:
    """Tests for metrics table formatting."""

    def test_format_metrics_table(self, sample_metrics: PerformanceMetrics) -> None:
        """Test metrics table formatting."""
        output = format_metrics_table(sample_metrics)

        assert "PERFORMANCE SUMMARY" in output
        assert "RETURNS" in output
        assert "RISK" in output
        assert "RISK-ADJUSTED RETURNS" in output
        assert "TRADE STATISTICS" in output
        assert "Total Return:" in output
        assert "Sharpe Ratio:" in output
        assert "Total Trades:" in output

    def test_format_metrics_with_benchmark(self, sample_equity_curve: pd.Series) -> None:
        """Test metrics table with benchmark comparison."""
        np.random.seed(123)
        benchmark_returns = pd.Series(
            np.random.randn(251) * 0.01,
            index=sample_equity_curve.index[1:]
        )

        metrics = calculate_metrics(
            equity_curve=sample_equity_curve,
            trades=[],
            benchmark_returns=benchmark_returns,
        )

        output = format_metrics_table(metrics)
        assert "BENCHMARK COMPARISON" in output
        assert "Alpha" in output
        assert "Beta" in output


class TestTradeFormatting:
    """Tests for trade summary formatting."""

    def test_format_trade_summary(self, sample_trades: list[Trade]) -> None:
        """Test trade summary formatting."""
        output = format_trade_summary(sample_trades)

        assert "TRADE HISTORY" in output
        assert "SPY" in output
        assert "QQQ" in output
        assert "IWM" in output
        assert "LONG" in output
        assert "SHORT" in output
        assert "Total P&L:" in output

    def test_format_trade_summary_limited(self, sample_trades: list[Trade]) -> None:
        """Test trade summary with max_trades limit."""
        output = format_trade_summary(sample_trades, max_trades=2)

        assert "SPY" in output
        assert "QQQ" in output
        assert "... and 1 more trades" in output

    def test_format_empty_trades(self) -> None:
        """Test formatting with no trades."""
        output = format_trade_summary([])
        assert "No trades to display" in output


class TestEquityCurveChart:
    """Tests for equity curve plotting."""

    def test_plot_equity_curve_basic(self, sample_equity_curve: pd.Series) -> None:
        """Test basic equity curve plot."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for testing

        fig = plot_equity_curve(sample_equity_curve)

        assert fig is not None
        assert len(fig.axes) == 1

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_equity_curve_with_benchmark(self, sample_equity_curve: pd.Series) -> None:
        """Test equity curve with benchmark overlay."""
        import matplotlib
        matplotlib.use("Agg")

        # Create simple benchmark
        benchmark = sample_equity_curve * 1.05

        fig = plot_equity_curve(sample_equity_curve, benchmark_curve=benchmark)

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDrawdownChart:
    """Tests for drawdown plotting."""

    def test_plot_drawdown(self, sample_equity_curve: pd.Series) -> None:
        """Test drawdown plot."""
        import matplotlib
        matplotlib.use("Agg")

        fig = plot_drawdown(sample_equity_curve)

        assert fig is not None
        assert len(fig.axes) == 1

        import matplotlib.pyplot as plt
        plt.close(fig)


class TestMonthlyReturnsChart:
    """Tests for monthly returns heatmap."""

    def test_plot_monthly_returns(self, sample_metrics: PerformanceMetrics) -> None:
        """Test monthly returns heatmap."""
        import matplotlib
        matplotlib.use("Agg")

        fig = plot_monthly_returns(sample_metrics.monthly_returns)

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_monthly_returns_empty(self) -> None:
        """Test monthly returns with empty data."""
        import matplotlib
        matplotlib.use("Agg")

        fig = plot_monthly_returns(pd.Series(dtype=float))

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)


class TestTradeDistributionChart:
    """Tests for trade distribution plotting."""

    def test_plot_trade_distribution(self, sample_trades: list[Trade]) -> None:
        """Test trade distribution histogram."""
        import matplotlib
        matplotlib.use("Agg")

        fig = plot_trade_distribution(sample_trades)

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_trade_distribution_empty(self) -> None:
        """Test trade distribution with no trades."""
        import matplotlib
        matplotlib.use("Agg")

        fig = plot_trade_distribution([])

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)


class TestRollingMetricsChart:
    """Tests for rolling metrics plotting."""

    def test_plot_rolling_metrics(self, sample_metrics: PerformanceMetrics) -> None:
        """Test rolling metrics plot."""
        import matplotlib
        matplotlib.use("Agg")

        fig = plot_rolling_metrics(sample_metrics.daily_returns)

        assert fig is not None
        assert len(fig.axes) == 2  # Two subplots

        import matplotlib.pyplot as plt
        plt.close(fig)


class TestYearlyReturnsChart:
    """Tests for yearly returns bar chart."""

    def test_plot_yearly_returns(self, sample_metrics: PerformanceMetrics) -> None:
        """Test yearly returns bar chart."""
        import matplotlib
        matplotlib.use("Agg")

        fig = plot_yearly_returns(sample_metrics.yearly_returns)

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)


class TestBacktestReport:
    """Tests for comprehensive backtest report."""

    def test_create_backtest_report(
        self,
        sample_metrics: PerformanceMetrics,
        sample_equity_curve: pd.Series,
        sample_trades: list[Trade],
    ) -> None:
        """Test full backtest report creation."""
        import matplotlib
        matplotlib.use("Agg")

        fig = create_backtest_report(
            metrics=sample_metrics,
            equity_curve=sample_equity_curve,
            trades=sample_trades,
        )

        assert fig is not None
        # Report has multiple subplots
        assert len(fig.axes) >= 6

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_create_backtest_report_with_benchmark(
        self,
        sample_metrics: PerformanceMetrics,
        sample_equity_curve: pd.Series,
        sample_trades: list[Trade],
    ) -> None:
        """Test backtest report with benchmark."""
        import matplotlib
        matplotlib.use("Agg")

        benchmark = sample_equity_curve * 0.95

        fig = create_backtest_report(
            metrics=sample_metrics,
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            benchmark_curve=benchmark,
        )

        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)
