"""Tests for backtesting engine and metrics."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    PerformanceMetrics,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_metrics,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_trade_statistics,
    calculate_volatility,
)
from src.portfolio.portfolio import Trade
from src.signals.base import SignalType


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """Create sample equity curve for testing."""
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
    # Simulate a portfolio that grows from 100k to 120k with some volatility
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0003  # ~7.5% annual return
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
        Trade(
            symbol="XLF", side=SignalType.LONG, shares=100,
            entry_date=date(2024, 1, 15), entry_price=35.0,
            exit_date=date(2024, 1, 30), exit_price=36.0,
            exit_reason="signal", pnl=100.0, pnl_pct=2.86, holding_days=15
        ),
    ]


class TestCalculateReturns:
    """Tests for return calculations."""

    def test_returns_calculation(self) -> None:
        """Test basic returns calculation."""
        equity = pd.Series([100, 110, 99, 105])
        returns = calculate_returns(equity)

        assert len(returns) == 3
        assert abs(returns.iloc[0] - 0.10) < 0.001
        assert abs(returns.iloc[1] - (-0.10)) < 0.001

    def test_returns_empty(self) -> None:
        """Test returns with insufficient data."""
        equity = pd.Series([100])
        returns = calculate_returns(equity)
        assert len(returns) == 0


class TestCalculateCAGR:
    """Tests for CAGR calculation."""

    def test_cagr_positive(self) -> None:
        """Test CAGR with positive returns."""
        dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
        equity = pd.Series([100000, 110000], index=[dates[0], dates[-1]])

        cagr = calculate_cagr(equity)

        # Should be approximately 10%
        assert abs(cagr - 0.10) < 0.01

    def test_cagr_negative(self) -> None:
        """Test CAGR with negative returns."""
        dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
        equity = pd.Series([100000, 90000], index=[dates[0], dates[-1]])

        cagr = calculate_cagr(equity)

        # Should be approximately -10%
        assert abs(cagr - (-0.10)) < 0.01

    def test_cagr_multi_year(self) -> None:
        """Test CAGR over multiple years."""
        dates = pd.date_range(start="2021-01-01", periods=730, freq="D")
        # Double over 2 years = ~41% CAGR
        equity = pd.Series([100000, 200000], index=[dates[0], dates[-1]])

        cagr = calculate_cagr(equity)

        assert abs(cagr - 0.41) < 0.02


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_max_drawdown_basic(self) -> None:
        """Test basic max drawdown calculation."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
        # Peak at 110, trough at 90 = 18.2% drawdown
        equity = pd.Series([100, 110, 95, 90, 100], index=dates)

        max_dd, peak_date, trough_date = calculate_max_drawdown(equity)

        assert abs(max_dd - 18.18) < 0.1
        assert peak_date == dates[1]
        assert trough_date == dates[3]

    def test_no_drawdown(self) -> None:
        """Test with monotonically increasing equity."""
        equity = pd.Series([100, 110, 120, 130, 140])
        max_dd, _, _ = calculate_max_drawdown(equity)

        assert max_dd == 0.0


class TestVolatility:
    """Tests for volatility calculation."""

    def test_volatility_calculation(self, sample_equity_curve: pd.Series) -> None:
        """Test volatility calculation."""
        returns = calculate_returns(sample_equity_curve)
        vol = calculate_volatility(returns, annualize=True)

        # Should be reasonable annual volatility
        assert 0.05 < vol < 0.50

    def test_volatility_annualization(self) -> None:
        """Test volatility annualization."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01)

        vol_daily = calculate_volatility(returns, annualize=False)
        vol_annual = calculate_volatility(returns, annualize=True)

        # Annual should be ~sqrt(252) times daily
        assert abs(vol_annual / vol_daily - np.sqrt(252)) < 0.1


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_positive(self) -> None:
        """Test Sharpe ratio with positive returns."""
        np.random.seed(42)
        # Positive expected return with some volatility
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

        # Should be positive
        assert sharpe > 0

    def test_sharpe_negative(self) -> None:
        """Test Sharpe ratio with negative returns."""
        np.random.seed(42)
        # Negative expected return
        returns = pd.Series(np.random.randn(252) * 0.01 - 0.001)

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

        # Should be negative
        assert sharpe < 0


class TestTradeStatistics:
    """Tests for trade statistics calculation."""

    def test_trade_stats_basic(self, sample_trades: list[Trade]) -> None:
        """Test basic trade statistics."""
        stats = calculate_trade_statistics(sample_trades)

        assert stats["total_trades"] == 4
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 1
        assert abs(stats["win_rate"] - 75.0) < 0.1

    def test_profit_factor(self, sample_trades: list[Trade]) -> None:
        """Test profit factor calculation."""
        stats = calculate_trade_statistics(sample_trades)

        # Gross profit = 1000 + 750 + 100 = 1850
        # Gross loss = 250
        # PF = 1850 / 250 = 7.4
        assert abs(stats["profit_factor"] - 7.4) < 0.1

    def test_empty_trades(self) -> None:
        """Test statistics with no trades."""
        stats = calculate_trade_statistics([])

        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0


class TestCalculateMetrics:
    """Tests for comprehensive metrics calculation."""

    def test_calculate_all_metrics(
        self,
        sample_equity_curve: pd.Series,
        sample_trades: list[Trade],
    ) -> None:
        """Test calculation of all metrics."""
        metrics = calculate_metrics(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            risk_free_rate=0.02,
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return_pct != 0
        assert metrics.cagr != 0
        assert metrics.max_drawdown_pct >= 0
        assert metrics.volatility > 0
        assert metrics.total_trades == 4

    def test_metrics_with_benchmark(self, sample_equity_curve: pd.Series) -> None:
        """Test metrics with benchmark comparison."""
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

        # Should have benchmark metrics
        assert metrics.beta != 0 or metrics.correlation != 0

    def test_metrics_insufficient_data(self) -> None:
        """Test metrics with insufficient data."""
        equity = pd.Series([100000])
        metrics = calculate_metrics(equity, [], None)

        assert metrics.total_return_pct == 0
        assert metrics.cagr == 0
