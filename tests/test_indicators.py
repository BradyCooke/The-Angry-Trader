"""Tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from src.signals.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_keltner_channels,
    calculate_returns,
    calculate_sma,
    calculate_true_range,
    calculate_volatility,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100

    # Generate trending price data
    base = 100
    trend = np.cumsum(np.random.randn(n) * 0.5)
    close = base + trend

    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.5,
        "high": close + np.abs(np.random.randn(n)) * 1.5,
        "low": close - np.abs(np.random.randn(n)) * 1.5,
        "close": close,
        "volume": np.random.randint(1000000, 5000000, n),
    }, index=pd.date_range(start="2024-01-01", periods=n, freq="D"))


class TestEMA:
    """Tests for EMA calculation."""

    def test_ema_length(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test that EMA returns same length as input."""
        ema = calculate_ema(sample_ohlcv["close"], period=20)
        assert len(ema) == len(sample_ohlcv)

    def test_ema_first_values_nan(self) -> None:
        """Test that early EMA values are not NaN (EWM handles warmup)."""
        series = pd.Series([1, 2, 3, 4, 5])
        ema = calculate_ema(series, period=3)
        # EWM doesn't produce NaN, it starts from first value
        assert not pd.isna(ema.iloc[0])

    def test_ema_responds_to_price_changes(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test that EMA follows price trend."""
        ema = calculate_ema(sample_ohlcv["close"], period=10)
        # EMA should generally follow close price direction
        assert ema.corr(sample_ohlcv["close"]) > 0.9


class TestSMA:
    """Tests for SMA calculation."""

    def test_sma_calculation(self) -> None:
        """Test SMA calculation correctness."""
        series = pd.Series([1, 2, 3, 4, 5])
        sma = calculate_sma(series, period=3)

        # First two values should be NaN
        assert pd.isna(sma.iloc[0])
        assert pd.isna(sma.iloc[1])
        # Third value should be (1+2+3)/3 = 2
        assert sma.iloc[2] == 2.0
        # Fourth value should be (2+3+4)/3 = 3
        assert sma.iloc[3] == 3.0


class TestTrueRange:
    """Tests for True Range calculation."""

    def test_true_range_normal(self) -> None:
        """Test True Range with normal price action."""
        high = pd.Series([102, 104, 103])
        low = pd.Series([98, 100, 99])
        close = pd.Series([100, 103, 101])

        tr = calculate_true_range(high, low, close)

        # First value: high - low = 102 - 98 = 4
        assert tr.iloc[0] == 4.0
        # Second value: max(104-100, |104-100|, |100-100|) = 4
        assert tr.iloc[1] == 4.0

    def test_true_range_gap_up(self) -> None:
        """Test True Range handles gap up."""
        high = pd.Series([100, 110])  # Gap up
        low = pd.Series([98, 108])
        close = pd.Series([99, 109])

        tr = calculate_true_range(high, low, close)

        # Second value should account for gap: |110 - 99| = 11
        assert tr.iloc[1] == 11.0


class TestATR:
    """Tests for ATR calculation."""

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test that ATR is always positive."""
        atr = calculate_atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            period=20
        )
        assert (atr.dropna() > 0).all()

    def test_atr_smoothing(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test that ATR is smoother than True Range."""
        tr = calculate_true_range(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )
        atr = calculate_atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            period=20
        )

        # ATR should have lower standard deviation than TR
        assert atr.std() < tr.std()


class TestKeltnerChannels:
    """Tests for Keltner Channel calculation."""

    def test_keltner_returns_three_series(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test that Keltner returns middle, upper, lower bands."""
        middle, upper, lower = calculate_keltner_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )

        assert len(middle) == len(sample_ohlcv)
        assert len(upper) == len(sample_ohlcv)
        assert len(lower) == len(sample_ohlcv)

    def test_keltner_band_ordering(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test that upper > middle > lower."""
        middle, upper, lower = calculate_keltner_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )

        # Skip NaN values
        valid_idx = ~(middle.isna() | upper.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_keltner_band_width(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test that band width depends on ATR multiplier."""
        middle1, upper1, lower1 = calculate_keltner_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            atr_multiplier=2.0,
        )

        middle2, upper2, lower2 = calculate_keltner_channels(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            atr_multiplier=3.0,
        )

        # Middle should be same
        assert (middle1 == middle2).all()

        # Width with multiplier 3 should be 1.5x width with multiplier 2
        width1 = upper1 - lower1
        width2 = upper2 - lower2

        ratio = (width2 / width1).dropna()
        assert np.allclose(ratio, 1.5, atol=0.01)


class TestReturns:
    """Tests for returns calculation."""

    def test_returns_calculation(self) -> None:
        """Test returns calculation correctness."""
        prices = pd.Series([100, 110, 99])
        returns = calculate_returns(prices)

        assert pd.isna(returns.iloc[0])
        assert np.isclose(returns.iloc[1], 0.10)  # 10% gain
        assert np.isclose(returns.iloc[2], -0.10)  # 10% loss


class TestVolatility:
    """Tests for volatility calculation."""

    def test_volatility_annualization(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test volatility annualization."""
        returns = calculate_returns(sample_ohlcv["close"])

        vol_daily = calculate_volatility(returns, period=20, annualize=False)
        vol_annual = calculate_volatility(returns, period=20, annualize=True)

        # Annualized should be ~sqrt(252) times daily
        ratio = (vol_annual / vol_daily).dropna()
        expected_ratio = np.sqrt(252)
        assert np.allclose(ratio, expected_ratio, atol=0.01)
