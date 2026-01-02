"""Tests for Keltner Channel signal generator."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.signals.base import Signal, SignalType
from src.signals.keltner import KeltnerConfig, KeltnerSignalGenerator


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


@pytest.fixture
def breakout_data() -> pd.DataFrame:
    """Create data with a clear breakout pattern."""
    n = 60
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

    # Range-bound for first 50 days, then breakout up
    close = np.concatenate([
        np.full(50, 100) + np.random.randn(50) * 0.5,  # Range bound
        np.linspace(100, 115, 10)  # Strong upward breakout
    ])

    return pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(n, 1000000),
    }, index=dates)


class TestKeltnerConfig:
    """Tests for KeltnerConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = KeltnerConfig()
        assert config.ema_period == 50
        assert config.atr_period == 20
        assert config.atr_multiplier == 2.0
        assert config.initial_stop_atr == 2.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = KeltnerConfig(
            ema_period=20,
            atr_period=10,
            atr_multiplier=3.0,
        )
        assert config.ema_period == 20
        assert config.atr_period == 10
        assert config.atr_multiplier == 3.0


class TestKeltnerSignalGenerator:
    """Tests for KeltnerSignalGenerator."""

    def test_calculate_indicators(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test indicator calculation."""
        generator = KeltnerSignalGenerator()
        df = generator.calculate_indicators(sample_ohlcv)

        assert "keltner_middle" in df.columns
        assert "keltner_upper" in df.columns
        assert "keltner_lower" in df.columns
        assert "atr" in df.columns

    def test_generate_signals_returns_list(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test that generate_signals returns a list."""
        generator = KeltnerSignalGenerator()
        signals = generator.generate_signals(sample_ohlcv, "TEST")

        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, Signal)

    def test_signal_has_required_fields(self, breakout_data: pd.DataFrame) -> None:
        """Test that signals have all required fields."""
        # Use shorter periods to ensure signal generation
        config = KeltnerConfig(ema_period=20, atr_period=10)
        generator = KeltnerSignalGenerator(config)
        signals = generator.generate_signals(breakout_data, "TEST")

        if signals:
            signal = signals[0]
            assert signal.symbol == "TEST"
            assert signal.signal_type in [SignalType.LONG, SignalType.SHORT]
            assert signal.price > 0
            assert signal.atr > 0
            assert signal.stop_price is not None

    def test_long_signal_on_breakout(self) -> None:
        """Test long signal generation on upward breakout."""
        # Create simple breakout scenario
        n = 55
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

        # Flat, then spike up
        close = np.concatenate([
            np.full(52, 100.0),
            [105.0, 108.0, 112.0]  # Break above upper band
        ])

        df = pd.DataFrame({
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000000),
        }, index=dates)

        config = KeltnerConfig(ema_period=50, atr_period=20, atr_multiplier=2.0)
        generator = KeltnerSignalGenerator(config)
        signals = generator.generate_signals(df, "TEST")

        # Should have at least one long signal
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1

    def test_short_signal_on_breakdown(self) -> None:
        """Test short signal generation on downward breakdown."""
        n = 55
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

        # Flat, then spike down
        close = np.concatenate([
            np.full(52, 100.0),
            [95.0, 92.0, 88.0]  # Break below lower band
        ])

        df = pd.DataFrame({
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000000),
        }, index=dates)

        config = KeltnerConfig(ema_period=50, atr_period=20, atr_multiplier=2.0)
        generator = KeltnerSignalGenerator(config)
        signals = generator.generate_signals(df, "TEST")

        # Should have at least one short signal
        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        assert len(short_signals) >= 1

    def test_no_signals_insufficient_data(self) -> None:
        """Test no signals with insufficient data."""
        # Only 10 days of data, need at least 50 for EMA
        df = pd.DataFrame({
            "open": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10,
            "volume": [1000000] * 10,
        }, index=pd.date_range(start="2024-01-01", periods=10, freq="D"))

        generator = KeltnerSignalGenerator()
        signals = generator.generate_signals(df, "TEST")

        assert len(signals) == 0

    def test_breakout_strength_calculation(self) -> None:
        """Test breakout strength is calculated correctly."""
        n = 55
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

        # Strong breakout
        close = np.concatenate([
            np.full(52, 100.0),
            [110.0, 115.0, 120.0]  # Very strong breakout
        ])

        df = pd.DataFrame({
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000000),
        }, index=dates)

        config = KeltnerConfig(ema_period=50, atr_period=20)
        generator = KeltnerSignalGenerator(config)
        signals = generator.generate_signals(df, "TEST")

        if signals:
            # Strength should be positive for long signals
            long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
            if long_signals:
                assert long_signals[0].strength > 0


class TestStopLossCalculations:
    """Tests for stop loss calculations."""

    def test_check_stop_hit_long_not_hit(self) -> None:
        """Test stop not hit for long position."""
        generator = KeltnerSignalGenerator()

        is_stopped, _ = generator.check_stop_hit(
            position_type=SignalType.LONG,
            entry_price=100.0,
            current_high=105.0,
            current_low=99.0,  # Above stop
            current_open=102.0,
            stop_price=95.0,
        )

        assert not is_stopped

    def test_check_stop_hit_long_hit(self) -> None:
        """Test stop hit for long position."""
        generator = KeltnerSignalGenerator()

        is_stopped, exit_price = generator.check_stop_hit(
            position_type=SignalType.LONG,
            entry_price=100.0,
            current_high=100.0,
            current_low=93.0,  # Below stop
            current_open=98.0,
            stop_price=95.0,
        )

        assert is_stopped
        assert exit_price == 95.0  # Exit at stop price

    def test_check_stop_hit_long_gap_below(self) -> None:
        """Test stop hit with gap below for long position."""
        generator = KeltnerSignalGenerator()

        is_stopped, exit_price = generator.check_stop_hit(
            position_type=SignalType.LONG,
            entry_price=100.0,
            current_high=92.0,
            current_low=90.0,  # Gapped below stop
            current_open=91.0,
            stop_price=95.0,
        )

        assert is_stopped
        assert exit_price == 91.0  # Exit at open (worse than stop)

    def test_check_stop_hit_short_hit(self) -> None:
        """Test stop hit for short position."""
        generator = KeltnerSignalGenerator()

        is_stopped, exit_price = generator.check_stop_hit(
            position_type=SignalType.SHORT,
            entry_price=100.0,
            current_high=107.0,  # Above stop
            current_low=103.0,
            current_open=104.0,
            stop_price=105.0,
        )

        assert is_stopped
        assert exit_price == 105.0  # Exit at stop price


class TestTrailingStop:
    """Tests for trailing stop calculations."""

    def test_trailing_stop_not_activated(self) -> None:
        """Test trailing stop not activated before profit threshold."""
        config = KeltnerConfig(
            trailing_activation_atr=1.0,
            trailing_stop_atr=2.0,
        )
        generator = KeltnerSignalGenerator(config)

        new_stop, updated = generator.calculate_trailing_stop(
            position_type=SignalType.LONG,
            entry_price=100.0,
            current_stop=96.0,
            highest_high=101.0,
            lowest_low=99.0,
            current_atr=2.0,
            current_close=101.0,  # Only 1% profit, need 2 (1 ATR)
        )

        assert not updated
        assert new_stop == 96.0

    def test_trailing_stop_activated_long(self) -> None:
        """Test trailing stop activation for long position."""
        config = KeltnerConfig(
            trailing_activation_atr=1.0,
            trailing_stop_atr=2.0,
        )
        generator = KeltnerSignalGenerator(config)

        new_stop, updated = generator.calculate_trailing_stop(
            position_type=SignalType.LONG,
            entry_price=100.0,
            current_stop=96.0,  # Initial stop
            highest_high=106.0,
            lowest_low=99.0,
            current_atr=2.0,
            current_close=105.0,  # 5% profit, > 1 ATR (2)
        )

        assert updated
        # New stop should be highest_high - 2*ATR = 106 - 4 = 102
        assert new_stop == 102.0

    def test_trailing_stop_only_moves_up(self) -> None:
        """Test trailing stop only increases for long positions."""
        config = KeltnerConfig(
            trailing_activation_atr=1.0,
            trailing_stop_atr=2.0,
        )
        generator = KeltnerSignalGenerator(config)

        # First update
        new_stop, _ = generator.calculate_trailing_stop(
            position_type=SignalType.LONG,
            entry_price=100.0,
            current_stop=96.0,
            highest_high=106.0,
            lowest_low=99.0,
            current_atr=2.0,
            current_close=105.0,
        )

        # Second update with lower high - should not decrease stop
        new_stop2, updated = generator.calculate_trailing_stop(
            position_type=SignalType.LONG,
            entry_price=100.0,
            current_stop=102.0,  # Previous stop at 102
            highest_high=104.0,  # Lower high
            lowest_low=99.0,
            current_atr=2.0,
            current_close=103.0,
        )

        # Stop should not decrease below 102
        assert new_stop2 == 102.0
        assert not updated
