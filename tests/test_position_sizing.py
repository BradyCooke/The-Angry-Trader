"""Tests for position sizing."""

import pytest

from src.risk.position_sizing import PositionSize, PositionSizer, PositionSizingConfig


class TestPositionSizingConfig:
    """Tests for PositionSizingConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PositionSizingConfig()
        assert config.volatility_target_pct == 2.0
        assert config.max_position_pct == 15.0
        assert config.max_gross_exposure_pct == 100.0


class TestPositionSizer:
    """Tests for PositionSizer."""

    def test_basic_position_sizing(self) -> None:
        """Test basic position size calculation."""
        sizer = PositionSizer()

        # $100,000 equity, $50 stock, $2 ATR
        # Target: 2% risk = $2,000
        # Position value = $2,000 * $50 / $2 = $50,000
        # But 50% exceeds 15% max, so will be capped
        result = sizer.calculate_position_size(
            equity=100000,
            price=50.0,
            atr=2.0,
        )

        assert result.shares > 0
        assert result.dollar_value > 0
        assert result.pct_of_equity <= 15.0  # Within max position limit
        assert result.was_capped  # Should be capped since 50% > 15%

    def test_position_capped_at_max(self) -> None:
        """Test that position is capped at maximum size."""
        config = PositionSizingConfig(
            volatility_target_pct=2.0,
            max_position_pct=15.0,
        )
        sizer = PositionSizer(config)

        # Low ATR would normally result in very large position
        result = sizer.calculate_position_size(
            equity=100000,
            price=100.0,
            atr=0.1,  # Very low ATR
        )

        # Should be capped at 15% = $15,000 = 150 shares
        assert result.pct_of_equity <= 15.0
        assert result.was_capped

    def test_zero_atr_returns_zero(self) -> None:
        """Test that zero ATR returns zero position."""
        sizer = PositionSizer()

        result = sizer.calculate_position_size(
            equity=100000,
            price=50.0,
            atr=0.0,
        )

        assert result.shares == 0
        assert result.dollar_value == 0.0

    def test_zero_equity_returns_zero(self) -> None:
        """Test that zero equity returns zero position."""
        sizer = PositionSizer()

        result = sizer.calculate_position_size(
            equity=0,
            price=50.0,
            atr=2.0,
        )

        assert result.shares == 0

    def test_exposure_limit_applied(self) -> None:
        """Test that gross exposure limit is applied."""
        config = PositionSizingConfig(
            volatility_target_pct=2.0,
            max_position_pct=15.0,
            max_gross_exposure_pct=100.0,
        )
        sizer = PositionSizer(config)

        # Already at 90% exposure
        result = sizer.calculate_position_size(
            equity=100000,
            price=50.0,
            atr=2.0,
            current_exposure_pct=90.0,
        )

        # Should only get up to 10% more
        assert result.pct_of_equity <= 10.1  # Small tolerance for rounding
        assert result.was_capped

    def test_position_sizing_high_volatility(self) -> None:
        """Test position sizing with high volatility stock."""
        sizer = PositionSizer()

        # High ATR should result in smaller position
        result = sizer.calculate_position_size(
            equity=100000,
            price=100.0,
            atr=10.0,  # 10% ATR
        )

        # With 2% target and 10% ATR:
        # Position = $2,000 * $100 / $10 = $20,000 (but capped at 15%)
        assert result.pct_of_equity <= 15.0

    def test_position_sizing_low_volatility(self) -> None:
        """Test position sizing with low volatility stock."""
        config = PositionSizingConfig(
            volatility_target_pct=2.0,
            max_position_pct=50.0,  # High cap for this test
        )
        sizer = PositionSizer(config)

        # Low ATR should result in larger position
        low_vol_result = sizer.calculate_position_size(
            equity=100000,
            price=100.0,
            atr=1.0,  # 1% ATR
        )

        high_vol_result = sizer.calculate_position_size(
            equity=100000,
            price=100.0,
            atr=5.0,  # 5% ATR
        )

        # Low vol should have larger position
        assert low_vol_result.dollar_value > high_vol_result.dollar_value

    def test_calculate_dollar_risk(self) -> None:
        """Test dollar risk calculation."""
        sizer = PositionSizer()

        # 100 shares, entry $50, stop $45 = $500 risk
        risk = sizer.calculate_dollar_risk(
            shares=100,
            entry_price=50.0,
            stop_price=45.0,
        )

        assert risk == 500.0

    def test_calculate_dollar_risk_short(self) -> None:
        """Test dollar risk for short position."""
        sizer = PositionSizer()

        # 100 shares short, entry $50, stop $55 = $500 risk
        risk = sizer.calculate_dollar_risk(
            shares=100,
            entry_price=50.0,
            stop_price=55.0,
        )

        assert risk == 500.0

    def test_calculate_risk_pct(self) -> None:
        """Test risk percentage calculation."""
        sizer = PositionSizer()

        # $500 risk on $100,000 = 0.5%
        risk_pct = sizer.calculate_risk_pct(
            shares=100,
            entry_price=50.0,
            stop_price=45.0,
            equity=100000,
        )

        assert abs(risk_pct - 0.5) < 0.01

    def test_can_add_position_yes(self) -> None:
        """Test can_add_position returns True when room available."""
        sizer = PositionSizer()

        can_add = sizer.can_add_position(
            equity=100000,
            current_exposure_pct=50.0,
        )

        assert can_add

    def test_can_add_position_no(self) -> None:
        """Test can_add_position returns False when at limit."""
        sizer = PositionSizer()

        can_add = sizer.can_add_position(
            equity=100000,
            current_exposure_pct=99.5,
            min_position_value=1000.0,
        )

        assert not can_add

    def test_shares_rounded_down(self) -> None:
        """Test that shares are always rounded down."""
        sizer = PositionSizer()

        result = sizer.calculate_position_size(
            equity=100000,
            price=33.33,  # Odd price
            atr=2.0,
        )

        # Shares should be integer
        assert isinstance(result.shares, int)
        # Actual value should be <= calculated
        assert result.dollar_value <= result.shares * 33.33 + 0.01
