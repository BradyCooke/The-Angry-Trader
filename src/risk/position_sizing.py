"""Position sizing based on ATR volatility."""

from dataclasses import dataclass

from src.utils.logging import get_logger

logger = get_logger("position_sizing")


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    volatility_target_pct: float = 2.0  # Target risk per position as % of equity
    max_position_pct: float = 15.0  # Maximum position size as % of equity
    max_gross_exposure_pct: float = 100.0  # Maximum total exposure


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    shares: int
    dollar_value: float
    pct_of_equity: float
    risk_per_share: float  # ATR or stop distance
    was_capped: bool  # True if position was reduced due to max position limit


class PositionSizer:
    """Calculate position sizes based on ATR volatility targeting.

    Uses the formula: Position Size = (Equity × volatility_target_pct) / ATR

    This equalizes the volatility contribution of each position regardless
    of the underlying's price or volatility level.
    """

    def __init__(self, config: PositionSizingConfig | None = None):
        """Initialize position sizer.

        Args:
            config: Position sizing configuration.
        """
        self.config = config or PositionSizingConfig()

    def calculate_position_size(
        self,
        equity: float,
        price: float,
        atr: float,
        current_exposure_pct: float = 0.0,
    ) -> PositionSize:
        """Calculate position size based on ATR volatility targeting.

        Args:
            equity: Current portfolio equity.
            price: Current price of the asset.
            atr: Current ATR (Average True Range) of the asset.
            current_exposure_pct: Current gross exposure as percentage of equity.

        Returns:
            PositionSize object with calculated size.
        """
        if price <= 0 or atr <= 0 or equity <= 0:
            return PositionSize(
                shares=0,
                dollar_value=0.0,
                pct_of_equity=0.0,
                risk_per_share=atr,
                was_capped=False,
            )

        # Calculate volatility-based position size
        # Position Value = (Equity × target_pct) / ATR × Price
        # This targets a specific volatility contribution
        volatility_target = self.config.volatility_target_pct / 100
        target_dollar_risk = equity * volatility_target

        # Dollar value = target_risk * price / atr
        # Because: position_value * atr/price = dollar_risk
        dollar_value = target_dollar_risk * price / atr

        was_capped = False

        # Apply maximum position size limit
        max_position_value = equity * (self.config.max_position_pct / 100)
        if dollar_value > max_position_value:
            dollar_value = max_position_value
            was_capped = True
            logger.debug(f"Position capped at {self.config.max_position_pct}% of equity")

        # Check if adding this position would exceed gross exposure limit
        available_exposure = (self.config.max_gross_exposure_pct - current_exposure_pct) / 100
        max_available_value = equity * available_exposure

        if dollar_value > max_available_value:
            dollar_value = max(0, max_available_value)
            was_capped = True
            logger.debug(f"Position limited by gross exposure constraint")

        # Calculate shares (round down to avoid exceeding limits)
        shares = int(dollar_value / price)
        actual_value = shares * price
        pct_of_equity = (actual_value / equity) * 100 if equity > 0 else 0

        return PositionSize(
            shares=shares,
            dollar_value=actual_value,
            pct_of_equity=pct_of_equity,
            risk_per_share=atr,
            was_capped=was_capped,
        )

    def calculate_dollar_risk(
        self,
        shares: int,
        entry_price: float,
        stop_price: float,
    ) -> float:
        """Calculate dollar risk for a position.

        Args:
            shares: Number of shares.
            entry_price: Entry price.
            stop_price: Stop loss price.

        Returns:
            Dollar risk (always positive).
        """
        return abs(shares * (entry_price - stop_price))

    def calculate_risk_pct(
        self,
        shares: int,
        entry_price: float,
        stop_price: float,
        equity: float,
    ) -> float:
        """Calculate risk as percentage of equity.

        Args:
            shares: Number of shares.
            entry_price: Entry price.
            stop_price: Stop loss price.
            equity: Portfolio equity.

        Returns:
            Risk as percentage of equity.
        """
        if equity <= 0:
            return 0.0
        dollar_risk = self.calculate_dollar_risk(shares, entry_price, stop_price)
        return (dollar_risk / equity) * 100

    def can_add_position(
        self,
        equity: float,
        current_exposure_pct: float,
        min_position_value: float = 1000.0,
    ) -> bool:
        """Check if there's room to add a new position.

        Args:
            equity: Current portfolio equity.
            current_exposure_pct: Current gross exposure percentage.
            min_position_value: Minimum position value to be worthwhile.

        Returns:
            True if a new position can be added.
        """
        available_exposure = self.config.max_gross_exposure_pct - current_exposure_pct
        max_available_value = equity * (available_exposure / 100)
        return max_available_value >= min_position_value
