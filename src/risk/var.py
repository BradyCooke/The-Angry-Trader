"""Value at Risk (VaR) calculations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("var")


@dataclass
class VaRConfig:
    """Configuration for VaR calculations."""
    confidence_level: float = 0.95  # 95% confidence
    time_horizon_days: int = 20  # 20-day VaR
    max_var_pct: float = 20.0  # Maximum acceptable VaR as % of equity
    min_history_days: int = 252  # Minimum history for reliable VaR


@dataclass
class VaRResult:
    """Result of VaR calculation."""
    var_pct: float  # VaR as percentage of portfolio
    var_dollar: float  # VaR in dollar terms
    exceeds_limit: bool  # True if VaR exceeds max threshold
    confidence_level: float
    time_horizon_days: int
    observations_used: int


class VaRCalculator:
    """Calculate Historical Value at Risk for portfolio.

    Uses historical simulation method:
    1. Calculate historical returns for each position
    2. Calculate portfolio returns based on current weights
    3. Find the return at the specified percentile (e.g., 5th for 95% VaR)
    4. Scale to time horizon if needed
    """

    def __init__(self, config: VaRConfig | None = None):
        """Initialize VaR calculator.

        Args:
            config: VaR configuration.
        """
        self.config = config or VaRConfig()

    def calculate_portfolio_var(
        self,
        returns_dict: dict[str, pd.Series],
        weights: dict[str, float],
        equity: float,
    ) -> VaRResult:
        """Calculate portfolio VaR using historical simulation.

        Args:
            returns_dict: Dictionary mapping symbol to daily returns Series.
            weights: Dictionary mapping symbol to portfolio weight (as decimal).
            equity: Current portfolio equity.

        Returns:
            VaRResult with VaR metrics.
        """
        if not returns_dict or not weights:
            return VaRResult(
                var_pct=0.0,
                var_dollar=0.0,
                exceeds_limit=False,
                confidence_level=self.config.confidence_level,
                time_horizon_days=self.config.time_horizon_days,
                observations_used=0,
            )

        # Align all returns to common dates
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        if len(returns_df) < self.config.min_history_days // 4:
            logger.warning(f"Insufficient history for VaR: {len(returns_df)} days")
            # Return conservative estimate
            return VaRResult(
                var_pct=self.config.max_var_pct,
                var_dollar=equity * self.config.max_var_pct / 100,
                exceeds_limit=True,
                confidence_level=self.config.confidence_level,
                time_horizon_days=self.config.time_horizon_days,
                observations_used=len(returns_df),
            )

        # Calculate portfolio returns for each day
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        for symbol, weight in weights.items():
            if symbol in returns_df.columns:
                portfolio_returns += returns_df[symbol] * weight

        # Calculate n-day rolling returns for time horizon
        if self.config.time_horizon_days > 1:
            # Compound returns over time horizon
            rolling_returns = (
                (1 + portfolio_returns)
                .rolling(window=self.config.time_horizon_days)
                .apply(lambda x: x.prod() - 1, raw=True)
            ).dropna()
        else:
            rolling_returns = portfolio_returns

        if len(rolling_returns) == 0:
            return VaRResult(
                var_pct=0.0,
                var_dollar=0.0,
                exceeds_limit=False,
                confidence_level=self.config.confidence_level,
                time_horizon_days=self.config.time_horizon_days,
                observations_used=0,
            )

        # Find VaR at specified confidence level
        # VaR is the loss at the (1 - confidence) percentile
        percentile = (1 - self.config.confidence_level) * 100
        var_return = np.percentile(rolling_returns, percentile)

        # VaR is typically expressed as positive number
        var_pct = abs(var_return) * 100
        var_dollar = equity * abs(var_return)

        exceeds_limit = var_pct > self.config.max_var_pct

        if exceeds_limit:
            logger.warning(
                f"Portfolio VaR ({var_pct:.1f}%) exceeds limit ({self.config.max_var_pct}%)"
            )

        return VaRResult(
            var_pct=var_pct,
            var_dollar=var_dollar,
            exceeds_limit=exceeds_limit,
            confidence_level=self.config.confidence_level,
            time_horizon_days=self.config.time_horizon_days,
            observations_used=len(rolling_returns),
        )

    def calculate_marginal_var(
        self,
        returns_dict: dict[str, pd.Series],
        current_weights: dict[str, float],
        new_symbol: str,
        new_weight: float,
        equity: float,
    ) -> tuple[VaRResult, VaRResult]:
        """Calculate VaR before and after adding a new position.

        Args:
            returns_dict: Dictionary mapping symbol to daily returns Series.
            current_weights: Current portfolio weights.
            new_symbol: Symbol to add.
            new_weight: Weight of new position.
            equity: Portfolio equity.

        Returns:
            Tuple of (current_var, new_var) after adding position.
        """
        # Current VaR
        current_var = self.calculate_portfolio_var(returns_dict, current_weights, equity)

        # Calculate new weights (scale down existing)
        total_new_weight = sum(current_weights.values()) + new_weight
        new_weights = {s: w / total_new_weight for s, w in current_weights.items()}
        new_weights[new_symbol] = new_weight / total_new_weight

        # New VaR
        new_var = self.calculate_portfolio_var(returns_dict, new_weights, equity)

        return current_var, new_var

    def check_var_limit(
        self,
        returns_dict: dict[str, pd.Series],
        weights: dict[str, float],
        equity: float,
    ) -> bool:
        """Check if portfolio VaR is within acceptable limits.

        Args:
            returns_dict: Dictionary mapping symbol to daily returns Series.
            weights: Portfolio weights.
            equity: Portfolio equity.

        Returns:
            True if VaR is within limits, False if exceeded.
        """
        result = self.calculate_portfolio_var(returns_dict, weights, equity)
        return not result.exceeds_limit

    def calculate_single_position_var(
        self,
        returns: pd.Series,
        position_value: float,
    ) -> float:
        """Calculate VaR for a single position.

        Args:
            returns: Daily returns series for the position.
            position_value: Dollar value of the position.

        Returns:
            VaR in dollars.
        """
        if len(returns) < self.config.min_history_days // 4:
            # Conservative estimate using 2x standard deviation
            return position_value * returns.std() * 2 * np.sqrt(self.config.time_horizon_days)

        # Calculate rolling returns for time horizon
        if self.config.time_horizon_days > 1:
            rolling_returns = (
                (1 + returns)
                .rolling(window=self.config.time_horizon_days)
                .apply(lambda x: x.prod() - 1, raw=True)
            ).dropna()
        else:
            rolling_returns = returns

        percentile = (1 - self.config.confidence_level) * 100
        var_return = np.percentile(rolling_returns, percentile)

        return position_value * abs(var_return)
