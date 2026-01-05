"""Backtesting engine."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from src.portfolio.portfolio import Portfolio, Trade
from src.risk.position_sizing import PositionSizer, PositionSizingConfig
from src.risk.var import VaRCalculator, VaRConfig
from src.signals.base import Signal, SignalType
from src.signals.keltner import KeltnerConfig, KeltnerSignalGenerator
from src.utils.config import Config
from src.utils.logging import get_logger, TradeLogger

from .metrics import PerformanceMetrics, calculate_metrics, calculate_returns

logger = get_logger("backtest")


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    metrics: PerformanceMetrics
    portfolio: Portfolio
    equity_curve: pd.Series
    trades: list[Trade]
    daily_returns: pd.Series
    config: dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtesting engine.

    Simulation flow per bar:
    1. Update prices for all holdings
    2. Check stop losses, execute any exits
    3. Update trailing stops for positions past activation threshold
    4. Calculate portfolio value and VaR
    5. Generate new signals from today's close
    6. If VaR < threshold, process new entries by priority
    7. Log trades and portfolio state
    8. Advance to next bar
    """

    def __init__(self, config: Config):
        """Initialize backtest engine.

        Args:
            config: System configuration.
        """
        self.config = config

        # Initialize components
        self.signal_generator = KeltnerSignalGenerator(
            KeltnerConfig(
                ema_period=config.strategy.keltner.ema_period,
                atr_period=config.strategy.keltner.atr_period,
                atr_multiplier=config.strategy.keltner.atr_multiplier,
                initial_stop_atr=config.strategy.stops.initial_atr_multiple,
                trailing_stop_atr=config.strategy.stops.trailing_atr_multiple,
                trailing_activation_atr=config.strategy.stops.trailing_activation_atr,
            )
        )

        self.position_sizer = PositionSizer(
            PositionSizingConfig(
                volatility_target_pct=config.risk_management.position_sizing.volatility_target_pct,
                max_position_pct=config.risk_management.position_sizing.max_position_pct,
                max_gross_exposure_pct=config.risk_management.portfolio.max_gross_exposure_pct,
            )
        )

        self.var_calculator = VaRCalculator(
            VaRConfig(
                confidence_level=config.risk_management.var.confidence_level,
                time_horizon_days=config.risk_management.var.time_horizon_days,
                max_var_pct=config.risk_management.var.max_var_pct,
            )
        )

        self.trade_logger = TradeLogger()

    def run(
        self,
        data: dict[str, pd.DataFrame],
        start_date: date | None = None,
        end_date: date | None = None,
        benchmark_data: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            data: Dictionary mapping symbol to OHLCV DataFrame.
            start_date: Optional start date for backtest.
            end_date: Optional end date for backtest.
            benchmark_data: Optional benchmark data for comparison.

        Returns:
            BacktestResult with all metrics and data.
        """
        # Initialize portfolio
        portfolio = Portfolio(
            starting_capital=self.config.portfolio.starting_capital,
            risk_free_rate=0.0,  # Will use T-bill rates if available
        )

        # Get all trading dates
        all_dates = self._get_trading_dates(data, start_date, end_date)

        if len(all_dates) == 0:
            logger.warning("No trading dates found in data")
            return self._empty_result(portfolio)

        logger.info(f"Running backtest from {all_dates[0]} to {all_dates[-1]}")
        logger.info(f"Trading {len(data)} symbols over {len(all_dates)} days")

        # Pre-calculate indicators for all symbols
        indicator_data = {}
        for symbol, df in data.items():
            indicator_data[symbol] = self.signal_generator.calculate_indicators(df)

        # Pre-compute ALL signals upfront (major optimization)
        logger.info("Pre-computing signals for all symbols...")
        precomputed_signals = self._precompute_all_signals(indicator_data)
        logger.info(f"Pre-computed {sum(len(v) for v in precomputed_signals.values())} signals")

        # Track returns for VaR calculation
        returns_data = {symbol: calculate_returns(df["close"]) for symbol, df in data.items()}

        # Main simulation loop
        equity_history = []

        for i, current_date in enumerate(all_dates):
            # Get current prices for all symbols
            prices = self._get_prices_for_date(data, current_date)

            if not prices:
                continue

            # Step 1-3: Process exits (stops)
            self._process_exits(portfolio, data, indicator_data, current_date, prices)

            # Step 4: Calculate portfolio state
            equity = portfolio.calculate_equity(prices)
            exposure_pct = portfolio.calculate_gross_exposure_pct(prices)

            # Calculate VaR if we have positions
            var_exceeded = False
            if portfolio.position_count > 0:
                weights = portfolio.get_position_weights(prices)
                var_result = self.var_calculator.calculate_portfolio_var(
                    returns_dict=returns_data,
                    weights=weights,
                    equity=equity,
                )
                var_exceeded = var_result.exceeds_limit

            # Step 5: Get pre-computed signals for this date
            signals = self._get_signals_for_date(precomputed_signals, current_date, portfolio)

            # Step 6: Process entries if VaR allows
            if not var_exceeded and signals:
                self._process_entries(
                    portfolio=portfolio,
                    signals=signals,
                    prices=prices,
                    current_date=current_date,
                    equity=equity,
                    exposure_pct=exposure_pct,
                )

            # Step 7: Record portfolio state
            equity_history.append((current_date, equity))

            # Accrue interest on cash
            if self.config.portfolio.cash_earns_risk_free:
                portfolio.accrue_cash_interest(days=1)

        # Build results
        equity_curve = pd.Series(
            {d: e for d, e in equity_history},
            name="equity"
        )

        # Calculate benchmark returns if available
        benchmark_returns = None
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_returns = calculate_returns(benchmark_data["close"])

        # Calculate metrics
        metrics = calculate_metrics(
            equity_curve=equity_curve,
            trades=portfolio.trades,
            benchmark_returns=benchmark_returns,
            risk_free_rate=0.0,
        )

        return BacktestResult(
            metrics=metrics,
            portfolio=portfolio,
            equity_curve=equity_curve,
            trades=portfolio.trades,
            daily_returns=metrics.daily_returns,
            config=self._get_config_dict(),
        )

    def _get_trading_dates(
        self,
        data: dict[str, pd.DataFrame],
        start_date: date | None,
        end_date: date | None,
    ) -> list[date]:
        """Get sorted list of all trading dates."""
        all_dates = set()
        for df in data.values():
            for d in df.index:
                if isinstance(d, pd.Timestamp):
                    d = d.date()
                all_dates.add(d)

        dates = sorted(all_dates)

        # Apply date filters
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        return dates

    def _get_prices_for_date(
        self,
        data: dict[str, pd.DataFrame],
        current_date: date,
    ) -> dict[str, float]:
        """Get closing prices for all symbols on a date."""
        prices = {}
        for symbol, df in data.items():
            if current_date in df.index:
                prices[symbol] = df.loc[current_date, "close"]
            elif isinstance(df.index, pd.DatetimeIndex):
                # Try with timestamp
                ts = pd.Timestamp(current_date)
                if ts in df.index:
                    prices[symbol] = df.loc[ts, "close"]
        return prices

    def _get_bar_for_date(
        self,
        data: dict[str, pd.DataFrame],
        symbol: str,
        current_date: date,
    ) -> pd.Series | None:
        """Get OHLCV bar for a symbol on a date."""
        df = data.get(symbol)
        if df is None:
            return None

        if current_date in df.index:
            return df.loc[current_date]
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = pd.Timestamp(current_date)
            if ts in df.index:
                return df.loc[ts]
        return None

    def _process_exits(
        self,
        portfolio: Portfolio,
        data: dict[str, pd.DataFrame],
        indicator_data: dict[str, pd.DataFrame],
        current_date: date,
        prices: dict[str, float],
    ) -> None:
        """Process stop losses and exits."""
        symbols_to_close = []

        for symbol, position in list(portfolio.positions.items()):
            bar = self._get_bar_for_date(data, symbol, current_date)
            if bar is None:
                continue

            # Check if stop is hit
            is_stopped, exit_price = self.signal_generator.check_stop_hit(
                position_type=position.side,
                entry_price=position.entry_price,
                current_high=bar["high"],
                current_low=bar["low"],
                current_open=bar["open"],
                stop_price=position.stop_price,
            )

            if is_stopped:
                symbols_to_close.append((symbol, exit_price, "stop"))
            else:
                # Update trailing stop
                ind_bar = self._get_bar_for_date(indicator_data, symbol, current_date)
                if ind_bar is not None and "atr" in ind_bar.index:
                    position.update_trailing_stop(
                        current_high=bar["high"],
                        current_low=bar["low"],
                        current_close=bar["close"],
                        current_atr=ind_bar["atr"],
                        trailing_activation_atr=self.config.strategy.stops.trailing_activation_atr,
                        trailing_stop_atr=self.config.strategy.stops.trailing_atr_multiple,
                    )

        # Close stopped positions
        for symbol, exit_price, reason in symbols_to_close:
            portfolio.close_position(symbol, exit_price, current_date, reason)

    def _precompute_all_signals(
        self,
        indicator_data: dict[str, pd.DataFrame],
    ) -> dict[date, list[Signal]]:
        """Pre-compute all signals for all symbols and dates.

        This is a major optimization - compute signals once upfront
        instead of recalculating for every date during simulation.

        Returns:
            Dictionary mapping date to list of signals for that date.
        """
        signals_by_date: dict[date, list[Signal]] = {}

        for symbol, df in indicator_data.items():
            # Generate all signals for this symbol at once
            signals = self.signal_generator.generate_signals(df, symbol)

            # Group by date
            for signal in signals:
                if signal.date not in signals_by_date:
                    signals_by_date[signal.date] = []
                signals_by_date[signal.date].append(signal)

        # Sort each date's signals by strength
        for d in signals_by_date:
            signals_by_date[d].sort(key=lambda s: s.strength, reverse=True)

        return signals_by_date

    def _get_signals_for_date(
        self,
        precomputed_signals: dict[date, list[Signal]],
        current_date: date,
        portfolio: Portfolio,
    ) -> list[Signal]:
        """Get pre-computed signals for a date, filtering out existing positions.

        Args:
            precomputed_signals: Pre-computed signals dictionary.
            current_date: Date to get signals for.
            portfolio: Current portfolio state.

        Returns:
            List of signals for symbols not already in portfolio.
        """
        if current_date not in precomputed_signals:
            return []

        # Filter out symbols we already have positions in
        return [
            s for s in precomputed_signals[current_date]
            if not portfolio.has_position(s.symbol)
        ]

    def _process_entries(
        self,
        portfolio: Portfolio,
        signals: list[Signal],
        prices: dict[str, float],
        current_date: date,
        equity: float,
        exposure_pct: float,
    ) -> None:
        """Process entry signals."""
        for signal in signals:
            # Check if we can still add positions
            if not self.position_sizer.can_add_position(equity, exposure_pct):
                break

            # Calculate position size
            size = self.position_sizer.calculate_position_size(
                equity=equity,
                price=signal.price,
                atr=signal.atr,
                current_exposure_pct=exposure_pct,
            )

            shares = size.shares

            # Apply short size multiplier for short positions
            if signal.signal_type == SignalType.SHORT:
                short_mult = self.config.risk_management.position_sizing.short_size_multiplier
                shares = int(shares * short_mult)

            if shares <= 0:
                continue

            # Check if we have enough cash
            cost = shares * signal.price
            if cost > portfolio.cash:
                continue

            # Open position
            try:
                portfolio.open_position(
                    symbol=signal.symbol,
                    side=signal.signal_type,
                    shares=shares,
                    entry_price=signal.price,
                    entry_date=current_date,
                    stop_price=signal.stop_price or (signal.price - 2 * signal.atr),
                    atr=signal.atr,
                    metadata=signal.metadata,
                )

                # Update exposure
                exposure_pct = portfolio.calculate_gross_exposure_pct(prices)

            except ValueError as e:
                logger.warning(f"Could not open position for {signal.symbol}: {e}")

    def _get_config_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary for results."""
        return {
            "strategy": {
                "ema_period": self.config.strategy.keltner.ema_period,
                "atr_period": self.config.strategy.keltner.atr_period,
                "atr_multiplier": self.config.strategy.keltner.atr_multiplier,
            },
            "risk": {
                "volatility_target_pct": self.config.risk_management.position_sizing.volatility_target_pct,
                "max_position_pct": self.config.risk_management.position_sizing.max_position_pct,
                "max_var_pct": self.config.risk_management.var.max_var_pct,
            },
            "portfolio": {
                "starting_capital": self.config.portfolio.starting_capital,
            },
        }

    def _empty_result(self, portfolio: Portfolio) -> BacktestResult:
        """Return empty result for edge cases."""
        return BacktestResult(
            metrics=PerformanceMetrics(),
            portfolio=portfolio,
            equity_curve=pd.Series(dtype=float),
            trades=[],
            daily_returns=pd.Series(dtype=float),
        )
