"""Keltner Channel signal generator."""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.utils.logging import get_logger
from .base import Signal, SignalGenerator, SignalType
from .indicators import calculate_atr, calculate_keltner_channels

logger = get_logger("keltner")


@dataclass
class KeltnerConfig:
    """Configuration for Keltner Channel signals."""
    ema_period: int = 50
    atr_period: int = 20
    atr_multiplier: float = 2.0
    initial_stop_atr: float = 2.0
    trailing_stop_atr: float = 2.0
    trailing_activation_atr: float = 1.0


class KeltnerSignalGenerator(SignalGenerator):
    """Generate trading signals based on Keltner Channel breakouts.

    Entry Signals:
    - Long: Price closes above upper Keltner band
    - Short: Price closes below lower Keltner band

    Re-entry Rules:
    - After stop-out, price must return inside bands first
    - Then a new breakout must occur for re-entry
    """

    def __init__(self, config: KeltnerConfig | None = None):
        """Initialize Keltner signal generator.

        Args:
            config: Keltner configuration. Uses defaults if not provided.
        """
        self.config = config or KeltnerConfig()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channel indicators.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            DataFrame with indicator columns added:
            - keltner_middle: EMA middle line
            - keltner_upper: Upper band
            - keltner_lower: Lower band
            - atr: Average True Range
        """
        df = df.copy()

        middle, upper, lower = calculate_keltner_channels(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            ema_period=self.config.ema_period,
            atr_period=self.config.atr_period,
            atr_multiplier=self.config.atr_multiplier,
        )

        df["keltner_middle"] = middle
        df["keltner_upper"] = upper
        df["keltner_lower"] = lower
        df["atr"] = calculate_atr(
            df["high"], df["low"], df["close"],
            period=self.config.atr_period
        )

        return df

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> list[Signal]:
        """Generate Keltner Channel breakout signals.

        Args:
            df: DataFrame with OHLCV data indexed by date.
            symbol: ETF symbol.

        Returns:
            List of Signal objects for breakout entries.
        """
        if len(df) < max(self.config.ema_period, self.config.atr_period):
            return []

        # Calculate indicators
        df = self.calculate_indicators(df)

        signals = []
        inside_band = True  # Track if price has returned inside bands

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # Skip if indicators not yet valid
            if pd.isna(row["keltner_upper"]) or pd.isna(row["atr"]):
                continue

            current_date = df.index[i]
            if isinstance(current_date, pd.Timestamp):
                current_date = current_date.date()

            close = row["close"]
            upper = row["keltner_upper"]
            lower = row["keltner_lower"]
            atr = row["atr"]

            prev_close = prev_row["close"]
            prev_upper = prev_row["keltner_upper"]
            prev_lower = prev_row["keltner_lower"]

            # Check if price is inside bands (for re-entry logic)
            if lower <= close <= upper:
                inside_band = True

            # Long breakout: close above upper band
            if close > upper and prev_close <= prev_upper and inside_band:
                # Calculate breakout strength (distance beyond band as % of ATR)
                strength = (close - upper) / atr if atr > 0 else 0

                # Calculate stop price
                stop_price = close - (self.config.initial_stop_atr * atr)

                signal = Signal(
                    date=current_date,
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    price=close,
                    strength=strength,
                    atr=atr,
                    stop_price=stop_price,
                    metadata={
                        "keltner_upper": upper,
                        "keltner_lower": lower,
                        "keltner_middle": row["keltner_middle"],
                    }
                )
                signals.append(signal)
                inside_band = False  # Reset for re-entry logic

            # Short breakout: close below lower band
            elif close < lower and prev_close >= prev_lower and inside_band:
                # Calculate breakout strength
                strength = (lower - close) / atr if atr > 0 else 0

                # Calculate stop price
                stop_price = close + (self.config.initial_stop_atr * atr)

                signal = Signal(
                    date=current_date,
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    price=close,
                    strength=strength,
                    atr=atr,
                    stop_price=stop_price,
                    metadata={
                        "keltner_upper": upper,
                        "keltner_lower": lower,
                        "keltner_middle": row["keltner_middle"],
                    }
                )
                signals.append(signal)
                inside_band = False  # Reset for re-entry logic

        logger.debug(f"Generated {len(signals)} signals for {symbol}")
        return signals

    def check_stop_hit(
        self,
        position_type: SignalType,
        entry_price: float,
        current_high: float,
        current_low: float,
        current_open: float,
        stop_price: float,
    ) -> tuple[bool, float]:
        """Check if stop loss is hit.

        Args:
            position_type: LONG or SHORT.
            entry_price: Original entry price.
            current_high: Current bar's high.
            current_low: Current bar's low.
            current_open: Current bar's open (used if gap through stop).
            stop_price: Current stop price level.

        Returns:
            Tuple of (is_stopped, exit_price).
        """
        if position_type == SignalType.LONG:
            if current_low <= stop_price:
                # Stopped out - use stop price or open if gapped below
                exit_price = min(stop_price, current_open)
                return True, exit_price
        elif position_type == SignalType.SHORT:
            if current_high >= stop_price:
                # Stopped out - use stop price or open if gapped above
                exit_price = max(stop_price, current_open)
                return True, exit_price

        return False, 0.0

    def calculate_trailing_stop(
        self,
        position_type: SignalType,
        entry_price: float,
        current_stop: float,
        highest_high: float,
        lowest_low: float,
        current_atr: float,
        current_close: float,
    ) -> tuple[float, bool]:
        """Calculate updated trailing stop.

        Trailing stop activates after 1×ATR profit is reached.

        Args:
            position_type: LONG or SHORT.
            entry_price: Original entry price.
            current_stop: Current stop price.
            highest_high: Highest high since entry (for longs).
            lowest_low: Lowest low since entry (for shorts).
            current_atr: Current ATR value.
            current_close: Current close price.

        Returns:
            Tuple of (new_stop_price, was_updated).
        """
        profit_threshold = entry_price + (self.config.trailing_activation_atr * current_atr)

        if position_type == SignalType.LONG:
            # Check if profit threshold reached
            if current_close >= profit_threshold:
                # Calculate trailing stop from highest high
                new_stop = highest_high - (self.config.trailing_stop_atr * current_atr)
                if new_stop > current_stop:
                    return new_stop, True

        elif position_type == SignalType.SHORT:
            # For shorts, profit threshold is below entry
            profit_threshold = entry_price - (self.config.trailing_activation_atr * current_atr)
            if current_close <= profit_threshold:
                # Calculate trailing stop from lowest low
                new_stop = lowest_low + (self.config.trailing_stop_atr * current_atr)
                if new_stop < current_stop:
                    return new_stop, True

        return current_stop, False
