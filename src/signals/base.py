"""Base classes for signal generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

import pandas as pd


class SignalType(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    NONE = "NONE"


@dataclass
class Signal:
    """Trading signal with metadata."""
    date: date
    symbol: str
    signal_type: SignalType
    price: float
    strength: float = 0.0  # Breakout strength for prioritization
    atr: float = 0.0  # ATR at signal time for position sizing
    stop_price: float | None = None  # Suggested stop price
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_entry(self) -> bool:
        """Check if signal is an entry signal."""
        return self.signal_type in (SignalType.LONG, SignalType.SHORT)

    @property
    def is_exit(self) -> bool:
        """Check if signal is an exit signal."""
        return self.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT)

    @property
    def is_long(self) -> bool:
        """Check if signal is long-related."""
        return self.signal_type in (SignalType.LONG, SignalType.EXIT_LONG)

    @property
    def is_short(self) -> bool:
        """Check if signal is short-related."""
        return self.signal_type in (SignalType.SHORT, SignalType.EXIT_SHORT)


class SignalGenerator(ABC):
    """Abstract base class for signal generators."""

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> list[Signal]:
        """Generate trading signals from price data.

        Args:
            df: DataFrame with OHLCV data indexed by date.
            symbol: ETF symbol.

        Returns:
            List of Signal objects.
        """
        pass

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators needed for signal generation.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            DataFrame with indicator columns added.
        """
        pass

    def get_signal_for_date(
        self,
        df: pd.DataFrame,
        symbol: str,
        target_date: date,
    ) -> Signal | None:
        """Get signal for a specific date.

        Args:
            df: DataFrame with OHLCV data.
            symbol: ETF symbol.
            target_date: Date to check for signal.

        Returns:
            Signal if one exists for the date, None otherwise.
        """
        signals = self.generate_signals(df, symbol)
        for signal in signals:
            if signal.date == target_date:
                return signal
        return None
