"""Signal generation module for trading signals."""

from .indicators import calculate_ema, calculate_atr, calculate_keltner_channels
from .base import Signal, SignalType, SignalGenerator
from .keltner import KeltnerSignalGenerator

__all__ = [
    "calculate_ema",
    "calculate_atr",
    "calculate_keltner_channels",
    "Signal",
    "SignalType",
    "SignalGenerator",
    "KeltnerSignalGenerator",
]
