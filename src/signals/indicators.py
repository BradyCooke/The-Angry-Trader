"""Technical indicator calculations."""

import numpy as np
import pandas as pd


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        series: Price series (typically close prices).
        period: EMA period (lookback window).

    Returns:
        Series with EMA values.
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        series: Price series.
        period: SMA period (lookback window).

    Returns:
        Series with SMA values.
    """
    return series.rolling(window=period).mean()


def calculate_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Calculate True Range.

    True Range is the greatest of:
    - Current High - Current Low
    - |Current High - Previous Close|
    - |Current Low - Previous Close|

    Args:
        high: High prices series.
        low: Low prices series.
        close: Close prices series.

    Returns:
        Series with True Range values.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Calculate Average True Range (ATR).

    Uses Wilder's smoothing method (exponential moving average).

    Args:
        high: High prices series.
        low: Low prices series.
        close: Close prices series.
        period: ATR period (default 20).

    Returns:
        Series with ATR values.
    """
    tr = calculate_true_range(high, low, close)
    # Wilder's smoothing: alpha = 1/period
    return tr.ewm(alpha=1/period, adjust=False).mean()


def calculate_keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 50,
    atr_period: int = 20,
    atr_multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Keltner Channels.

    Keltner Channels consist of:
    - Middle line: EMA of close prices
    - Upper band: EMA + (ATR × multiplier)
    - Lower band: EMA - (ATR × multiplier)

    Args:
        high: High prices series.
        low: Low prices series.
        close: Close prices series.
        ema_period: Period for EMA calculation (default 50).
        atr_period: Period for ATR calculation (default 20).
        atr_multiplier: ATR multiplier for band width (default 2.0).

    Returns:
        Tuple of (middle, upper, lower) band Series.
    """
    middle = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)

    upper = middle + (atr * atr_multiplier)
    lower = middle - (atr * atr_multiplier)

    return middle, upper, lower


def calculate_highest_high(high: pd.Series, period: int) -> pd.Series:
    """Calculate rolling highest high.

    Args:
        high: High prices series.
        period: Lookback period.

    Returns:
        Series with rolling highest high.
    """
    return high.rolling(window=period).max()


def calculate_lowest_low(low: pd.Series, period: int) -> pd.Series:
    """Calculate rolling lowest low.

    Args:
        low: Low prices series.
        period: Lookback period.

    Returns:
        Series with rolling lowest low.
    """
    return low.rolling(window=period).min()


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage returns.

    Args:
        prices: Price series.
        periods: Number of periods for return calculation (default 1).

    Returns:
        Series with percentage returns.
    """
    return prices.pct_change(periods=periods)


def calculate_volatility(returns: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    """Calculate rolling volatility (standard deviation of returns).

    Args:
        returns: Returns series.
        period: Lookback period for rolling calculation.
        annualize: If True, annualize volatility assuming 252 trading days.

    Returns:
        Series with volatility values.
    """
    vol = returns.rolling(window=period).std()

    if annualize:
        vol = vol * np.sqrt(252)

    return vol
