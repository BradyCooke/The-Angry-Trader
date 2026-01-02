"""Data fetching from Yahoo Finance using yfinance."""

from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd
import yfinance as yf

from src.utils.logging import get_logger
from .database import Database

logger = get_logger("fetcher")


class DataFetcher:
    """Fetch market data from Yahoo Finance."""

    def __init__(self, database: Database):
        """Initialize data fetcher.

        Args:
            database: Database instance for storing data.
        """
        self.db = database

    def fetch_symbol(
        self,
        symbol: str,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> pd.DataFrame:
        """Fetch data for a single symbol from Yahoo Finance.

        Args:
            symbol: ETF symbol to fetch.
            start_date: Start date for data fetch. If None, fetches all available.
            end_date: End date for data fetch. If None, uses today.

        Returns:
            DataFrame with OHLCV data.
        """
        logger.info(f"Fetching data for {symbol}")

        # Convert string dates if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        try:
            ticker = yf.Ticker(symbol)

            # Fetch historical data
            if start_date:
                df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            else:
                df = ticker.history(period="max", auto_adjust=False)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Standardize column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Keep only OHLCV columns
            required_cols = ["open", "high", "low", "close", "volume"]
            df = df[required_cols]

            # Remove timezone info from index if present
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Convert index to date (remove time component)
            df.index = pd.to_datetime(df.index).date

            logger.info(f"Fetched {len(df)} rows for {symbol} from {df.index[0]} to {df.index[-1]}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_and_store(
        self,
        symbol: str,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> int:
        """Fetch data for a symbol and store in database.

        Args:
            symbol: ETF symbol to fetch.
            start_date: Start date for data fetch.
            end_date: End date for data fetch.

        Returns:
            Number of rows stored.
        """
        df = self.fetch_symbol(symbol, start_date, end_date)

        if df.empty:
            return 0

        return self.db.insert_price_data(df, symbol)

    def update_symbol(self, symbol: str) -> int:
        """Update data for a symbol with only new data since last update.

        Args:
            symbol: ETF symbol to update.

        Returns:
            Number of new rows added.
        """
        last_date = self.db.get_last_date(symbol)

        if last_date:
            # Fetch from day after last date
            start_date = last_date + timedelta(days=1)
            logger.info(f"Updating {symbol} from {start_date}")
        else:
            # No existing data, fetch all
            start_date = None
            logger.info(f"No existing data for {symbol}, fetching all")

        return self.fetch_and_store(symbol, start_date=start_date)

    def fetch_multiple(
        self,
        symbols: list[str],
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, int]:
        """Fetch data for multiple symbols.

        Args:
            symbols: List of ETF symbols to fetch.
            start_date: Start date for data fetch.
            end_date: End date for data fetch.
            progress_callback: Optional callback(symbol, current, total) for progress.

        Returns:
            Dictionary mapping symbol to number of rows stored.
        """
        results = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(symbol, i + 1, total)

            rows = self.fetch_and_store(symbol, start_date, end_date)
            results[symbol] = rows

        logger.info(f"Fetched data for {len(symbols)} symbols")
        return results

    def update_all(
        self,
        symbols: list[str],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, int]:
        """Update data for all symbols (incremental update).

        Args:
            symbols: List of ETF symbols to update.
            progress_callback: Optional callback(symbol, current, total) for progress.

        Returns:
            Dictionary mapping symbol to number of new rows added.
        """
        results = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(symbol, i + 1, total)

            rows = self.update_symbol(symbol)
            results[symbol] = rows

        total_rows = sum(results.values())
        logger.info(f"Updated {len(symbols)} symbols, {total_rows} total new rows")
        return results


def print_progress(symbol: str, current: int, total: int) -> None:
    """Default progress callback that prints to console.

    Args:
        symbol: Current symbol being processed.
        current: Current progress count.
        total: Total number of symbols.
    """
    print(f"[{current}/{total}] Fetching {symbol}...")
