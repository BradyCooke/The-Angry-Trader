"""Tests for data fetcher."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.database import Database
from src.data.fetcher import DataFetcher


@pytest.fixture
def temp_db(tmp_path: Path) -> Database:
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    yield db
    db.close()


@pytest.fixture
def fetcher(temp_db: Database) -> DataFetcher:
    """Create a DataFetcher with temporary database."""
    return DataFetcher(temp_db)


@pytest.fixture
def mock_yf_data() -> pd.DataFrame:
    """Create mock yfinance return data."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D", tz="UTC")
    return pd.DataFrame({
        "Open": [100.0, 101.0, 102.0, 101.5, 103.0],
        "High": [101.0, 102.0, 103.0, 102.5, 104.0],
        "Low": [99.0, 100.0, 101.0, 100.5, 102.0],
        "Close": [100.5, 101.5, 102.5, 102.0, 103.5],
        "Volume": [1000000, 1100000, 1200000, 1050000, 1300000],
        "Dividends": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Stock Splits": [0.0, 0.0, 0.0, 0.0, 0.0],
    }, index=dates)


class TestDataFetcher:
    """Tests for DataFetcher class."""

    def test_fetch_symbol_returns_dataframe(self, fetcher: DataFetcher, mock_yf_data: pd.DataFrame) -> None:
        """Test that fetch_symbol returns a DataFrame."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_yf_data

            result = fetcher.fetch_symbol("SPY")

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 5
            assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_fetch_symbol_normalizes_columns(self, fetcher: DataFetcher, mock_yf_data: pd.DataFrame) -> None:
        """Test that column names are normalized to lowercase."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_yf_data

            result = fetcher.fetch_symbol("SPY")

            # All columns should be lowercase
            assert all(c.islower() for c in result.columns)

    def test_fetch_symbol_removes_timezone(self, fetcher: DataFetcher, mock_yf_data: pd.DataFrame) -> None:
        """Test that timezone is removed from index."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_yf_data

            result = fetcher.fetch_symbol("SPY")

            # Index should be dates without timezone
            assert isinstance(result.index[0], date)

    def test_fetch_symbol_empty_data(self, fetcher: DataFetcher) -> None:
        """Test handling of empty data response."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()

            result = fetcher.fetch_symbol("INVALID")

            assert result.empty

    def test_fetch_symbol_error_handling(self, fetcher: DataFetcher) -> None:
        """Test error handling during fetch."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.side_effect = Exception("Network error")

            result = fetcher.fetch_symbol("SPY")

            assert result.empty

    def test_fetch_and_store(self, fetcher: DataFetcher, temp_db: Database, mock_yf_data: pd.DataFrame) -> None:
        """Test fetching and storing data."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_yf_data

            rows = fetcher.fetch_and_store("SPY")

            assert rows == 5
            assert len(temp_db.get_price_data("SPY")) == 5

    def test_update_symbol_incremental(self, fetcher: DataFetcher, temp_db: Database, mock_yf_data: pd.DataFrame) -> None:
        """Test incremental update only fetches new data."""
        # First, insert some initial data
        initial_dates = pd.date_range(start="2024-01-01", periods=3, freq="D")
        initial_data = pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000000, 1100000, 1200000],
        }, index=initial_dates)
        temp_db.insert_price_data(initial_data, "SPY")

        # Now update should request data from 2024-01-04
        with patch("yfinance.Ticker") as mock_ticker:
            # Return new data
            new_dates = pd.date_range(start="2024-01-04", periods=2, freq="D", tz="UTC")
            new_data = pd.DataFrame({
                "Open": [103.0, 104.0],
                "High": [104.0, 105.0],
                "Low": [102.0, 103.0],
                "Close": [103.5, 104.5],
                "Volume": [1300000, 1400000],
                "Dividends": [0.0, 0.0],
                "Stock Splits": [0.0, 0.0],
            }, index=new_dates)
            mock_ticker.return_value.history.return_value = new_data

            rows = fetcher.update_symbol("SPY")

            # Should have added 2 new rows
            assert rows == 2
            # Total should now be 5
            assert len(temp_db.get_price_data("SPY")) == 5

    def test_fetch_multiple(self, fetcher: DataFetcher, mock_yf_data: pd.DataFrame) -> None:
        """Test fetching multiple symbols."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_yf_data

            results = fetcher.fetch_multiple(["SPY", "QQQ", "IWM"])

            assert len(results) == 3
            assert all(r == 5 for r in results.values())

    def test_fetch_multiple_with_progress(self, fetcher: DataFetcher, mock_yf_data: pd.DataFrame) -> None:
        """Test progress callback during fetch."""
        progress_calls = []

        def track_progress(symbol: str, current: int, total: int) -> None:
            progress_calls.append((symbol, current, total))

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_yf_data

            fetcher.fetch_multiple(["SPY", "QQQ"], progress_callback=track_progress)

            assert len(progress_calls) == 2
            assert progress_calls[0] == ("SPY", 1, 2)
            assert progress_calls[1] == ("QQQ", 2, 2)
