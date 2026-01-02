"""Tests for database operations."""

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.data.database import Database


@pytest.fixture
def temp_db(tmp_path: Path) -> Database:
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    yield db
    db.close()


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample OHLCV data."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 101.5, 103.0],
        "high": [101.0, 102.0, 103.0, 102.5, 104.0],
        "low": [99.0, 100.0, 101.0, 100.5, 102.0],
        "close": [100.5, 101.5, 102.5, 102.0, 103.5],
        "volume": [1000000, 1100000, 1200000, 1050000, 1300000],
    }, index=dates)


class TestDatabase:
    """Tests for Database class."""

    def test_init_creates_file(self, tmp_path: Path) -> None:
        """Test that database initialization creates the file."""
        db_path = tmp_path / "subdir" / "test.db"
        db = Database(db_path)
        assert db_path.exists()
        db.close()

    def test_insert_and_retrieve_data(self, temp_db: Database, sample_data: pd.DataFrame) -> None:
        """Test inserting and retrieving price data."""
        rows = temp_db.insert_price_data(sample_data, "TEST")
        assert rows == 5

        retrieved = temp_db.get_price_data("TEST")
        assert len(retrieved) == 5
        assert list(retrieved.columns) == ["open", "high", "low", "close", "volume"]

    def test_get_price_data_with_date_filter(self, temp_db: Database, sample_data: pd.DataFrame) -> None:
        """Test retrieving data with date filters."""
        temp_db.insert_price_data(sample_data, "TEST")

        # Filter by start date
        result = temp_db.get_price_data("TEST", start_date=date(2024, 1, 3))
        assert len(result) == 3

        # Filter by end date
        result = temp_db.get_price_data("TEST", end_date=date(2024, 1, 3))
        assert len(result) == 3

        # Filter by both
        result = temp_db.get_price_data("TEST", start_date=date(2024, 1, 2), end_date=date(2024, 1, 4))
        assert len(result) == 3

    def test_get_nonexistent_symbol(self, temp_db: Database) -> None:
        """Test retrieving data for a symbol that doesn't exist."""
        result = temp_db.get_price_data("NONEXISTENT")
        assert result.empty

    def test_upsert_behavior(self, temp_db: Database, sample_data: pd.DataFrame) -> None:
        """Test that inserting duplicate dates updates existing data."""
        temp_db.insert_price_data(sample_data, "TEST")

        # Modify and reinsert
        sample_data["close"] = [200.0, 201.0, 202.0, 203.0, 204.0]
        temp_db.insert_price_data(sample_data, "TEST")

        result = temp_db.get_price_data("TEST")
        assert len(result) == 5  # Still 5 rows (not 10)
        assert result["close"].iloc[0] == 200.0  # Updated value

    def test_get_last_date(self, temp_db: Database, sample_data: pd.DataFrame) -> None:
        """Test getting the last date for a symbol."""
        temp_db.insert_price_data(sample_data, "TEST")

        last_date = temp_db.get_last_date("TEST")
        assert last_date == date(2024, 1, 5)

    def test_get_last_date_nonexistent(self, temp_db: Database) -> None:
        """Test getting last date for nonexistent symbol."""
        last_date = temp_db.get_last_date("NONEXISTENT")
        assert last_date is None

    def test_get_symbols_in_database(self, temp_db: Database, sample_data: pd.DataFrame) -> None:
        """Test getting list of symbols in database."""
        temp_db.insert_price_data(sample_data, "AAA")
        temp_db.insert_price_data(sample_data, "BBB")
        temp_db.insert_price_data(sample_data, "CCC")

        symbols = temp_db.get_symbols_in_database()
        assert symbols == ["AAA", "BBB", "CCC"]

    def test_get_all_symbols_data(self, temp_db: Database, sample_data: pd.DataFrame) -> None:
        """Test getting data for multiple symbols."""
        temp_db.insert_price_data(sample_data, "AAA")
        temp_db.insert_price_data(sample_data, "BBB")

        data = temp_db.get_all_symbols_data(["AAA", "BBB", "CCC"])
        assert "AAA" in data
        assert "BBB" in data
        assert "CCC" in data
        assert len(data["AAA"]) == 5
        assert len(data["BBB"]) == 5
        assert len(data["CCC"]) == 0  # Doesn't exist

    def test_delete_symbol(self, temp_db: Database, sample_data: pd.DataFrame) -> None:
        """Test deleting a symbol's data."""
        temp_db.insert_price_data(sample_data, "TEST")
        assert len(temp_db.get_price_data("TEST")) == 5

        deleted = temp_db.delete_symbol("TEST")
        assert deleted == 5
        assert temp_db.get_price_data("TEST").empty

    def test_metadata_tracking(self, temp_db: Database, sample_data: pd.DataFrame) -> None:
        """Test that metadata is tracked correctly."""
        temp_db.insert_price_data(sample_data, "TEST")

        metadata = temp_db.get_metadata("TEST")
        assert metadata is not None
        assert metadata["symbol"] == "TEST"
        assert metadata["row_count"] == 5
        assert metadata["first_date"] == date(2024, 1, 1)
        assert metadata["last_date"] == date(2024, 1, 5)

    def test_context_manager(self, tmp_path: Path, sample_data: pd.DataFrame) -> None:
        """Test database context manager."""
        db_path = tmp_path / "context_test.db"

        with Database(db_path) as db:
            db.insert_price_data(sample_data, "TEST")

        # Verify data persisted after context exit
        with Database(db_path) as db:
            result = db.get_price_data("TEST")
            assert len(result) == 5
