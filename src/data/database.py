"""SQLite database operations for market data storage."""

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("database")


class Database:
    """SQLite database for storing market data."""

    def __init__(self, db_path: str | Path):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create price data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_data_symbol
            ON price_data (symbol)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_data_date
            ON price_data (date)
        """)

        # Create metadata table for tracking updates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                symbol TEXT PRIMARY KEY,
                last_update TIMESTAMP,
                first_date DATE,
                last_date DATE,
                row_count INTEGER
            )
        """)

        conn.commit()
        logger.debug(f"Database schema initialized at {self.db_path}")

    def insert_price_data(self, df: pd.DataFrame, symbol: str) -> int:
        """Insert or update price data for a symbol.

        Args:
            df: DataFrame with columns: date, open, high, low, close, volume.
            symbol: ETF symbol.

        Returns:
            Number of rows inserted/updated.
        """
        if df.empty:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        # Prepare data for insertion
        rows = []
        for idx, row in df.iterrows():
            # Handle both datetime index and date column
            if isinstance(idx, (datetime, date)):
                row_date = idx.date() if isinstance(idx, datetime) else idx
            elif "date" in df.columns:
                row_date = row["date"]
            else:
                row_date = idx

            rows.append((
                symbol,
                row_date,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
            ))

        # Use INSERT OR REPLACE for upsert behavior
        cursor.executemany("""
            INSERT OR REPLACE INTO price_data
            (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)

        # Update metadata
        cursor.execute("""
            INSERT OR REPLACE INTO metadata
            (symbol, last_update, first_date, last_date, row_count)
            VALUES (?, ?, ?, ?, ?)
        """, (
            symbol,
            datetime.now(),
            min(r[1] for r in rows),
            max(r[1] for r in rows),
            len(rows),
        ))

        conn.commit()
        logger.info(f"Inserted {len(rows)} rows for {symbol}")
        return len(rows)

    def get_price_data(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Get price data for a symbol.

        Args:
            symbol: ETF symbol.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            DataFrame with OHLCV data indexed by date.
        """
        conn = self._get_connection()

        query = "SELECT date, open, high, low, close, volume FROM price_data WHERE symbol = ?"
        params: list = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])

        if not df.empty:
            df.set_index("date", inplace=True)

        return df

    def get_all_symbols_data(
        self,
        symbols: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Get price data for multiple symbols.

        Args:
            symbols: List of ETF symbols.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        return {
            symbol: self.get_price_data(symbol, start_date, end_date)
            for symbol in symbols
        }

    def get_last_date(self, symbol: str) -> date | None:
        """Get the last date of data for a symbol.

        Args:
            symbol: ETF symbol.

        Returns:
            Last date in database or None if no data.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT MAX(date) as max_date FROM price_data WHERE symbol = ?",
            (symbol,)
        )
        row = cursor.fetchone()

        if row and row["max_date"]:
            max_date = row["max_date"]
            # Convert string to date if needed
            if isinstance(max_date, str):
                return datetime.strptime(max_date, "%Y-%m-%d").date()
            elif isinstance(max_date, datetime):
                return max_date.date()
            return max_date
        return None

    def get_symbols_in_database(self) -> list[str]:
        """Get list of symbols with data in the database.

        Returns:
            List of symbol strings.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT symbol FROM price_data ORDER BY symbol")
        return [row["symbol"] for row in cursor.fetchall()]

    def get_metadata(self, symbol: str) -> dict | None:
        """Get metadata for a symbol.

        Args:
            symbol: ETF symbol.

        Returns:
            Dictionary with metadata or None if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM metadata WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def delete_symbol(self, symbol: str) -> int:
        """Delete all data for a symbol.

        Args:
            symbol: ETF symbol to delete.

        Returns:
            Number of rows deleted.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM price_data WHERE symbol = ?", (symbol,))
        deleted = cursor.rowcount

        cursor.execute("DELETE FROM metadata WHERE symbol = ?", (symbol,))

        conn.commit()
        logger.info(f"Deleted {deleted} rows for {symbol}")
        return deleted

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    def __enter__(self) -> "Database":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
