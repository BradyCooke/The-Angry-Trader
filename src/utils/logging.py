"""Logging setup and utilities."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """Set up logging for the trading system.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional path to log file.
        format_string: Optional custom format string.

    Returns:
        Configured root logger.
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger for the package
    logger = logging.getLogger("trend_follower")
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Name of the module (will be prefixed with 'trend_follower.').

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"trend_follower.{name}")


class TradeLogger:
    """Specialized logger for trade events."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize trade logger.

        Args:
            logger: Optional logger instance. Creates new one if not provided.
        """
        self.logger = logger or get_logger("trades")

    def log_entry(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        stop_price: float,
    ) -> None:
        """Log a trade entry.

        Args:
            symbol: ETF symbol.
            side: LONG or SHORT.
            quantity: Number of shares.
            price: Entry price.
            stop_price: Initial stop price.
        """
        self.logger.info(
            f"ENTRY | {symbol} | {side} | qty={quantity} | "
            f"price={price:.2f} | stop={stop_price:.2f}"
        )

    def log_exit(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        """Log a trade exit.

        Args:
            symbol: ETF symbol.
            side: LONG or SHORT.
            quantity: Number of shares.
            entry_price: Original entry price.
            exit_price: Exit price.
            pnl: Dollar P&L.
            pnl_pct: Percentage P&L.
            reason: Exit reason (stop, signal, etc.).
        """
        self.logger.info(
            f"EXIT  | {symbol} | {side} | qty={quantity} | "
            f"entry={entry_price:.2f} | exit={exit_price:.2f} | "
            f"pnl=${pnl:+.2f} ({pnl_pct:+.2f}%) | {reason}"
        )

    def log_stop_update(
        self,
        symbol: str,
        side: str,
        old_stop: float,
        new_stop: float,
    ) -> None:
        """Log a trailing stop update.

        Args:
            symbol: ETF symbol.
            side: LONG or SHORT.
            old_stop: Previous stop price.
            new_stop: New stop price.
        """
        self.logger.debug(
            f"STOP UPDATE | {symbol} | {side} | "
            f"{old_stop:.2f} -> {new_stop:.2f}"
        )
