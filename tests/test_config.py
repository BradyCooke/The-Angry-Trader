"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    Config,
    DataConfig,
    KeltnerConfig,
    StrategyConfig,
    load_config,
    validate_config,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Test loading a valid configuration file."""
        config_data = {
            "data": {
                "source": "yfinance",
                "database_path": "data/test.db",
                "etf_symbols": ["SPY", "QQQ"],
                "benchmark_symbol": "SPY",
            },
            "strategy": {
                "name": "keltner_breakout",
                "keltner": {
                    "ema_period": 50,
                    "atr_period": 20,
                    "atr_multiplier": 2.0,
                },
            },
            "portfolio": {
                "starting_capital": 100000,
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        assert config.data.source == "yfinance"
        assert config.data.database_path == "data/test.db"
        assert config.data.etf_symbols == ["SPY", "QQQ"]
        assert config.strategy.keltner.ema_period == 50
        assert config.portfolio.starting_capital == 100000

    def test_load_config_file_not_found(self) -> None:
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_load_empty_config(self, tmp_path: Path) -> None:
        """Test that empty config file raises ValueError."""
        config_file = tmp_path / "empty.yaml"
        config_file.touch()

        with pytest.raises(ValueError, match="empty"):
            load_config(config_file)

    def test_load_config_with_defaults(self, tmp_path: Path) -> None:
        """Test that missing fields use defaults."""
        config_data = {
            "data": {
                "etf_symbols": ["SPY"],
            }
        }

        config_file = tmp_path / "minimal.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        # Check defaults are applied
        assert config.strategy.keltner.ema_period == 50
        assert config.strategy.keltner.atr_period == 20
        assert config.portfolio.starting_capital == 100000
        assert config.risk_management.var.confidence_level == 0.95


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config(self) -> None:
        """Test validation of a valid configuration."""
        config = Config()
        config.data.etf_symbols = ["SPY", "QQQ"]
        config.data.database_path = "data/test.db"

        errors = validate_config(config)
        assert len(errors) == 0

    def test_missing_etf_symbols(self) -> None:
        """Test validation catches missing ETF symbols."""
        config = Config()
        config.data.etf_symbols = []
        config.data.database_path = "data/test.db"

        errors = validate_config(config)
        assert any("etf_symbols" in e for e in errors)

    def test_invalid_ema_period(self) -> None:
        """Test validation catches invalid EMA period."""
        config = Config()
        config.data.etf_symbols = ["SPY"]
        config.data.database_path = "data/test.db"
        config.strategy.keltner.ema_period = 0

        errors = validate_config(config)
        assert any("ema_period" in e for e in errors)

    def test_invalid_var_confidence(self) -> None:
        """Test validation catches invalid VaR confidence level."""
        config = Config()
        config.data.etf_symbols = ["SPY"]
        config.data.database_path = "data/test.db"
        config.risk_management.var.confidence_level = 1.5

        errors = validate_config(config)
        assert any("confidence_level" in e for e in errors)

    def test_negative_starting_capital(self) -> None:
        """Test validation catches negative starting capital."""
        config = Config()
        config.data.etf_symbols = ["SPY"]
        config.data.database_path = "data/test.db"
        config.portfolio.starting_capital = -1000

        errors = validate_config(config)
        assert any("starting_capital" in e for e in errors)

    def test_invalid_out_of_sample_split(self) -> None:
        """Test validation catches invalid out-of-sample split."""
        config = Config()
        config.data.etf_symbols = ["SPY"]
        config.data.database_path = "data/test.db"
        config.optimization.out_of_sample_split = 1.5

        errors = validate_config(config)
        assert any("out_of_sample_split" in e for e in errors)


class TestConfigDataclasses:
    """Tests for configuration dataclasses."""

    def test_keltner_config_defaults(self) -> None:
        """Test KeltnerConfig default values."""
        keltner = KeltnerConfig()
        assert keltner.ema_period == 50
        assert keltner.atr_period == 20
        assert keltner.atr_multiplier == 2.0

    def test_data_config_defaults(self) -> None:
        """Test DataConfig default values."""
        data = DataConfig()
        assert data.source == "yfinance"
        assert data.benchmark_symbol == "SPY"
        assert data.etf_symbols == []

    def test_config_nested_access(self) -> None:
        """Test nested config access."""
        config = Config()
        assert config.strategy.keltner.ema_period == 50
        assert config.risk_management.var.confidence_level == 0.95
        assert config.risk_management.position_sizing.max_position_pct == 15.0
