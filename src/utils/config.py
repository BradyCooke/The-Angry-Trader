"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class KeltnerConfig:
    """Keltner Channel parameters."""
    ema_period: int = 50
    atr_period: int = 20
    atr_multiplier: float = 2.0


@dataclass
class StopsConfig:
    """Stop loss parameters."""
    initial_atr_multiple: float = 2.0
    trailing_atr_multiple: float = 2.0
    trailing_activation_atr: float = 1.0


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str = "keltner_breakout"
    keltner: KeltnerConfig = field(default_factory=KeltnerConfig)
    stops: StopsConfig = field(default_factory=StopsConfig)


@dataclass
class PositionSizingConfig:
    """Position sizing parameters."""
    method: str = "volatility_atr"
    volatility_target_pct: float = 2.0
    max_position_pct: float = 15.0
    short_size_multiplier: float = 1.0  # Multiplier for short position sizes (0.25 = quarter size)


@dataclass
class VaRConfig:
    """Value at Risk parameters."""
    confidence_level: float = 0.95
    time_horizon_days: int = 20
    max_var_pct: float = 20.0
    breach_action: str = "block_new_positions"


@dataclass
class PortfolioRiskConfig:
    """Portfolio-level risk parameters."""
    max_gross_exposure_pct: float = 100.0


@dataclass
class RiskManagementConfig:
    """Risk management configuration."""
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    portfolio: PortfolioRiskConfig = field(default_factory=PortfolioRiskConfig)
    var: VaRConfig = field(default_factory=VaRConfig)


@dataclass
class DataConfig:
    """Data source configuration."""
    source: str = "yfinance"
    database_path: str = "data/market_data.db"
    etf_symbols: list[str] = field(default_factory=list)
    benchmark_symbol: str = "SPY"


@dataclass
class PortfolioConfig:
    """Portfolio configuration."""
    starting_capital: float = 100000.0
    signal_priority: str = "strongest_first"
    cash_earns_risk_free: bool = True
    include_dividends: bool = False
    transaction_costs: float = 0.0
    slippage: float = 0.0


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    execution_timing: str = "next_open"
    benchmark: str = "SPY"


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    method: str = "random_search"
    iterations: int = 100
    out_of_sample_split: float = 0.2


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


def _dict_to_dataclass(data: dict[str, Any], cls: type) -> Any:
    """Recursively convert a dictionary to a dataclass instance."""
    if not hasattr(cls, "__dataclass_fields__"):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            # Handle nested dataclasses
            if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Config object with all settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError("Configuration file is empty")

    # Build nested config objects
    config = Config()

    if "data" in raw_config:
        config.data = _dict_to_dataclass(raw_config["data"], DataConfig)

    if "strategy" in raw_config:
        strategy_data = raw_config["strategy"]
        config.strategy = StrategyConfig(
            name=strategy_data.get("name", "keltner_breakout"),
            keltner=_dict_to_dataclass(strategy_data.get("keltner", {}), KeltnerConfig),
            stops=_dict_to_dataclass(strategy_data.get("stops", {}), StopsConfig),
        )

    if "risk_management" in raw_config:
        rm_data = raw_config["risk_management"]
        config.risk_management = RiskManagementConfig(
            position_sizing=_dict_to_dataclass(rm_data.get("position_sizing", {}), PositionSizingConfig),
            portfolio=_dict_to_dataclass(rm_data.get("portfolio", {}), PortfolioRiskConfig),
            var=_dict_to_dataclass(rm_data.get("var", {}), VaRConfig),
        )

    if "portfolio" in raw_config:
        config.portfolio = _dict_to_dataclass(raw_config["portfolio"], PortfolioConfig)

    if "backtest" in raw_config:
        config.backtest = _dict_to_dataclass(raw_config["backtest"], BacktestConfig)

    if "optimization" in raw_config:
        config.optimization = _dict_to_dataclass(raw_config["optimization"], OptimizationConfig)

    return config


def validate_config(config: Config) -> list[str]:
    """Validate configuration and return list of errors.

    Args:
        config: Configuration object to validate.

    Returns:
        List of error messages. Empty list if valid.
    """
    errors = []

    # Data validation
    if not config.data.etf_symbols:
        errors.append("No ETF symbols specified in data.etf_symbols")

    if not config.data.database_path:
        errors.append("No database path specified in data.database_path")

    # Strategy validation
    if config.strategy.keltner.ema_period <= 0:
        errors.append("strategy.keltner.ema_period must be positive")

    if config.strategy.keltner.atr_period <= 0:
        errors.append("strategy.keltner.atr_period must be positive")

    if config.strategy.keltner.atr_multiplier <= 0:
        errors.append("strategy.keltner.atr_multiplier must be positive")

    # Risk management validation
    if not 0 < config.risk_management.position_sizing.volatility_target_pct <= 100:
        errors.append("risk_management.position_sizing.volatility_target_pct must be between 0 and 100")

    if not 0 < config.risk_management.position_sizing.max_position_pct <= 100:
        errors.append("risk_management.position_sizing.max_position_pct must be between 0 and 100")

    if not 0 < config.risk_management.var.confidence_level < 1:
        errors.append("risk_management.var.confidence_level must be between 0 and 1")

    if config.risk_management.var.time_horizon_days <= 0:
        errors.append("risk_management.var.time_horizon_days must be positive")

    # Portfolio validation
    if config.portfolio.starting_capital <= 0:
        errors.append("portfolio.starting_capital must be positive")

    if config.portfolio.transaction_costs < 0:
        errors.append("portfolio.transaction_costs cannot be negative")

    if config.portfolio.slippage < 0:
        errors.append("portfolio.slippage cannot be negative")

    # Optimization validation
    if not 0 < config.optimization.out_of_sample_split < 1:
        errors.append("optimization.out_of_sample_split must be between 0 and 1")

    if config.optimization.iterations <= 0:
        errors.append("optimization.iterations must be positive")

    return errors
