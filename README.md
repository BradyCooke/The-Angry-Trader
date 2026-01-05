# Trend Following Trading System

A systematic trend-following trading system based on Tom Basso's "All-Weather Trader" principles. Trades 30 sector ETFs long and short using Keltner Channel breakouts with ATR-based position sizing and VaR risk management.

## Performance (2010-2025)

| Metric | Strategy | SPY (Buy & Hold) |
|--------|----------|------------------|
| CAGR | 9.49% | 11.89% |
| Max Drawdown | **20.4%** | 34.1% |
| Sharpe Ratio | 0.70 | 0.74 |
| Calmar Ratio | **0.47** | 0.35 |
| Profit Factor | 1.03 | - |

The strategy achieves **significantly lower drawdowns** (20% vs 34%) and **better risk-adjusted returns** (Calmar 0.47 vs 0.35). Quarter-size shorts reduce drawdown while maintaining profitability.

## Features

- **Keltner Channel Breakouts**: Enter positions when price breaks above/below the channel
- **ATR-Based Position Sizing**: Target 2% volatility per position, max 15% allocation
- **Quarter-Size Shorts**: Reduced short position sizes to minimize drawdown from market's upward bias
- **VaR Risk Management**: 95% confidence VaR with 20% portfolio limit
- **Trailing Stops**: Activate after 1×ATR profit, trail at 2.5×ATR
- **Event-Driven Backtesting**: Full historical simulation with comprehensive metrics
- **Paper Trading**: Daily signal monitoring with portfolio tracking
- **30 Sector ETFs**: Diversified universe across all market sectors

## Quick Start

1. **Setup Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Backtest**
   ```bash
   jupyter lab notebooks/quick_start.ipynb
   ```

3. **Paper Trading**
   ```bash
   python paper_trade.py           # Show today's signals
   python paper_trade.py --execute # Execute signals (update portfolio)
   python paper_trade.py --status  # Show portfolio status only
   python paper_trade.py --reset   # Reset paper portfolio
   ```

4. **Run Tests**
   ```bash
   python -m pytest tests/ -v
   ```

## Project Structure

```
trend-follower/
├── config.yaml              # Strategy configuration
├── paper_trade.py           # Daily signal monitoring
├── backtest_*_variant.py    # LLM optimization experiments
├── src/
│   ├── utils/               # Config loading, logging
│   ├── data/                # Database, data fetching
│   ├── signals/             # Indicators, signal generation
│   ├── risk/                # Position sizing, VaR
│   ├── portfolio/           # Portfolio management, orders
│   ├── backtest/            # Backtesting engine, metrics
│   ├── reporting/           # Console output, charts
│   └── execution/           # Broker interface (stub)
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks
└── data/                    # SQLite database
```

## Configuration

Edit `config.yaml` to customize. Current optimized parameters:

```yaml
strategy:
  keltner:
    ema_period: 35          # EMA lookback (optimized)
    atr_period: 20          # ATR lookback
    atr_multiplier: 1.5     # Channel width (tighter = lower drawdown)
  stops:
    initial_atr_multiple: 2.5    # Initial stop distance
    trailing_atr_multiple: 2.5   # Trailing stop distance
    trailing_activation_atr: 1.0 # Profit before trailing activates

risk_management:
  position_sizing:
    volatility_target_pct: 2.0   # Per-position volatility
    max_position_pct: 15.0       # Max position size
    short_size_multiplier: 0.25  # Quarter-size shorts
  var:
    confidence_level: 0.95       # VaR confidence
    max_var_pct: 20.0            # Max portfolio VaR

portfolio:
  starting_capital: 100000
```

## ETF Universe

30 sector ETFs covering:
- Technology (XLK, XSD, XSW)
- Healthcare (XLV, XBI, XPH)
- Financials (XLF, KBE, KRE)
- Energy (XLE, XES, XOP)
- Consumer (XLY, XLP, XRT)
- Industrials (XLI, XAR, XTN)
- Materials (XLB, XME)
- Utilities (XLU)
- Communications (XLC)
- Real Estate (XHB)
- International (EEM, SPDW)
- And more...

## Strategy Logic

### Entry Rules
1. Calculate Keltner Channels: EMA(35) ± ATR(20) × 1.5
2. **Long**: Close breaks above upper band
3. **Short**: Close breaks below lower band
4. Must be inside bands before re-entry allowed

### Position Sizing
```
Position Size = (Equity × 2%) / ATR × Price
Capped at 15% of equity
Short positions sized at 25% (quarter-size) to reduce drawdown
```

### Stop Loss
- **Initial**: Entry price ± 2.5×ATR
- **Trailing**: Activates after 1×ATR profit, trails at 2.5×ATR from high/low

### Risk Management
- Calculate 95% historical VaR daily
- Block new positions when portfolio VaR > 20%
- Prioritize signals by breakout strength

## Performance Metrics

The system calculates:
- **Returns**: Total return, CAGR
- **Risk**: Max drawdown, volatility, VaR
- **Risk-Adjusted**: Sharpe, Sortino, Calmar ratios
- **Trade Stats**: Win rate, profit factor, avg trade duration
- **Benchmark**: Alpha, beta, correlation vs SPY

## Modules

### Data (`src/data/`)
- `database.py`: SQLite storage for OHLCV data
- `fetcher.py`: Yahoo Finance data fetching

### Signals (`src/signals/`)
- `indicators.py`: EMA, ATR, Keltner Channels
- `keltner.py`: Signal generation with breakout detection

### Risk (`src/risk/`)
- `position_sizing.py`: ATR-based volatility sizing
- `var.py`: Historical VaR calculations

### Portfolio (`src/portfolio/`)
- `portfolio.py`: Position tracking, P&L calculation
- `orders.py`: Order management

### Backtest (`src/backtest/`)
- `engine.py`: Event-driven simulation loop
- `metrics.py`: Performance calculations

### Reporting (`src/reporting/`)
- `console.py`: Text output formatting
- `charts.py`: Matplotlib visualizations

### Execution (`src/execution/`)
- `broker_base.py`: Abstract broker interface
- `ib_broker.py`: Interactive Brokers stub (for future live trading)

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_backtest.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## Dependencies

- `pandas`, `numpy`: Data manipulation
- `yfinance`: Market data
- `matplotlib`: Visualization
- `pyyaml`: Configuration
- `pytest`: Testing
- `schedule`: Scheduling (optional)

## License

MIT
