# Multi-Chain Price Oracle Expansion

Complete expansion of the X1 price prediction engine to support multi-chain price feeds with correlation tracking and paper trading.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRICE ORACLE ENGINE                       │
├─────────────────────────────────────────────────────────────┤
│  PriceFetcher ◄──► Multi-Chain RPCs (Solana/ETH/BSC/BTC)    │
│       │                                                      │
│       ▼                                                      │
│  MultiTimeframeData ◄──► 5m/10m/25m/1h/4h candles           │
│       │                                                      │
│       ▼                                                      │
│  PredictionEngine ──► Technical analysis (SMA/vol/volume)   │
│       │                                                      │
│       ▼                                                      │
│  CorrelationTracker ──► Cross-pair correlation matrix       │
│       │                                                      │
│       ▼                                                      │
│  PaperTrader (Testnet) ──► Virtual XNT, real predictions    │
└─────────────────────────────────────────────────────────────┘
```

## Supported Token Pairs

| Pair | Source Chain | Quote | Status |
|------|--------------|-------|--------|
| BTC/XNT | Bitcoin (via wrapped) | XNT | ✅ Active |
| SOL/XNT | Solana (native SVM) | XNT | ✅ Active |
| ETH/XNT | Ethereum | XNT | ✅ Active |
| BNB/XNT | BSC | XNT | ✅ Active |

### Future Expansion (Configured)

```python
ADDITIONAL_L1_TOKENS = {
    "AVAX": {"chain": "avalanche", "decimals": 18},
    "MATIC": {"chain": "polygon", "decimals": 18},
    "ARB": {"chain": "arbitrum", "decimals": 18},
    "OP": {"chain": "optimism", "decimals": 18},
    "FTM": {"chain": "fantom", "decimals": 18},
    "NEAR": {"chain": "near", "decimals": 24},
    "APT": {"chain": "aptos", "decimals": 8},
    "SUI": {"chain": "sui", "decimals": 9},
}
```

## Correlation Matrix

### Decoupling Detection

The oracle tracks rolling correlations between all pairs:

```python
CORRELATION_CONFIG = {
    "lookback_window": 100,          # 100 periods
    "min_correlation_threshold": 0.7, # Strong correlation
    "decoupling_threshold": 0.3,      # Weak = decoupling
}
```

**Signals Generated:**
- `BTC_XNT:SOL_XNT` correlation drops below 0.3 → Decoupling alert
- Which pair is leading the breakaway flagged in `decoupling_signal`

### Bridge Stress Index

Composite score (0-100) combining:

```python
BRIDGE_STRESS_CONFIG = {
    "max_expected_divergence": 0.02,  # 2% max expected
    "volume_spike_threshold": 3.0,    # 3x average = spike
    "stress_levels": {
        "low": 0.0,
        "moderate": 0.3,
        "high": 0.6,
        "critical": 0.8,
    },
}
```

**Formula:**
```
stress = (price_divergence / max_divergence) * 0.5 + 
         (volume_spike_factor) * 0.3 + 
         (correlation_drop) * 0.2
```

**Interpretation:**
- 0-30: Normal operation
- 30-60: Elevated stress (monitor)
- 60-80: High stress (arbitrage opportunity likely)
- 80+: Critical (potential bridge issues)

## Paper Trading Testnet

### Quick Start

```bash
cd price-oracle/testnet

# Reset and initialize
python3 testnet_trader.py --reset

# Start trading loop (runs until Ctrl+C)
python3 testnet_trader.py --run

# Generate accuracy report
python3 testnet_trader.py --report
```

### Configuration

```python
PAPER_TRADING = {
    "initial_balance_usd": 10000.0,
    "initial_balance_xnt": 100000.0,
    "fee_rate": 0.003,          # 0.3% per trade
    "slippage": 0.01,           # 1% slippage
    "max_position_size": 0.25,  # 25% of portfolio max
    "stop_loss_pct": 0.05,      # 5% stop loss
    "take_profit_pct": 0.15,    # 15% take profit
}
```

### Trading Logic

**Entry Conditions:**
- Confidence >= 70%
- Prediction != "neutral"
- No existing position in pair
- Max 4 concurrent positions

**Exit Conditions:**
- Stop loss hit (5%)
- Take profit hit (15%)
- Position held > 25m (timeframe expiry)

**Position Sizing:**
- 25% of portfolio per position
- Virtual XNT balance tracked
- Fees deducted on entry and exit

### Report Format

```json
{
  "balance": {
    "xnt": 98750.50,
    "usd": 10000.00,
    "initial_xnt": 100000.0
  },
  "performance": {
    "total_trades": 47,
    "win_rate": 0.617,
    "total_pnl_xnt": -1249.50,
    "roi_pct": -1.25
  },
  "prediction_accuracy": {
    "directional_accuracy": 0.681,
    "correct_predictions": 32
  },
  "by_confidence": {
    "high_confidence_80+": {
      "trades": 18,
      "accuracy": 0.778,
      "avg_pnl": 125.5
    }
  }
}
```

## Decoupling Signals

### When Pairs Decouple

**Common Scenarios:**

1. **Bridge Issues**
   - ETH/XNT decouples from BTC/XNT
   - Bridge contract paused or congested
   - Stress index > 70

2. **Chain-Specific News**
   - SOL/XNT moves independently (Solana outage/news)
   - Other pairs maintain correlation

3. **Arbitrage Opportunity**
   - Price divergence exceeds fees + slippage
   - High volume spike on one pair
   - Stress index signals opportunity

### Trading Implications

| Signal | Action | Rationale |
|--------|--------|-----------|
| BTC-SOL decouple, SOL leading | Long SOL/XNT | Solana-specific catalyst |
| Bridge stress > 80 | Flatten positions | Potential liquidity issues |
| Correlation reversion | Enter pairs trade | Mean reversion expected |

## API Reference

### PriceOracle

```python
from price_oracle import PriceOracle, get_oracle

# Start the oracle
oracle = get_oracle()
await oracle.start()  # Runs update loop

# Get predictions
pred = oracle.get_prediction("BTC_XNT", "25m")
# Returns: Prediction object with direction, confidence, features

# Get all predictions
all_preds = oracle.get_all_predictions()

# Get correlation report
report = oracle.get_correlation_report()
```

### PaperTrader

```python
from testnet.testnet_trader import PaperTrader

trader = PaperTrader()

# Start trading
await trader.start()

# Get report
report = trader.get_report()
```

## File Structure

```
price-oracle/
├── src/
│   ├── price_oracle.py      # Core engine (25KB)
│   └── config.py            # All configuration (8KB)
├── testnet/
│   └── testnet_trader.py    # Paper trading (9KB)
├── docs/
│   └── EXPANSION.md         # This document
└── data/
    └── (runtime state files)
```

## Validation & Testing

### Unit Tests

```bash
# Test price fetching
python3 -c "from src.price_oracle import PriceFetcher; ..."

# Test correlation calculation
python3 -c "from src.price_oracle import CorrelationTracker; ..."

# Test predictions
python3 -c "from src.price_oracle import PredictionEngine; ..."
```

### Integration Test

```bash
# Run paper trader for 1 hour, generate report
python3 testnet/testnet_trader.py --reset
python3 testnet/testnet_trader.py --run &
sleep 3600
kill %1
python3 testnet/testnet_trader.py --report
```

## Production Deployment

### Requirements

```
python >= 3.8
aiohttp >= 3.8
numpy >= 1.20
```

### Environment Variables

```bash
export X1_RPC_URL="https://rpc.mainnet.x1.xyz"
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
export ETH_RPC_URL="https://eth.llamarpc.com"
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY testnet/ ./testnet/

CMD ["python3", "-m", "testnet.testnet_trader", "--run"]
```

## Changelog

### v1.0.0 (2026-02-18)

- ✅ Multi-pair support (BTC, SOL, ETH, BNB)
- ✅ Multi-timeframe architecture (5m/10m/25m)
- ✅ Real-time correlation tracking
- ✅ Decoupling detection with signals
- ✅ Bridge stress index
- ✅ Paper trading testnet
- ✅ Accuracy reporting by confidence level

## Credits

- **Architecture:** H T Armstrong & augerd
- **Integration:** obbba Gitbook prediction code
- **Imperial Seal:** Pending Emperor review

---

*For the Empire. 🔴⚖️*
