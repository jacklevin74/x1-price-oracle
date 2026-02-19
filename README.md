# Multi-Chain Price Oracle

Real-time price prediction engine for X1 blockchain with multi-chain support and correlation tracking.

## Features

- **Multi-Chain Feeds:** BTC, SOL, ETH, BNB → XNT
- **Multi-Timeframe:** 5m/10m/25m prediction windows
- **Correlation Matrix:** Real-time pair correlation with decoupling detection
- **Bridge Stress Index:** Composite score for bridge health
- **Paper Trading:** Testnet with virtual XNT balances

## Quick Start

```bash
cd price-oracle
pip install -r requirements.txt

# Test the oracle
cd testnet
python3 testnet_trader.py --reset
python3 testnet_trader.py --report
```

## Structure

- `src/price_oracle.py` - Core prediction engine
- `src/config.py` - All configuration
- `testnet/testnet_trader.py` - Paper trading
- `docs/EXPANSION.md` - Full documentation

## Credits

H T Armstrong • augerd • obbba • Cyberdyne Unlimited
