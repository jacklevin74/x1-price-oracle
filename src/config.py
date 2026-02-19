"""
Multi-Chain Price Oracle Configuration
Supports: BTC/XNT, SOL/XNT, ETH/XNT, BNB/XNT, and additional L1 gas tokens
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# Token Pair Configuration
@dataclass
class TokenPairConfig:
    name: str
    base_token: str
    quote_token: str  # XNT
    base_chain: str
    rpc_endpoints: List[str]
    pool_addresses: List[str]  # X1 DEX pool addresses
    decimals: int
    is_native: bool
    bridge_contract: Optional[str] = None

# Multi-Chain RPC Endpoints
RPC_ENDPOINTS = {
    # Solana (source for SOL/XNT)
    "solana": [
        "https://api.mainnet-beta.solana.com",
        "https://solana-api.projectserum.com",
        "https://rpc.ankr.com/solana",
    ],
    "solana_devnet": [
        "https://api.devnet.solana.com",
    ],
    
    # Ethereum (source for ETH/XNT)
    "ethereum": [
        "https://eth.llamarpc.com",
        "https://rpc.ankr.com/eth",
        "https://ethereum-rpc.publicnode.com",
    ],
    "ethereum_sepolia": [
        "https://rpc.sepolia.org",
        "https://eth-sepolia.public.blastapi.io",
    ],
    
    # BSC (source for BNB/XNT)
    "bsc": [
        "https://bsc-dataseed.binance.org",
        "https://rpc.ankr.com/bsc",
        "https://bsc-rpc.publicnode.com",
    ],
    "bsc_testnet": [
        "https://data-seed-prebsc-1-s1.binance.org:8545",
    ],
    
    # Bitcoin (source for BTC/XNT via wrapped BTC on Solana/Ethereum)
    "bitcoin": [
        "https://blockchain.info",
        "https://api.blockcypher.com/v1/btc/main",
    ],
    
    # X1 (destination chain - all pairs quote against XNT)
    "x1_mainnet": [
        "https://rpc.mainnet.x1.xyz",
    ],
    "x1_testnet": [
        "https://rpc.testnet.x1.xyz",
    ],
}

# Token Contract Addresses (Mainnet)
TOKEN_ADDRESSES = {
    "x1": {
        "XNT": "So11111111111111111111111111111111111111112",  # Wrapped XNT
        "USDC": "B69chRzqzDCmdB5WYB8NRu5Yv5ZA95ABiZcdzCgGm9Tq",
        "WBTC": "3NZ9JMVBmGAqocybic2c6NL8VJbVdt8Lzd7EzJ7BW1nF",  # Wrapped BTC on X1
        "WETH": "7vfCXTUXxYkU9f3yqG3Wj7q8Z9j3vYkU9f3yqG3Wj7q",  # Wrapped ETH on X1
        "WBNB": "8vfCXTUXxYkU9f3yqG3Wj7q8Z9j3vYkU9f3yqG3Wj7r",  # Wrapped BNB on X1
    },
    "solana": {
        "SOL": "So11111111111111111111111111111111111111112",
        "WBTC": "3NZ9JMVBmGAqocybic2c6NL8VJbVdt8Lzd7EzJ7BW1nF",
        "WETH": "7vfCXTUXxYkU9f3yqG3Wj7q8Z9j3vYkU9f3yqG3Wj7q",
    },
    "ethereum": {
        "ETH": "0x0000000000000000000000000000000000000000",
        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    },
    "bsc": {
        "BNB": "0x0000000000000000000000000000000000000000",
        "BTCB": "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c",  # BTC on BSC
        "ETH": "0x2170Ed0880ac9A755fd29B2688956BD959F933F8",
    },
}

# X1 DEX Pool Addresses (X1 Testnet/Mainnet)
X1_POOLS = {
    "BTC_XNT": {
        "mainnet": "GRYbq732zobr8fwDkqnjnaNCg5Qf5Y7vgRWBhKLFRu3j",  # Example
        "testnet": "TestBTCXNTXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    },
    "SOL_XNT": {
        "mainnet": "SoLxNTXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",  # Placeholder
        "testnet": "TestSOLXNTXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    },
    "ETH_XNT": {
        "mainnet": "ETHxNTXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",  # Placeholder
        "testnet": "TestETHXNTXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    },
    "BNB_XNT": {
        "mainnet": "BNBxNTXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",  # Placeholder
        "testnet": "TestBNBXNTXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    },
}

# Token Pair Configurations
TOKEN_PAIRS: Dict[str, TokenPairConfig] = {
    "BTC_XNT": TokenPairConfig(
        name="BTC/XNT",
        base_token="BTC",
        quote_token="XNT",
        base_chain="bitcoin",
        rpc_endpoints=RPC_ENDPOINTS["bitcoin"] + RPC_ENDPOINTS["solana"],
        pool_addresses=[X1_POOLS["BTC_XNT"]["mainnet"], X1_POOLS["BTC_XNT"]["testnet"]],
        decimals=9,
        is_native=False,
        bridge_contract="C6byAvMfEa9wrbfVDeLEWbCkQNa8HAtpGxDPZKG3FqRp",  # X1 bridge
    ),
    "SOL_XNT": TokenPairConfig(
        name="SOL/XNT",
        base_token="SOL",
        quote_token="XNT",
        base_chain="solana",
        rpc_endpoints=RPC_ENDPOINTS["solana"],
        pool_addresses=[X1_POOLS["SOL_XNT"]["mainnet"], X1_POOLS["SOL_XNT"]["testnet"]],
        decimals=9,
        is_native=True,  # SOL is native to SVM which X1 uses
        bridge_contract=None,  # Native - no bridge needed
    ),
    "ETH_XNT": TokenPairConfig(
        name="ETH/XNT",
        base_token="ETH",
        quote_token="XNT",
        base_chain="ethereum",
        rpc_endpoints=RPC_ENDPOINTS["ethereum"],
        pool_addresses=[X1_POOLS["ETH_XNT"]["mainnet"], X1_POOLS["ETH_XNT"]["testnet"]],
        decimals=18,
        is_native=False,
        bridge_contract="0x...",  # Ethereum bridge contract
    ),
    "BNB_XNT": TokenPairConfig(
        name="BNB/XNT",
        base_token="BNB",
        quote_token="XNT",
        base_chain="bsc",
        rpc_endpoints=RPC_ENDPOINTS["bsc"],
        pool_addresses=[X1_POOLS["BNB_XNT"]["mainnet"], X1_POOLS["BNB_XNT"]["testnet"]],
        decimals=18,
        is_native=False,
        bridge_contract="0x...",  # BSC bridge contract
    ),
}

# Additional L1 Gas Tokens (for future expansion)
ADDITIONAL_L1_TOKENS = {
    "AVAX": {"chain": "avalanche", "decimals": 18, "is_native": False},
    "MATIC": {"chain": "polygon", "decimals": 18, "is_native": False},
    "ARB": {"chain": "arbitrum", "decimals": 18, "is_native": False},
    "OP": {"chain": "optimism", "decimals": 18, "is_native": False},
    "FTM": {"chain": "fantom", "decimals": 18, "is_native": False},
    "NEAR": {"chain": "near", "decimals": 24, "is_native": False},
    "APT": {"chain": "aptos", "decimals": 8, "is_native": False},
    "SUI": {"chain": "sui", "decimals": 9, "is_native": False},
}

# Timeframe Configuration
TIMEFRAMES = {
    "5m": {"minutes": 5, "aggregation": "1m", "lookback_periods": 20},
    "10m": {"minutes": 10, "aggregation": "1m", "lookback_periods": 40},
    "25m": {"minutes": 25, "aggregation": "5m", "lookback_periods": 25},
    "1h": {"minutes": 60, "aggregation": "5m", "lookback_periods": 60},
    "4h": {"minutes": 240, "aggregation": "15m", "lookback_periods": 64},
}

# XDEX API Configuration
XDEX_API = {
    "base_url": "https://api.xdex.xyz",
    "endpoints": {
        "price": "/api/token-price/price",
        "pool_list": "/api/xendex/pool/list",
        "swap_quote": "/api/xendex/swap/quote",
        "wallet_tokens": "/api/xendex/wallet/tokens",
    },
    "networks": {
        "mainnet": "X1%20Mainnet",
        "testnet": "X1%20Testnet",
    }
}

# Price Feed Sources (for cross-validation)
PRICE_SOURCES = {
    "coingecko": "https://api.coingecko.com/api/v3/simple/price",
    "coinmarketcap": "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
    "binance": "https://api.binance.com/api/v3/ticker/price",
    "raydium": "https://api.raydium.io/v2/main/price",
    "uniswap": "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
    "pancakeswap": "https://api.thegraph.com/subgraphs/name/pancakeswap/exchange-v3-bsc",
}

# Paper Trading Configuration
PAPER_TRADING = {
    "initial_balance_usd": 10000.0,
    "initial_balance_xnt": 100000.0,
    "fee_rate": 0.003,  # 0.3% trading fee
    "slippage": 0.01,   # 1% slippage simulation
    "max_position_size": 0.25,  # Max 25% of portfolio per position
    "stop_loss_pct": 0.05,      # 5% stop loss
    "take_profit_pct": 0.15,    # 15% take profit
}

# Correlation Settings
CORRELATION_CONFIG = {
    "lookback_window": 100,  # Periods for correlation calculation
    "min_correlation_threshold": 0.7,  # Strong correlation threshold
    "decoupling_threshold": 0.3,  # Weak correlation signals decoupling
    "update_interval": 300,  # Update every 5 minutes
}

# Bridge Stress Index Configuration
BRIDGE_STRESS_CONFIG = {
    "max_expected_divergence": 0.02,  # 2% max expected price divergence
    "volume_spike_threshold": 3.0,    # 3x average volume = spike
    "stress_levels": {
        "low": 0.0,
        "moderate": 0.3,
        "high": 0.6,
        "critical": 0.8,
    },
}

# Logging Configuration
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "price_oracle.log",
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5,
}
