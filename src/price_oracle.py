"""
Multi-Chain Price Oracle Engine
Supports multi-timeframe architecture (5m/10m/25m) for BTC/XNT and expanded pairs
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import json
import aiohttp
import numpy as np
from dataclasses import asdict

from config import (
    TokenPairConfig, TOKEN_PAIRS, RPC_ENDPOINTS, XDEX_API,
    TIMEFRAMES, PRICE_SOURCES, CORRELATION_CONFIG
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PriceOracle')


@dataclass
class PricePoint:
    """Single price data point"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    pair: str
    source: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'pair': self.pair,
            'source': self.source,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class Prediction:
    """Price prediction for a specific timeframe"""
    pair: str
    timeframe: str
    timestamp: float
    predicted_direction: str  # 'up', 'down', 'neutral'
    predicted_price: float
    confidence: float
    features_used: List[str]
    model_version: str
    
    def to_dict(self) -> Dict:
        return {
            'pair': self.pair,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp,
            'predicted_direction': self.predicted_direction,
            'predicted_price': self.predicted_price,
            'confidence': self.confidence,
            'features_used': self.features_used,
            'model_version': self.model_version,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class CorrelationData:
    """Correlation data between two pairs"""
    pair_a: str
    pair_b: str
    correlation: float
    timestamp: float
    lookback_periods: int
    is_decoupled: bool
    decoupling_signal: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'pair_a': self.pair_a,
            'pair_b': self.pair_b,
            'correlation': self.correlation,
            'timestamp': self.timestamp,
            'lookback_periods': self.lookback_periods,
            'is_decoupled': self.is_decoupled,
            'decoupling_signal': self.decoupling_signal,
        }


class MultiTimeframeData:
    """Manages price data across multiple timeframes"""
    
    def __init__(self, pair: str, max_history: int = 1000):
        self.pair = pair
        self.max_history = max_history
        self.timeframe_data: Dict[str, deque] = {}
        self.raw_ticks: deque = deque(maxlen=10000)  # Raw 1-minute ticks
        
        # Initialize timeframe buckets
        for tf in TIMEFRAMES.keys():
            self.timeframe_data[tf] = deque(maxlen=max_history)
    
    def add_tick(self, price_point: PricePoint):
        """Add a raw tick and aggregate to timeframes"""
        self.raw_ticks.append(price_point)
        self._aggregate_to_timeframes(price_point)
    
    def _aggregate_to_timeframes(self, tick: PricePoint):
        """Aggregate raw tick to all timeframes"""
        for tf_name, tf_config in TIMEFRAMES.items():
            self._update_timeframe_candle(tf_name, tf_config, tick)
    
    def _update_timeframe_candle(self, tf_name: str, tf_config: dict, tick: PricePoint):
        """Update or create candle for a specific timeframe"""
        minutes = tf_config['minutes']
        bucket_time = (int(tick.timestamp) // (minutes * 60)) * (minutes * 60)
        
        # Check if we need a new candle
        if not self.timeframe_data[tf_name]:
            self._create_new_candle(tf_name, bucket_time, tick)
        else:
            last_candle = self.timeframe_data[tf_name][-1]
            if last_candle.timestamp == bucket_time:
                # Update existing candle
                last_candle.high = max(last_candle.high, tick.close)
                last_candle.low = min(last_candle.low, tick.close)
                last_candle.close = tick.close
                last_candle.volume += tick.volume
            else:
                # Create new candle
                self._create_new_candle(tf_name, bucket_time, tick)
    
    def _create_new_candle(self, tf_name: str, bucket_time: float, tick: PricePoint):
        """Create a new OHLCV candle"""
        candle = PricePoint(
            timestamp=bucket_time,
            open=tick.close,
            high=tick.close,
            low=tick.close,
            close=tick.close,
            volume=tick.volume,
            pair=self.pair,
            source=tick.source
        )
        self.timeframe_data[tf_name].append(candle)
    
    def get_data(self, timeframe: str) -> List[PricePoint]:
        """Get all data points for a specific timeframe"""
        return list(self.timeframe_data.get(timeframe, deque()))
    
    def get_latest(self, timeframe: str) -> Optional[PricePoint]:
        """Get the latest price point for a timeframe"""
        data = self.timeframe_data.get(timeframe, deque())
        return data[-1] if data else None
    
    def get_ohlcv_array(self, timeframe: str) -> Tuple[np.ndarray, ...]:
        """Get OHLCV data as numpy arrays for ML processing"""
        data = list(self.timeframe_data.get(timeframe, deque()))
        if not data:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        opens = np.array([d.open for d in data])
        highs = np.array([d.high for d in data])
        lows = np.array([d.low for d in data])
        closes = np.array([d.close for d in data])
        volumes = np.array([d.volume for d in data])
        
        return opens, highs, lows, closes, volumes


class PriceFetcher:
    """Fetches price data from multiple sources"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.price_cache: Dict[str, Tuple[float, float]] = {}  # pair -> (price, timestamp)
        self.cache_ttl = 10  # 10 second cache
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_x1_dex_price(self, pair: str, network: str = "mainnet") -> Optional[float]:
        """Fetch price from X1 DEX API"""
        try:
            config = TOKEN_PAIRS.get(pair)
            if not config:
                logger.warning(f"No config found for pair: {pair}")
                return None
            
            # Use XDEX API to get pool data
            network_param = XDEX_API['networks'].get(network, 'X1%20Mainnet')
            
            # Find token mint address
            token_mint = self._get_token_mint(config.base_token, 'x1')
            if not token_mint:
                return None
            
            url = f"{XDEX_API['base_url']}{XDEX_API['endpoints']['price']}"
            params = {
                'network': network_param,
                'address': token_mint
            }
            
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = data.get('price')
                    if price:
                        self.price_cache[pair] = (float(price), time.time())
                        return float(price)
                else:
                    logger.warning(f"XDEX API returned status {resp.status} for {pair}")
                    
        except Exception as e:
            logger.error(f"Error fetching X1 DEX price for {pair}: {e}")
        
        return None
    
    async def fetch_coingecko_price(self, token_id: str) -> Optional[float]:
        """Fetch price from CoinGecko"""
        try:
            url = f"{PRICE_SOURCES['coingecko']}"
            params = {
                'ids': token_id,
                'vs_currencies': 'usd'
            }
            
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get(token_id, {}).get('usd')
                    
        except Exception as e:
            logger.error(f"Error fetching CoinGecko price for {token_id}: {e}")
        
        return None
    
    async def fetch_binance_price(self, symbol: str) -> Optional[float]:
        """Fetch price from Binance"""
        try:
            url = f"{PRICE_SOURCES['binance']}"
            params = {'symbol': symbol}
            
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get('price', 0))
                    
        except Exception as e:
            logger.error(f"Error fetching Binance price for {symbol}: {e}")
        
        return None
    
    async def fetch_solana_price(self, token_address: str) -> Optional[float]:
        """Fetch price from Solana RPC (using Jupiter or Raydium)"""
        try:
            # Use Jupiter price API for Solana tokens
            url = f"https://price.jup.ag/v4/price?ids={token_address}"
            
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price_data = data.get('data', {}).get(token_address, {})
                    return price_data.get('price')
                    
        except Exception as e:
            logger.error(f"Error fetching Solana price for {token_address}: {e}")
        
        return None
    
    def _get_token_mint(self, token: str, chain: str) -> Optional[str]:
        """Get token mint address for a chain"""
        from config import TOKEN_ADDRESSES
        return TOKEN_ADDRESSES.get(chain, {}).get(token)
    
    async def fetch_pair_price(self, pair: str) -> Optional[PricePoint]:
        """Fetch current price for a token pair from multiple sources"""
        config = TOKEN_PAIRS.get(pair)
        if not config:
            return None
        
        # Check cache first
        if pair in self.price_cache:
            price, ts = self.price_cache[pair]
            if time.time() - ts < self.cache_ttl:
                return PricePoint(
                    timestamp=time.time(),
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0,
                    pair=pair,
                    source='cache'
                )
        
        # Try multiple sources
        price = None
        source = 'unknown'
        
        # Source 1: X1 DEX
        if not price:
            price = await self.fetch_x1_dex_price(pair)
            if price:
                source = 'x1_dex'
        
        # Source 2: CoinGecko
        if not price:
            token_map = {
                'BTC': 'bitcoin',
                'SOL': 'solana',
                'ETH': 'ethereum',
                'BNB': 'binancecoin'
            }
            token_id = token_map.get(config.base_token)
            if token_id:
                price = await self.fetch_coingecko_price(token_id)
                if price:
                    source = 'coingecko'
        
        # Source 3: Binance
        if not price:
            symbol_map = {
                'BTC': 'BTCUSDT',
                'SOL': 'SOLUSDT',
                'ETH': 'ETHUSDT',
                'BNB': 'BNBUSDT'
            }
            symbol = symbol_map.get(config.base_token)
            if symbol:
                price = await self.fetch_binance_price(symbol)
                if price:
                    source = 'binance'
        
        if price:
            return PricePoint(
                timestamp=time.time(),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0,  # Volume requires separate call
                pair=pair,
                source=source
            )
        
        return None


class PredictionEngine:
    """Generates predictions using technical analysis and ML signals"""
    
    def __init__(self):
        self.model_version = "v1.0.0-technical"
    
    def predict(self, pair: str, timeframe: str, data: MultiTimeframeData) -> Optional[Prediction]:
        """Generate prediction for a pair and timeframe"""
        ohlcv = data.get_ohlcv_array(timeframe)
        closes = ohlcv[3]  # Index 3 is closes
        volumes = ohlcv[4]  # Index 4 is volumes
        
        if len(closes) < 20:
            return None
        
        # Calculate technical indicators
        features = []
        
        # 1. Simple Moving Averages
        sma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
        sma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        
        features.append(f"sma5_{sma_5:.4f}")
        features.append(f"sma10_{sma_10:.4f}")
        features.append(f"sma20_{sma_20:.4f}")
        
        # 2. Price vs SMA position
        current = closes[-1]
        above_sma5 = current > sma_5
        above_sma10 = current > sma_10
        above_sma20 = current > sma_20
        
        features.append(f"above_sma5_{above_sma5}")
        features.append(f"above_sma10_{above_sma10}")
        features.append(f"above_sma20_{above_sma20}")
        
        # 3. Trend direction (short term)
        price_change = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        features.append(f"change_5period_{price_change:.4f}")
        
        # 4. Volatility (ATR-like)
        if len(closes) >= 5:
            highs = ohlcv[1][-5:]
            lows = ohlcv[2][-5:]
            volatility = np.mean((highs - lows) / lows)
            features.append(f"volatility_{volatility:.4f}")
        
        # 5. Volume trend
        if len(volumes) >= 5:
            vol_sma = np.mean(volumes[-5:])
            vol_spike = volumes[-1] > vol_sma * 1.5 if vol_sma > 0 else False
            features.append(f"volume_spike_{vol_spike}")
        
        # Determine prediction
        signals = []
        
        # Golden/Death cross signals
        if sma_5 > sma_10 > sma_20:
            signals.append('strong_bullish')
        elif sma_5 < sma_10 < sma_20:
            signals.append('strong_bearish')
        elif sma_5 > sma_10:
            signals.append('bullish')
        elif sma_5 < sma_10:
            signals.append('bearish')
        
        # Price momentum
        if price_change > 0.02:  # >2% up
            signals.append('momentum_up')
        elif price_change < -0.02:  # >2% down
            signals.append('momentum_down')
        
        # Calculate confidence based on signal alignment
        bullish_count = sum(1 for s in signals if 'bullish' in s or 'up' in s)
        bearish_count = sum(1 for s in signals if 'bearish' in s or 'down' in s)
        
        # Direction and confidence
        if bullish_count > bearish_count:
            direction = 'up'
            confidence = min(0.5 + (bullish_count - bearish_count) * 0.15, 0.95)
        elif bearish_count > bullish_count:
            direction = 'down'
            confidence = min(0.5 + (bearish_count - bullish_count) * 0.15, 0.95)
        else:
            direction = 'neutral'
            confidence = 0.5
        
        # Predicted price (simple extrapolation)
        if direction == 'up':
            predicted = current * (1 + abs(price_change) * 0.5)
        elif direction == 'down':
            predicted = current * (1 - abs(price_change) * 0.5)
        else:
            predicted = current
        
        return Prediction(
            pair=pair,
            timeframe=timeframe,
            timestamp=time.time(),
            predicted_direction=direction,
            predicted_price=predicted,
            confidence=confidence,
            features_used=features,
            model_version=self.model_version
        )


class CorrelationTracker:
    """Tracks correlations between token pairs and detects decoupling"""
    
    def __init__(self):
        self.correlations: Dict[str, CorrelationData] = {}
        self.price_history: Dict[str, deque] = {}
        self.lookback = CORRELATION_CONFIG['lookback_window']
        
        # Initialize price history for all pairs
        for pair in TOKEN_PAIRS.keys():
            self.price_history[pair] = deque(maxlen=self.lookback)
    
    def update_price(self, pair: str, price: float):
        """Add new price point for correlation calculation"""
        if pair in self.price_history:
            self.price_history[pair].append(price)
    
    def calculate_all_correlations(self) -> List[CorrelationData]:
        """Calculate correlations between all pair combinations"""
        pairs = list(TOKEN_PAIRS.keys())
        results = []
        
        for i, pair_a in enumerate(pairs):
            for pair_b in pairs[i+1:]:
                corr_data = self._calculate_pair_correlation(pair_a, pair_b)
                if corr_data:
                    results.append(corr_data)
                    key = f"{pair_a}:{pair_b}"
                    self.correlations[key] = corr_data
        
        return results
    
    def _calculate_pair_correlation(self, pair_a: str, pair_b: str) -> Optional[CorrelationData]:
        """Calculate correlation between two pairs"""
        prices_a = list(self.price_history[pair_a])
        prices_b = list(self.price_history[pair_b])
        
        if len(prices_a) < 10 or len(prices_b) < 10:
            return None
        
        # Ensure same length
        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[-min_len:]
        prices_b = prices_b[-min_len:]
        
        # Calculate returns
        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]
        
        if len(returns_a) < 2 or len(returns_b) < 2:
            return None
        
        # Calculate correlation
        correlation = np.corrcoef(returns_a, returns_b)[0, 1]
        
        # Handle NaN
        if np.isnan(correlation):
            correlation = 0.0
        
        # Detect decoupling
        is_decoupled = abs(correlation) < CORRELATION_CONFIG['decoupling_threshold']
        
        signal = None
        if is_decoupled:
            # Determine which pair is leading
            recent_vol_a = np.std(returns_a[-5:]) if len(returns_a) >= 5 else 0
            recent_vol_b = np.std(returns_b[-5:]) if len(returns_b) >= 5 else 0
            
            if recent_vol_a > recent_vol_b:
                signal = f"{pair_a}_leading_breakaway"
            else:
                signal = f"{pair_b}_leading_breakaway"
        
        return CorrelationData(
            pair_a=pair_a,
            pair_b=pair_b,
            correlation=correlation,
            timestamp=time.time(),
            lookback_periods=min_len,
            is_decoupled=is_decoupled,
            decoupling_signal=signal
        )
    
    def get_correlation_matrix(self) -> Dict[str, float]:
        """Get correlation matrix as dictionary"""
        matrix = {}
        for key, data in self.correlations.items():
            matrix[key] = data.correlation
        return matrix
    
    def get_decoupling_signals(self) -> List[CorrelationData]:
        """Get all current decoupling signals"""
        return [c for c in self.correlations.values() if c.is_decoupled]


class PriceOracle:
    """Main Price Oracle class that coordinates all components"""
    
    def __init__(self):
        self.price_data: Dict[str, MultiTimeframeData] = {}
        self.fetcher = PriceFetcher()
        self.predictor = PredictionEngine()
        self.correlation_tracker = CorrelationTracker()
        self.running = False
        self.update_interval = 60  # seconds
        
        # Initialize data structures for all pairs
        for pair in TOKEN_PAIRS.keys():
            self.price_data[pair] = MultiTimeframeData(pair)
    
    async def start(self):
        """Start the price oracle update loop"""
        self.running = True
        logger.info("Starting Price Oracle...")
        
        async with self.fetcher:
            while self.running:
                await self._update_cycle()
                await asyncio.sleep(self.update_interval)
    
    async def _update_cycle(self):
        """Single update cycle - fetch prices for all pairs"""
        tasks = []
        for pair in TOKEN_PAIRS.keys():
            task = self._update_pair(pair)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update correlations after all prices are updated
        correlations = self.correlation_tracker.calculate_all_correlations()
        
        # Log decoupling signals
        decoupled = self.correlation_tracker.get_decoupling_signals()
        if decoupled:
            for d in decoupled:
                logger.info(f"DECOUPLING DETECTED: {d.pair_a} vs {d.pair_b} "
                           f"(correlation: {d.correlation:.3f}, signal: {d.decoupling_signal})")
    
    async def _update_pair(self, pair: str):
        """Update price for a single pair"""
        try:
            price_point = await self.fetcher.fetch_pair_price(pair)
            
            if price_point:
                self.price_data[pair].add_tick(price_point)
                self.correlation_tracker.update_price(pair, price_point.close)
                
                logger.debug(f"Updated {pair}: ${price_point.close:.4f} from {price_point.source}")
            else:
                logger.warning(f"Failed to fetch price for {pair}")
                
        except Exception as e:
            logger.error(f"Error updating {pair}: {e}")
    
    def stop(self):
        """Stop the price oracle"""
        self.running = False
        logger.info("Stopping Price Oracle...")
    
    def get_prediction(self, pair: str, timeframe: str) -> Optional[Prediction]:
        """Get prediction for a pair and timeframe"""
        if pair not in self.price_data:
            return None
        
        return self.predictor.predict(pair, timeframe, self.price_data[pair])
    
    def get_all_predictions(self) -> Dict[str, List[Prediction]]:
        """Get predictions for all pairs and timeframes"""
        results = {}
        
        for pair in TOKEN_PAIRS.keys():
            pair_predictions = []
            for tf in TIMEFRAMES.keys():
                pred = self.get_prediction(pair, tf)
                if pred:
                    pair_predictions.append(pred)
            
            if pair_predictions:
                results[pair] = pair_predictions
        
        return results
    
    def get_latest_prices(self) -> Dict[str, PricePoint]:
        """Get latest price for all pairs"""
        results = {}
        for pair, data in self.price_data.items():
            latest = data.get_latest('5m')  # Use 5m as default
            if latest:
                results[pair] = latest
        return results
    
    def get_correlation_report(self) -> Dict:
        """Get full correlation report"""
        return {
            'timestamp': time.time(),
            'matrix': self.correlation_tracker.get_correlation_matrix(),
            'decoupling_signals': [
                d.to_dict() for d in self.correlation_tracker.get_decoupling_signals()
            ],
            'strong_correlations': [
                {'pairs': k, 'correlation': v}
                for k, v in self.correlation_tracker.get_correlation_matrix().items()
                if abs(v) > CORRELATION_CONFIG['min_correlation_threshold']
            ]
        }


# Singleton instance
_oracle_instance: Optional[PriceOracle] = None

def get_oracle() -> PriceOracle:
    """Get or create singleton oracle instance"""
    global _oracle_instance
    if _oracle_instance is None:
        _oracle_instance = PriceOracle()
    return _oracle_instance
