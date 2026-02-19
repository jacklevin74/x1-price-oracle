#!/usr/bin/env python3
"""
Paper Trading Testnet for Multi-Chain Price Oracle
Virtual balances, real predictions, no capital risk.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from price_oracle import PriceOracle, get_oracle, Prediction
from config import PAPER_TRADING, TOKEN_PAIRS, TIMEFRAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PaperTrader')


@dataclass
class Position:
    """Paper trading position"""
    pair: str
    direction: str  # 'long' or 'short'
    entry_price: float
    size: float  # Amount of XNT risked
    entry_time: float
    timeframe: str
    predicted_direction: str
    confidence: float
    stop_loss: float
    take_profit: float
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    pnl: float = 0.0
    status: str = "open"  # open, closed, stopped
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeLog:
    """Record of completed trade"""
    timestamp: float
    pair: str
    direction: str
    entry: float
    exit: float
    size: float
    pnl: float
    pnl_pct: float
    predicted_direction: str
    actual_direction: str
    correct_prediction: bool
    confidence: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class PaperTrader:
    """Paper trading engine with virtual XNT balances"""
    
    def __init__(self, state_file: str = "testnet_state.json"):
        self.state_file = Path(state_file)
        self.balance_xnt: float = 0.0
        self.balance_usd: float = 0.0
        self.positions: List[Position] = []
        self.trade_history: List[TradeLog] = []
        self.oracle: Optional[PriceOracle] = None
        self.config = PAPER_TRADING
        self.trading_active = False
        self.last_prediction_check: float = 0
        
        # Load or initialize state
        self._load_state()
    
    def _load_state(self):
        """Load testnet state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.balance_xnt = state.get('balance_xnt', self.config['initial_balance_xnt'])
                self.balance_usd = state.get('balance_usd', self.config['initial_balance_usd'])
                self.trade_history = [TradeLog(**t) for t in state.get('trade_history', [])]
                
                # Reconstruct positions
                self.positions = [Position(**p) for p in state.get('positions', [])]
                
                logger.info(f"Loaded testnet state: {self.balance_xnt:.2f} vXNT, {self.balance_usd:.2f} vUSD")
                logger.info(f"Active positions: {len(self.positions)}, Completed trades: {len(self.trade_history)}")
                
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                self._reset_balances()
        else:
            self._reset_balances()
    
    def _reset_balances(self):
        """Reset to initial virtual balances"""
        self.balance_xnt = self.config['initial_balance_xnt']
        self.balance_usd = self.config['initial_balance_usd']
        self.positions = []
        self.trade_history = []
        logger.info(f"Initialized fresh testnet: {self.balance_xnt:.2f} vXNT, {self.balance_usd:.2f} vUSD")
    
    def save_state(self):
        """Persist testnet state to disk"""
        state = {
            'timestamp': time.time(),
            'balance_xnt': self.balance_xnt,
            'balance_usd': self.balance_usd,
            'positions': [p.to_dict() for p in self.positions],
            'trade_history': [t.to_dict() for t in self.trade_history],
            'config': self.config,
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.debug(f"Saved testnet state to {self.state_file}")
    
    def reset(self):
        """Clear all state and reset balances"""
        logger.warning("RESETTING TESTNET STATE - All positions and history cleared")
        self._reset_balances()
        if self.state_file.exists():
            self.state_file.unlink()
        self.save_state()
        print(f"✓ Testnet reset: {self.balance_xnt:,.2f} vXNT, ${self.balance_usd:,.2f} vUSD")
    
    async def start(self):
        """Start paper trading loop"""
        self.trading_active = True
        self.oracle = get_oracle()
        
        logger.info("Starting paper trading testnet...")
        
        async with self.oracle.fetcher:
            while self.trading_active:
                await self._trading_cycle()
                self.save_state()
                await asyncio.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop trading loop"""
        self.trading_active = False
        self.save_state()
        logger.info("Paper trading stopped")
    
    async def _trading_cycle(self):
        """One trading cycle - check predictions, manage positions"""
        now = time.time()
        
        # Update prices for all pairs
        for pair in TOKEN_PAIRS.keys():
            await self.oracle._update_pair(pair)
        
        # Update correlations
        self.oracle.correlation_tracker.calculate_all_correlations()
        
        # Check existing positions for exit conditions
        await self._check_position_exits()
        
        # Look for new entry opportunities
        if now - self.last_prediction_check > 300:  # Every 5 minutes
            await self._evaluate_entries()
            self.last_prediction_check = now
    
    async def _check_position_exits(self):
        """Check if any positions should be closed"""
        for position in self.positions:
            if position.status != "open":
                continue
            
            # Get current price
            latest = self.oracle.price_data[position.pair].get_latest('5m')
            if not latest:
                continue
            
            current_price = latest.close
            
            # Check stop loss
            if position.direction == "long":
                if current_price <= position.stop_loss:
                    await self._close_position(position, current_price, "stopped")
                    continue
                
                # Check take profit
                if current_price >= position.take_profit:
                    await self._close_position(position, current_price, "profit")
                    continue
            
            else:  # short
                if current_price >= position.stop_loss:
                    await self._close_position(position, current_price, "stopped")
                    continue
                
                if current_price <= position.take_profit:
                    await self._close_position(position, current_price, "profit")
                    continue
    
    async def _close_position(self, position: Position, exit_price: float, reason: str):
        """Close a position and record P&L"""
        position.exit_price = exit_price
        position.exit_time = time.time()
        
        # Calculate P&L
        if position.direction == "long":
            position.pnl = (exit_price - position.entry_price) / position.entry_price * position.size
        else:
            position.pnl = (position.entry_price - exit_price) / position.entry_price * position.size
        
        # Apply fees (entry + exit)
        fees = position.size * self.config['fee_rate'] * 2
        position.pnl -= fees
        
        position.status = reason
        
        # Update balance
        self.balance_xnt += position.pnl
        
        # Determine if prediction was correct
        actual_direction = "up" if exit_price > position.entry_price else "down"
        if position.direction == "short":
            actual_direction = "up" if exit_price < position.entry_price else "down"
        
        correct = position.predicted_direction == actual_direction
        
        # Log trade
        trade = TradeLog(
            timestamp=position.exit_time,
            pair=position.pair,
            direction=position.direction,
            entry=position.entry_price,
            exit=exit_price,
            size=position.size,
            pnl=position.pnl,
            pnl_pct=(position.pnl / position.size * 100) if position.size > 0 else 0,
            predicted_direction=position.predicted_direction,
            actual_direction=actual_direction,
            correct_prediction=correct,
            confidence=position.confidence
        )
        
        self.trade_history.append(trade)
        
        logger.info(f"Position closed [{reason}]: {position.pair} {position.direction} "
                   f"P&L: {position.pnl:.2f} vXNT ({trade.pnl_pct:.1f}%)")
    
    async def _evaluate_entries(self):
        """Evaluate new entry opportunities"""
        max_positions = 4  # Max concurrent positions
        
        if len([p for p in self.positions if p.status == "open"]) >= max_positions:
            return
        
        for pair in TOKEN_PAIRS.keys():
            # Skip if already have position in this pair
            if any(p.pair == pair and p.status == "open" for p in self.positions):
                continue
            
            # Get prediction for 25m timeframe
            prediction = self.oracle.get_prediction(pair, '25m')
            if not prediction:
                continue
            
            # Only enter on high confidence
            if prediction.confidence < 0.7:
                continue
            
            # Don't trade neutral
            if prediction.predicted_direction == "neutral":
                continue
            
            # Get current price
            latest = self.oracle.price_data[pair].get_latest('5m')
            if not latest:
                continue
            
            current_price = latest.close
            
            # Calculate position size (max 25% of portfolio)
            portfolio_value = self.balance_xnt * current_price + self.balance_usd
            position_size = portfolio_value * self.config['max_position_size']
            
            # Direction
            direction = "long" if prediction.predicted_direction == "up" else "short"
            
            # Stop loss and take profit
            if direction == "long":
                stop_loss = current_price * (1 - self.config['stop_loss_pct'])
                take_profit = current_price * (1 + self.config['take_profit_pct'])
            else:
                stop_loss = current_price * (1 + self.config['stop_loss_pct'])
                take_profit = current_price * (1 - self.config['take_profit_pct'])
            
            # Create position
            position = Position(
                pair=pair,
                direction=direction,
                entry_price=current_price,
                size=position_size / current_price,  # Convert to XNT amount
                entry_time=time.time(),
                timeframe='25m',
                predicted_direction=prediction.predicted_direction,
                confidence=prediction.confidence,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.positions.append(position)
            
            logger.info(f"New position: {pair} {direction} @ {current_price:.4f} "
                       f"(confidence: {prediction.confidence:.1%})")
    
    def get_report(self) -> Dict:
        """Generate accuracy report"""
        if not self.trade_history:
            return {
                'status': 'No trades completed yet',
                'balance_xnt': self.balance_xnt,
                'balance_usd': self.balance_usd,
                'active_positions': len([p for p in self.positions if p.status == "open"])
            }
        
        trades = self.trade_history
        
        # Overall stats
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # Prediction accuracy
        correct_predictions = len([t for t in trades if t.correct_prediction])
        directional_accuracy = correct_predictions / total_trades if total_trades > 0 else 0
        
        # By pair
        by_pair = {}
        for trade in trades:
            if trade.pair not in by_pair:
                by_pair[trade.pair] = {'trades': [], 'pnl': 0}
            by_pair[trade.pair]['trades'].append(trade)
            by_pair[trade.pair]['pnl'] += trade.pnl
        
        # By confidence level
        high_conf = [t for t in trades if t.confidence >= 0.8]
        med_conf = [t for t in trades if 0.7 <= t.confidence < 0.8]
        
        return {
            'generated_at': datetime.now().isoformat(),
            'balance': {
                'xnt': self.balance_xnt,
                'usd': self.balance_usd,
                'initial_xnt': self.config['initial_balance_xnt'],
                'initial_usd': self.config['initial_balance_usd'],
            },
            'performance': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl_xnt': total_pnl,
                'avg_pnl_per_trade': avg_pnl,
                'roi_pct': (total_pnl / self.config['initial_balance_xnt']) * 100,
            },
            'prediction_accuracy': {
                'directional_accuracy': directional_accuracy,
                'correct_predictions': correct_predictions,
                'incorrect_predictions': total_trades - correct_predictions,
            },
            'by_confidence': {
                'high_confidence_80+': {
                    'trades': len(high_conf),
                    'accuracy': len([t for t in high_conf if t.correct_prediction]) / len(high_conf) if high_conf else 0,
                    'avg_pnl': sum(t.pnl for t in high_conf) / len(high_conf) if high_conf else 0,
                },
                'medium_confidence_70_80': {
                    'trades': len(med_conf),
                    'accuracy': len([t for t in med_conf if t.correct_prediction]) / len(med_conf) if med_conf else 0,
                    'avg_pnl': sum(t.pnl for t in med_conf) / len(med_conf) if med_conf else 0,
                },
            },
            'by_pair': {
                pair: {
                    'trades': len(data['trades']),
                    'pnl': data['pnl'],
                    'accuracy': len([t for t in data['trades'] if t.correct_prediction]) / len(data['trades']),
                }
                for pair, data in by_pair.items()
            },
            'active_positions': len([p for p in self.positions if p.status == "open"]),
        }


def print_report(report: Dict):
    """Pretty print the accuracy report"""
    print("\n" + "="*60)
    print("PAPER TRADING ACCURACY REPORT")
    print("="*60)
    
    bal = report['balance']
    perf = report['performance']
    acc = report['prediction_accuracy']
    
    print(f"\n📊 BALANCE")
    print(f"   Current: {bal['xnt']:,.2f} vXNT (${bal['usd']:,.2f} vUSD)")
    print(f"   Initial: {bal['initial_xnt']:,.2f} vXNT (${bal['initial_usd']:,.2f} vUSD)")
    
    print(f"\n📈 PERFORMANCE")
    print(f"   Total Trades: {perf['total_trades']}")
    print(f"   Win Rate: {perf['win_rate']:.1%} ({perf['winning_trades']}W / {perf['losing_trades']}L)")
    print(f"   Total P&L: {perf['total_pnl_xnt']:+.2f} vXNT ({perf['roi_pct']:+.1f}% ROI)")
    print(f"   Avg per Trade: {perf['avg_pnl_per_trade']:+.2f} vXNT")
    
    print(f"\n🎯 PREDICTION ACCURACY")
    print(f"   Directional: {acc['directional_accuracy']:.1%}")
    print(f"   Correct: {acc['correct_predictions']} / {acc['incorrect_predictions']} incorrect")
    
    if 'by_confidence' in report:
        print(f"\n📊 BY CONFIDENCE LEVEL")
        hc = report['by_confidence']['high_confidence_80+']
        mc = report['by_confidence']['medium_confidence_70_80']
        print(f"   High (80%+): {hc['trades']} trades, {hc['accuracy']:.1%} accuracy")
        print(f"   Medium (70-80%): {mc['trades']} trades, {mc['accuracy']:.1%} accuracy")
    
    if 'by_pair' in report:
        print(f"\n🔁 BY PAIR")
        for pair, data in report['by_pair'].items():
            print(f"   {pair}: {data['trades']} trades, {data['pnl']:+.2f} vXNT, {data['accuracy']:.1%} accuracy")
    
    print(f"\n   Active Positions: {report['active_positions']}")
    print("="*60 + "\n")


async def main():
    parser = argparse.ArgumentParser(description='Paper Trading Testnet')
    parser.add_argument('--reset', action='store_true', help='Reset all balances and history')
    parser.add_argument('--report', action='store_true', help='Print accuracy report and exit')
    parser.add_argument('--run', action='store_true', help='Start trading loop')
    
    args = parser.parse_args()
    
    trader = PaperTrader()
    
    if args.reset:
        trader.reset()
        return
    
    if args.report:
        report = trader.get_report()
        print_report(report)
        return
    
    if args.run:
        try:
            await trader.start()
        except KeyboardInterrupt:
            trader.stop()
            print("\n\nTrading stopped. Use --report for final stats.")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
