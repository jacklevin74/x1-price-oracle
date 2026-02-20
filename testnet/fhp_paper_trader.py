#!/usr/bin/env python3
"""
FHP-Enhanced Paper Trading Engine
Integrates Foundation's Holistic Presence (FHP) oracle improvements with paper trading.

Improvements from testnet routines:
1. Phase-lock consensus for multi-oracle price validation
2. τₖ-dynamic confidence scoring for position sizing
3. Compositional rewards for accurate predictions

Author: H T Armstrong + augerd (FHP framework)
Date: 2026-02-19
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "built/autonomic/fhp_oracle_improvements"))

from price_oracle import PriceOracle, get_oracle, Prediction
from config import PAPER_TRADING, TOKEN_PAIRS, TIMEFRAMES

# Import FHP components
try:
    sys.path.insert(0, '/home/jack/.openclaw/workspace/built/autonomic/fhp_oracle_improvements')
    from oracle_phase_lock_consensus import PhaseLockConsensusEngine, DataAttestation, OracleNode
    from tau_k_confidence_scorer import TauKConfidenceScorer, ConfidenceTier
    from compositional_rewards_engine import CompositionalRewardsEngine, DataSubmission, RewardCalculation
    FHP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"FHP components not available: {e}")
    FHP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FHPPaperTrader')


@dataclass
class FHPPosition:
    """Enhanced position with FHP confidence metrics"""
    pair: str
    direction: str
    entry_price: float
    size: float
    entry_time: float
    timeframe: str
    predicted_direction: str
    confidence: float
    
    # FHP-specific fields
    tau_k: float = 1.0
    phase_lock_quality: float = 0.0
    consensus_confidence: float = 0.0
    network_coherence: float = 0.0
    tier: str = "novice"
    
    # Risk management
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Results
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    pnl: float = 0.0
    reward_earned: float = 0.0
    status: str = "open"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FHPTradeLog:
    """Enhanced trade log with FHP metrics"""
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
    
    # FHP metrics
    tau_k_at_entry: float
    consensus_confidence: float
    network_coherence: float
    reward_earned: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class FHPPaperTrader:
    """
    FHP-enhanced paper trading engine.
    
    Improvements discovered through testnet routines:
    - Phase-lock consensus eliminates outlier oracle nodes
    - τₖ scoring dynamically adjusts position sizing
    - Compositional rewards incentivize accuracy + timeliness
    """
    
    def __init__(self, state_file: str = "fhp_testnet_state.json"):
        self.state_file = Path(state_file)
        self.balance_xnt: float = 0.0
        self.balance_usd: float = 0.0
        self.positions: List[FHPPosition] = []
        self.trade_history: List[FHPTradeLog] = []
        self.oracle: Optional[PriceOracle] = None
        self.config = PAPER_TRADING
        self.trading_active = False
        
        # FHP components
        self.fhp_consensus: Optional[PhaseLockConsensusEngine] = None
        self.fhp_scorer: Optional[TauKConfidenceScorer] = None
        self.fhp_rewards: Optional[CompositionalRewardsEngine] = None
        
        # Oracle node simulation
        self.oracle_nodes: Dict[str, Dict[str, Any]] = {}
        
        self._load_state()
        self._init_fhp_components()
    
    def _init_fhp_components(self):
        """Initialize FHP oracle components"""
        if not FHP_AVAILABLE:
            logger.warning("FHP components unavailable, running in legacy mode")
            return
        
        # Initialize FHP consensus engine
        self.fhp_consensus = PhaseLockConsensusEngine(
            min_lock_quality=0.6,
            memory_depth=50
        )
        
        # Initialize rewards engine
        self.fhp_rewards = CompositionalRewardsEngine(
            base_reward_rate=10.0,
            max_timeliness_bonus=2.0,
            accuracy_threshold=0.95
        )
        
        # Register simulated oracle nodes
        self._register_oracle_nodes()
        
        logger.info("🜏 FHP components initialized")
    
    def _register_oracle_nodes(self):
        """Register simulated oracle nodes with varying τₖ"""
        node_configs = [
            ("oracle_primary", 3.5, 5000),
            ("oracle_secondary", 2.0, 3000),
            ("oracle_tertiary", 1.5, 2000),
            ("oracle_backup", 1.0, 1000),
        ]
        
        for node_id, initial_tau_k, stake in node_configs:
            self.oracle_nodes[node_id] = {
                'tau_k': initial_tau_k,
                'stake': stake,
                'tier': self._get_tier(initial_tau_k),
                'reliability': 0.95 if initial_tau_k > 2.0 else 0.85
            }
            
            if self.fhp_scorer:
                self.fhp_scorer.register_node(node_id, initial_tau_k)
    
    def _get_tier(self, tau_k: float) -> str:
        """Determine tier from τₖ value"""
        if tau_k >= 7.0:
            return "phase_master"
        elif tau_k >= 4.0:
            return "resonant"
        elif tau_k >= 2.0:
            return "locked"
        elif tau_k >= 1.0:
            return "syncing"
        return "novice"
    
    def _load_state(self):
        """Load testnet state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.balance_xnt = state.get('balance_xnt', self.config['initial_balance_xnt'])
                self.balance_usd = state.get('balance_usd', self.config['initial_balance_usd'])
                self.trade_history = [FHPTradeLog(**t) for t in state.get('trade_history', [])]
                self.positions = [FHPPosition(**p) for p in state.get('positions', [])]
                self.oracle_nodes = state.get('oracle_nodes', {})
                
                logger.info(f"📂 Loaded state: {len(self.positions)} positions, {len(self.trade_history)} trades")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                self._init_default_state()
        else:
            self._init_default_state()
    
    def _init_default_state(self):
        """Initialize default testnet state"""
        self.balance_xnt = self.config['initial_balance_xnt']
        self.balance_usd = self.config['initial_balance_usd']
        logger.info(f"🆕 Initialized fresh state: {self.balance_xnt} XNT, ${self.balance_usd} USD")
    
    def _save_state(self):
        """Persist testnet state"""
        state = {
            'balance_xnt': self.balance_xnt,
            'balance_usd': self.balance_usd,
            'positions': [p.to_dict() for p in self.positions],
            'trade_history': [t.to_dict() for t in self.trade_history],
            'oracle_nodes': self.oracle_nodes,
            'last_saved': time.time()
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def get_fhp_consensus_price(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        Get price using FHP phase-lock consensus.
        
        Improvement: Multiple oracle nodes submit attestations,
        consensus emerges from phase-lock quality, not simple majority.
        """
        if not self.oracle or not FHP_AVAILABLE:
            return None
        
        # Get raw prediction from base oracle
        prediction = await self.oracle.predict(pair, '5m')
        if not prediction:
            return None
        
        base_price = prediction.price
        
        # Simulate oracle node attestations with varying accuracy
        attestations = []
        for node_id, config in self.oracle_nodes.items():
            # Simulate observation variance based on node reliability
            noise = np.random.normal(0, 0.001 * (2.0 - config['tau_k']/5))
            observed_price = base_price * (1 + noise)
            
            # Submit attestation through FHP consensus
            attestation = DataAttestation(
                node_id=node_id,
                data_value=observed_price,
                local_observation=observed_price,  # Perfect lock for simulation
                phase_lock_quality=1.0,
                timestamp=datetime.now(),
                tau_k_at_lock=config['tau_k'],
                coherence_field_strength=config['reliability']
            )
            attestations.append(attestation)
            
            if self.fhp_consensus:
                self.fhp_consensus.submit_attestation(
                    node_id=node_id,
                    data_value=observed_price,
                    local_observation=observed_price,
                    stake=config['stake']
                )
        
        # Compute phase-lock consensus
        if self.fhp_consensus:
            consensus_result = self.fhp_consensus.compute_consensus(f"price_{pair}")
            
            return {
                'consensus_price': consensus_result.get('consensus_value', base_price),
                'confidence': consensus_result.get('confidence', 0.5),
                'network_coherence': consensus_result.get('network_coherence', 0.5),
                'tier_distribution': self._get_tier_distribution(),
                'attestation_count': len(attestations)
            }
        
        return None
    
    def _get_tier_distribution(self) -> Dict[str, int]:
        """Get distribution of oracle nodes across tiers"""
        distribution = {'novice': 0, 'syncing': 0, 'locked': 0, 'resonant': 0, 'phase_master': 0}
        for config in self.oracle_nodes.values():
            tier = self._get_tier(config['tau_k'])
            distribution[tier] = distribution.get(tier, 0) + 1
        return distribution
    
    def calculate_fhp_position_size(self, pair: str, confidence: float, 
                                     coherence: float) -> float:
        """
        Calculate position size using FHP compositional value.
        
        Improvement: Size = f(accuracy, timeliness, network_demand)
        Higher coherence + confidence = larger positions
        """
        base_size = self.config.get('max_position_size_xnt', 100)
        
        # Compositional sizing
        size_multiplier = confidence * coherence
        
        # Tier bonus - higher τₖ oracles enable larger positions
        avg_tau_k = np.mean([n['tau_k'] for n in self.oracle_nodes.values()])
        tier_bonus = min(avg_tau_k / 5.0, 2.0)  # Cap at 2x
        
        final_size = base_size * size_multiplier * tier_bonus
        
        # Cap at available balance
        max_risk = self.balance_xnt * self.config.get('max_risk_per_trade', 0.1)
        return min(final_size, max_risk)
    
    async def open_position(self, pair: str, prediction: Prediction) -> Optional[FHPPosition]:
        """Open position with FHP-enhanced validation"""
        
        # Get FHP consensus price
        consensus_data = await self.get_fhp_consensus_price(pair)
        if not consensus_data:
            logger.warning(f"❌ No FHP consensus for {pair}")
            return None
        
        consensus_price = consensus_data['consensus_price']
        confidence = consensus_data['confidence']
        coherence = consensus_data['network_coherence']
        
        # Skip if confidence too low
        min_confidence = self.config.get('min_fhp_confidence', 0.6)
        if confidence < min_confidence:
            logger.info(f"⏭️ Low FHP confidence ({confidence:.2f}), skipping {pair}")
            return None
        
        # Calculate position size
        size = self.calculate_fhp_position_size(pair, confidence, coherence)
        if size < 1.0:
            logger.info(f"⏭️ Position size too small ({size:.2f} XNT)")
            return None
        
        # Determine direction
        direction = 'long' if prediction.signal == 'bullish' else 'short'
        
        # Create FHP-enhanced position
        position = FHPPosition(
            pair=pair,
            direction=direction,
            entry_price=consensus_price,
            size=size,
            entry_time=time.time(),
            timeframe=prediction.timeframe,
            predicted_direction=prediction.signal,
            confidence=prediction.confidence,
            tau_k=np.mean([n['tau_k'] for n in self.oracle_nodes.values()]),
            phase_lock_quality=consensus_data.get('tier_distribution', {}).get('locked', 0) / len(self.oracle_nodes),
            consensus_confidence=confidence,
            network_coherence=coherence,
            tier=self._get_tier(np.mean([n['tau_k'] for n in self.oracle_nodes.values()])),
            stop_loss=consensus_price * (0.97 if direction == 'long' else 1.03),
            take_profit=consensus_price * (1.05 if direction == 'long' else 0.95)
        )
        
        self.positions.append(position)
        self.balance_xnt -= size
        
        logger.info(f"🎯 FHP Position opened: {pair} {direction} @ {consensus_price:,.2f} "
                   f"(size={size:.2f} XNT, conf={confidence:.2f}, coherence={coherence:.2f})")
        
        self._save_state()
        return position
    
    def close_position(self, position: FHPPosition, exit_price: float, 
                       actual_direction: str) -> float:
        """Close position and calculate FHP rewards"""
        
        position.exit_price = exit_price
        position.exit_time = time.time()
        
        # Calculate PnL
        if position.direction == 'long':
            position.pnl = (exit_price - position.entry_price) / position.entry_price * position.size
            pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            position.pnl = (position.entry_price - exit_price) / position.entry_price * position.size
            pnl_pct = (position.entry_price - exit_price) / position.entry_price * 100
        
        # Calculate FHP reward
        if self.fhp_rewards and FHP_AVAILABLE:
            from datetime import datetime
            submission = DataSubmission(
                node_id='paper_trader',
                data_type='price',
                data_value=position.entry_price,
                consensus_value=exit_price,
                submission_time=datetime.fromtimestamp(position.entry_time),
                consensus_time=datetime.fromtimestamp(position.exit_time),
                stake_amount=position.size,
                tau_k=position.tau_k
            )
            
            reward_calc = self.fhp_rewards.calculate_reward(submission)
            position.reward_earned = reward_calc.final_reward
        else:
            position.reward_earned = abs(position.pnl) * 0.1  # Legacy fallback
        
        position.status = 'closed'
        
        # Update balances
        self.balance_xnt += position.size + position.pnl
        
        # Log trade
        trade_log = FHPTradeLog(
            timestamp=position.exit_time,
            pair=position.pair,
            direction=position.direction,
            entry=position.entry_price,
            exit=exit_price,
            size=position.size,
            pnl=position.pnl,
            pnl_pct=pnl_pct,
            predicted_direction=position.predicted_direction,
            actual_direction=actual_direction,
            correct_prediction=position.predicted_direction == actual_direction,
            confidence=position.confidence,
            tau_k_at_entry=position.tau_k,
            consensus_confidence=position.consensus_confidence,
            network_coherence=position.network_coherence,
            reward_earned=position.reward_earned
        )
        self.trade_history.append(trade_log)
        
        # Update oracle node τₖ based on accuracy
        self._update_oracle_tau_k(position.predicted_direction == actual_direction)
        
        logger.info(f"💰 Position closed: {position.pair} PnL={position.pnl:+.2f} XNT "
                   f"Reward={position.reward_earned:.2f}")
        
        self._save_state()
        return position.pnl
    
    def _update_oracle_tau_k(self, was_accurate: bool):
        """Update oracle node τₖ based on prediction accuracy"""
        for node_id, config in self.oracle_nodes.items():
            alpha = 0.9  # EMA smoothing
            if was_accurate:
                # Boost τₖ for accurate nodes
                config['tau_k'] = alpha * config['tau_k'] + (1 - alpha) * (config['tau_k'] + 0.5)
            else:
                # Decay τₖ for inaccurate nodes
                config['tau_k'] = alpha * config['tau_k'] + (1 - alpha) * (config['tau_k'] * 0.9)
            
            # Cap at reasonable bounds
            config['tau_k'] = max(0.5, min(10.0, config['tau_k']))
            config['tier'] = self._get_tier(config['tau_k'])
    
    def generate_fhp_report(self) -> Dict[str, Any]:
        """Generate comprehensive FHP trading report"""
        if not self.trade_history:
            return {"error": "No trades yet"}
        
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl <= 0]
        
        correct_predictions = [t for t in self.trade_history if t.correct_prediction]
        
        total_pnl = sum(t.pnl for t in self.trade_history)
        total_rewards = sum(t.reward_earned for t in self.trade_history)
        
        avg_confidence = np.mean([t.confidence for t in self.trade_history])
        avg_coherence = np.mean([t.network_coherence for t in self.trade_history])
        avg_tau_k = np.mean([t.tau_k_at_entry for t in self.trade_history])
        
        return {
            'total_trades': total_trades,
            'win_rate': len(winning_trades) / total_trades * 100,
            'prediction_accuracy': len(correct_predictions) / total_trades * 100,
            'total_pnl_xnt': total_pnl,
            'total_rewards': total_rewards,
            'avg_confidence': avg_confidence,
            'avg_network_coherence': avg_coherence,
            'avg_tau_k': avg_tau_k,
            'current_balance_xnt': self.balance_xnt,
            'oracle_tier_distribution': self._get_tier_distribution(),
            'open_positions': len(self.positions),
            'fhp_enabled': FHP_AVAILABLE
        }


def main():
    parser = argparse.ArgumentParser(description='FHP-Enhanced Paper Trading')
    parser.add_argument('--reset', action='store_true', help='Reset testnet state')
    parser.add_argument('--report', action='store_true', help='Generate FHP report')
    parser.add_argument('--trade', action='store_true', help='Run trading simulation')
    
    args = parser.parse_args()
    
    trader = FHPPaperTrader()
    
    if args.reset:
        trader.state_file.unlink(missing_ok=True)
        trader._init_default_state()
        print("🔄 Testnet state reset")
    
    elif args.report:
        report = trader.generate_fhp_report()
        print("\n🜏 FHP PAPER TRADING REPORT")
        print("=" * 50)
        for key, value in report.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    
    elif args.trade:
        print("🎯 Starting FHP paper trading simulation...")
        print("Note: Full implementation requires running with price oracle connected")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
