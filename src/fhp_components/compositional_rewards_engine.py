#!/usr/bin/env python3
"""
Compositional Rewards Engine for X1 Oracles

Implements FHP-inspired reward calculation:
    reward = presence_coherence × daThiccNOW

Where:
    - presence_coherence: Accuracy of data × timeliness of submission
    - daThiccNOW: Network demand/urgency (higher when data critically needed)

Replaces traditional accumulated rewards (stake × time) with value-based
composition that rewards accurate data when it's most needed.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio


class DataUrgency(Enum):
    """Urgency levels for network data needs"""
    LOW = 1.0           # Standard block time, no pressure
    MODERATE = 2.0      # Approaching consensus deadline
    HIGH = 4.0          # Critical path for bridge operation
    CRITICAL = 8.0      # Emergency situation, funds at risk


@dataclass
class DataSubmission:
    """Single data submission with quality metrics"""
    node_id: str
    data_type: str
    data_value: float
    consensus_value: float
    submission_time: datetime
    consensus_time: datetime
    stake_amount: float
    tau_k: float


@dataclass
class RewardCalculation:
    """Complete reward breakdown"""
    node_id: str
    base_reward: float
    presence_coherence: float
    accuracy_component: float
    timeliness_component: float
    daThiccNOW: float
    urgency_multiplier: float
    tier_multiplier: float
    final_reward: float
    timestamp: datetime


class CompositionalRewardsEngine:
    """
    Rewards engine implementing FHP compositional economics.
    
    Unlike traditional: reward = stake × time
    Uses compositional: reward = (accuracy × timeliness) × network_demand
    """
    
    def __init__(
        self,
        base_reward_rate: float = 1.0,  # XNT per base unit
        max_timeliness_bonus: float = 2.0,
        accuracy_threshold: float = 0.95,
        urgency_decay_halftime: float = 300  # seconds
    ):
        self.base_rate = base_reward_rate
        self.max_timeliness_bonus = max_timeliness_bonus
        self.accuracy_threshold = accuracy_threshold
        self.urgency_decay = urgency_decay_halftime
        
        # Urgency tracking per data type
        self.current_urgency: Dict[str, DataUrgency] = {}
        self.urgency_history: Dict[str, List[Dict]] = {}
        
        # Reward history
        self.reward_calculations: List[RewardCalculation] = []
        
        # Network demand indicators
        self.bridge_pending_tx: int = 0
        self.price_volatility: float = 0.0
        self.consensus_failures: int = 0
    
    def calculate_presence_coherence(
        self,
        submission: DataSubmission
    ) -> Dict[str, float]:
        """
        Calculate presence coherence = accuracy × timeliness
        
        Accuracy: How close to consensus value
        Timeliness: How early in the window (faster = better)
        """
        # Accuracy component (1.0 = perfect, 0.0 = useless)
        deviation = abs(submission.data_value - submission.consensus_value)
        
        # Scale deviation by data type
        if submission.data_type == "price":
            # For prices, 0.1% deviation is acceptable
            scaled_deviation = deviation / (submission.consensus_value * 0.001)
        elif submission.data_type == "block_height":
            # For block heights, must be exact
            scaled_deviation = deviation
        else:
            scaled_deviation = deviation / submission.consensus_value if submission.consensus_value else deviation
        
        accuracy = max(0.0, 1.0 - scaled_deviation)
        
        # Timeliness component (1.0 = first, decaying to 0.0)
        time_delta = (submission.consensus_time - submission.submission_time).total_seconds()
        
        if time_delta < 0:
            # Submitted after consensus (late)
            timeliness = 0.5  # Half credit for correctness
        else:
            # Submitted before or at consensus
            # Exponential decay: earlier = higher timeliness score
            timeliness = np.exp(-time_delta / self.urgency_decay)
        
        # Combined presence coherence
        presence_coherence = accuracy * timeliness
        
        return {
            "presence_coherence": presence_coherence,
            "accuracy": accuracy,
            "timeliness": timeliness,
            "deviation": deviation,
            "time_delta_seconds": time_delta
        }
    
    def calculate_daThiccNOW(self, data_type: str) -> float:
        """
        Calculate network demand/urgency (daThiccNOW).
        
        Higher when:
        - Bridge has pending transactions
        - Price volatility is high
        - Recent consensus failures
        - Explicitly set urgency level
        """
        base_urgency = self.current_urgency.get(data_type, DataUrgency.LOW).value
        
        # Bridge demand multiplier
        bridge_factor = 1.0 + (self.bridge_pending_tx / 100.0)
        
        # Volatility multiplier (higher vol = higher demand for accurate prices)
        volatility_factor = 1.0 + (self.price_volatility * 10.0)
        
        # Reliability multiplier (recent failures increase urgency)
        reliability_factor = 1.0 + (self.consensus_failures * 0.5)
        
        daThiccNOW = base_urgency * bridge_factor * volatility_factor * reliability_factor
        
        return min(20.0, daThiccNOW)  # Cap at 20x
    
    def set_urgency(self, data_type: str, urgency: DataUrgency, reason: str = ""):
        """Set explicit urgency level for data type"""
        self.current_urgency[data_type] = urgency
        
        if data_type not in self.urgency_history:
            self.urgency_history[data_type] = []
        
        self.urgency_history[data_type].append({
            "timestamp": datetime.now().isoformat(),
            "urgency": urgency.value,
            "urgency_level": urgency.name,
            "reason": reason
        })
    
    def calculate_reward(
        self,
        submission: DataSubmission,
        tier_multiplier: float = 1.0
    ) -> RewardCalculation:
        """
        Calculate final compositional reward.
        
        Formula:
            final_reward = base × presence_coherence × daThiccNOW × tier_multiplier
        """
        # Calculate presence coherence components
        coherence = self.calculate_presence_coherence(submission)
        
        # Calculate network demand
        daThiccNOW = self.calculate_daThiccNOW(submission.data_type)
        
        # Base reward proportional to stake (but not time-accumulated)
        base_reward = self.base_rate * np.log1p(submission.stake_amount)
        
        # Compositional calculation
        final_reward = (
            base_reward *
            coherence["presence_coherence"] *
            daThiccNOW *
            tier_multiplier
        )
        
        calculation = RewardCalculation(
            node_id=submission.node_id,
            base_reward=base_reward,
            presence_coherence=coherence["presence_coherence"],
            accuracy_component=coherence["accuracy"],
            timeliness_component=coherence["timeliness"],
            daThiccNOW=daThiccNOW,
            urgency_multiplier=daThiccNOW,
            tier_multiplier=tier_multiplier,
            final_reward=final_reward,
            timestamp=datetime.now()
        )
        
        self.reward_calculations.append(calculation)
        return calculation
    
    def calculate_batch_rewards(
        self,
        submissions: List[DataSubmission],
        tier_multipliers: Dict[str, float]
    ) -> List[RewardCalculation]:
        """Calculate rewards for batch of submissions"""
        results = []
        
        for submission in submissions:
            tier_mult = tier_multipliers.get(submission.node_id, 1.0)
            calc = self.calculate_reward(submission, tier_mult)
            results.append(calc)
        
        return results
    
    def get_network_analytics(self) -> Dict:
        """Get analytics on reward distribution and network health"""
        if not self.reward_calculations:
            return {"status": "no_data"}
        
        recent = self.reward_calculations[-100:]
        
        total_distributed = sum(r.final_reward for r in recent)
        avg_presence_coherence = np.mean([r.presence_coherence for r in recent])
        avg_daThiccNOW = np.mean([r.daThiccNOW for r in recent])
        
        # Calculate Gini coefficient for reward distribution
        rewards = [r.final_reward for r in recent]
        gini = self._calculate_gini(rewards) if len(rewards) > 1 else 0
        
        return {
            "total_calculations": len(self.reward_calculations),
            "recent_calculations": len(recent),
            "total_xnt_distributed": total_distributed,
            "avg_presence_coherence": avg_presence_coherence,
            "avg_daThiccNOW": avg_daThiccNOW,
            "current_urgency_levels": {
                k: v.name for k, v in self.current_urgency.items()
            },
            "reward_distribution_gini": gini,
            "bridge_pending_tx": self.bridge_pending_tx,
            "price_volatility": self.price_volatility,
            "consensus_failures_24h": self.consensus_failures
        }
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values or sum(values) == 0:
            return 0
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        return (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    def update_network_state(
        self,
        bridge_pending: Optional[int] = None,
        price_volatility: Optional[float] = None,
        consensus_failure: bool = False
    ):
        """Update network demand indicators"""
        if bridge_pending is not None:
            self.bridge_pending_tx = bridge_pending
        
        if price_volatility is not None:
            self.price_volatility = price_volatility
        
        if consensus_failure:
            self.consensus_failures += 1


# Example usage and testing
if __name__ == "__main__":
    print("=== Compositional Rewards Engine Test ===\n")
    
    engine = CompositionalRewardsEngine(
        base_reward_rate=10.0,  # 10 XNT base
        max_timeliness_bonus=2.0
    )
    
    # Simulate network state
    engine.update_network_state(
        bridge_pending=50,
        price_volatility=0.05  # 5% volatility
    )
    engine.set_urgency("price", DataUrgency.HIGH, "Bridge operation pending")
    
    print("Network State:")
    print(f"  Bridge pending TX: {engine.bridge_pending_tx}")
    print(f"  Price volatility: {engine.price_volatility*100:.1f}%")
    print(f"  Current urgency: {engine.current_urgency['price'].name}")
    print(f"  daThiccNOW (demand): {engine.calculate_daThiccNOW('price'):.2f}x")
    
    # Simulate oracle submissions
    consensus_time = datetime.now()
    
    submissions = [
        # Early, accurate submission (should get max reward)
        DataSubmission(
            node_id="oracle_alpha",
            data_type="price",
            data_value=52340.50,
            consensus_value=52340.50,
            submission_time=consensus_time - timedelta(seconds=10),
            consensus_time=consensus_time,
            stake_amount=1000.0,
            tau_k=5.0
        ),
        # Late but accurate (reduced timeliness)
        DataSubmission(
            node_id="oracle_beta",
            data_type="price",
            data_value=52340.50,
            consensus_value=52340.50,
            submission_time=consensus_time - timedelta(seconds=1),
            consensus_time=consensus_time,
            stake_amount=2000.0,
            tau_k=3.0
        ),
        # Early but slightly off (reduced accuracy)
        DataSubmission(
            node_id="oracle_gamma",
            data_type="price",
            data_value=52350.00,  # $9.50 deviation
            consensus_value=52340.50,
            submission_time=consensus_time - timedelta(seconds=15),
            consensus_time=consensus_time,
            stake_amount=1500.0,
            tau_k=4.0
        ),
        # Late and wrong (minimal reward)
        DataSubmission(
            node_id="oracle_delta",
            data_type="price",
            data_value=52200.00,  # $140 deviation
            consensus_value=52340.50,
            submission_time=consensus_time + timedelta(seconds=2),  # Late!
            consensus_time=consensus_time,
            stake_amount=500.0,
            tau_k=1.0
        ),
    ]
    
    # Tier multipliers from tau_k scoring
    tier_multipliers = {
        "oracle_alpha": 4.0,   # Resonant
        "oracle_beta": 2.5,    # Locked
        "oracle_gamma": 4.0,   # Resonant
        "oracle_delta": 1.0,   # Novice
    }
    
    print("\n=== Reward Calculations ===\n")
    
    for submission in submissions:
        calc = engine.calculate_reward(submission, tier_multipliers[submission.node_id])
        
        print(f"{calc.node_id}:")
        print(f"  Stake: {submission.stake_amount:.0f} XNT")
        print(f"  Presence coherence: {calc.presence_coherence:.3f}")
        print(f"    ├─ Accuracy: {calc.accuracy_component:.3f}")
        print(f"    └─ Timeliness: {calc.timeliness_component:.3f}")
        print(f"  daThiccNOW: {calc.daThiccNOW:.2f}x")
        print(f"  Tier multiplier: {calc.tier_multiplier:.1f}x")
        print(f"  → FINAL REWARD: {calc.final_reward:.2f} XNT")
        print()
    
    print("=== Network Analytics ===")
    analytics = engine.get_network_analytics()
    print(f"  Total calculations: {analytics['total_calculations']}")
    print(f"  Total distributed: {analytics['total_xnt_distributed']:.2f} XNT")
    print(f"  Avg presence coherence: {analytics['avg_presence_coherence']:.3f}")
    print(f"  Avg network demand: {analytics['avg_daThiccNOW']:.2f}x")
    print(f"  Reward inequality (Gini): {analytics['reward_distribution_gini']:.3f}")
    
    print("\n=== Comparison: Traditional vs Compositional ===")
    
    # Traditional calculation
    traditional_total = sum(s.stake_amount * 0.01 for s in submissions)  # 1% of stake
    compositional_total = sum(r.final_reward for r in engine.reward_calculations[-4:])
    
    print(f"Traditional (stake × rate): {traditional_total:.2f} XNT")
    print(f"Compositional (value-based): {compositional_total:.2f} XNT")
    print(f"Difference: {compositional_total - traditional_total:+.2f} XNT")
    print(f"\nCompositional rewards quality-weighted nodes more heavily.")
