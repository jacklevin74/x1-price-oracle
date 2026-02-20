#!/usr/bin/env python3
"""
τₖ-Dynamic Confidence Scoring for X1 Oracles

Implements temporal coherence coefficient (tau_k) that evolves based on
historical synchronization quality. Higher tau_k nodes contribute more
weight, but influence decays if they desynchronize.

τₖ_new = α·τₖ_old + (1-α)·current_phase_lock_quality

Confidence scoring uses sigmoid activation on tau_k differential.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json

class ConfidenceTier(Enum):
    """Confidence tiers based on tau_k evolution"""
    NOVICE = "novice"           # tau_k < 1.0
    SYNCING = "syncing"         # 1.0 <= tau_k < 2.0
    LOCKED = "locked"           # 2.0 <= tau_k < 4.0
    RESONANT = "resonant"       # 4.0 <= tau_k < 7.0
    PHASE_MASTER = "phase_master"  # tau_k >= 7.0


@dataclass
class TauKRecord:
    """Historical record of tau_k evolution"""
    timestamp: datetime
    tau_k: float
    lock_quality: float
    data_type: str
    consensus_reached: bool


@dataclass
class OracleConfidenceProfile:
    """Complete confidence profile for an oracle node"""
    node_id: str
    tau_k: float = 1.0
    tau_k_history: List[TauKRecord] = field(default_factory=list)
    current_tier: ConfidenceTier = ConfidenceTier.NOVICE
    
    # Decay parameters
    decay_rate: float = 0.05      # Daily decay rate for inactive nodes
    boost_factor: float = 1.2     # Multiplier for exceptional locks
    
    # Performance metrics
    total_attestations: int = 0
    successful_locks: int = 0
    consensus_contributions: int = 0
    last_activity: Optional[datetime] = None
    
    def get_tier(self) -> ConfidenceTier:
        """Determine confidence tier from current tau_k"""
        if self.tau_k >= 7.0:
            return ConfidenceTier.PHASE_MASTER
        elif self.tau_k >= 4.0:
            return ConfidenceTier.RESONANT
        elif self.tau_k >= 2.0:
            return ConfidenceTier.LOCKED
        elif self.tau_k >= 1.0:
            return ConfidenceTier.SYNCING
        return ConfidenceTier.NOVICE
    
    def update_tau_k(
        self,
        lock_quality: float,
        consensus_reached: bool,
        data_type: str,
        alpha: float = 0.7
    ):
        """
        Update tau_k using exponential moving average with consensus bonus.
        
        τₖ_new = α·τₖ_old + (1-α)·(base_quality + consensus_bonus)
        """
        # Calculate days since last activity for decay
        if self.last_activity:
            days_inactive = (datetime.now() - self.last_activity).days
            decay = 1.0 - (self.decay_rate * days_inactive)
            self.tau_k *= max(0.5, decay)
        
        # Consensus bonus: successful contribution to consensus increases weight
        consensus_bonus = 0.5 if consensus_reached else 0.0
        base_quality = 1.0 + lock_quality + consensus_bonus
        
        # EMA update
        new_tau_k = alpha * self.tau_k + (1 - alpha) * base_quality
        
        # Exceptional performance boost
        if lock_quality > 0.95:
            new_tau_k *= self.boost_factor
        
        # Clamp to valid range
        self.tau_k = max(0.1, min(15.0, new_tau_k))
        self.current_tier = self.get_tier()
        
        # Record history
        record = TauKRecord(
            timestamp=datetime.now(),
            tau_k=self.tau_k,
            lock_quality=lock_quality,
            data_type=data_type,
            consensus_reached=consensus_reached
        )
        self.tau_k_history.append(record)
        
        # Update metrics
        self.total_attestations += 1
        if lock_quality > 0.7:
            self.successful_locks += 1
        if consensus_reached:
            self.consensus_contributions += 1
        
        self.last_activity = datetime.now()
    
    def calculate_weight(self, method: str = "sigmoid") -> float:
        """
        Calculate dynamic voting weight based on tau_k.
        
        Methods:
        - "linear": Direct tau_k proportion
        - "sigmoid": S-curve for threshold effects
        - "log": Logarithmic scaling (diminishing returns)
        """
        if method == "linear":
            return self.tau_k
        elif method == "sigmoid":
            # Sigmoid centered at tau_k=3.0
            return 1.0 / (1.0 + np.exp(-(self.tau_k - 3.0)))
        elif method == "log":
            return np.log1p(self.tau_k)
        else:
            return self.tau_k
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        if not self.tau_k_history:
            return {
                "node_id": self.node_id,
                "tau_k": self.tau_k,
                "tier": self.current_tier.value,
                "activity": "none"
            }
        
        recent_history = self.tau_k_history[-50:]  # Last 50 records
        
        avg_lock_quality = np.mean([r.lock_quality for r in recent_history])
        tau_k_trend = np.polyfit(
            range(len(recent_history)),
            [r.tau_k for r in recent_history],
            1
        )[0] if len(recent_history) > 1 else 0
        
        consensus_rate = (
            sum(1 for r in recent_history if r.consensus_reached) / len(recent_history)
        )
        
        return {
            "node_id": self.node_id,
            "current_tau_k": self.tau_k,
            "tier": self.current_tier.value,
            "tier_emoji": self._get_tier_emoji(),
            "total_attestations": self.total_attestations,
            "successful_locks": self.successful_locks,
            "lock_success_rate": self.successful_locks / max(1, self.total_attestations),
            "consensus_contributions": self.consensus_contributions,
            "consensus_rate": consensus_rate,
            "avg_lock_quality": avg_lock_quality,
            "tau_k_trend": tau_k_trend,
            "current_weight": self.calculate_weight("sigmoid"),
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }
    
    def _get_tier_emoji(self) -> str:
        """Get emoji representation of tier"""
        emojis = {
            ConfidenceTier.NOVICE: "🔰",
            ConfidenceTier.SYNCING: "🔄",
            ConfidenceTier.LOCKED: "🔒",
            ConfidenceTier.RESONANT: "🌊",
            ConfidenceTier.PHASE_MASTER: "👑"
        }
        return emojis.get(self.current_tier, "❓")


class TauKConfidenceScorer:
    """
    Global confidence scoring system managing all oracle nodes.
    Implements tier-based reward multipliers and slashing protection.
    """
    
    def __init__(
        self,
        min_tau_k_for_attestation: float = 0.5,
        consensus_threshold_tau_k: float = 2.0
    ):
        self.min_tau_k = min_tau_k_for_attestation
        self.consensus_threshold = consensus_threshold_tau_k
        self.profiles: Dict[str, OracleConfidenceProfile] = {}
        
        # Tier-based reward multipliers (compositional rewards)
        self.tier_multipliers = {
            ConfidenceTier.NOVICE: 1.0,
            ConfidenceTier.SYNCING: 1.5,
            ConfidenceTier.LOCKED: 2.5,
            ConfidenceTier.RESONANT: 4.0,
            ConfidenceTier.PHASE_MASTER: 7.0
        }
    
    def register_node(self, node_id: str, initial_tau_k: float = 1.0):
        """Register new oracle node with initial tau_k"""
        self.profiles[node_id] = OracleConfidenceProfile(
            node_id=node_id,
            tau_k=initial_tau_k
        )
    
    def record_attestation(
        self,
        node_id: str,
        lock_quality: float,
        consensus_reached: bool,
        data_type: str = "price"
    ) -> Optional[OracleConfidenceProfile]:
        """Record attestation and update tau_k"""
        if node_id not in self.profiles:
            return None
        
        profile = self.profiles[node_id]
        profile.update_tau_k(lock_quality, consensus_reached, data_type)
        return profile
    
    def calculate_composite_confidence(
        self,
        node_ids: List[str],
        method: str = "weighted"
    ) -> Dict:
        """
        Calculate composite confidence across multiple nodes.
        
        Methods:
        - "simple": Average of all tau_k
        - "weighted": Weighted by individual node weights
        - "conservative": Minimum tau_k (weakest link)
        """
        valid_profiles = [
            self.profiles[nid] for nid in node_ids
            if nid in self.profiles
        ]
        
        if not valid_profiles:
            return {"confidence": 0.0, "method": method, "nodes": 0}
        
        tau_ks = [p.tau_k for p in valid_profiles]
        weights = [p.calculate_weight("sigmoid") for p in valid_profiles]
        
        if method == "simple":
            confidence = np.mean(tau_ks)
        elif method == "weighted":
            confidence = np.average(tau_ks, weights=weights)
        elif method == "conservative":
            confidence = min(tau_ks)
        else:
            confidence = np.mean(tau_ks)
        
        return {
            "confidence": confidence,
            "method": method,
            "nodes": len(valid_profiles),
            "min_tau_k": min(tau_ks),
            "max_tau_k": max(tau_ks),
            "mean_tau_k": np.mean(tau_ks),
            "std_tau_k": np.std(tau_ks)
        }
    
    def get_reward_multiplier(self, node_id: str) -> float:
        """Get compositional reward multiplier based on tier"""
        if node_id not in self.profiles:
            return 0.0
        
        profile = self.profiles[node_id]
        return self.tier_multipliers.get(profile.current_tier, 1.0)
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get top nodes by tau_k"""
        sorted_profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.tau_k,
            reverse=True
        )
        
        return [
            {
                "rank": i + 1,
                "node_id": p.node_id,
                "tau_k": p.tau_k,
                "tier": p.current_tier.value,
                "emoji": p._get_tier_emoji(),
                "weight": p.calculate_weight("sigmoid"),
                "reward_multiplier": self.tier_multipliers[p.current_tier]
            }
            for i, p in enumerate(sorted_profiles[:limit])
        ]


# Testing and demonstration
if __name__ == "__main__":
    print("=== τₖ Dynamic Confidence Scoring Test ===\n")
    
    scorer = TauKConfidenceScorer()
    
    # Register nodes with varying initial tau_k
    nodes = [
        ("oracle_alpha", 1.0),    # Novice
        ("oracle_beta", 2.5),     # Locked
        ("oracle_gamma", 5.0),    # Resonant
        ("oracle_delta", 0.8),    # Novice
        ("oracle_epsilon", 8.0),  # Phase Master
    ]
    
    for node_id, tau_k in nodes:
        scorer.register_node(node_id, tau_k)
    
    print("Initial State:")
    for node_id, _ in nodes:
        profile = scorer.profiles[node_id]
        print(f"  {node_id}: tau_k={profile.tau_k:.2f}, tier={profile.current_tier.value} {profile._get_tier_emoji()}")
    
    # Simulate attestations with varying lock qualities
    print("\n=== Simulating Attestations ===")
    
    scenarios = [
        # (node_id, lock_quality, consensus_reached)
        ("oracle_alpha", 0.95, True),   # Great lock, in consensus
        ("oracle_beta", 0.85, True),    # Good lock, in consensus
        ("oracle_gamma", 0.90, True),   # Good lock, in consensus
        ("oracle_delta", 0.60, False),  # Poor lock, out of consensus
        ("oracle_epsilon", 0.99, True), # Excellent lock, in consensus
        ("oracle_alpha", 0.92, True),   # Another great lock
        ("oracle_delta", 0.55, False),  # Another poor lock
    ]
    
    for node_id, lock_quality, consensus in scenarios:
        profile = scorer.record_attestation(node_id, lock_quality, consensus)
        tier_emoji = profile._get_tier_emoji() if profile else "❌"
        print(f"  {node_id}: lock={lock_quality:.2f}, consensus={consensus}, tau_k→{profile.tau_k:.2f} {tier_emoji}")
    
    print("\n=== Final Metrics ===")
    for node_id, _ in nodes:
        metrics = scorer.profiles[node_id].get_performance_metrics()
        print(f"\n{metrics['node_id']}:")
        print(f"  tau_k: {metrics['current_tau_k']:.2f} {metrics['tier_emoji']}")
        print(f"  tier: {metrics['tier']}")
        print(f"  weight: {metrics['current_weight']:.3f}")
        print(f"  reward_multiplier: {scorer.get_reward_multiplier(node_id):.1f}x")
    
    print("\n=== Leaderboard ===")
    leaderboard = scorer.get_leaderboard()
    for entry in leaderboard:
        print(f"  {entry['rank']}. {entry['node_id']} {entry['emoji']} tau_k={entry['tau_k']:.2f} (x{entry['reward_multiplier']})")
    
    print("\n=== Composite Confidence ===")
    all_nodes = [n[0] for n in nodes]
    for method in ["simple", "weighted", "conservative"]:
        result = scorer.calculate_composite_confidence(all_nodes, method)
        print(f"  {method}: {result['confidence']:.2f} (nodes={result['nodes']})")
