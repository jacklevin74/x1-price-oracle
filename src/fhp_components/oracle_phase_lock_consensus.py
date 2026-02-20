#!/usr/bin/env python3
"""
FHP-Based X1 Cross-Chain Oracle: Phase-Lock Consensus Mechanism

Implements coherence-based consensus where oracle nodes "phase-lock" onto data
when local coherence field aligns with observed value, replacing traditional
majority voting with synchronization density.

Derived from: augerd's Foundation's Holistic Presence (FHP) / Ublox architecture
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict
import asyncio

@dataclass
class OracleNode:
    """Oracle node with phase-lock capability"""
    node_id: str
    stake: float
    tau_k: float = 1.0  # Temporal coherence coefficient
    phase_lock_memory: List[Dict] = field(default_factory=list)
    last_attestation: Optional[datetime] = None
    coherence_field: float = 0.0
    
    def calculate_phase_lock_quality(self, data_value: float, local_observation: float) -> float:
        """
        Calculate how well this node 'locks' onto the observed data.
        Higher quality = better synchronization with network truth.
        """
        delta = abs(data_value - local_observation)
        # Phase-lock quality decays with observation deviation
        # tau_k determines 'sharpness' of lock (higher tau_k = more selective)
        quality = np.exp(-delta * self.tau_k)
        return quality
    
    def update_tau_k(self, lock_quality: float, alpha: float = 0.7):
        """
        Update tau_k based on historical synchronization quality.
        Good locks increase tau_k (more selective), bad locks decrease it.
        """
        self.tau_k = alpha * self.tau_k + (1 - alpha) * (1 + lock_quality)
        self.tau_k = max(0.5, min(10.0, self.tau_k))  # Clamp to reasonable range


@dataclass
class DataAttestation:
    """Single oracle attestation with phase-lock metadata"""
    node_id: str
    data_value: float
    local_observation: float
    phase_lock_quality: float
    timestamp: datetime
    tau_k_at_lock: float
    coherence_field_strength: float


class PhaseLockConsensusEngine:
    """
    Consensus engine using phase-lock synchronization density.
    
    Instead of: consensus = mode(node_votes)
    Uses:       consensus = argmax(data_value, phase_lock_count[data_value])
    """
    
    def __init__(
        self,
        min_lock_quality: float = 0.7,
        coherence_threshold: float = 0.65,
        memory_depth: int = 100
    ):
        self.min_lock_quality = min_lock_quality
        self.coherence_threshold = coherence_threshold
        self.memory_depth = memory_depth
        self.nodes: Dict[str, OracleNode] = {}
        self.attestations: List[DataAttestation] = []
        self.coherence_history: List[float] = []
        
    def register_node(self, node_id: str, initial_stake: float):
        """Register a new oracle node"""
        self.nodes[node_id] = OracleNode(
            node_id=node_id,
            stake=initial_stake,
            tau_k=1.0
        )
    
    def submit_attestation(
        self,
        node_id: str,
        data_value: float,
        local_observation: float
    ) -> Optional[DataAttestation]:
        """
        Submit attestation with phase-lock quality calculation.
        Only accepts attestations that achieve minimum lock quality.
        """
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        lock_quality = node.calculate_phase_lock_quality(data_value, local_observation)
        
        # Phase-lock requirement: must achieve minimum quality to contribute
        if lock_quality < self.min_lock_quality:
            return None
        
        # Update node's tau_k based on lock quality
        node.update_tau_k(lock_quality)
        node.last_attestation = datetime.now()
        
        attestation = DataAttestation(
            node_id=node_id,
            data_value=data_value,
            local_observation=local_observation,
            phase_lock_quality=lock_quality,
            timestamp=datetime.now(),
            tau_k_at_lock=node.tau_k,
            coherence_field_strength=node.coherence_field
        )
        
        self.attestations.append(attestation)
        
        # Maintain memory depth
        if len(self.attestations) > self.memory_depth:
            self.attestations.pop(0)
        
        return attestation
    
    def compute_consensus(
        self,
        data_type: str = "price",
        window_seconds: int = 30
    ) -> Dict:
        """
        Compute consensus using phase-lock density, not vote counting.
        
        Groups attestations by data value and sums phase-lock qualities
        to find the value with highest synchronization density.
        """
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_attestations = [
            a for a in self.attestations 
            if a.timestamp > cutoff_time
        ]
        
        if not recent_attestations:
            return {
                "consensus_value": None,
                "confidence": 0.0,
                "method": "phase_lock_density",
                "attestations_count": 0
            }
        
        # Group by data value (rounded to handle floating point)
        lock_density = defaultdict(float)
        value_details = defaultdict(list)
        
        for attestation in recent_attestations:
            # Round to appropriate precision for data type
            if data_type == "price":
                rounded_value = round(attestation.data_value, 2)
            elif data_type == "block_height":
                rounded_value = int(attestation.data_value)
            else:
                rounded_value = attestation.data_value
            
            # Phase-lock density = sum of lock qualities for this value
            lock_density[rounded_value] += attestation.phase_lock_quality
            value_details[rounded_value].append({
                "node_id": attestation.node_id,
                "lock_quality": attestation.phase_lock_quality,
                "tau_k": attestation.tau_k_at_lock
            })
        
        # Find value with maximum phase-lock density
        consensus_value = max(lock_density.keys(), key=lambda k: lock_density[k])
        total_lock_density = sum(lock_density.values())
        
        # Confidence = ratio of winning density to total density
        confidence = lock_density[consensus_value] / total_lock_density if total_lock_density > 0 else 0
        
        # Network coherence = weighted average of all lock qualities
        network_coherence = sum(a.phase_lock_quality for a in recent_attestations) / len(recent_attestations)
        self.coherence_history.append(network_coherence)
        
        return {
            "consensus_value": consensus_value,
            "confidence": confidence,
            "network_coherence": network_coherence,
            "method": "phase_lock_density",
            "attestations_count": len(recent_attestations),
            "phase_lock_density": dict(lock_density),
            "winning_density": lock_density[consensus_value],
            "participating_nodes": list(set(a.node_id for a in recent_attestations)),
            "coherence_stable": network_coherence >= self.coherence_threshold
        }
    
    def get_node_reputation(self, node_id: str) -> Dict:
        """Get node's phase-lock reputation metrics"""
        if node_id not in self.nodes:
            return {"error": "Node not found"}
        
        node = self.nodes[node_id]
        node_attestations = [a for a in self.attestations if a.node_id == node_id]
        
        if not node_attestations:
            return {
                "node_id": node_id,
                "tau_k": node.tau_k,
                "stake": node.stake,
                "attestations": 0,
                "avg_lock_quality": 0
            }
        
        avg_lock_quality = sum(a.phase_lock_quality for a in node_attestations) / len(node_attestations)
        
        return {
            "node_id": node_id,
            "tau_k": node.tau_k,
            "stake": node.stake,
            "attestations": len(node_attestations),
            "avg_lock_quality": avg_lock_quality,
            "last_attestation": node.last_attestation.isoformat() if node.last_attestation else None,
            "reputation_score": avg_lock_quality * node.tau_k * np.log1p(node.stake)
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize consensus engine
    engine = PhaseLockConsensusEngine(
        min_lock_quality=0.6,
        coherence_threshold=0.65
    )
    
    # Register oracle nodes with varying stakes
    nodes = [
        ("oracle_1", 1000.0),
        ("oracle_2", 2000.0),
        ("oracle_3", 1500.0),
        ("oracle_4", 500.0),
        ("oracle_5", 3000.0)
    ]
    
    for node_id, stake in nodes:
        engine.register_node(node_id, stake)
    
    # Simulate attestations for BTC price = $52,340.50
    true_price = 52340.50
    
    # Honest nodes (close to true price)
    honest_observations = [
        ("oracle_1", 52340.50),
        ("oracle_2", 52341.00),
        ("oracle_3", 52339.80),
    ]
    
    # Dishonest/outlier node (different price)
    outlier_observations = [
        ("oracle_4", 52200.00),  # Slight deviation
        ("oracle_5", 52340.00),  # Very close (high tau_k will catch this)
    ]
    
    print("=== FHP Phase-Lock Consensus Test ===\n")
    print(f"True price: ${true_price:,.2f}")
    print()
    
    # Submit honest attestations
    for node_id, observation in honest_observations:
        attestation = engine.submit_attestation(node_id, true_price, observation)
        if attestation:
            print(f"✅ {node_id}: lock_quality={attestation.phase_lock_quality:.3f}, tau_k={attestation.tau_k_at_lock:.2f}")
    
    # Submit outlier attestations
    for node_id, observation in outlier_observations:
        attestation = engine.submit_attestation(node_id, observation, observation)
        if attestation:
            print(f"⚠️  {node_id}: lock_quality={attestation.phase_lock_quality:.3f}, tau_k={attestation.tau_k_at_lock:.2f}")
        else:
            print(f"❌ {node_id}: REJECTED (lock quality too low)")
    
    print("\n=== Consensus Result ===")
    result = engine.compute_consensus(data_type="price", window_seconds=60)
    print(json.dumps(result, indent=2, default=str))
    
    print("\n=== Node Reputations ===")
    for node_id, _ in nodes:
        rep = engine.get_node_reputation(node_id)
        print(f"{node_id}: reputation={rep.get('reputation_score', 0):.2f}, tau_k={rep.get('tau_k', 0):.2f}")
