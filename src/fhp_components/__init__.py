"""
FHP (Foundation's Holistic Presence) Oracle Components

Integration of augerd's FHP framework with X1 Price Oracle.

Components:
- PhaseLockConsensusEngine: Synchronization-based consensus
- TauKConfidenceScorer: Dynamic trust scoring with tiers
- CompositionalRewardsEngine: Value-based reward calculation

Author: H T Armstrong + augerd
Date: 2026-02-19
"""

from .oracle_phase_lock_consensus import PhaseLockConsensusEngine, Attestation
from .tau_k_confidence_scorer import TauKConfidenceScorer, ConfidenceTier
from .compositional_rewards_engine import CompositionalRewardsEngine, RewardSubmission

__all__ = [
    'PhaseLockConsensusEngine',
    'Attestation',
    'TauKConfidenceScorer',
    'ConfidenceTier',
    'CompositionalRewardsEngine',
    'RewardSubmission'
]
