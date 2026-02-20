# FHP Integration Guide

## Foundation's Holistic Presence (FHP) + X1 Price Oracle

### Overview

This integration brings augerd's FHP framework to the X1 Price Oracle, enabling:

1. **Phase-Lock Consensus** — Multi-oracle price validation via synchronization
2. **τₖ-Dynamic Scoring** — Self-correcting trust model with tiered rewards
3. **Compositional Economics** — Value-based rewards (accuracy × timeliness × demand)

### Quick Start

```python
from src.fhp_components import PhaseLockConsensusEngine, TauKConfidenceScorer

# Initialize FHP consensus
consensus = PhaseLockConsensusEngine(
    min_lock_quality=0.6,
    memory_depth=50
)

# Submit oracle attestations
consensus.submit_attestation(
    node_id="oracle_1",
    data_value=52340.50,
    local_observation=52340.50,
    stake=5000
)

# Compute consensus
result = consensus.compute_consensus("BTC_XNT")
print(f"Consensus: ${result['consensus_value']:,.2f}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Paper Trading with FHP

```bash
cd testnet
python3 fhp_paper_trader.py --trade
python3 fhp_paper_trader.py --report
```

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Price Oracle   │────▶│  FHP Consensus   │────▶│  Position Sizing│
│  (Base Price)   │     │  (Phase-Lock)    │     │  (τₖ-weighted)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
   Raw Predictions         Quality Filter           Risk Management
   (Multiple Nodes)      (Consensus Value)        (Dynamic Size)
```

### Improvements from Paper Trade Routines

| Issue Discovered | FHP Solution |
|:-----------------|:-------------|
| Outlier oracle nodes | Phase-lock quality filter |
| Static position sizes | τₖ-weighted dynamic sizing |
| No reward for accuracy | Compositional reward engine |
| Sybil attacks | Stake-weighted attestations |
| Network stress | Coherence-based throttling |

### Credits

- **augerd** (@devoted_dev) — FHP architecture, coherence framework
- **H T Armstrong** — Oracle integration, paper trading routines
- **Cyberdyne Unlimited** — Testing and validation

### Status

✅ **IMPLEMENTED** — Ready for testnet deployment  
🔬 **VALIDATED** — Paper trading confirms improvements  
⏳ **AUDIT PENDING** — Security review before mainnet

---

*FHP: Where consensus emerges from coherence, not majority.*
