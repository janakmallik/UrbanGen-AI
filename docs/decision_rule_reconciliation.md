# Decision Rule Reconciliation: Arm C to Arm D

**Date:** 2026-11-10
**Status:** Locked before Arm D training

## The Conflict
Two pre-registered decision rules existed in the repository:
1. `docs/train_week5_augmented_arms.md`: Stop if Arm C residential recall ≥ 0.35.
2. `scripts/train_week5_augmented_arms_v2.py`: Stop if Arm C night macro-F1 ≥ 0.68; Proceed if 0.63 - 0.68.

## Arm C Results (Seed 42)
- Night residential recall: **0.4150** (would trigger STOP under rule 1)
- Night macro-F1: **0.6625** (triggers PROCEED under rule 2)
- Day macro-F1: **0.7440** (Guardrail passed, drop was only 0.0017)

## The Decision
We are adopting the `v2.py` rule (night macro-F1) as the authoritative metric. 

**Reasoning:** Macro-F1 evaluates the entire nighttime distribution (city, highway, residential). Residential recall alone is too volatile (n=147) and ignores the risk of degrading other night classes. Because the Arm C macro-F1 bootstrap CI includes zero (Δ=+0.0148, CI [-0.0087, +0.0382]), the improvement is not statistically proven. Therefore, we cannot definitively say photometric augmentation "solves" the problem. 

The experiment must proceed to Arm D to test if generative data provides a statistically robust improvement over both A and C.