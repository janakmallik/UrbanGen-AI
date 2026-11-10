# Arm D Findings: Synthetic Night Augmentation

**Date:** 2026-11-10
**Experiment:** Adding 276 curated CycleGAN nighttime images to ResNet-50 training.
**Status:** Completed (Seed 42). Pending multi-seed verification.

## Executive Summary
Adding generative synthetic data achieved a statistically supported improvement in nighttime residential recall, successfully addressing the core weakness identified in the baseline. However, this came at the cost of daytime accuracy, violating the pre-registered guardrail. The result is a documented **trade-off**, not a free win.

## The Data (Seed 42)

| Metric | Arm A (Baseline) | Arm C (Conventional Aug) | Arm D (Synthetic Aug) |
| :--- | :--- | :--- | :--- |
| **Night Residential Recall** | 0.3605 | 0.4150 | **0.5170** |
| Night Macro-F1 | 0.6475 | 0.6625 | **0.6757** |
| **Day Macro-F1** | **0.7457** | 0.7440 | 0.7216 |
| Test Macro-F1 | 0.7183 | 0.7214 | 0.7088 |

### Paired Bootstrap (Night Set, n=147)
- **D vs A (Night Residential):** +0.1577, CI [+0.0848, +0.2292] **(Excludes Zero)**
- **D vs C (Night Residential):** +0.1036, CI [+0.0357, +0.1678] **(Excludes Zero)**
- **D vs C (Night Macro-F1):** +0.0140, CI [-0.0093, +0.0380] (Includes Zero)

*Conclusion:* Arm D is statistically significantly better than both A and C at finding residential scenes at night. The overall night macro-F1 improvement over C is directional but not statistically proven.

## The Trade-off (Guardrail Violation)
The pre-registered guardrail allowed a maximum daytime macro-F1 drop of 0.02. 
- Arm D dropped to 0.7216, a loss of **0.0224 vs C**.
- **Guardrail: VIOLATED.**

### Why did it drop? (Confusion Matrix Diagnostic)
The synthetic data taught the model to be sensitive to residential features. This is highly beneficial at night, but creates false positives during the day.
- **City Street Recall:** 81.3% → 74.5% (-6.8 pp). The model now mistakes daytime city streets for residential.
- **Residential Precision:** 63.8% → 56.8% (-7.0 pp). 
- **Residential Recall:** 63.5% → 67.0% (+3.5 pp).

The model shifted its decision boundary toward residential. 

## Verdict
**Qualified Success.** Generative data successfully solved the nighttime residential weakness (recall jumped from 36% to 51%), but the cost to daytime accuracy is real. The model is fundamentally trading daytime city-street precision for nighttime residential recall.

## Next Steps
1. Run seed 123 (Arm C and D) to confirm the trade-off is stable and not an artifact of seed 42.
2. If stable, document this as the final finding: "Synthetic nighttime data works, but requires a trade-off mitigation strategy (e.g., stratified weighting) for production."