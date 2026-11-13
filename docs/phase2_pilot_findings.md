# Phase 2 Pilot Findings: Augmentation, Dose, and Class Mix

**Date:** 2026-11-14
**Status:** Pilot Study (Single Seed). INCONCLUSIVE, awaiting multi-seed.

## 1. Executive Summary
We investigated whether CycleGAN synthetic data could improve nighttime residential recognition, and whether the improvement was due to a "shortcut" (the model just learning that "dark synthetic = residential"). 

We trained five arms: Real-only (A), Photometric Aug (C), 276 Residential Synthetic (D), 138 Residential Synthetic (D-138), and 138 Residential + 138 Mixed Synthetic (D-bal).

**Key Findings:**
1. **Synthetic augmentation is promising.** D-138 improved nighttime residential recall from 0.361 (Baseline A) to **0.578**. The image-level paired bootstrap 95% CI excludes zero. This does not establish statistical significance across training seeds.
2. **Increasing synthetic dose did not show a clear benefit on this seed.** D-138 (138 images) performed slightly better than D (276 images) on recall (0.578 vs 0.517). However, the difference is small (+0.061) and the CI includes zero. 
3. **The "Shortcut" hypothesis remains unresolved.** Adding non-residential automated-filtered synthetic data (D-bal) dropped recall to 0.497. However, because the non-residential images were not manually curated, the drop could be due to image quality rather than class mixing.
4. **The Daytime Trade-off is real.** Every synthetic arm (D, D-138, D-bal) violated the pre-registered guardrail, dropping daytime macro-F1 by 0.024 to 0.031.

## 2. Experimental Integrity Checks
*Before reading the results, we must establish that the comparisons are valid.*
- [x] **D-138 Subset:** Verified D-138 is an exact 138-image subset of D.
- [x] **D-bal Residential:** Verified D-bal uses the same 138 residential images as D-138.
- [x] **D-bal Mixed Classes:** Verified D-bal contains exactly 69 City + 69 Highway images.
- [x] **Evaluation Identity:** Verified all five arms evaluated the exact same night/day/test filenames in the same order.
- [x] **Reproducibility:** Training configuration, seed (42), and evaluation artifacts are logged.

## 3. Results Table (Seed 42)

| Arm | Night Res Recall | Delta vs A | Night Macro-F1 | Delta vs A | Day Macro-F1 | Day Drop vs A | Guardrail (≤0.02) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** (Real Only) | 0.3605 | — | 0.6475 | — | 0.7457 | — | PASS |
| **C** (Photometric) | 0.4150 | +0.0545 | 0.6625 | +0.0150 | 0.7440 | -0.0017 | PASS |
| **D** (276 Res) | 0.5170 | +0.1565 | 0.6757 | +0.0282 | 0.7216 | -0.0241 | **FAIL** |
| **D-138** (138 Res) | **0.5782** | **+0.2177** | **0.6788** | **+0.0313** | 0.7148 | -0.0309 | **FAIL** |
| **D-bal** (138 Res + 138 Mixed) | 0.4966 | +0.1361 | 0.6635 | +0.0160 | 0.7221 | -0.0236 | **FAIL** |

*Note: Deltas are percentage-point changes. Guardrail threshold was a maximum drop of 0.02.*

## 4. Statistical Analysis
- **D-138 vs A:** Night recall +0.2177, 95% CI [0.141, 0.292] (Image-level CI excludes zero).
- **D vs A:** Night recall +0.1565, 95% CI [0.085, 0.229] (Image-level CI excludes zero).
- **D-bal vs D-138:** Night recall -0.0786, 95% CI [-0.158, 0.000] (Touches zero).
- **D vs C:** Night macro-F1 +0.014, CI includes zero.

## 5. Limitations & Threats to Validity
1. **Single Seed:** All results are from seed 42. We have not measured training-seed variance. A multi-seed run (seeds 123, 456) is required to confirm these rankings.
2. **D-bal Quality Confound:** The D-bal city/highway images passed the automated filter but were not manually curated. The drop in D-bal performance may be because the model was fed garbage city/highway images, not because of class mixing.
3. **Daytime Regression:** The model trades daytime accuracy for nighttime accuracy. This is a consistent feature of all synthetic augmentation arms in this study.

## 6. Future Work Roadmap (What we could do next)
To elevate this pilot to a formal conference paper, the following steps are defined but not yet executed:

### Phase A: Multi-Seed Replication (4 GPU hours)
Run seeds 123 and 456 for Arms A, C, D, D-138, and D-bal.
- **Goal:** Compute mean ± std across seeds to confirm that D-138 is reliably the best arm.
- **Files:** `scripts/train_week5_augmented_arms_v3.py --arm [X] --seed [123/456]`

### Phase B: Class-Balanced Baseline (1 GPU hour)
Run Arm B (Real + Real Oversampled).
- **Goal:** Prove that CycleGAN synthetic data is better than simply oversampling the existing real nighttime images.
- **Files:** `scripts/train_week5_augmented_arms_v3.py --arm B --seed 42`

### Phase C: Manually Curate D-bal (2-3 hours CPU)
Manually review the 162 accepted Highway and 235 accepted City images.
- **Goal:** Rebuild the D-bal manifest using only visually valid images to definitively test the shortcut hypothesis without image-quality confounds.

### Phase D: Daytime Mitigation
Experiment with synthetic sampling ratios (e.g., use 10% synthetic per epoch instead of 100%) to see if the nighttime gain can be preserved while recovering daytime F1.

### Phase E: Independent Validation
Test on an independent nighttime dataset (e.g., ACDC) or different geographic domain to see if the results generalize.

## 7. Conclusion
The seed-42 evaluation shows a substantial improvement in nighttime residential recall with synthetic residential augmentation. However, the single-seed nature, the daytime trade-off, and the unresolved shortcut hypothesis mean this is currently a promising pilot, not a finalized scientific claim.