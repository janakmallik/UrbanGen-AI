# Phase 2 Experimental Protocol (Frozen)

**Date Frozen:** 2026-11-13
**Status:** Locked before any training runs

## 1. Research Questions
1. **Effectiveness:** Does curated CycleGAN augmentation improve nighttime scene recognition beyond real-data baselines and conventional augmentation?
2. **Curation Necessity:** Does human curation of synthetic images contribute measurable value over uncurated synthetic images?
3. **Shortcut Check:** Does the improvement survive when synthetic images are drawn from non-residential classes? (This is a control, not a proof of mechanism.)

## 2. Experimental Arms (8 arms)
All arms use ResNet-50 (IMAGENET1K_V2), batch 8, 5 epochs, best-epoch selection by **validation macro-F1**. Training budgets are documented per arm in `phase2_data_manifest.md`.

| Arm | Training Data | Purpose |
|---|---|---|
| **A** | 68,969 real images (full training split) | Baseline |
| **B** | Real + 276 real nighttime images, oversampled ×3 | Cheap rebalancing control |
| **C** | Real + strong photometric augmentation (see below) | Conventional augmentation control |
| **D** | Real + 276 curated CycleGAN residential images | Proposed method |
| **D-unc** | Real + 276 uncurated CycleGAN residential images (deterministically sampled from the 282) | Curation control |
| **D-bal** | Real + 138 residential synthetic + 69 city + 69 highway = 276 total (dose-matched to Arm D) | Shortcut control |
| **D-25** | Real + 69 curated images | Low-dose ablation |
| **D-50** | Real + 138 curated images | Mid-dose ablation |

## 3. Arm Specifications (Reproducible)

### Arm B — Real oversampling
- The 276 unique real nighttime residential images from the training split (matching the number of synthetic images in Arm D).
- Oversampled ×3 → 828 total presentations (276 unique + 552 duplicates per epoch).
- `random.Random(42)` used to select the 276 unique images.

### Arm C — Photometric augmentation
Applied to all training images at load time:
- `transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)` — p=1.0
- Random brightness multiplier `t * (0.5 + 0.5 * rand)` — p=0.5
- `RandomHorizontalFlip` — p=0.5 (same as baseline)

### D-bal construction (dose-matched shortcut control)
- **Total synthetic dose:** 276 images (identical to Arm D).
- **Composition:** 138 residential + 69 city + 69 highway = 276.
- **Residential slice:** Randomly sampled 138 of the existing 276 curated residential images using `random.Random(42).sample(residential_276, 138)`.
- **City slice:** 69 human-approved city synthetic images selected from the generated candidates using `random.Random(42).sample(approved_city, 69)`.
- **Highway slice:** 69 human-approved highway synthetic images using `random.Random(42).sample(approved_highway, 69)`.
- **Why dose-matched:** Doubling the synthetic dose (as in an unbalanced design) would confound "shortcut removal" with "more data." This design tests whether the same dose of synthetic data, mixed across classes, preserves Arm D's gain.
- **Class labels recorded in the manifest**, never inferred from folder names.
- **CycleGAN checkpoint:** `outputs/method_2_resume/checkpoints/epoch_20.pth` (SHA-256 recorded in generation manifest).
- **Source images:** Drawn exclusively from the training split; verified disjoint from validation/test and from the 500 residential source images.

## 4. Metrics
- **Primary:** Night Macro-F1 over 1,931 nighttime test images.
- **Secondary:** Night residential recall, day macro-F1, day residential precision, per-class metrics, confusion matrices.
- **Guardrail:** Day Macro-F1 drop vs Arm A must be ≤ 0.02.

## 5. Pre-Registered Decision Rules

### Target effect
- **D vs A:** Night Macro-F1 improvement ≥ +0.02.
- **D vs B:** Must beat real-data oversampling.
- **D vs C:** Must beat conventional augmentation.
- **D vs D-unc:** Must beat uncurated to justify curation (Δ ≥ +0.01).
- **D vs D-bal:** If Arm D's gain over Arm A disappears in D-bal, this is evidence that the gain was shortcut-driven (synthetic-appearance cues), not a real nighttime-feature effect. Because D-bal is dose-matched to Arm D, this comparison isolates the "shortcut" hypothesis from the "more-data" hypothesis.

The D vs D-short comparison is a **shortcut control**, not proof of mechanism. If D performs similarly to D-short, it is evidence consistent with a label-shortcut hypothesis, not definitive proof.

### Guardrail
If Arm D day macro-F1 drops by > 0.02 vs Arm A, the result is reported as "documented trade-off," not "clean success."

## 6. Screening and Replication Rules

### Seed 42 = exploratory screening
- All 8 arms run on seed 42.
- Screening uses **validation nighttime macro-F1**, never test results.
- Test results are computed only after all 8 arms complete and are used only for final reporting.

### Replication triggers (frozen before execution)
- **Always replicated:** Arms A, C, D on seeds 123 and 456, regardless of seed-42 results.
- **Conditionally replicated:** Arms B, D-unc, D-short only if Arm D's seed-42 validation macro-F1 improvement vs Arm A ≥ +0.02.
- **Dose-response (D-25, D-50):** Not replicated. Their purpose is exploratory dose-response analysis, not statistical confirmation.

### Statistical reporting
- Report **individual seed results** for every arm.
- Report **mean and paired seed-level differences** for replicated arms.
- Do not present 3-seed t-tests as definitive proof. State explicitly: n=3 is preliminary.

## 7. Test-Set Isolation
The 4,745 test images (including all 1,931 nighttime images) are touched only:
1. Once per completed arm at the end of seed-42 training for reporting.
2. Once per completed arm at the end of seeds 123 and 456 for replication.

Never used for hyperparameter tuning, checkpoint selection, or arm selection.

## 8. Failure Conditions
CycleGAN augmentation is declared **unsuccessful** if any of:
- Arm D does not beat Arm B or Arm C on Night Macro-F1 at seed 42.
- Arm D's day macro-F1 drops by > 0.02.
- Arm D's seed-42 validation improvement vs Arm A is < +0.02.
- The shortcut hypothesis (that Arm D learned "synthetic = residential" rather than nighttime residential features) is **not tested** in Phase 2. This is documented as a limitation in `docs/final_report.md`. A properly dose-matched control would require three arms (D, D-bal, D-half) and is deferred to future work.

If unsuccessful, Method 2 synthetic augmentation is reported as a documented negative result.

## 9. Re-run vs Previously Reported Results
Arm D was already run at seed 42 in a prior experiment (`outputs/week5_arms/results_D_seed42.json`). That prior run did not save checkpoints or full daytime predictions. The new Phase 2 runs are **explicit reproducibility reruns**, not continuations. Both sets of results will be reported; discrepancies will be documented as reproducibility limitations.

## 10. Pre-Existing Limitations
- Segmentation label-transfer is not performed (only 22 nighttime images exist in the segmentation subset).
- Only one architecture (ResNet-50) and one dataset (BDD100K).
- Only 3 training seeds.
- Human curation was performed by a single reviewer.
- CycleGAN is not a modern frontier generator.
- Shortcut hypothesis not tested. The current 7-arm design cannot distinguish "CycleGAN teaches nighttime residential features" from "CycleGAN teaches that synthetic nighttime images should be classified as residential." This is a documented limitation.