# Phase 2 Data Manifest (Frozen)

**Date Frozen:** 2026-11-13

## 1. Real Data Splits — Verified Hashes

| Split | Count | SHA-256 |
|---|---|---|
| Train | 68,969 | TBD (verify against `bdd100k_labels_images_train.json`) |
| Val Clean | 4,727 | TBD (verify against `resnet_val_clean.json`) |
| Test Clean | 4,745 | TBD (verify against `resnet_test_clean.json`) |
| Night Test | 1,931 | Derived: Test Clean ∩ timeofday=night |

**Train count reconciliation:** The raw BDD100K training set contains 69,863 images. Our training split uses 68,969. The 894-image difference is: images missing on disk (0 per prior audit), images whose scene attribute is not one of the 3 viable classes (city/highway/residential). Full reconciliation in `docs/claims.md`.

**Actual hashes must be filled in before freezing this document. Run `Get-FileHash` on each file and paste values above.**

## 2. Seed Distinctions
This project uses **three distinct seeds** that must not be conflated:
- **Source-selection seed (999):** Used in Method 3 to pick source images for generation.
- **Synthetic-generation seed (42):** Used in Method 2 CycleGAN training.
- **Classifier-training seeds (42, 123, 456):** Used in Phase 2 training runs.

## 3. Synthetic Data — Curated Set (Arm D)
- **Count:** 276 images
- **Manifest:** `outputs/method_2_synthetic/curated_manifest.json`
- **Source selection:** From `filter_log.csv`, the 282 accepted images minus 6 manually rejected (labels 6, 7, 36, 72, 84, 149).
- **Rejection reasons (per earlier decision):** `too_dark_or_artifacts`.
- **SHA-256 of curated manifest:** TBD (fill from `Get-FileHash`)

## 4. Synthetic Data — Uncurated Set (Arm D-unc)
- **Count:** 276 images
- **Selection method:** Deterministic random sample using `random.Random(1000).sample(range(282), 276)` from the 282 accepted images, minus none. This ensures dose-matching with Arm D.
- **Excluded:** None (dose-matched).
- **SHA-256 of uncurated manifest:** TBD

## 5. Synthetic Data — D-bal Set (Dose-Matched Shortcut Control)

**Status:** To be produced after gates G1–G8 pass (see `phase2_execution_checklist.md`).

- **Total count:** 276 synthetic images (dose-matched to Arm D).
- **Composition:**
  - 138 residential images (random 50% sample of the existing 276 curated set, seed 42)
  - 69 city synthetic images (human-approved from generated candidates, seed 42)
  - 69 highway synthetic images (human-approved from generated candidates, seed 42)
- **Generation source:** `outputs/method_2_resume/checkpoints/epoch_20.pth`
- **Generation pipeline:** `scripts/shortcut_control_common.py` (shared filter functions) called by `scripts/generate_shortcut_candidates.py`
- **Candidate count:** 400 city + 400 highway generated, filtered, then human-curated. If approved count < 69 for either class, D-bal is **cancelled** and documented as a limitation.
- **Output manifests:**
  - `outputs/method_2_synthetic/shortcut_control/city/generation_manifest.json`
  - `outputs/method_2_synthetic/shortcut_control/highway/generation_manifest.json`
  - `outputs/method_2_synthetic/d_bal_manifest.json` (final combined training set)
- **Label integrity:** Every image in `d_bal_manifest.json` has an explicit `"label"` field (`"residential"`, `"city"`, or `"highway"`). Labels are never inferred from folder names.
- **Hashes required in final manifest:** Source-file SHA-256, generated-image SHA-256, checkpoint SHA-256, git commit SHA, filter threshold values.

## 6. Data Provenance
- **CycleGAN checkpoint SHA-256:** `91fee41198559e8064064d68c677d422c8e68aebe3305ea9a1469040ad9b8d97`
- **Source-selection manifest:** `outputs/method_2_synthetic/source_selection.json`
- **Filter log:** `outputs/method_2_synthetic/filter_log.csv`

**Known provenance limitation:** The Method 2 provenance audit previously flagged 2 source images (labels 15 and 230) as "uncertain" before manual curation resolved them. Both were kept after review. The classification of these images is documented in `curated_manifest.json`.

## 7. Preprocessing (Applied to all arms identically)
- Resize: `Resize(256)` then `CenterCrop(224)`
- Normalization: ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- Color channels: RGB
- Invalid image handling: none expected; corruption would raise during loading
- Label mapping: `{0: city street, 1: highway, 2: residential}`

## 8. Leakage Prevention (verified via `verify_source_split_membership.py`)
- Zero overlap between synthetic source images and Val Clean.
- Zero overlap between synthetic source images and Test Clean.
- Zero overlap between synthetic source images and Method 2 reserved splits (3,400 images).

**Basename-level verification only.** Sequence-level overlap was not verified (BDD100K filename prefixes were not established as independent video sequences).

## 9. Environment (Frozen for reproducibility)
- PyTorch: 2.6.0+cu124
- torchvision: 0.21
- CUDA: 12.4
- Diffusers: 0.42.1
- Transformers: 5.22.2
- Python: 3.12.4

**All hashes above must be computed and filled before this file is committed.**