# Phase 2 Execution Checklist (Frozen)

**Date Frozen:** 2026-11-13

## Order of Operations (Mandatory)

1. Method 2 provenance audit completed (already done — 2 flagged images resolved)
2. Synthetic data audit completed (`synthetic_audit_report.json` exists)
3. Three protocol documents committed and tagged
4. Training script audited against this checklist
5. **Only then** run Phase 2 experiments

## Pre-Flight Code Audit

- [ ] `train_week5_augmented_arms_v2.py` saves `best_state_dict` permanently
- [ ] Full test predictions saved to results JSON (all 4,745 images, not just night)
- [ ] All 8 arms implemented (A, B, C, D, D-unc, D-short, D-25, D-50)
- [ ] Image cache strategy implemented (`.npy` pre-resized 224×224)
- [ ] Checkpoint selection uses validation macro-F1 only
- [ ] No test images are loaded during training

## Pre-Flight D-bal Audit (Gates G1–G8)

- [ ] **G1** `phase2_protocol.md` and `phase2_data_manifest.md` reflect the dose-matched design (138+69+69 = 276)
- [ ] **G2** `tests/unit/test_shortcut_parity.py` passes (filter parity with Method 2)
- [ ] **G3** Source-pool verification: 400+ city, 400+ highway, disjoint from residential/reserved/val/test
- [ ] **G4** One-image smoke test passes (1 city + 1 highway generated and inspected)
- [ ] **G5** 20+20 pilot passes (survival rate allows reaching 69+69, visually inspected)
- [ ] **G6** Accepted AND rejected contact sheets generated for both classes
- [ ] **G7** `human_curation_template.json` filled in; ≥69 approved per class
- [ ] **G8** `d_bal_manifest.json` written with all 276 filenames, SHA-256 hashes, and explicit labels

**If any of G1–G8 fail, D-bal is cancelled and documented as a limitation in `docs/final_report.md`. Do not proceed with a partial pipeline.**

## Pre-Flight Runtime Benchmark

- [ ] Measure one full epoch with image caching enabled
- [ ] Record actual epoch time
- [ ] Document whether the 8 seed-42 runs are feasible in one session

*(Do not promise runtime under any threshold before measuring.)*

## Pre-Flight Data Audit

- [ ] All hashes in `phase2_data_manifest.md` are filled in
- [ ] 276 curated images exist (`outputs/method_2_synthetic/synth_*.jpg`)
- [ ] 276 uncurated images exist (or this arm is cancelled)
- [ ] 276 D-short images exist (or this arm is cancelled)
- [ ] CycleGAN checkpoint hash matches manifest

## Pre-Flight Environment Audit

- [ ] `uv run ruff check src/ scripts/ tests/` passes
- [ ] `uv run pytest tests/unit/test_week5_pipeline.py -v` passes
- [ ] `git status` is clean
- [ ] Protocol files committed with tag `v0.3.0-phase2-protocol`

## Execution Rules

- [ ] Run seed 42 for arms A, B, C, D, D-unc (or cancelled), D-short (or cancelled), D-25, D-50 sequentially
- [ ] After each arm completes, immediately save the results JSON
- [ ] Do NOT inspect test results until all seed-42 runs complete
- [ ] Screening decision based on validation macro-F1 only (never test)
- [ ] Run seed 123 and 456 for Arms A, C, D (mandatory) and B, D-unc, D-short (conditional on screening)
- [ ] All test evaluation occurs only after each seed's runs complete

## Statistical Analysis Rules

- [ ] Report individual seed results per arm
- [ ] Report mean and paired seed-level differences for replicated arms
- [ ] Do NOT present 3-seed t-tests as definitive proof
- [ ] Explicitly state n=3 is preliminary
- [ ] Report per-class metrics and confusion matrices
- [ ] Distinguish test-sample uncertainty (bootstrap CI) from training-seed uncertainty (std across seeds)

## Post-Execution Rules

- [ ] Update `docs/final_report.md` with actual results
- [ ] Explicitly document all failure modes and negative findings
- [ ] Document the reproducibility rerun status vs prior Arm D results
- [ ] Tag final release `v0.4.0-phase2-results`
- [ ] Do not modify the protocol after seeing results

## Stopping Rules

- If Arm D fails any pre-registered criterion, report as documented negative result
- If Arm D-unc outperforms Arm D, report that curation was not beneficial
- If Arm D-short performs similarly, report it as evidence consistent with shortcut hypothesis
- No post-hoc metric changes permitted