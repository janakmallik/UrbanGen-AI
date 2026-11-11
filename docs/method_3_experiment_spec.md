# Method 3 Experimental Specification (Frozen)

**Date frozen:** 2026-11-11
**Status:** Pre-registered before implementation
**Purpose:** Test whether depth-conditioned diffusion (ControlNet + SD 1.5) produces more useful synthetic nighttime data than CycleGAN for improving nighttime residential recall.

This document freezes the experimental protocol. Do not modify after Phase 1 begins.

## 1. Source Image Selection (locked)

- **Source domain:** BDD100K train split, `scene == "residential"`, `timeofday == "daytime"`.
- **Exclusions:** All images in `outputs/method_2_resume/split_manifest.json` (day_train, day_holdout, night_train, night_holdout = 3,400 images). This prevents overlap with Method 2's training and holdout.
- **Holdout exclusion:** All images in `resnet_val_clean.json` (4,727) and `resnet_test_clean.json` (4,745).
- **Seed:** 999 (distinct from Method 2's seed 42, so source sets do not overlap).
- **Pilot source count:** 5 images.
- **Full-scale source count (if pilot passes):** 500 images.

## 2. Model Versions (locked)

- **Base model:** `stable-diffusion-v1-5/stable-diffusion-v1-5` (revision: main)
- **ControlNet:** `lllyasviel/control_v11f1p_sd15_depth`
- **Depth estimator:** `Intel/dpt-hybrid-midas`
- **Framework:** PyTorch + Hugging Face Diffusers
- **Precision:** FP16 (with attention slicing and CPU offload if needed)

## 3. Generation Configuration

### Pilot (Phase 3)
- **Sources:** 5 daytime residential images.
- **Strengths:** 0.60, 0.75.
- **ControlNet conditioning scales:** 0.6, 0.8, 1.0.
- **Total outputs:** 5 × 2 × 3 = 30.
- **Output resolution:** 512 × 288.
- **Seed:** 42 (fixed for reproducibility).
- **Prompt:** "nighttime residential street, dark sky, streetlights, headlights, dashcam view, original scene geometry preserved"
- **Negative prompt:** "daytime, sunny, hallucinated objects, distorted structure, blurry, cartoon"

### Full scale (Phase 5, if pilot passes)
- Same parameters as winning pilot config.
- 500 source images.

## 4. Pilot Decision Gate (Phase 4)

The pilot passes if and only if:

1. **Human inspection (primary gate):** At least **5 of 30** outputs are judged by the human reviewer as plausible nighttime scenes with preserved structural geometry.
2. **Source diversity (additional requirement):** The 5 accepted outputs must come from **at least 3 of the 5 source images**. A single source producing 5 great outputs does not constitute a valid pilot result.
3. **Metric support (secondary gate):** Mean CLIP night score of accepted outputs > 0.85, and mean SSIM between source and output > 0.4.

If any of the three gates fails, Method 3 fails the pilot and closes as a documented negative result.

## 5. Full-scale Decision (Phase 6)

Only if Phase 4 passes:
- Generate 500 synthetic images using the winning configuration.
- Run automated filters (CLIP night, luminance, SegFormer paved surface).
- **Manual curation of all 500 via contact sheets** (expect automated filters to be unreliable).
- Freeze curated manifest as `outputs/method_3_synthetic/curated_manifest.json`.
- Train Arm D-M3 = Arm A recipe + curated synthetic images.
- Evaluate using **identical protocol** to Arm D-M2 (same splits, same bootstrap, same guardrails).

## 6. Fair Comparison Rules (locked)

- **Same training recipe:** Arm A hyperparameters (IMAGENET1K_V2, class weights, 5 epochs, batch size 8, lr 1e-4, best-epoch selection).
- **Same evaluation splits:** Val clean (4,727), Test clean (4,745), Night subset (n=1,931), Night residential (n=147).
- **Same bootstrap:** 1,000 paired resamples, 95% CI.
- **Same guardrails:** Day macro-F1 must not drop by more than 0.02.
- **Primary comparison:** D-M3 vs D-M2 vs Arm A vs Arm C.

## 7. Artifact Structure

### Committed (small, text-based)
- This specification document.
- `outputs/method_3_calibration/` — the metric calibration set (already exists).
- `outputs/method_3_pilot/metrics.csv` — 30-row metrics table.
- `outputs/method_3_pilot/manifest.json` — source filenames, generation configs, seeds, model revisions, output filenames, peak VRAM, runtime per output.
- `docs/method_3_pilot_findings.md` — human inspection verdict.
- If full-scale run happens: `outputs/method_3_synthetic/curated_manifest.json`, `outputs/week5_arms/results_D-M3_seed42.json`.

### Not committed to Git (too large)
- The 30 pilot-generated JPEGs (in `outputs/method_3_pilot/`).
- The 500 generated images (in `outputs/method_3_synthetic/`).
- The depth maps.
- Any checkpoints.
- These remain on disk, referenced by manifest, and are listed as reproducible via the manifest + generation script.

## 8. Failure Mode Documentation

If any phase fails:
- Commit the failure evidence (metrics.csv, manifest.json, human inspection notes).
- Write `docs/method_3_pilot_findings.md` explaining the specific failure mode.
- Do not attempt to "fix" the failure by re-running with different parameters — that would violate the pre-registration.
- Close Method 3 as a documented negative result.

## 9. What Method 3 Does NOT Do

- Does not replace Method 2 or Arm D-M2.
- Does not invalidate the existing `v0.2.0-g2-arm-d-seed42` tag.
- Does not require changing the existing `results_{A,C,D}_seed42.json` files.
- Does not involve training a new generative model (inference only).

## 10. Stop Conditions

The experiment stops immediately if:
- Single-image smoke test produces unusable output (hallucinated or stayed daytime).
- VRAM exceeds 6 GB and cannot be reduced with attention slicing + CPU offload.
- Phase 4 decision gate fails.
- Any phase takes more than 2× its estimated time.

## 11. Sign-off

This specification is frozen as of 2026-11-11. It will not be modified
after Phase 1 begins. Any deviation from this specification must be
recorded in `docs/method_3_deviations.md` and justified in writing.

If Method 3 is abandoned at any phase, the experiment is still a
documented result. The failure evidence (metrics, manifests, human
inspection notes) must be committed before closing.

---

**Frozen by:** Janak Mallik
**Freeze commit:** (this commit)