# UrbanGen-AI: Final Project Summary

**Status:** Closed as a single-seed research pilot (v1.0.0-final).
**Core contribution:** A controlled empirical study of synthetic data augmentation for nighttime urban scene classification.

## 1. The Problem
ResNet-50 (UrbanMind-AI Week 5) achieves 78% overall accuracy on BDD100K, but only **18.4% recall on nighttime residential scenes**. We asked: can generative models produce synthetic nighttime residential images that fix this weakness?

## 2. What We Tested
We evaluated four generative approaches:
1. **cVAE from scratch** — Failed (blurry representation mismatch).
2. **SD 1.5 Img2Img** — Failed (hallucinated buildings, deleted signs).
3. **Depth ControlNet** — Failed (cartoon artifacts at 512×288, 0/30 passed visual gate).
4. **CycleGAN from scratch** — **Succeeded** (preserved scene geometry).

## 3. The Downstream Result
We added 276 curated CycleGAN synthetic nighttime images to the training set (Arm D). 
- **Nighttime residential recall:** 0.3605 → **0.5170** (paired bootstrap 95% CI excludes zero).
- **The Trade-off:** Daytime macro-F1 dropped by **-0.0241**, violating the pre-registered 0.02 guardrail.

We then ran a dose-matched pilot (Arm D-138) with only 138 images:
- **Nighttime residential recall:** 0.3605 → **0.5782** (best pilot result).
- **Daytime macro-F1 drop:** -0.0309 (VIOLATED).

We also tested mixed-class synthetic data (Arm D-bal):
- **Nighttime residential recall:** 0.4966.
- **Confound:** City/highway images were not manually curated. The result is inconclusive regarding the "shortcut" hypothesis.

## 4. The Diagnostics Toolkit
The project produced reusable artifacts:
- **Parity tests** (`test_shortcut_parity.py`) — verify a new filter matches the original.
- **Golden tests** (`test_week5_pipeline.py`) — 6 tests proving Week 5 checkpoint + pipeline reproducibility.
- **Shared preprocessing** (`preprocessing.py`) — consumed by UrbanMind-AI Week 12.

## 5. Honest Limitations
- **Single seed only.** No multi-seed variance was measured. All claims are seed-42 pilot observations.
- **D-bal is confounded.** The city/highway synthetic images were not manually curated.
- **No Arm B.** We did not compare against real-image oversampling.
- **No external validation.** Results are specific to BDD100K and ResNet-50.
- **Checkpoints were deleted post-training.** Daytime statistical audits were not run.

## 6. Verdict
CycleGAN synthetic augmentation is a promising, statistically supported single-seed intervention for nighttime residential recognition, at a documented daytime cost. It is a **pilot**, not a definitive claim. Full multi-seed validation is documented as future work.

## 7. Where to Look
- **Every claim with evidence:** `docs/claims.md`
- **Full Phase 2 pilot details:** `docs/phase2_pilot_findings.md`
- **Reproduction commands:** `docs/reproducibility.md`
- **Journal of experiments:** `docs/experiments.md`
- **Negative result logs:** `docs/method_1_day_to_night_log.md`, `docs/method_2_cyclegan_log.md`, `docs/method_3_controlnet_log.md`