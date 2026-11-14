# Lessons — UrbanGen-AI

A postmortem of the pipeline-mismatch episode, the diagnostic discipline that caught it, and the engineering lessons from the generative video showcase. This is the project's contribution beyond the negative generative findings.

## Episode 1 — The blur-confounded evaluator

**What happened:** During G1c, a Week 5 classifier was used to evaluate synthetic residential samples. The classifier was trained on sharp 224×224 images. The synthetic samples were blurry 128×128 outputs. The classifier predicted 0% residential for the synthetics.

**Initial interpretation:** The cVAE failed to generate residential samples.

**What was wrong:** The classifier also fails on *real* residential images passed through the same 128×128 blur — a distribution shift between sharp training and blurry inference.

**Fix:** Train a "blur-aware" evaluator on reconstructions of real training images. This eliminated the distribution-shift artifact. Blur-aware balanced accuracy was 0.604 vs sharp baseline 0.611 (ratio 0.988) — the class signal survived reconstruction; the original evaluator was fragile.

**The lesson:** A classifier trained on sharp images is not a valid evaluator for blurry generated content. Match the evaluator's training distribution to the input distribution.

**Files:** `scripts/diagnose_blur_aware_evaluator.py`, `outputs/diagnostic_blur_aware/`

## Episode 2 — The silent pipeline mismatch in the robustness report

**What happened:** A robustness report evaluated the Week 5 checkpoint on the 128×128 cache content strip upsampled to 224×224. It reported residential recall 0.124 and macro-F1 0.571 — numbers far below Week 5's published 0.6984.

**Initial interpretation:** The model has a severe residential weakness.

**What was wrong:** Week 5 was trained on native 1280×720 JPEGs processed through `Resize(256) → CenterCrop(224)`. The robustness report fed it a 72×128 content strip from a 128×128 cache, upsampled to 224×224. That's a different input distribution. The model was not trained on this pipeline.

**Fix:** Three-condition verification (`scripts/verify_week5_pipeline.py`) evaluated the same 4,932 test images through:
- (a) native JPEG, Resize(256)→CenterCrop(224) — Week 5 training pipeline
- (b) native JPEG, resize to 224×224 squashed
- (c) 128×128 cache → 224×224

Condition (a) reproduced published Week 5 exactly (0.7800 / 0.6984 / 0.4121). Condition (c) dropped residential recall by 3.3×. **The 0.124 was an artifact.**

**The lesson:** Evaluation pipeline must match training pipeline exactly. Verify by reproducing published numbers to the decimal place before trusting any new metric. A silent pipeline mismatch is the same class of bug as train-serve skew — the failure mode is the same, the domain is just different.

**Files:** `scripts/verify_week5_pipeline.py`, `outputs/pipeline_verification/pipeline_verification.json`

## Episode 3 — The over-loose suspect rule

**What happened:** A label audit script flagged residential "suspects" using a rule of `P(true) < 0.5 AND P(true) < max_other`. It flagged 29.7% of residential images as suspects — suggesting widespread label noise.

**Initial interpretation:** The residential label is noisy.

**What was wrong:** The same rule flagged 36.0% of city street and 24.2% of highway images. If a rule flags roughly a third of everything, it is measuring "hard images," not "mislabeled images."

**Fix:** Visual inspection of the suspect and confident grids. Both grids showed residential-looking scenes. The label is consistent; what's real is that the model's per-class precision is low (residential precision ≈ 0.37, so high recall with low precision).

**The lesson:** Automated label-noise detectors need calibration against human visual inspection. A rule that flags everything is not detecting noise; it's detecting difficulty. Never publish a "label noise" claim without looking at the flagged examples.

**Files:** `scripts/audit_residential_labels.py`, `outputs/label_audit/suspect_residential.png`, `outputs/label_audit/confident_residential.png`

## Episode 4 — Multiple comparisons without correction

**What happened:** The corrected robustness report tested 6 within-class comparisons (3 classes × 2 weather/time pairs) at nominal 95% CI.

**What was wrong:** 6 comparisons at nominal 95% have ~26% chance of at least one false positive. The report's stars are labeled as exploratory and unadjusted, which is honest, but a reader might treat them as definitive.

**Fix:** Add a note to every table: "Comparisons are exploratory and not adjusted for multiple testing." For the final project, this is adequate because we're not making deployment decisions from these stars. If we were, we'd apply Holm-Bonferroni correction.

**The lesson:** Whenever you run more than ~3 comparisons in a single report, either correct or label as exploratory. Do not silently let readers treat unadjusted stars as definitive.

**Files:** `docs/robustness_report_v2.md` interpretation guide

## Episode 5 — The class-weight sensitivity hidden improvement

**What happened:** The frozen-feature probe showed arm B (real+synthetic) improved residential F1 by +0.0092 under `class_weight="balanced"` with 5/5 folds positive. This looked like a real signal.

**What was wrong:** Under `class_weight=None` (neutral), arm B is *worse* than arm A by −0.0099. Under `class_weight={0:1,1:1,2:4}`, it's worse by −0.0023. The improvement depends entirely on the class_weight setting.

**Fix:** The frozen-feature probe's "improvement" is a classifier interaction with the class_weight choice, not a synthetic-data effect. The full fine-tune was not run.

**The lesson:** Every augmentation claim must be tested under at least two class_weight settings. If the direction flips, the "improvement" is a hyperparameter artifact.

**Files:** `scripts/audit_probe.py`, `outputs/g1c_probe/audit_results.json`

## Episode 6 — The WanGP integration trap

**What happened:** During the generative video showcase, attempts were made to run the Wan 2.2 FastWan model directly using the `diffusers` library, or by reverse-engineering WanGP's internal `queue.json` format.

**Initial interpretation:** WanGP is just a UI wrapper; we can run the model ourselves with standard Python libraries.

**What was wrong:** WanGP is not a standard Gradio frontend. It contains a deeply integrated custom memory manager (`mmgp`) that uses block-swapping, VRAM preloading, and INT8 quantization to run a 5B model on a 6 GB GPU. The standard `diffusers` library cannot replicate this and attempted to download a 34 GB model, causing the script to hang.

**Fix:** Import WanGP's internal API directly (`from shared.api import init`). This bypasses the UI entirely but preserves the `mmgp` memory engine, allowing headless generation in ~1 minute per video.

**The lesson:** Always look for the library's official headless entry point before building custom wrappers. Do not assume that a UI-based application is just a frontend; it may contain a highly optimized backend that is impossible to replicate with standard libraries.

**Files:** `scripts/generate_i2v_showcase.py`, `docs/reproducibility.md`

## What these episodes have in common

Every one of them was caught by **comparing against an external reference or looking at raw examples**:

- Episode 1: comparison against a matched-distribution evaluator
- Episode 2: comparison against published Week 5 numbers
- Episode 3: visual inspection of the flagged images
- Episode 4: the discipline of labeling exploratory vs confirmed
- Episode 5: comparison across multiple hyperparameter settings
- Episode 6: reading the WanGP source code to find the correct API entry point

The common failure mode is **plausible-looking numbers (or code) that nobody verified against an external ground truth.** Every audit correction in this project came from asking "does this number reproduce something we already know?" or "does this API actually exist?"

## Recommended practices going forward

1. **Reference checks are non-negotiable.** Every evaluation script must include a check that reproduces a known published number on a known input before trusting new metrics.

2. **Golden test for the pipeline.** `tests/unit/test_week5_pipeline.py` runs Week 5 on a fixed subset and asserts the outputs match. If the pipeline drifts, CI fails before any new claim gets made.

3. **Shared preprocessing function.** `src/urbangen/data/preprocessing.py` is the single source of truth for pipeline transforms. Both evaluation scripts and future serving code import from it. Train-serve skew becomes a compile-time error, not a silent metric degradation.

4. **Claims ledger.** `docs/claims.md` records every quantitative claim with evidence and status. A claim without a corresponding output file is removed before commit.

5. **Visual inspection is a required gate.** Any claim about "the model generates X" or "the label is Y" requires at least 5–10 example images looked at by a human. Metrics without visual confirmation are provisional.

6. **Respect the official entry points.** When integrating a third-party engine (like WanGP), look for its documented headless API before attempting to reverse-engineer its internal files or rebuild its memory manager.

## Episode 7 — The unreliable filter

**What happened:** After generating 500 synthetic night images with CycleGAN, we applied three filters (CLIP night score, luminance, SegFormer paved-surface fraction). 282 images passed. When we inspected the rejections and acceptances, we found:
- 90% of the "paved_surface" rejections were visually good images
- 80% of the "luminance" rejections were visually good images
- 15-25% of the "accepted" images had structural failures

**Initial interpretation:** The filters were working as designed (rejecting bad, accepting good).

**What was wrong:** 
- SegFormer-B0 was trained on ADE20K (daytime scenes). It misclassifies nighttime roads, producing bias.
- CLIP night score is a relative softmax (daytime prompt vs nighttime prompt) — not an absolute nighttime probability.
- Luminance rejects images with bright spots (streetlights, lit buildings), which are exactly the features we want.

**The lesson:** For generative augmentation, no automated filter is reliable. Every filter must be validated by inspecting the images it accepts and rejects. A filter that flags 40% of images as "bad" is not a good filter — it's an untested heuristic that will destroy training data.

**The fix:** Manual curation. Show source + generated pairs, let human eyes decide good/bad. This is ~30 min for 500 images using a contact sheet.

## Episode 8 — The confounded ablation

**What happened:** After Arm D (276 residential synthetic) improved nighttime recall, we built D-bal (138 residential + 138 mixed) to test whether the model was "learning a shortcut." D-bal underperformed D-138 (138 residential only). We initially concluded the shortcut hypothesis was confirmed.

**What was wrong:** The D-bal mix had two changes at once:
1. Different class distribution (residential vs residential + city/highway)
2. Different image quality (residential went through manual curation; city/highway did not)

The performance drop could be from either factor. We cannot isolate the "shortcut" explanation from the "garbage images" explanation.

**The lesson:** When comparing two arms, change exactly one variable. If you're testing "does class composition matter?", keep the total dose AND the image quality identical. If you're testing "does curation matter?", change only the curation protocol, not the class distribution.

**Files:** `docs/phase2_pilot_findings.md` §5 (limitations), `docs/claims.md` (retracted shortcut-confirmed claim)