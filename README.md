
# UrbanGen-AI

**Diagnosing whether generative models produce useful synthetic urban data for urban scene understanding.**

UrbanGen-AI trained a conditional VAE on BDD100K scene images and asked whether its synthetic residential samples improve downstream recognition. It does not: the cVAE's samples are 0.6% residential in a real-trained classifier's judgment. The project then pivoted to **CycleGAN**, which successfully learned day-to-night translation while preserving scene geometry. The project documents all negative results, the successful generative pipeline, and a reusable diagnostic toolkit.

It is a **separate project** from [UrbanMind-AI](https://github.com/janakmallik/UrbanMind-AI). UrbanGen-AI consumes UrbanMind-AI's published data splits and Week 5 ResNet-50 baseline as frozen provenance, and never modifies UrbanMind-AI.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/janakmallik/UrbanGen-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/janakmallik/UrbanGen-AI/actions/workflows/ci.yml)

---

## The research question

> **Do class-conditional generative models trained from scratch on BDD100K produce synthetic residential images that measurably improve residential recall on the real BDD100K test set — and does any generative family beat conventional augmentation?**

The project tested three generative approaches:
1. **cVAE from scratch** (negative result)
2. **Stable Diffusion Img2Img** (failed structural preservation)
3. **CycleGAN from scratch** (successful structural preservation)

## Project status

**v1.0.0-final** — Research pilot. Closed as a single-seed study.

The project's central research question — whether synthetic urban data improves downstream recognition — has a partial answer: **yes, for nighttime residential recall specifically, at a documented cost to daytime precision.**

**Final results (seed 42):**
- **Arm A (real only):** 0.3605 night residential recall
- **Arm C (photometric aug):** 0.4150
- **Arm D (276 residential synthetic):** 0.5170
- **Arm D-138 (138 residential synthetic):** **0.5782** ← best pilot result
- **Arm D-bal (138 res + 138 mixed):** 0.4966

**The trade-off:** All synthetic arms violated the pre-registered daytime guardrail (>0.02 drop).

**Limitations:** Single seed only. Multi-seed confirmation, manual curation of D-bal, and independent dataset validation are documented as future work.

Full details: `docs/phase2_pilot_findings.md`, `docs/arm_d_findings.md`, `docs/claims.md`.

## Arm D result (qualified success, seed 42)

Adding 276 manually curated CycleGAN synthetic nighttime images to ResNet-50 training improved nighttime residential recall from **0.3605 → 0.5170** (paired bootstrap Δ=+0.158, 95% CI [+0.085, +0.229], excludes zero), a +43% relative improvement over the Arm A baseline. The improvement over Arm C (photometric augmentation, 0.4150 → 0.5170) is also statistically supported at Δ=+0.104.

**Trade-off:** Daytime macro-F1 dropped from 0.7440 (Arm C) to 0.7216 (Arm D) — a 0.0224 drop, violating the pre-registered 0.02 guardrail. The confusion matrix shows the classifier shifts toward residential predictions, which helps at night but creates false positives during the day (day city-street recall drops from 81.3% → 74.5%).

**Verdict:** Synthetic augmentation works for the specific problem it targets, at a real cost. Full details in `docs/arm_d_findings.md`. Multi-seed confirmation (seed 123) is the next step.

## The four-method plan (all tested)

| Method | Approach | Outcome |
|---|---|---|
| **Method 1** | SD 1.5 Img2Img (day→night) | ❌ Failed (hallucinated buildings, deleted signs) — see `docs/method_1_day_to_night_log.md` |
| **Method 2** | CycleGAN from scratch | ✅ **Successful** — see `docs/method_2_cyclegan_log.md` |
| **Method 3** | Depth ControlNet | ⏸️ Deferred after Arm D-M2 success (calibration complete, generation not attempted) |
| **Method 4** | WanGP I2V video generation | ✅ Showcase pipeline documented |

## Milestones

| Milestone | Tag | Outcome |
|---|---|---|
| **G0** — Data provenance | — | Cache, splits, checkpoint frozen and verified |
| **G1-pre** — 128×128 memmap cache | — | 3.59 GiB, aspect-preserving |
| **G1a** — cVAE model + tests | — | 34 tests, CI green |
| **G1b** — cVAE 30-epoch training | `v0.1.0-g1b-cvae` | Converged, 7.6× better than class-mean baseline |
| **G1c** — Pilot diagnostic | `v0.1.1-g1c-closed` | **Closed as negative**: 0.6% synthetic residential |
| **G1-final** — Close + Showcase | `v0.1.2-g1-final` | Diagnostic toolkit, corrected robustness report, I2V showcase |
| **G2** — Generative methods | `v0.1.3-g1-final` | Method 1 failed, Method 2 succeeded, Method 3 pending |
| **G2-ArmD** — Synthetic augmentation | `v0.2.0-g2-arm-d-seed42` | Qualified success at seed 42: night residential recall 0.36 → 0.52, day macro-F1 guardrail violated |
| **G2-Phase2** — Dose + class mix pilot | `v0.5.0-phase2-seed42-pilot` | D-138 (138 res) hits 0.578 night res recall; D-bal inconclusive; shortcut hypothesis unresolved |
| **v1.0.0-final** — Project close | `v1.0.0-final` | Research pilot closed. All claims committed and reproducible. |

## The cVAE finding (G1c — negative)

A conditional VAE trained correctly but failed to produce useful synthetic data:

- **Trains correctly.** Converged by epoch 25 (val loss 273.9), 49% reconstruction error reduction, KL stable, 128/128 active latent dims.
- **Produces diverse, non-memorized samples.** Pairwise cosine distance 0.365, 0/3000 above 0.95 nearest-neighbor similarity.
- **But fails to produce residential samples.** A real-trained classifier assigns **0.6% of 3,000 synthetic residential samples to residential**.

**Diagnosis:** representation mismatch. The cVAE at 128×128 with MSE reconstruction cannot produce class-faithful urban scenes.

**Files:** `scripts/train_cvae.py`, `scripts/diagnose_g1c.py`, `docs/model_cards/bdd100k_cvae_v1.md`

## Method 1 (SD 1.5 Img2Img — failed)

Tested Stable Diffusion 1.5 Img2Img at multiple strengths (0.25, 0.4, 0.6). All failed the structural preservation test:
- Hallucinated houses and cars that were not in the original
- Deleted highway signs and specific vehicles
- At strength=0.6, produced beautiful night images but lost all scene identity

**Verdict:** Cannot preserve scene geometry while changing lighting.

**Full log:** `docs/method_1_day_to_night_log.md`

## Method 2 (CycleGAN — successful)

Trained a from-scratch CycleGAN on 1,500 daytime + 1,500 nighttime residential images for 20 epochs with learning-rate decay. Results:

- **Structure:** Cars, buildings, and road signs **preserved exactly**
- **Appearance:** Convincing nighttime illumination, dark sky, streetlights
- **Purple artifacts:** Eliminated by epoch 20 through LR decay
- **Training time:** 57 minutes on RTX 4050

**Verdict:** Successfully learns the day-to-night mapping while preserving scene identity.

**Full log:** `docs/method_2_cyclegan_log.md`

## The pipeline-mismatch finding

The project's own robustness report initially reported residential recall 0.124. This was an artifact of evaluating on the wrong input pipeline.

**Corrected evaluation** (native JPEGs, `Resize(256) → CenterCrop(224) → ImageNet norm`) reproduces published Week 5 exactly:

| Condition | n | Accuracy | Macro-F1 | Residential recall |
|---|---|---:|---:|---:|
| (a) native JPEG, Resize(256)→CenterCrop(224) | 4,932 | **0.7800** | **0.6984** | **0.4121** |
| (a) clean test | 4,745 | 0.7819 | 0.6986 | 0.4078 |
| (b) native JPEG, resize to 224×224 (squashed) | 4,745 | 0.7966 | 0.7283 | 0.5178 |
| (c) 128×128 cache → 224×224 | 4,745 | 0.7503 | 0.5710 | 0.1235 |

**The pipeline mismatch reduced residential recall by 3.3× (0.41 → 0.12).** The corrected value is 0.408 at condition (a). Documented in `docs/lessons.md`.

## The corrected robustness finding

At condition (a), Week 5's robustness profile on the clean test set (n=4,745):

**By time of day:**

| Time | n | macro-F1 | Residential recall |
|---|---:|---:|---:|
| dawn/dusk | 367 | 0.722 | 0.532 |
| daytime | 2,447 | 0.725 | 0.476 |
| **night** | 1,931 | **0.610** | **0.184** |

**Within-class comparisons (unpaired bootstrap, exploratory):**

| Comparison | Class | Δ | 95% CI | n_A | n_B |
|---|---|---:|---|---:|---:|
| Clear → rainy | Highway | −0.235 | [−0.368, −0.106] | 740 | 63 ⚠️ |
| Daytime → night | Highway | −0.136 | [−0.188, −0.079] | 614 | 506 |
| Daytime → night | **Residential** | **−0.292** | **[−0.370, −0.212]** | 397 | 147 |

**Night is the dominant robustness weakness.** Residential recall drops 0.29 (0.476 → 0.184) at night vs daytime.

Full report: `docs/robustness_report_v2.md`

## The Week 5 recipe improvement (Arm A)

Retrained ResNet-50 with a modified recipe (class weights, IMAGENET1K_V2, best-epoch selection) achieved **night residential recall 0.3605** — nearly **2× the published baseline** of 0.184.

This finding contextualizes the whole project: the "0.184 baseline" that motivated the generative work was a weak recipe, not a fundamental limitation of the classifier.

**Full details:** `docs/week5_recipe_improvement.md`

## Generative Video Showcase (Wan 2.2 FastWan)

The project includes a working Image-to-Video pipeline (`scripts/generate_i2v_showcase.py`) using WanGP's internal `shared.api` to generate 9-second, 720p synthetic driving videos in ~1 minute on a 6 GB GPU.

Generated showcase videos are in `docs/assets/generated_examples/`. Full environment and reproduction: `docs/reproducibility.md`.

## The diagnostic toolkit

Three scripts answer "does a generative model produce useful synthetic data?" in ~30 minutes:

| Script | Purpose |
|---|---|
| `scripts/probe_pilot_features.py` | Frozen ResNet-50 features + logistic regression, four augmentation arms |
| `scripts/diagnose_g1c.py` | **The decisive test.** Real-only classifier on synthetic samples + NN grid + centroid distances |
| `scripts/verify_week5_pipeline.py` | Three-condition pipeline verification |

Plus supporting diagnostics:
- `scripts/audit_probe.py` — class-weight sensitivity
- `scripts/audit_residential_labels.py` — OOF label consistency
- `scripts/robustness_report_v2.py` — per-weather/per-time subgroup analysis
- `scripts/diagnose_blur_aware_evaluator.py` — matched-distribution classifier

## Reusable artifacts

- **`docs/robustness_report_v2.md`** — corrected Week 5 evaluation
- **`docs/claims.md`** — every quantitative claim with verified/provisional/retracted status
- **`docs/lessons.md`** — six diagnostic episodes
- **`docs/findings.md`** — one-page summary
- **`docs/reproducibility.md`** — WanGP I2V pipeline record
- **`docs/method_1_day_to_night_log.md`** — Method 1 failed (SD Img2Img)
- **`docs/method_2_cyclegan_log.md`** — Method 2 successful (CycleGAN)
- **`docs/week5_recipe_improvement.md`** — Arm A baseline finding
- **`docs/model_cards/bdd100k_cvae_v1.md`** — cVAE model card
- **`docs/model_cards/bdd100k_cyclegan_v1.md`** — CycleGAN model card
- **`docs/model_cards/wan_i2v_showcase_v1.md`** — WanGP I2V model card
- **`docs/dataset_cards/bdd100k_synthetic_cyclegan.md`** — Synthetic night dataset card
- **`src/urbangen/data/preprocessing.py`** — Shared Week 5 pipeline (importable by UrbanMind-AI Week 12)
- **`docs/assets/method_results/`** — comparison images from all methods

## The cross-project finding

UrbanMind-AI's Week 5 ResNet-50 is a **city-street detector**: 88% recall on city street, 72% on highway, **41% on residential** at native resolution. Night residential recall drops to 18%. Documented in the corrected robustness report and relevant to UrbanMind-AI's Week 12 serving layer (should flag residential and night predictions as "needs review").

## What this project is NOT

- **Not a claim that the cVAE approach works.** It failed.
- **Not a claim that Stable Diffusion cannot do day-to-night.** Method 1 tested SD 1.5 Img2Img specifically; ControlNet is Method 3 (not attempted).
- **Not a claim that CycleGAN solves the residential weakness without cost.** Arm D showed +0.158 night residential recall at seed 42, but with a documented daytime macro-F1 drop of 0.0224. The trade-off is real.
- **Not a claim that the shortcut hypothesis is resolved.** The D-bal pilot was inconclusive because the city/highway synthetic images were not manually curated.
- **Not a production system.** This is a single-seed research pilot, not a validated production classifier. Multi-seed variance and independent dataset validation are deferred.
- **Not deployed.** No API, no Docker, no serving. UrbanMind-AI's Week 12 is the natural home.

## Setup

### Requirements
- Python 3.12
- `uv` for dependency management
- CUDA-capable GPU with ≥ 6 GB VRAM (for training and inference; CI runs on CPU)
- Local BDD100K checkout and UrbanMind-AI checkout

### Installation

``````powershell
git clone https://github.com/janakmallik/UrbanGen-AI.git
cd UrbanGen-AI

uv sync --extra dev

Copy-Item .env.example .env
# Edit .env:
#   URBANINTEL_DATA_ROOT = parent directory containing bdd100k/
#   URBANMIND_REPO       = path to a local UrbanMind-AI checkout

# Verify the cache and provenance
uv run python scripts\audit_data_provenance.py
# expected: AUDIT PASS -- all checks green

# Run tests
uv run pytest -v
``````

### Reproduce the findings

``````powershell
# 1. Verify cache and provenance
uv run python scripts\audit_data_provenance.py

# 2. Verify Week 5 pipeline
uv run python scripts\verify_week5_pipeline.py

# 3. Corrected robustness report
uv run python scripts\robustness_report_v2.py

# 4. Reproduce cVAE training (30 min)
uv run python scripts\train_cvae.py --epochs 30

# 5. Reproduce the G1c negative finding (~15 min)
uv run python scripts\diagnose_g1c.py

# 6. Audit the residential label
uv run python scripts\audit_residential_labels.py

# 7. Train CycleGAN (Method 2, ~60 min)
uv run python scripts\method_2_cyclegan_resume.py
``````

## Project structure

``````text
UrbanGen-AI/
├── data/splits/                      # Frozen split manifests (committed)
├── docs/
│   ├── assets/method_results/        # Comparison images for all methods
│   ├── dataset_cards/                # Dataset provenance
│   ├── model_cards/                  # cVAE, CycleGAN, WanGP cards
│   ├── experiments.md                # Running journal
│   ├── claims.md                     # Claims ledger
│   ├── lessons.md                    # Postmortem
│   ├── findings.md                   # One-page summary
│   ├── reproducibility.md            # WanGP I2V record
│   ├── method_1_day_to_night_log.md  # Method 1 failed
│   ├── method_2_cyclegan_log.md      # Method 2 successful
│   ├── robustness_report_v2.md       # Corrected robustness
│   └── week5_recipe_improvement.md   # Arm A finding
├── scripts/                          # 20+ scripts (see file map)
├── src/urbangen/                     # Package source
└── tests/unit/                       # 40 tests
``````

## Lessons learned

Six diagnostic episodes. Full version in `docs/lessons.md`.

1. **Blur-confounded evaluator** — matched-distribution classifier fixes blur artifacts.
2. **Pipeline mismatch** — 3.3× residential recall drop from wrong input pipeline.
3. **Over-loose suspect rule** — flags difficulty, not label noise.
4. **Multiple comparisons** — 6 tests at nominal 95% have ~26% false positive chance.
5. **Class-weight sensitivity** — improvements that flip sign across settings are artifacts.
6. **WanGP integration trap** — use the official API, don't reverse-engineer.

## What future work should test

**Deferred experiments** (documented in `docs/phase2_pilot_findings.md`):

1. **Multi-seed confirmation** — rerun Arms A, C, D, D-138, D-bal at seeds 123 and 456 to measure training-seed variance.
2. **Manual curation of D-bal** — manually review the 397 city/highway accepted images to resolve the shortcut hypothesis without quality confounds.
3. **Arm B (real oversampling)** — compare synthetic data against simply duplicating real nighttime images.
4. **Daytime trade-off mitigation** — sampling ratios or fine-tuning to recover the 0.02 day macro-F1 cost.
5. **Independent validation** — test on ACDC or another nighttime dataset.

**Not recommended:**
- Additional generative architectures (CycleGAN is already the winner).
- Higher-resolution cVAE (representation mismatch is architectural, not resolution-limited).

## License

MIT — see [LICENSE](LICENSE).

## Citation

``````bibtex
@software{urbangen_2026,
  title   = {UrbanGen-AI: Diagnosing Whether From-Scratch Generative Models Produce Useful Synthetic Urban Data},
  author  = {Mallik, Janak},
  year    = {2026},
  url     = {https://github.com/janakmallik/UrbanGen-AI},
  note    = {Release v0.2.0-g2-arm-d-seed42. Consumes BDD100K splits from UrbanMind-AI v0.10.0-week10-calibration}
}
``````

## Acknowledgments

Built as a diagnostic-first investigation into generative-model augmentation for urban scene understanding. Uses BDD100K (Yu et al., CVPR 2020) and the Week 5 ResNet-50 baseline from UrbanMind-AI.
