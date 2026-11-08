# UrbanGen-AI

**Diagnosing whether from-scratch generative models produce useful synthetic urban data.**

UrbanGen-AI trained a conditional VAE on BDD100K scene images and asked whether its synthetic residential samples improve downstream recognition. It does not: the cVAE's samples are 0.6% residential in a real-trained classifier's judgment, despite being diverse, non-memorized, and blurry. The project documents that negative result alongside a reusable diagnostic toolkit, a corrected Week 5 robustness report, and a working image-to-video showcase pipeline.

It is a **separate project** from [UrbanMind-AI](https://github.com/janakmallik/UrbanMind-AI). UrbanMind-AI is the multimodal understanding platform; UrbanGen-AI is the generative-synthetic-data research project. UrbanGen-AI consumes UrbanMind-AI's published data splits and Week 5 ResNet-50 baseline as frozen provenance, and never modifies UrbanMind-AI.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/janakmallik/UrbanGen-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/janakmallik/UrbanGen-AI/actions/workflows/ci.yml)

---

## The research question

> **Do class-conditional generative models trained from scratch on BDD100K produce synthetic residential images that measurably improve residential recall on the real BDD100K test set — and does any generative family beat conventional augmentation?**

The pilot was designed with four arms (real-only, real+synthetic, real+conventional-augmentation, real+class-weighted) and a pre-registered success criterion. The pilot was not run to completion because the cVAE samples failed class-fidelity tests before the augmentation step could be meaningfully evaluated.

## Project status

**Complete.** Tagged `v0.1.2-g1-final`.

The project closes as a documented negative result with four contributions:

1. **A reusable diagnostic toolkit** (three scripts) for evaluating any generative pipeline's downstream utility before committing to a fine-tune.
2. **A corrected Week 5 robustness report** — verified to reproduce published numbers exactly, revealing night as the dominant robustness weakness.
3. **A documented pipeline-mismatch episode** with the reference-check pattern that caught it.
4. **A working Image-to-Video showcase pipeline** using WanGP and Wan2.2 FastWan 5B, proving that consumer hardware (6 GB VRAM) can generate high-quality synthetic nighttime video from real BDD100K images.

## Milestones

| Milestone | Tag | Outcome |
|---|---|---|
| **G0** — Data provenance | — | Cache, splits, checkpoint frozen and verified |
| **G1-pre** — 128×128 memmap cache | — | 3.59 GiB, aspect-preserving |
| **G1a** — cVAE model + tests | — | 34 tests, CI green |
| **G1b** — cVAE 30-epoch training | `v0.1.0-g1b-cvae` | Converged, 7.6× better than class-mean baseline |
| **G1c** — Pilot diagnostic | `v0.1.1-g1c-closed` | **Closed as negative**: 0.6% synthetic residential |
| **G1-final** — Close + Showcase | `v0.1.2-g1-final` | Negative findings documented, diagnostic toolkit shipped, I2V showcase pipeline working |

### Session 2 deliverables (close-out work)

| # | Deliverable | Artifact |
|---|---|---|
| D1 | Corrected robustness report | `docs/robustness_report_v2.md` |
| D2 | Golden check test | `tests/unit/test_week5_pipeline.py` |
| D3 | Shared preprocessing module | `src/urbangen/data/preprocessing.py` |
| D4 | Claims ledger | `docs/claims.md` |
| D5 | Lessons postmortem | `docs/lessons.md` |
| D6 | Findings page + README + tag | `docs/findings.md`, `README.md`, tag `v0.1.2-g1-final` |

## The main finding

A conditional VAE trained correctly but failed to produce useful synthetic data:

- **Trains correctly.** Converged by epoch 25 (val loss 273.9). Train reconstruction error drops 49% (0.0152 → 0.0078). KL stable at 76–88 (no posterior collapse). 128/128 active latent dimensions. 7.6× better than class-mean baseline (0.0068 vs 0.0592 per-pixel MSE).
- **Produces diverse, non-memorized samples.** Pairwise cosine distance 0.365 within class. 0/3000 above 0.95 nearest-neighbor similarity to real training images.
- **But fails to produce residential samples.** A logistic classifier trained only on real ResNet-50 features assigns **0.6% of 3,000 synthetic residential samples to residential** (90.0% to highway, 9.4% to city street). Visual inspection confirms: every synthetic is a blurred brown-gray gradient with no lane structure, buildings, or scene content.

**Diagnosis:** representation mismatch. The cVAE at 128×128 with 128-dim latent and MSE reconstruction produces samples that do not occupy the residential feature region. Not a training failure — a limit of the architecture and loss function at this resolution.

**Files:**
- `scripts/train_cvae.py` — training loop
- `scripts/diagnose_g1c.py` — the decisive "is it residential?" test
- `outputs/g1c_diagnostic/nn_grid.png` — 8 synthetics × 7 nearest real neighbors
- `docs/model_cards/bdd100k_cvae_v1.md` — the model card

## The pipeline-mismatch finding

The project's own robustness report initially reported residential recall 0.124 for the Week 5 model. This was an artifact of evaluating the model on the wrong input pipeline.

**Corrected evaluation** (native JPEGs, `Resize(256) → CenterCrop(224) → ImageNet norm`) reproduces published Week 5 exactly:

| Condition | n | Accuracy | Macro-F1 | Residential recall |
|---|---|---:|---:|---:|
| (a) native JPEG, Resize(256)→CenterCrop(224) | 4,932 | **0.7800** | **0.6984** | **0.4121** |
| (a) clean test | 4,745 | 0.7819 | 0.6986 | 0.4078 |
| (b) native JPEG, resize to 224×224 (squashed) | 4,745 | 0.7966 | 0.7283 | 0.5178 |
| (c) 128×128 cache → 224×224 | 4,745 | 0.7503 | 0.5710 | 0.1235 |

**The pipeline mismatch reduced residential recall by 3.3× (0.41 → 0.12).** The corrected value is 0.408 at condition (a). Condition (b) — squashed full frame — outperforms the training pipeline by +0.030 macro-F1 and +0.110 residential recall, but this was observed on test and is exploratory only.

The pipeline-mismatch episode is documented in `docs/lessons.md`.

## The corrected robustness finding

At condition (a), the Week 5 model's robustness profile on the clean test set (n=4,745):

**By weather:**

| Weather | n | macro-F1 | Residential recall |
|---|---:|---:|---:|
| clear | 2,627 | 0.685 | 0.372 |
| overcast | 582 | 0.738 | 0.526 |
| partly cloudy | 350 | 0.739 | 0.432 |
| **rainy** | 326 | **0.599** | **0.270** |
| snowy | 366 | 0.654 | 0.377 |
| undefined | 487 | 0.700 | 0.488 |
| foggy | 7 | 1.000 | 1.000 *(n<30, qualitative only)* |

**By time of day:**

| Time | n | macro-F1 | Residential recall |
|---|---:|---:|---:|
| dawn/dusk | 367 | 0.722 | 0.532 |
| daytime | 2,447 | 0.725 | 0.476 |
| **night** | 1,931 | **0.610** | **0.184** |

**Within-class comparisons (unpaired bootstrap, nominal 95% CI, exploratory and unadjusted):**

| Comparison | Class | Δ | 95% CI | n_A | n_B |
|---|---|---:|---|---:|---:|
| Clear → rainy | City street | +0.042 | [+0.006, +0.078] | 1,599 | 226 |
| Clear → rainy | **Highway** | **−0.235** | **[−0.368, −0.106]** | 740 | 63 ⚠️ |
| Clear → rainy | Residential | −0.101 | [−0.248, +0.051] | 288 | 37 |
| Daytime → night | City street | +0.043 | [+0.020, +0.066] | 1,436 | 1,278 |
| Daytime → night | Highway | −0.136 | [−0.188, −0.079] | 614 | 506 |
| Daytime → night | **Residential** | **−0.292** | **[−0.370, −0.212]** | 397 | 147 |

**Findings:**

1. **Night is the dominant robustness weakness.** Residential recall drops 0.29 (from 0.476 to 0.184) at night vs daytime, with class-specific n > 145 on both sides. Highway recall drops 0.14. Both statistically supported.
2. **Rainy highway is a real but underpowered weakness.** Δ = −0.235 with n_B=63 — the estimate is uncertain (flagged ⚠️ small n).
3. **Rainy city street is significantly *better* than clear** (Δ = +0.042). Wet roads + streetlights + urban density may form a recognizable pattern.

Full report: `docs/robustness_report_v2.md`. Machine-readable: `outputs/robustness_v2/report.json`.

## Generative Video Showcase (Wan 2.2 FastWan)

The project includes a working Image-to-Video pipeline (`scripts/generate_i2v_showcase.py`) that uses the WanGP engine (`shared.api`) to generate 9-second, 720p synthetic driving videos in ~1 minute on a 6 GB GPU.

| Method | Output | Cost | Label Preservation |
|---|---|---|---|
| WanGP / FastWan2.2 5B | Synthetic Video | ~1 min / video | Weak (requires filtering) |

The generated showcase videos are available in `docs/assets/generated_examples/`.

## The diagnostic toolkit

Three scripts answer "does a generative model produce useful synthetic data?" in ~30 minutes, before committing to a 2-hour fine-tune:

| Script | Purpose | Runtime |
|---|---|---|
| `scripts/probe_pilot_features.py` | Frozen ResNet-50 features + logistic regression head. Compares augmentation arms (real-only / +synthetic / +oversampled) with class-weight sensitivity. | ~10 min |
| `scripts/diagnose_g1c.py` | **The decisive test.** Trains a real-only classifier, runs it on synthetic samples, reports what fraction are classified as the requested class. Plus a nearest-neighbor grid (synthetic vs 3 nearest real residential vs 3 nearest real non-residential) and feature-space centroid distances. | ~10 min |
| `scripts/verify_week5_pipeline.py` | Three-condition pipeline verification. Reproduces published Week 5 numbers on the correct pipeline, isolates resolution/crop effects. | ~15 min |

Plus four supporting diagnostics:

- `scripts/audit_probe.py` — class-weight sensitivity across four settings
- `scripts/audit_residential_labels.py` — OOF label consistency and grids
- `scripts/robustness_report_v2.py` — per-weather / per-time subgroup analysis
- `scripts/diagnose_blur_aware_evaluator.py` — matched-distribution classifier (fixes blur-confounded evaluations)

**Any future generative pipeline** (DDPM, ControlNet, GAN, img2img) can be tested with these in ~30 min before committing to any downstream training.

## The reused infrastructure

UrbanGen-AI shares the following with UrbanMind-AI:

- **Data:** `C:\Users\Janak Mallik\ml-data\bdd100k` (via `URBANINTEL_DATA_ROOT`)
- **Split manifests:** `data/splits/resnet_{val,test}_{clean,excluded}.json` (frozen from UrbanMind-AI commit `dfcfaf7`)
- **Cache:** `data/bdd100k_128/` — 128×128 uint8 memmap, verified by `scripts/precompute_bdd100k_128.py`
- **Week 5 classifier:** `bdd_resnet50_best.pt` from UrbanMind-AI, evaluated on 4,745 clean test images

## Reusable artifacts

- **`docs/robustness_report_v2.md`** — corrected Week 5 evaluation, per-weather and per-time subgroups, within-class unpaired bootstrap
- **`docs/claims.md`** — every quantitative claim with evidence file and verified/provisional/retracted status
- **`docs/lessons.md`** — six diagnostic episodes and the discipline that caught them
- **`docs/findings.md`** — one-page summary of the project's negative findings and contributions
- **`docs/reproducibility.md`** — WanGP Image-to-Video pipeline reproducibility record
- **`docs/model_cards/bdd100k_cvae_v1.md`** — cVAE training, sample quality, memorization, diversity, class conditioning
- **`src/urbangen/data/preprocessing.py`** — shared Week 5 preprocessing (importable by UrbanMind-AI Week 12 to prevent train-serve skew)
- **`outputs/label_audit/`** — OOF per-class recall, confusion matrix, suspect/confident residential grids
- **`outputs/g1c_diagnostic/nn_grid.png`** — 8 synthetics × 7 nearest real neighbors

## The cross-project finding

UrbanMind-AI's Week 5 ResNet-50 is a **city-street detector**: 88% recall on city street, 72% on highway, **41% on residential** at native resolution. Night residential recall drops to 18%. These are now documented in the corrected robustness report and are relevant to UrbanMind-AI's Week 12 serving layer (which should flag residential and night predictions as "needs review").

## What this project is NOT

- **Not a generative success.** The cVAE failed to produce class-faithful samples.
- **Not a claim that all generative augmentation is useless.** DDPM, ControlNet, and image-to-image approaches were not tested. The diagnostic toolkit is what they can be tested with.
- **Not a claim that the residential class is unlearnable.** The label is consistent; the Week 5 model is weak (residential recall 0.41 at native resolution).
- **Not a claim that Week 5 is broken.** It reproduces published numbers exactly, with documented night and residential weaknesses.
- **Not "trained from scratch"** for future work. Pretrained diffusion approaches are a legitimate next direction, documented separately.
- **Not deployed.** No API, no Docker, no serving. UrbanMind-AI's Week 12 is the natural home for serving work.

## Setup

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- CUDA-capable GPU with ≥ 6 GB VRAM (for training and inference; CI runs on CPU)
- A local BDD100K checkout and a local UrbanMind-AI checkout

### Installation

```powershell
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
```

`.env` is gitignored. Never commit your local paths.

## Reproduce the findings

```powershell
# 1. Verify cache and provenance
uv run python scripts\audit_data_provenance.py

# 2. Verify Week 5 pipeline (reproduces published numbers exactly)
uv run python scripts\verify_week5_pipeline.py

# 3. Corrected robustness report (per-weather, per-time, within-class)
uv run python scripts\robustness_report_v2.py

# 4. Reproduce cVAE training (30 min)
uv run python scripts\train_cvae.py --epochs 30

# 5. Reproduce the G1c negative finding (~15 min)
uv run python scripts\diagnose_g1c.py

# 6. Audit the residential label
uv run python scripts\audit_residential_labels.py
```

## Project structure

```text
UrbanGen-AI/
├── configs/
├── data/
│   └── splits/                 # Frozen split manifests (committed)
├── docs/
│   ├── dataset_cards/          # Dataset provenance
│   ├── model_cards/            # cVAE model card
│   ├── experiments.md          # Running journal (G0 through Step 1)
│   ├── robustness_report.md    # SUPERSEDED (retained with banner)
│   ├── robustness_report_v2.md # Corrected report
│   ├── claims.md               # Claims ledger
│   ├── lessons.md              # Postmortem
│   ├── findings.md             # One-page summary
│   ├── reproducibility.md      # WanGP I2V pipeline record
│   └── assets/
│       └── generated_examples/ # Showcase videos
├── scripts/
│   ├── audit_data_provenance.py
│   ├── audit_residential_labels.py
│   ├── robustness_report.py    # Original (superseded)
│   ├── robustness_report_v2.py # Corrected
│   ├── verify_week5_pipeline.py
│   ├── probe_pilot_features.py
│   ├── audit_probe.py
│   ├── diagnose_g1c.py
│   ├── diagnose_class_conditioning.py
│   ├── diagnose_blur_aware_evaluator.py
│   ├── precompute_bdd100k_128.py
│   ├── train_cvae.py
│   ├── generate_cvae_samples.py
│   ├── evaluate_cvae.py
│   ├── check_class_mapping.py
│   ├── profile_gpu_cvae.py
│   └── generate_i2v_showcase.py
├── src/urbangen/
│   ├── config.py
│   ├── data/
│   │   ├── bdd100k_generative.py
│   │   └── preprocessing.py    # Shared Week 5 pipeline
│   ├── models/cvae/
│   └── training/losses.py
├── tests/unit/                 # Golden check + unit tests
├── .env.example
├── pyproject.toml
└── README.md
```

## Lessons learned

Six diagnostic episodes and how each was caught. Full version in `docs/lessons.md`.

1. **Blur-confounded evaluator** — a classifier trained on sharp images fails on blurry reconstructions of the same content. Fixed by a matched blur-aware evaluator.
2. **Pipeline mismatch** — evaluating on the wrong input pipeline silently degraded residential recall 3.3×. Fixed by a three-condition verification against published numbers.
3. **Over-loose suspect rule** — a rule that flags 36% of city images and 30% of residential images measures difficulty, not label noise. Fixed by tighter thresholds and visual inspection.
4. **Multiple comparisons** — six within-class tests at nominal 95% have ~26% chance of at least one false positive. Labeled exploratory and unadjusted.
5. **Class-weight sensitivity** — an apparent improvement that flips sign across hyperparameter settings is not a real effect. Fixed by testing under multiple settings.
6. **WanGP integration trap** — attempting to run WanGP via `diffusers`, Gradio API hacks, or reverse-engineering `queue.json` failed. The correct solution was to import WanGP's internal `shared.api` directly, which preserves the `mmgp` memory engine. Always look for the library's official entry point before building custom wrappers.

## What future work should test

1. **Night augmentation.** Night is the dominant weakness for Week 5 (residential recall 0.48 → 0.18). Simple tone-curve + ColorJitter augmentation, or a pretrained img2img model, could be tested on val first, then confirmed on test. **UrbanMind-AI scope.**
2. **Residential redefinition.** A binary "urban scene (city+residential) vs highway" task might be well-posed where the three-class task is not. **UrbanMind-AI scope.**
3. **Object-level features.** Week 9a's segmentation + detection features improved residential recall from 0.41 to 0.67. Non-generative path worth investigating. **UrbanMind-AI scope.**
4. **DDPM at higher resolution.** 128×128 is a hard limit for the cVAE. Diffusion at 128×128 or higher may produce class-faithful samples — testable with `scripts/diagnose_g1c.py` and `scripts/probe_pilot_features.py`. **Future project.**

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you reference this project, cite BDD100K (Yu et al., CVPR 2020) as the dataset, and this repository by its tag for the software artifact:

```bibtex
@software{urbangen_2026,
  title   = {UrbanGen-AI: Diagnosing Whether From-Scratch Generative Models Produce Useful Synthetic Urban Data},
  author  = {Mallik, Janak},
  year    = {2026},
  url     = {https://github.com/janakmallik/UrbanGen-AI},
  note    = {Release v0.1.2-g1-final. Consumes BDD100K splits from UrbanMind-AI v0.10.0-week10-calibration}
}
```

See `CITATION.cff` for the full reference.

## Acknowledgments

Built as a diagnostic-first investigation into generative-model augmentation for urban scene understanding. Uses BDD100K (Yu et al., CVPR 2020) and the Week 5 ResNet-50 baseline from UrbanMind-AI.
