# UrbanGen-AI

**Diagnosing whether from-scratch generative models produce useful synthetic urban data.**

UrbanGen-AI trained a conditional VAE on BDD100K scene images and asked whether its synthetic residential samples improve downstream recognition. It does not: the cVAE's samples are 0.6% residential in a real-trained classifier's judgment, despite being diverse, non-memorized, and blurry. This project documents that negative result alongside the reusable diagnostic toolkit that produced it.

It is a **separate project** from [UrbanMind-AI](https://github.com/janakmallik/UrbanMind-AI). UrbanMind-AI is the multimodal understanding platform; UrbanGen-AI is the generative-synthetic-data research project. UrbanGen-AI consumes UrbanMind-AI's published data splits and Week 5 ResNet-50 baseline as frozen provenance, and never modifies UrbanMind-AI.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/janakmallik/UrbanGen-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/janakmallik/UrbanGen-AI/actions/workflows/ci.yml)

---

## The research question

> **Do class-conditional generative models trained from scratch on BDD100K produce synthetic residential images that measurably improve residential recall on the real BDD100K test set — and does any generative family beat conventional augmentation?**

The pilot was designed with four arms (real-only, real+synthetic, real+conventional-augmentation, real+class-weighted) and a pre-registered success criterion. The pilot was not run to completion because two intermediate diagnostics showed the synthetic samples were not residential-like.

## What actually shipped

| Milestone | Tag | Status | Finding |
|---|---|---|---|
| **G0** | — | Shipped | Data provenance audit, 3.59 GiB cache, frozen splits and Week 5 checkpoint |
| **G1-pre** | — | Shipped | 128×128 uint8 memmap cache with strict integrity |
| **G1a** | — | Shipped | Conditional VAE model + dataset + tests |
| **G1b** | `v0.1.0-g1b-cvae` | Shipped | 30-epoch cVAE training, diagnostics, model card |
| **G1c** | `v0.1.1-g1c-closed` | **Closed (negative)** | Frozen-feature probe + audit + diagnostic: **synthetics are 0.6% residential** |
| **Step 0** | — | Shipped | Residential label audit + Week 5 robustness report |

The project's core contribution is the **diagnostic toolkit** — three scripts that answer "does a generative model produce useful synthetic data?" in ~30 minutes, before committing to a 2-hour fine-tune.

## The finding

A 30-epoch conditional VAE at 128×128 with 128-dim latent and MSE reconstruction:

- **Trains correctly.** Converges by epoch 25 (best val loss 273.9), 49% reduction in reconstruction error, no KL collapse, 128/128 active latent dims.
- **Produces diverse, non-memorized samples.** Pairwise cosine distance 0.365, 0/3000 above 0.95 nearest-neighbor similarity to real training images.
- **Fails to produce residential samples.** A logistic regression classifier trained on real images only assigns **0.6% of 3,000 synthetic residential samples to residential** (90% to highway, 9.4% to city street).
- **Is visually unrelated to the class.** See `outputs/g1c_diagnostic/nn_grid.png` — every synthetic is a blurred brown-gray gradient; the nearest real neighbors are sharp road scenes.

The frozen-feature probe showed a weak, class-weight-dependent improvement under `balanced` weighting (+0.013 residential F1, 5/5 folds), but this reversed under `neutral` weighting (−0.010) and did not survive the representation-mismatch diagnostic.

**Conclusion:** the cVAE at this scale (128×128, 128-dim latent, MSE) cannot generate class-faithful urban scenes. Blurry gradient samples that are classified 90% as highway are not useful augmentation for residential recognition. This is documented as `v0.1.1-g1c-closed`.

## The diagnostic toolkit

Three scripts answer "does a generative model produce useful synthetic data?" in ~30 minutes:

| Script | Purpose | Output |
|---|---|---|
| `scripts/probe_pilot_features.py` | Frozen ResNet-50 features + logistic regression head; compares augmentation arms in ~10 min | Per-arm F1 with class-weight sensitivity |
| `scripts/diagnose_g1c.py` | Real-only classifier on synthetic samples; nearest-neighbor grid; centroid distances | **The decisive "is it residential?" test** |
| `scripts/audit_residential_labels.py` | OOF recall and confusion matrix on the residential label; suspect/confident grids | **Is the class itself well-posed?** |

**Any future generative pipeline** (DDPM, ControlNet, GAN) can be tested with these before committing GPU hours to downstream training.

## The reused infrastructure

UrbanGen-AI shares the following with UrbanMind-AI:

- **Data:** `C:\Users\Janak Mallik\ml-data\bdd100k` (via `URBANINTEL_DATA_ROOT`)
- **Split manifests:** `data/splits/resnet_{val,test}_{clean,excluded}.json` (frozen from UrbanMind-AI commit `dfcfaf7`)
- **Cache:** `data/bdd100k_128/` — 128×128 uint8 memmap, verified by `scripts/precompute_bdd100k_128.py`
- **Week 5 classifier:** `bdd_resnet50_best.pt` from UrbanMind-AI, evaluated on 4,745 clean test images

## Reusable artifacts

- **`docs/robustness_report.md`** — Week 5 ResNet-50 macro-F1 by real BDD100K weather and time-of-day subgroups, with bootstrap CIs and cross-tabs
- **`docs/model_cards/bdd100k_cvae_v1.md`** — the cVAE's training, sample quality, memorization, diversity, and class conditioning
- **`outputs/label_audit/`** — OOF per-class recall, confusion matrix, suspect/confident residential grids, per-image probabilities
- **`outputs/g1c_diagnostic/nn_grid.png`** — 8 synthetics × 7 nearest real neighbors (residential and non-residential)

## The cross-project finding

The Week 5 ResNet-50 from UrbanMind-AI is a **city-street detector**: 95% recall on city, 56% on highway, **12% on residential** on the clean test split. This is consistent with all prior weak-residential-recall results (Week 5, Week 9a, G1b) and is now documented in the robustness report.

The residential label itself is boundary-ambiguous: out-of-fold logistic regression on ResNet-50 features achieves 0.70 residential recall, but ~30% of labeled residential images are confidently classified as city street. The class is fuzzy at the BDD100K definition, not mislabeled.

## What this project is NOT

- **Not a generative-model success.** The cVAE failed to produce class-faithful samples. G1c documents why.
- **Not a claim that all generative augmentation is useless.** DDPM, ControlNet, and image-to-image approaches were not tested. The diagnostic toolkit is what they can be tested with.
- **Not a claim that the residential class is unlearnable.** The label is fuzzy at the boundary, but the confident 70% are genuinely residential.
- **Not "trained from scratch"** for future work — pretrained diffusion approaches are a legitimate next direction, documented separately.

## How to read this repo

1. **Start with the research question** (above). Then read [`docs/experiments.md`](docs/experiments.md) for the scoping decision and G1c's diagnostic.
2. **Read the cVAE model card** at [`docs/model_cards/bdd100k_cvae_v1.md`](docs/model_cards/bdd100k_cvae_v1.md) for what the trained model can and cannot do.
3. **Read the robustness report** at [`docs/robustness_report.md`](docs/robustness_report.md) for the Week 5 model's per-weather and per-time performance.
4. **Inspect the diagnostic scripts** — `scripts/diagnose_g1c.py` and `scripts/probe_pilot_features.py` are the reusable toolkit.
5. **Look at `outputs/g1c_diagnostic/nn_grid.png`** — the visual that made the failure visible.

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

uv run python scripts\audit_data_provenance.py
# expected: AUDIT PASS -- all checks green

uv run pytest -v
```

`.env` is gitignored. Never commit your local paths.

## Reproduce the diagnostics

```powershell
# 1. Verify the cache
uv run python scripts\audit_data_provenance.py

# 2. Label audit (Step 0a) — 15 min
uv run python scripts\audit_residential_labels.py

# 3. Robustness report (Step 0b) — 5 min
uv run python scripts\robustness_report.py

# 4. Reproduce the cVAE training (30 min)
uv run python scripts\train_cvae.py --epochs 30

# 5. Reproduce the negative finding (~15 min)
uv run python scripts\diagnose_g1c.py
```

## Project structure

```text
UrbanGen-AI/
├── configs/
├── data/
│   └── splits/                 # Frozen split manifests
├── docs/
│   ├── dataset_cards/          # Dataset provenance
│   ├── model_cards/            # cVAE model card
│   ├── experiments.md          # Running journal (G0 through Step 0)
│   └── robustness_report.md    # Week 5 subgroup evaluation
├── scripts/
│   ├── audit_data_provenance.py
│   ├── audit_residential_labels.py
│   ├── robustness_report.py
│   ├── probe_pilot_features.py
│   ├── diagnose_g1c.py
│   ├── train_cvae.py
│   ├── generate_cvae_samples.py
│   ├── evaluate_cvae.py
│   ├── diagnose_class_conditioning.py
│   ├── diagnose_blur_aware_evaluator.py
│   ├── diagnose_classifier_pipeline.py
│   ├── precompute_bdd100k_128.py
│   └── profile_gpu_cvae.py
├── src/urbangen/
│   ├── config.py
│   ├── data/bdd100k_generative.py
│   ├── models/cvae/model.py
│   └── training/losses.py
├── tests/unit/                 # 34 tests
├── .env.example
├── pyproject.toml
└── README.md
```

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
  note    = {Release v0.1.1-g1c-closed. Consumes BDD100K splits from UrbanMind-AI v0.10.0-week10-calibration}
}
```

See `CITATION.cff` for the full reference.