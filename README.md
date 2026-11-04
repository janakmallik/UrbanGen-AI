# UrbanGen-AI

**Class-conditional generative models trained from scratch on urban scenes, evaluated by whether their synthetic data measurably improves downstream recognition.**

UrbanGen-AI trains three generative architectures (cVAE, conditional DDPM in latent space, conditional DCGAN) on BDD100K scene images, then tests whether the residential-scene images they synthesize improve residential recall on the real BDD100K test set — compared to real-data-only training and to conventional augmentation.

It is a **separate project** from [UrbanMind-AI](https://github.com/janakmallik/UrbanMind-AI). UrbanMind-AI is the multimodal understanding platform; UrbanGen-AI is the generative-synthetic-data research project. UrbanGen-AI consumes UrbanMind-AI's published data splits and Week 5 ResNet-50 baseline as frozen provenance, and never modifies UrbanMind-AI.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/janakmallik/UrbanGen-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/janakmallik/UrbanGen-AI/actions/workflows/ci.yml)

---

## The research question

> **Do class-conditional generative models trained from scratch produce synthetic residential-scene images that measurably improve residential recall on the real BDD100K test set — and does any generative family beat conventional augmentation?**

**Why residential?** UrbanMind-AI's Week 5 ResNet-50 (macro-F1 0.6984) identified residential as the weakest class. Residential is only 11.71% of BDD100K train. If generative augmentation can help any class, it should be this one.

**The five downstream arms** (each trains a fresh ResNet-50 on the same real train split, differing only in what synthetic data is added):

| Arm | Training data | Purpose |
|---|---|---|
| **A** | Real BDD100K train only | Baseline |
| **B** | Real + conventional augmentation (crop/flip/jitter) | Control — does synthetic beat trivial augmentation? |
| **C** | Real + cVAE synthetic residential | VAE arm |
| **D** | Real + DDPM synthetic residential | Diffusion arm |
| **E** | Real + DCGAN synthetic residential | GAN arm |

**Pre-stated success criterion (frozen — not movable after results are seen):** a generative arm "works" if residential recall improves by ≥ +0.03 with a 95% bootstrap CI excluding zero, **and** macro-F1 does not degrade by more than −0.01. Evaluated on the real clean BDD100K test split (4,745 images).

**A negative result is a valid outcome.** If no generative arm beats arm A, the finding is: *"class-conditional generative models at this scale do not produce downstream-useful synthetic data for this task."* That is a real, documented research result — the same discipline UrbanMind-AI applied to its fusion negative results.

---

## Project status — G0 complete, G1 next

| Milestone | Deliverable | Status |
|---|---|---|
| **G0** | Project setup, data provenance audit, scoping decision | ✅ Complete (CI green) |
| **G1** | Conditional VAE training + sample quality | Next |
| **G2** | Conditional DDPM in latent space + sample quality | Not started |
| **G3** | Conditional DCGAN + mode-collapse analysis | Not started |
| **G4** | Shared generative evaluation framework (FID/KID, memorization, class fidelity) | Not started |
| **G5** | Five-arm downstream ResNet-50 experiment + bootstrap CIs | Not started |
| **G6** | Comparative report + `v0.1.0-urbangen` release tag | Not started |

---

## What G0 established

The foundational discipline: **no generative model code before data and checkpoint provenance is verified.** G0's audit (`scripts/audit_data_provenance.py`) verifies:

- BDD100K label JSON SHA-256 (`c1998792…` train, `c835be65…` val)
- The four `resnet_*` split manifests partition the 9,864-image validation pool into four disjoint sets (verified partition integrity)
- Class counts (train: 43,516 city street / 17,379 highway / 8,074 residential = 68,969)
- Week 5 ResNet-50 checkpoint SHA-256 (`f760b94c…`), structure (plain state_dict, 320 tensors, `fc` (3, 2048)), and loadability

Any future drift in manifests or checkpoint fails the audit and halts the experiment. The full provenance record is in [`docs/dataset_cards/bdd100k_generative.md`](docs/dataset_cards/bdd100k_generative.md). The scoping decision (research question, five arms, success criterion) is frozen in [`docs/experiments.md`](docs/experiments.md).

**Not claimed:** historical reproduction of UrbanMind-AI's Week 5 `train_test_split(seed=42)` output, or reproduction of its 0.6984 test macro-F1. The manifests are consumed as frozen artifacts; the checkpoint is reused but its historical F1 is a reference value, not re-derived by UrbanGen-AI.

---

## Generative models — results

Results will be populated as each milestone ships. The table is deliberately honest: `[not yet run]` rows are placeholders, not claims.

| Model | Dataset | Status | FID | Sample quality | Model card |
|---|---|---|---|---|---|
| **cVAE** (conditional, 64×64) | BDD100K train | [G1 — not yet run] | — | — | — |
| **Conditional DDPM** (latent space) | BDD100K train + cVAE latents | [G2 — not yet run] | — | — | — |
| **Conditional DCGAN** (64×64) | BDD100K train | [G3 — not yet run] | — | — | — |

---

## Downstream augmentation experiment

The five-arm ResNet-50 comparison (G5). Primary metric: residential recall. Guardrail: macro-F1 not degrading by more than 0.01. Bootstrap CIs (1,000 paired resamples).

| Arm | Residential recall | Δ vs A | 95% CI | Macro-F1 | Δ vs A |
|---|---|---|---|---|---|
| A — Real only | [G5 — not yet run] | — | — | — | — |
| B — Conventional aug | — | — | — | — | — |
| C — cVAE aug | — | — | — | — | — |
| D — DDPM aug | — | — | — | — | — |
| E — DCGAN aug | — | — | — | — | — |

---

## Provenance (frozen)

UrbanGen-AI consumes the following published artifacts from UrbanMind-AI at commit `dfcfaf7` (tag `v0.10.0-week10-calibration`):

| Artifact | SHA-256 (prefix) | Role |
|---|---|---|
| `bdd100k_labels_images_train.json` | `c1998792…` | Source of the 68,969 training images |
| `bdd100k_labels_images_val.json` | `c835be65…` | Source of val/test partition |
| `resnet_val_clean.json` (4,727 entries) | `9ee1b6c6…` | Validation split |
| `resnet_test_clean.json` (4,745 entries) | `ca47e16a…` | **Final test split** — never used for training or model selection |
| `resnet_val_excluded.json` (205) | `05d54fda…` | Contamination record |
| `resnet_test_excluded.json` (187) | `ec48bc01…` | Contamination record |
| `bdd_resnet50_best.pt` | `f760b94c…` | Week 5 ResNet-50 baseline (arm A) |

Full hashes, split semantics, and the documented class-distribution discrepancy vs UrbanMind-AI's Week 5 card are in [`docs/dataset_cards/bdd100k_generative.md`](docs/dataset_cards/bdd100k_generative.md).

**The manifests are frozen inputs.** Their partition integrity is verified; historical reproduction of the Week 5 seed-42 `train_test_split` is out of scope.

---

## Setup

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- CUDA-capable GPU with ≥ 6 GB VRAM (for generative training; CI runs on CPU)
- A local BDD100K checkout and a local UrbanMind-AI checkout (provenance is consumed from there)

### Installation

```powershell
git clone https://github.com/janakmallik/UrbanGen-AI.git
cd UrbanGen-AI

# Install dependencies (uses CPU torch on CI; CUDA torch on a GPU machine via the pinned index)
uv sync --extra dev

# Configure local paths
Copy-Item .env.example .env
# Edit .env:
#   URBANINTEL_DATA_ROOT = parent directory containing bdd100k/
#   URBANMIND_REPO       = path to a local UrbanMind-AI checkout

# Verify setup
uv run python scripts\audit_data_provenance.py
# expected: AUDIT PASS -- all checks green

# Run unit tests
uv run pytest -v
```

`.env` is gitignored. Never commit your local paths.

---

## How to read this repo

1. **Start with the research question** (above). If the question doesn't interest you, the rest won't either.
2. **Read the scoping decision** in [`docs/experiments.md`](docs/experiments.md) — the G0 entry has the five-arm design, pre-stated success criterion, and out-of-scope list.
3. **Read the data provenance record** in [`docs/dataset_cards/bdd100k_generative.md`](docs/dataset_cards/bdd100k_generative.md) — what's frozen, what's verified, what's not claimed.
4. **Read one model card** (once G1 ships) — the cVAE card will be the first.
5. **Look at `scripts/audit_data_provenance.py`** — eight sections, all fatal on failure. It's the foundation.

The design notes in [`CONCEPTS.md`](CONCEPTS.md) cover the generative-model concepts this project exercises (VAEs, diffusion, GANs, mixed precision, etc.) in plain language.

---

## What this project is NOT

- **Not a Stable Diffusion project.** No pretrained SD fine-tuning, no Stable Diffusion from scratch. All three generative models are trained on BDD100K from random initialization.
- **Not a claim that generative augmentation improves urban perception.** G5 tests this; the outcome may be negative.
- **Not a replacement for UrbanMind-AI.** It consumes UrbanMind-AI's published artifacts; it never modifies them.
- **Not a claim about arbitrary urban scenes.** BDD100K is dashcam footage from a small set of cities. Generative models trained on it reflect that distribution.
- **Not deployed.** No API, no Docker, no serving. Generation happens offline.

---

## Project structure

```text
UrbanGen-AI/
├── configs/                    # YAML configs (data, models, experiments)
├── data/
│   └── splits/                 # Frozen split manifests (committed)
├── docs/
│   ├── dataset_cards/          # Dataset provenance
│   ├── model_cards/            # One per trained generative model
│   └── experiments.md          # Running journal
├── scripts/
│   ├── audit_data_provenance.py
│   └── (train_*.py, generate_*.py as milestones land)
├── src/urbangen/
│   ├── config.py               # Env-var config (reads .env)
│   ├── data/                   # BDD100K generative loader
│   ├── models/
│   │   ├── cvae/               # [G1]
│   │   ├── diffusion/          # [G2]
│   │   └── dcgan/              # [G3]
│   ├── training/
│   ├── generation/
│   ├── evaluation/             # [G4]
│   └── experiments/
├── tests/unit/                 # 3 smoke tests at G0
├── .env.example
├── pyproject.toml
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

### The dataset

BDD100K is the source dataset and should be cited directly:

```bibtex
@inproceedings{yu2020bdd100k,
  title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

### This project

UrbanGen-AI is not a peer-reviewed publication and does not make priority claims. If you use these models, samples, or results, please reference the project by its **release tag**, which pins a specific, reproducible state:

```bibtex
@software{urbangen_2026,
  title   = {UrbanGen-AI: Class-Conditional Generative Models for Urban Scene Augmentation},
  author  = {Mallik, Janak},
  year    = {2026},
  url     = {https://github.com/janakmallik/UrbanGen-AI},
  note    = {Release <tag>, commit <sha>. Consumes BDD100K splits from
             UrbanMind-AI v0.10.0-week10-calibration}
}
```

The `<tag>` and `<sha>` placeholders fill in at G6 when `v0.1.0-urbangen` ships. Until then, cite by repository URL + current `main` branch.

**For exact artifact provenance:**
- Checkpoint SHA-256s are in each model card.
- Split manifest SHA-256s are in `docs/dataset_cards/bdd100k_generative.md`.
- Full provenance chain is recorded in `docs/experiments.md`.
- A `CITATION.cff` file at repo root provides the "Cite this repository" button on GitHub.

### What this project does NOT claim

- **Not a peer-reviewed result.** No methods novelty claim beyond the integration and evaluation approach.
- **Not a priority claim.** If you independently train a cVAE on BDD100K, that's fine — this project isn't claiming to have done it first.
- **Not a result about urban perception in general.** All claims are scoped to BDD100K scene classification under the five-arm protocol in `docs/experiments.md`.

**If you want to formally cite it in an academic paper, the right reference is the repository tag, not a formal BibTeX entry.** Reviewers understand "GitHub repository, tag vX.Y.Z" as a legitimate reference for software artifacts — the [Software Heritage](https://www.softwareheritage.org/) archive and Zenodo DOI-on-release patterns exist for exactly this. Adding a `CITATION.cff` file (see below) makes it automatic.