# UrbanGen-AI

**Synthetic urban-scene generation via class-conditional generative models, evaluated against real downstream recognition.**

UrbanGen-AI trains class-conditional generative models (cVAE, conditional DDPM in latent space, conditional DCGAN) from scratch on BDD100K scene images, then tests whether the synthetic residential-scene images they produce improve residential recall on the real BDD100K test set. It consumes [UrbanMind-AI](https://github.com/janakmallik/UrbanMind-AI)'s published data split manifests and Week 5 ResNet-50 baseline as frozen provenance — it does not modify UrbanMind-AI.

## What it is

A generative-modeling research project. The central question is: **can class-conditional generative models produce synthetic urban-scene images that measurably improve downstream urban-scene recognition, or does conventional augmentation beat them?**

Three generative models trained from scratch:
- Conditional VAE (cVAE) — variational inference, latent representation, class conditioning
- Conditional DDPM — diffusion in the cVAE's latent space (no pretrained Stable Diffusion)
- Conditional DCGAN — adversarial generation, mode collapse analysis

Five downstream ResNet-50 training arms:
- A: real BDD100K train only (baseline)
- B: real + conventional augmentation (crop/flip/jitter)
- C: real + cVAE synthetic residential images
- D: real + DDPM synthetic residential images
- E: real + DCGAN synthetic residential images

Success is measured by residential recall improvement on the real BDD100K test set with 95% bootstrap CIs excluding zero, and macro-F1 not degrading by more than 0.01. A negative result (no generative arm beats conventional augmentation) is a documented outcome, not a failure.

## Setup

UrbanGen-AI reads paths from environment variables, not from committed code.

```powershell
# 1. Clone the repo
git clone https://github.com/janakmallik/UrbanGen-AI.git
cd UrbanGen-AI

# 2. Install dependencies
uv sync --extra dev

# 3. Configure local paths
Copy-Item .env.example .env
# Edit .env:
#   URBANINTEL_DATA_ROOT = parent directory containing bdd100k/
#   URBANMIND_REPO       = path to a local UrbanMind-AI checkout

# 4. Verify setup
uv run python scripts\audit_data_provenance.py
# expected: AUDIT PASS
```

## Provenance

This project consumes the following published artifacts from UrbanMind-AI at commit `dfcfaf7` (tag `v0.10.0-week10-calibration`):

- BDD100K scene split manifests (`data/splits/resnet_{val,test}_{clean,excluded}.json`):
  - `resnet_val_clean.json` — 4,727 entries, SHA-256 `9ee1b6c6...`
  - `resnet_val_excluded.json` — 205 entries, SHA-256 `05d54fda...`
  - `resnet_test_clean.json` — 4,745 entries, SHA-256 `ca47e16a...`
  - `resnet_test_excluded.json` — 187 entries, SHA-256 `ec48bc01...`
- Week 5 ResNet-50 checkpoint: `bdd_resnet50_best.pt`
  SHA-256: `F760B94C4BB334C95C5D6C3C7884534FE6F51CE397641249610AAB1A98447F84`
- Week 5 evaluation protocol (bootstrap CIs, per-class recall)

The four manifests are consumed as **frozen inputs**. Their partition of the 9,864-image 3-class validation pool into four disjoint sets is verified by `scripts/audit_data_provenance.py`. The historical Week 5 `train_test_split(seed=42, stratify=label_id)` output is **not** independently reconstructed.

See `docs/dataset_cards/bdd100k_generative.md` for the full frozen provenance record and the documented class-distribution discrepancy vs UrbanMind-AI's Week 5 dataset card.

## Status

Week G0 (setup and scoping) — in progress.

## License

MIT — see [LICENSE](LICENSE).