# Dataset Card: BDD100K (Generative Use)

## Identity

| Field | Value |
|---|---|
| Source | BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning (Yu et al., CVPR 2020) |
| Official site | https://bdd-data.berkeley.edu/ |
| Local data root | `$URBANINTEL_DATA_ROOT/bdd100k` |
| Local images path | `$URBANINTEL_DATA_ROOT/bdd100k/images/` (100,000 unique filenames, verified) |
| Local labels path | `$URBANINTEL_DATA_ROOT/bdd100k/labels/` |
| License | UC Berkeley Regents License (research use) |

## Purpose

This is UrbanGen-AI's frozen BDD100K provenance card. It is a **distinct** card from UrbanMind-AI's primary BDD100K scene-classification card: that card describes the Week 5 ResNet-50 classification use; this card describes what UrbanGen-AI consumes to **train generative models and evaluate downstream augmentation**.

UrbanGen-AI consumes BDD100K from a **shared local directory**. It does not re-download, re-split, or re-label the dataset. All images and labels are exactly those used by UrbanMind-AI's Week 5 ResNet-50 training and evaluation.

## Source provenance (frozen)

| Field | Value |
|---|---|
| Source project | UrbanMind-AI |
| Source commit | `dfcfaf7` |
| Source tag | `v0.10.0-week10-calibration` |
| Source repo URL | https://github.com/janakmallik/UrbanMind-AI |

## Source label JSONs (frozen SHA-256)

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `bdd100k_labels_images_train.json` | 1,451,202,234 | `c1998792e385fd79213187576ca07d5249563d0ab83ae976688ad60668baca9e` |
| `bdd100k_labels_images_val.json` | 208,257,864 | `c835be652f0c002897be51e74deedd16a1dcee0b7230f19c0456fba772f694af` |

## Split construction (verified)

UrbanGen-AI uses **Option B** — the four `resnet_*` manifests are treated as frozen inputs. The historical Week 5 `train_test_split(seed=42, stratify=label_id)` output is **not** independently reconstructed. The audit verifies that the manifests partition the 3-class validation pool into four disjoint sets with all entries present in `bdd100k_labels_images_val.json`.

| Split | Source | Count | Role in UrbanGen-AI |
|---|---|---:|---|
| **Train** | `bdd100k_labels_images_train.json` filtered to 3 scene classes | 68,969 | Training data for all five downstream arms; **the cVAE and DDPM are trained on this split only** |
| **Val** | `resnet_val_clean.json` | 4,727 | Model selection, checkpoint selection, generation-quality evaluation |
| **Test** | `resnet_test_clean.json` | 4,745 | **Final downstream evaluation only** |
| Val excluded | `resnet_val_excluded.json` | 205 | Contamination record (SegFormer train overlap) — not used for training or evaluation |
| Test excluded | `resnet_test_excluded.json` | 187 | Contamination record — not used for training or evaluation |

**Note on historical reconstruction:** the four manifests cleanly partition the 9,864-image 3-class validation pool into four disjoint sets (`|val_clean ∪ val_excluded| = 4,932`, `|test_clean ∪ test_excluded| = 4,932`, `|all four| = 9,864`, all pairwise intersections zero). They are consumed as frozen artifacts. Independent reconstruction of the exact Week 5 seed-42 split is out of scope for this audit and is not claimed.

## Frozen manifest SHA-256 (verified against UrbanMind-AI source)

| File | Entries | SHA-256 |
|---|---:|---|
| `resnet_val_clean.json` | 4,727 | `9ee1b6c69f4e58e83edb704fbd8d42b3972a9b0410aa838a939143205f924682` |
| `resnet_val_excluded.json` | 205 | `05d54fda00e07083d73f2114607636a6c4056802b5bfd5f5614de2c79d0c4760` |
| `resnet_test_clean.json` | 4,745 | `ca47e16a4c6148520b2a0cc3de74b78db20392507c93ea4d5a4039c695d03979` |
| `resnet_test_excluded.json` | 187 | `ec48bc0166145448e726bb277a7e62084deb727a457424f38f4184e1a6d9daea` |

Local copies live in `data/splits/`. Any mismatch between local and source hashes halts the experiment.

## Week 5 ResNet-50 baseline (arm A)

UrbanMind-AI's Week 5 fine-tuned ResNet-50 is the **real-only baseline** for UrbanGen-AI's downstream augmentation experiment (arm A).

| Field | Value |
|---|---|
| Path | `$URBANMIND_REPO/bdd_resnet50_best.pt` |
| Size | 94,367,514 bytes |
| SHA-256 | `f760b94c4bb334c95c5d6c3c7884534fe6f51ce397641249610aab1a98447f84` |
| Format | Plain `state_dict` (320 parameter tensors, all values are tensors) |
| Architecture | ResNet-50, `fc` layer (3, 2048) — 3-class scene classification |
| Total parameters | 23,514,179 |
| Reported test macro-F1 | 0.6984 (UrbanMind-AI Week 5) |

**Provenance caveat:** this checkpoint's test-set macro-F1 (0.6984) was reported by UrbanMind-AI on the clean test split (4,745 images). UrbanGen-AI reuses the checkpoint but does not re-run the Week 5 evaluation. Any UrbanGen experiment that reports a "baseline" number must be re-derived from an UrbanGen evaluation of this checkpoint on the clean test split; the 0.6984 value is the historical reference only.

## Class distribution (verified from source label JSONs)

### Train (68,969 images in 3 target classes)

| Class | Count | Share |
|---|---:|---:|
| city street | 43,516 | 63.10% |
| highway | 17,379 | 25.20% |
| residential | 8,074 | 11.71% |
| **Total** | **68,969** | 100% |

### Val pool (9,864 images in 3 target classes, from `bdd100k_labels_images_val.json`)

| Class | Count | Share |
|---|---:|---:|
| city street | 6,112 | 61.96% |
| highway | 2,499 | 25.33% |
| residential | 1,253 | 12.70% |
| **Total** | **9,864** | 100% |

**Note on UrbanMind-AI's Week 5 dataset card:** that card reports train class shares of 62.3% / 24.9% / 11.6%. The verified counts above recomputed from `bdd100k_labels_images_train.json` (SHA-256 `c1998792…`) yield 63.10% / 25.20% / 11.71%. The discrepancy is documented here, not silently reconciled. UrbanGen-AI uses the recomputed values.

## Memory and compute (confirmed at audit time)

| Field | Value |
|---|---|
| GPU |  |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.12.4 |

## Leakage prevention rule

The real BDD100K test set (`resnet_test_clean.json`, 4,745 images) is **never** used for:
- Generative model training
- Augmentation filtering thresholds
- Class-fidelity evaluator development
- Checkpoint selection
- Model comparison / arm selection

It is used **exactly once per arm** for the final downstream evaluation, after the full protocol (five arms, bootstrap methodology, success criterion) is frozen in `docs/experiments.md`.

## Known limitations

1. **Class imbalance** — city street is 63.10% of train. Any augmentation experiment must not degrade macro-F1 while chasing residential recall.
2. **Duplicate-adjacent frames** — BDD100K keyframes are sampled from videos. The split manifests are UrbanMind-AI's audit-verified versions; duplicate-image analysis is not part of UrbanGen-AI's audit.
3. **Domain specificity** — BDD100K is dashcam footage from a small set of cities. Generative models trained on it will reflect that distribution.
4. **No public test labels** — the "test" split is a 50% holdout of BDD100K's official validation annotations, matching UrbanMind-AI's Week 5 protocol.
5. **Prior test-set use** — UrbanMind-AI's Week 5 model and Week 9a fusion experiments evaluated on this same test split. UrbanGen-AI's downstream comparison against the historical baseline is not a pristine out-of-sample evaluation; it is a controlled comparison using the same test set. Disclosed here.

## What this card does NOT claim

- That BDD100K is representative of all urban scenes.
- That the Week 5 baseline (macro-F1 0.6984) represents SOTA.
- That generated images inherit correct class labels. Class fidelity is measured independently in the evaluation framework.
- That the historical Week 5 seed-42 `train_test_split` has been reproduced. The manifests are frozen inputs.

## Reproduce the audit

```powershell
uv run python scripts/audit_data_provenance.py
```

The script prints all hashes, partition-integrity checks, class counts, and checkpoint verification. Output should match every value in this card. Any mismatch indicates data drift and halts the experiment.

## Citation

```bibtex
@inproceedings{yu2020bdd100k,
  title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

## Changelog

| Date | Notes |
|---|---|
| 2026-11-03 | Initial card. UrbanGen-AI G0.2/G0.4 audit. Provenance frozen at UrbanMind commit `dfcfaf7`, tag `v0.10.0-week10-calibration`. Train class shares recomputed (63.10/25.20/11.71) and discrepancy vs UrbanMind Week 5 card (62.3/24.9/11.6) documented. |