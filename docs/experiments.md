# Experiments Journal

Short entries for every experiment. Structure: hypothesis -> setup -> result -> takeaway.

---

## 2026-11-03 — G0: Project scoping and data provenance audit

Hypothesis: UrbanMind-AI's Week 5 BDD100K data and ResNet-50 baseline can be frozen as reproducible inputs for a separate generative-modeling project (UrbanGen-AI), with all provenance auditable from a single script and no silent data drift.

### Scoping decision (frozen before any model code)

**Central research question:** Can class-conditional generative models trained from scratch on BDD100K (cVAE, conditional DDPM in latent space, conditional DCGAN) produce synthetic residential-scene images that measurably improve residential recall on the real BDD100K test set, compared to training on real data alone and to conventional augmentation?

**Five downstream ResNet-50 arms:**

| Arm | Training data | Purpose |
|---|---|---|
| A | Real BDD100K train only | Baseline |
| B | Real + conventional augmentation (crop/flip/jitter) | Control (does synthetic beat trivial augmentation?) |
| C | Real + cVAE synthetic residential | VAE arm |
| D | Real + DDPM synthetic residential | Diffusion arm |
| E | Real + DCGAN synthetic residential | GAN arm |

**Pre-stated success criterion (not movable after results are seen):** a generative arm "works" if residential recall improves by ≥ +0.03 with a 95% bootstrap CI excluding zero, **and** macro-F1 does not degrade by more than −0.01. Evaluated on the real clean test split (`resnet_test_clean.json`, 4,745 images). Paired bootstrap, 1,000 resamples, clip-level.

**Explicitly out of scope:** pretrained Stable Diffusion fine-tuning (not compute-feasible on 6 GB VRAM for meaningful training); any generative model on video, audio, or time series; any change to UrbanMind-AI's Week 5 model or its shipped artifacts.

### Setup

- **Source project:** UrbanMind-AI at commit `dfcfaf7` (tag `v0.10.0-week10-calibration`)
- **Shared data root:** `$URBANINTEL_DATA_ROOT/bdd100k`
- **Frozen label JSONs:**
  - `bdd100k_labels_images_train.json` — SHA-256 `c1998792e385fd79213187576ca07d5249563d0ab83ae976688ad60668baca9e`
  - `bdd100k_labels_images_val.json` — SHA-256 `c835be652f0c002897be51e74deedd16a1dcee0b7230f19c0456fba772f694af`
- **Frozen split manifests** (from UrbanMind-AI `data/splits/`, copied locally with source hashes verified):
  - `resnet_val_clean.json` — 4,727 entries, SHA-256 `9ee1b6c69f4e58e83edb704fbd8d42b3972a9b0410aa838a939143205f924682`
  - `resnet_val_excluded.json` — 205 entries, SHA-256 `05d54fda00e07083d73f2114607636a6c4056802b5bfd5f5614de2c79d0c4760`
  - `resnet_test_clean.json` — 4,745 entries, SHA-256 `ca47e16a4c6148520b2a0cc3de74b78db20392507c93ea4d5a4039c695d03979`
  - `resnet_test_excluded.json` — 187 entries, SHA-256 `ec48bc0166145448e726bb277a7e62084deb727a457424f38f4184e1a6d9daea`
- **Baseline checkpoint:** `$URBANMIND_REPO/bdd_resnet50_best.pt` — SHA-256 `f760b94c4bb334c95c5d6c3c7884534fe6f51ce397641249610aab1a98447f84`, 94,367,514 bytes, plain state_dict (320 tensors), ResNet-50 with `fc.weight` shape `(3, 2048)`, 23,514,179 parameters
- **Environment:** Python 3.12.4, PyTorch 2.6.0+cu124, CUDA 12.4 available, single CUDA-capable GPU
- **Audit script:** `scripts/audit_data_provenance.py` (CPU-only, no dataset file writes; prints all frozen values)

### Result

**AUDIT PASS — all checks green.**

| Check | Result |
|---|---|
| BDD100K `images/` + `labels/` present | ✅ |
| 100,000 unique image filenames (no duplicates in tree) | ✅ |
| Train + val label JSON SHA-256 match frozen values | ✅ |
| All four manifest SHA-256 match UrbanMind-AI source | ✅ |
| Manifest schema: JSON lists of strings, no internal duplicates, entry counts as expected | ✅ |
| Partition integrity: all four pairwise intersections = 0 | ✅ |
| `\|val_clean ∪ val_excluded\| = 4932`, `\|test_clean ∪ test_excluded\| = 4932`, `\|all four\| = 9864` | ✅ |
| All manifest entries present in `bdd100k_labels_images_val.json` | ✅ |
| Train class counts: 43,516 / 17,379 / 8,074 (total 68,969) | ✅ |
| Val class counts: 6,112 / 2,499 / 1,253 (total 9,864) | ✅ |
| Checkpoint size, SHA-256, format (plain state_dict), tensor count (320) | ✅ |
| `load_state_dict` succeeds, `fc.weight` shape `(3, 2048)`, params 23,514,179 | ✅ |

### Takeaway

The G0 foundation is frozen and auditable. UrbanGen-AI's data and baseline provenance are verified by a single reproducible script; any future drift in manifests, label JSONs, or checkpoint will fail the audit and halt the experiment rather than silently producing unreliable downstream results.

This is the provenance-first discipline that UrbanMind-AI's Week 9c contamination audit established (encoder provenance verification) applied to a new project from day one. **No generative model code was written before data and checkpoint provenance were verified.**

The train class shares recomputed here (63.10% / 25.20% / 11.71%) differ slightly from UrbanMind-AI's Week 5 dataset card (62.3% / 24.9% / 11.6%). The discrepancy is documented in the dataset card, not silently reconciled. UrbanGen-AI uses the recomputed values.

### Limitations

- The four `resnet_*` manifests are consumed as frozen inputs. Their partition of the 3-class validation pool is verified; the historical Week 5 `train_test_split(seed=42, stratify=label_id)` output is **not** independently reconstructed. This is documented as out of scope.
- The BDD100K "test" split is a 50% holdout of the official validation annotations, not the official BDD100K test set (which has no public labels). This matches UrbanMind-AI's Week 5 protocol.
- UrbanMind-AI's Week 5 and Week 9a experiments evaluated on this same test split. UrbanGen-AI's downstream comparison against the historical baseline is a controlled comparison, not a pristine out-of-sample evaluation. Disclosed.
- Residential class is 11.71% of train (the target weakness); class imbalance will constrain the eventual success criterion.

### Deliverables

- `scripts/audit_data_provenance.py` — reproducible provenance audit
- `docs/dataset_cards/bdd100k_generative.md` — frozen provenance record
- `data/splits/*.json` — local frozen copies of UrbanMind-AI split manifests

### Next

G1 (cVAE training). No generative model code before this audit was green.

---