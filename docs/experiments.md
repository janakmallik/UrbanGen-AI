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

## 2026-11-05 — G1 scoping: cVAE pilot for residential augmentation

Hypothesis: A class-conditional VAE trained on BDD100K scene images can generate residential-scene images whose addition to the real training set measurably improves residential recall on BDD100K validation — beyond what conventional augmentation, class-weighted loss, or residential oversampling achieve on their own. The pilot uses validation as the go/no-go signal. The final frozen evaluation on the clean test set (previously used in UrbanMind-AI; reserved for this project's final evaluation) occurs only if the pilot passes.

### Frozen decisions (not movable after results are seen)

| Decision | Value | Rationale |
|---|---|---|
| Resolution | 128×128 | Cache built at this size; profiling shows all configs fit easily on 6 GB. Aspect-ratio preserved via padding. |
| cVAE channels | 32-64-128-256 | Standard for this resolution; wider adds compute without proven benefit for a first pilot. |
| Latent dim | 128 | Sufficient for 3 classes at this image size. |
| β (KL weight) | 1.0 | Standard VAE. β-sweep deferred until pilot shows signal. |
| Seed | 42 | Single seed for pilot; multi-seed only after pilot passes. |
| Batch size | 32 | Measured 0.27 GB peak (FP32 profiling); 5+ GB headroom available. |
| Precision | FP16 AMP, KL computed in FP32 | Standard VAE discipline; avoids KL underflow. |
| Conditioning | Class embedding (dim 16) concatenated with flattened encoder features (before μ/logσ²) and with latent z (before decoder) | Standard; sufficient for 3 classes. |
| Synthetic count | 3,000 residential images | Initial target; count may be adjusted on validation only, not test. |
| Pilot arms | **5 arms:** A real-only, B +synthetic, C +conventional aug (RandomResizedCrop 0.75-1.0, RandomHorizontalFlip, ColorJitter), D +class-weighted loss, E +residential oversampling | Required to test whether synthetic beats simple rebalancing. |
| Success criterion (pilot) | Arm B beats **each of** arms C, D, and E individually on residential recall with paired-bootstrap 95% CI excluding zero on **validation**, after **Holm correction** for the three comparisons, and macro-F1 does not degrade by more than 0.01 | Not just beating arm A, and not "beat the best control" (which would be a post-hoc selection trap). Must beat each non-generative control individually. |
| Go/no-go evaluation | **Validation set only** (587 residential val images after cache build) | Test set (591 residential test images) reserved for final multi-seed evaluation only. |
| Training budget per arm | Same gradient steps, same optimizer, same schedule across all arms | Prevents training-budget confound from inflating any arm. |
| Reconstruction loss | Masked MSE on content rows only (rows 28:100 of 128). Padding rows (0:28, 100:128) excluded from loss. | Documented because 43.75% of every image is padding; unmasked MSE would let the model trivially minimize loss by learning black padding. |

### Frozen data provenance (from G1-pre)

| Item | Value |
|---|---|
| Cache commit | `085a906` |
| Cache manifest | `data/bdd100k_128/manifest.json` (committed) |
| Train images | 68,969 (class counts: 43,516 / 17,379 / 8,074) |
| Val images | 4,727 (class counts: 2,951 / 1,189 / 587) |
| Test images | 4,745 (class counts: 2,945 / 1,209 / 591) |
| Preprocessing | 1280×720 → resize to (128, 72) → center-pad to (128, 128) with black |
| Test cache | Marked `read_only_downstream: true`; dataset loader refuses to load it |

### Corrections to prior drafts (documented honestly)

Four errors in earlier versions of this protocol, corrected before any code was written:

1. **Statistical power was miscalculated.** The +0.03 effect size detection I claimed was based on the wrong formula. The pilot's go/no-go uses paired bootstrap on validation; the multi-seed final evaluation (if the pilot passes) uses test with both seed and sampling variance reported. No single pilot can establish a definitive +0.03 detection.
2. **3 seeds do not reduce test-sampling uncertainty by √3.** Seeds measure training variance; test sampling variance is separate. Both reported in the multi-seed phase.
3. **ResNet preprocessing** must match UrbanMind-AI's original training pipeline (RandomResizedCrop 224 scale 0.75-1.0, RandomHorizontalFlip, ImageNet normalization). To be verified against `train_bdd100k_resnet50.py` before G1c, not assumed.
4. **Training budget per arm** must be identical (same gradient steps, same optimizer, same schedule). Otherwise arm B's apparent improvement could come from extra updates, not synthetic data quality.

### Out of scope (deferred until pilot result)

- β-sweep (β ∈ {0.5, 1.0, 4.0})
- Multi-seed (3 seeds: 42, 123, 7)
- Conditional DDPM in latent space
- Conditional DCGAN
- FID/KID computation
- Final multi-seed downstream evaluation on test set

These are adopted only if the pilot shows a signal — defined as arm B beating **each of** arms C/D/E on validation, per the success criterion above.

### Deliverables for G1a (this session)

- `src/urbangen/data/bdd100k_generative.py` — Dataset reading the cache
- `src/urbangen/models/cvae/model.py` — `ConditionalVAE`
- `tests/unit/test_cvae.py` — model unit tests
- `tests/unit/test_bdd100k_generative.py` — dataset unit tests
- This scoping entry
- No training run; no sample generation; no model card

### Deliverables for G1b (next session)

- `scripts/train_cvae.py` — training loop with AMP, MLflow, checkpointing
- One run: β=1, seed 42, 30 epochs (or convergence-based epoch budget determined empirically)
- Sample grids per class
- Nearest-neighbor memorization check
- Model card `docs/model_cards/bdd100k_cvae_v1.md`

### Deliverables for G1c (pilot, after G1b)

- `scripts/generate_cvae_samples.py` — generate 3,000 residential images
- `scripts/run_pilot_arms.py` — 5-arm comparison on validation
- Go/no-go decision recorded

### Profiling evidence (scripts/profile_gpu_cvae.py)

Before locking 128×128 batch 32, a cVAE-shaped profiler measured FP32
training-step time and peak VRAM across five configurations. The
profiling model is a complete encoder→latent→decoder path with the same
channel widths as the planned cVAE — not an encoder-only placeholder.

| Resolution | Batch | Peak VRAM (FP32) | ms/step | Compute-only epoch |
|---|---|---|---|---|
| 64×64 | 32 | 0.09 GB | 12.2 | 0.4 min |
| 64×64 | 64 | 0.13 GB | 8.5 | 0.2 min |
| 64×64 | 128 | 0.20 GB | 14.3 | 0.1 min |
| 128×128 | 16 | 0.20 GB | 13.6 | 1.0 min |
| 128×128 | 32 | 0.27 GB | 16.6 | 0.6 min |

Every configuration fits easily inside 6 GB VRAM. 128×128 was chosen for
image fidelity, not memory constraints: the downstream ResNet-50 receives
224×224 inputs, and 128×128 retains more scene detail than 64×64 after
upscale. Batch 32 leaves ~5.7 GB of VRAM headroom.

The "compute-only epoch" column excludes DataLoader overhead (image
decode, augmentation, CPU→GPU transfer). Real epoch time will be
measured during the G1b training run and reported in the G1b entry.


### Pre-run amendment — loss reduction convention (2026-11-05)

**Change:** The reconstruction term in the training loss is a **sum of
squared error over content pixels** (per sample, averaged over batch),
not a per-pixel mean. KL remains a sum over latent dims (per sample,
averaged over batch). The frozen objective is:

```
recon_sse = Σ_{c,h,w ∈ content} (x - x̂)², mean over batch
kl        = Σ_{j=1..128} KL_j, mean over batch
total     = recon_sse + β·kl, β = 1.0
```

**Reason:** Under the previous convention (per-pixel MSE + KL sum), the
reconstruction term was ~0.05 and the KL term was ~tens of nats.
Optimizing that total drives the optimizer to zero out KL (posterior
collapse) and accept a class-average blurry output — total loss ~0.05
versus ~30 with a healthy latent. The ELBO in its correct form sums
squared error over the sample's data dimensions, which places recon in
the thousands and makes β = 1.0 a meaningful trade-off.

**Diagnostic additions:**

- `recon_mse` (per-pixel) is still logged for interpretability.
- `active_dims` = number of latent dimensions with std(μ) > 0.01 across
  the validation set. Reported per epoch. Under 100/128 at any point
  would be a posterior-collapse warning.
- `train_one_epoch` raises `RuntimeError` on non-finite loss.
- Gradient clipping at norm 1.0 is applied before optimizer step.
- `logvar` is clamped to [-10, 10] inside the model's `encode()`.

**Discipline:** No training has been run under either convention. The
amendment is recorded pre-run rather than as a post-hoc fix.

---

## 2026-11-05 — G1b: cVAE training run (β=1, seed 42)

Hypothesis (same as G1 scoping): see the G1 scoping entry above.

### Setup (frozen — see G1 scoping)

- Model: ConditionalVAE(128, latent_dim=128, channels=(32,64,128,256))
- Loss reduction (see pre-run amendment above):
  - `recon_sse = sum over (channels × content pixels) per sample, mean over batch`
  - `kl        = sum over latent dims per sample, mean over batch`
  - `total     = recon_sse + β·kl`, β = 1.0
  - Mask = 1 on rows 28:100; 0 elsewhere
  - `recon_mse` (per-pixel) logged for interpretability only
- Optimizer: AdamW(lr=1e-3, wd=1e-4), CosineAnnealingLR over 30 epochs
- Batch size 32, DataLoader num_workers=0, shuffle=True (train only)
- AMP: FP16 autocast for forward/backward; KL divergence computed in FP32
- Seeds: torch 42, numpy 42, cuda 42
- Validation: shuffle=False, deterministic (μ not sampled z), reports recon/kl/total
- Checkpointing: best by val total, plus last checkpoint each epoch
- Sample grid every 5 epochs (16 samples × 3 classes) → `outputs/cvae_beta1_seed42/samples/`
- Early stopping: **disabled** for this first run — full 30 epochs unless
  NaN loss or CUDA failure. Convergence compared at epochs 15 and 30.
- Runtime estimate (compute-only from G1 profiling): ~18 min of GPU step
  time. Actual wall time (with DataLoader + logging) to be measured.

### Result

**PENDING — training run not yet executed.**

This entry will be filled in after `scripts/train_cvae.py --epochs 30`
completes. Expected content: per-epoch loss curves, best val loss and
epoch, sample grids at epochs 5/10/15/20/25/30, nearest-neighbor
memorization analysis, and a convergence decision (30 epochs sufficient?).

### Limitations / non-claims

- **A successfully trained cVAE is not evidence that synthetic data
  improves downstream recognition.** That is G1c's pilot's question.
- Sample grids from a first cVAE run at 128×128 are expected to be
  blurry; this is a property of VAE reconstruction quality at this
  scale, not a failure.
- The padding region (43.75% of each image) is excluded from loss but
  is still part of the generated output. The model may or may not learn
  to output near-black there; this will be documented, not corrected.

---

