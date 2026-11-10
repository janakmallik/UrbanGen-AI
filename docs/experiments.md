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

### Result — training

30 epochs completed. Converged by epoch 25 (best val loss 273.9).

| Metric | Epoch 1 | Epoch 25 (best) | Epoch 30 |
|---|---|---|---|
| Train recon_mse | 0.0152 | 0.0078 | 0.0078 |
| Train KL | 76.2 | 87.7 | 87.8 |
| Val recon_mse | 0.0094 | 0.0068 | 0.0068 |
| Val KL | 73.5 | 87.2 | 87.2 |
| Active dims | 128/128 | 128/128 | 128/128 |
| Best val loss | — | **273.9** | — |

- Train recon_mse reduced 49% (0.0152 → 0.0078)
- KL stable at 76–88, no posterior collapse
- **7.6× better than class-mean baseline** (0.0068 vs 0.0592 per-pixel MSE)
- All 128 latent dims active throughout
- Scaler stabilized after epoch 1 (fp16 AMP); no NaN or divergence

### Result — sample quality

Sample grids at epochs 5/10/15/20/25/30 in
`outputs/cvae_beta1_seed42/samples/`. Final grid committed at
`docs/model_cards/cvae_samples_beta1_seed42.png`.

Samples are visibly blurry — expected for a standard VAE with MSE
reconstruction at 128×128 with a 128-dim latent. This is a property
of the loss function and resolution, not a training failure.

**Memorization analysis (residential, n=64):**
- MSE to nearest real residential: mean 0.026, median 0.027
- Perceptual nearest-neighbor (ResNet-50 features): mean cosine 0.624,
  **0/64 above 0.90** — no near-duplicate of any real training image

**Diversity analysis:**
- Pairwise cosine distance within class: city 0.109, highway 0.113,
  residential 0.121
- Samples within each class are meaningfully different — no mode collapse

### Result — class conditioning: definitive diagnostic

Two diagnostics were run. The first trained a small evaluator on **sharp**
real images and showed it blurry reconstructions — a distribution shift
that inflated the "reconstruction collapses" finding. That diagnostic
was **inconclusive** (evaluator confounded), and its original
"definitive bottleneck" framing was corrected.

The corrected diagnostic trains a **matched evaluator on reconstructions**
of real training images, then tests on held-out reconstructions. This
eliminates the distribution-shift artifact.

**Blur-aware evaluator (matched control):**

| Condition | Accuracy | Balanced acc | Macro-F1 |
|---|---|---|---|
| Sharp control (trained and tested on sharp) | 0.620 | 0.611 | 0.565 |
| Blur-aware (trained and tested on recons) | 0.682 | 0.604 | 0.590 |
| **Ratio (blur/sharp)** | — | **0.988** | — |

**Class signal survives reconstruction.** A matched evaluator trained
on reconstructions classifies held-out reconstructions at 0.604 balanced
accuracy — 98.8% of its sharp-image performance (0.611). The earlier
0.258–0.275 reconstruction accuracy was evaluator fragility on blurry
inputs, not class-signal destruction.

**Prior-sample test (the pilot-relevant check):**

Generated 64 prior samples per class with matched z (same z vector
across all three class labels). Classified each decode with the
blur-aware evaluator:

| Requested class | Correctly classified | Accuracy |
|---|---|---|
| City street | 54/64 | **0.844** |
| Residential | 23/64 | **0.359** |
| Highway | 11/64 | **0.172** |
| **Overall** | 88/192 | **0.458** (chance 0.333) |

**Findings:**

1. **The cVAE does condition on class.** Prior samples carry class signal
   at 0.458 overall vs 0.333 chance (37.6% relative improvement).
2. **City street is strongly class-distinct** (0.844 accuracy). The
   model generates clear city-street-like samples when conditioned on
   city street.
3. **Residential is weakly class-distinct** (0.359, barely above 0.333).
   The model's residential conditioning is present but fragile.
4. **Highway is actively confused with city street** (0.172, worse than
   chance). Highway-conditioned samples look more like city street to
   the evaluator than like real highway.

**Cross-project finding:** The Week 5 ResNet-50 classifier from
UrbanMind-AI achieves only **5.4% accuracy on real residential images**
(92.2% on real city street images). It is effectively a city-street
detector and **cannot be used to evaluate residential generation**.
This is documented in `outputs/diagnostic/classifier_pipeline_diagnostic.json`
and is a legitimate finding about UrbanMind-AI's Week 5 model,
independent of UrbanGen-AI.

### What G1b establishes

1. **A working cVAE training pipeline.** 30 epochs, converged, no collapse,
   no divergence, reproducible (seed 42).
2. **Diverse but blurry samples.** No memorization (0/64 above 0.90
   perceptual similarity), meaningful within-class diversity (0.109–0.121
   pairwise distance).
3. **The latent is used.** 7.6× better than class-mean baseline,
   128/128 active dims.
4. **Class signal survives reconstruction.** Blur-aware/sharp balanced
   accuracy ratio 0.988.
5. **Prior samples are class-distinct.** Overall 0.458 accuracy vs
   chance 0.333. Asymmetric: city strong (0.844), residential weak
   (0.359), highway actively confused (0.172).
6. **The Week 5 classifier is invalid for residential evaluation.**
   5.4% accuracy on real residential images.

### What G1b does NOT establish

- **Whether synthetic residential samples improve downstream recognition.**
  That is G1c's pilot's question. The weak residential conditioning (0.359)
  means a null pilot result could reflect sample weakness, not augmentation
  ineffectiveness.
- **Whether the class conditioning is sufficient for a meaningful pilot.**
  The prior samples carry signal, but only weakly for residential.

### Limitations

1. **Blur** — expected for standard VAE + MSE at 128×128. Not a training
   failure. The blur is a property of the loss function, not the data or
   the training procedure.
2. **Weak residential conditioning** (0.359 vs chance 0.333) — the model
   can generate residential-like samples, but the signal is fragile.
   A 16-dim class embedding concatenated at two points may be insufficient.
3. **Highway confusion** (0.172 vs chance 0.333) — the model generates
   samples that look more city-street-like than highway-like when
   conditioned on highway.
4. **Class imbalance in the training data** (62% city street) may bias
   the model toward city-like outputs regardless of conditioning.
5. **Padding region (43.75% of each image)** excluded from loss but
   present in output. Generated images have arbitrary content in the
   padding rows; visualization zeroes them.

### Preserved baseline

`v0.1.0-g1b-cvae` is the honest G1b milestone tag. The checkpoint,
model card, sample grids, and this experiment entry are the record of
what was trained. **No retraining, no architectural changes** in G1b.

### Next step: G1c pilot

The blur-aware diagnostic establishes that prior samples carry class
signal. **The G1c pilot is meaningful.** It will test whether adding
3,000 cVAE-generated residential samples to ResNet-50 training improves
residential recall on the real clean test set, against three control
arms (real-only, conventional augmentation, class-weighted loss).

The weak residential conditioning (0.359) is documented as a caveat:
a null pilot result may reflect sample weakness rather than augmentation
ineffectiveness.

**Files:**
- Diagnostic script: `scripts/diagnose_blur_aware_evaluator.py`
- Diagnostic JSON: `outputs/diagnostic_blur_aware/blur_aware_diagnostic.json`
- Original diagnostic (superseded, preserved): `scripts/diagnose_class_conditioning.py`,
  `outputs/diagnostic_conditioning/conditioning_diagnostic.json`
- Classifier diagnostic: `outputs/diagnostic/classifier_pipeline_diagnostic.json`
- Model card: `docs/model_cards/bdd100k_cvae_v1.md`
- Sample grid: `docs/model_cards/cvae_samples_beta1_seed42.png`


---

## 2026-11-06 — G1c: frozen-feature pilot and diagnostic (closed)

Hypothesis: Adding 3,000 cVAE-generated synthetic residential images to real BDD100K training data measurably improves residential recall on the real val split, beyond baseline and non-generative controls.

### Probe setup

Frozen ImageNet-pretrained ResNet-50 features (2048-d), logistic regression head.
Real train: 68,969 images. Real val: 4,727 (587 residential).
No fine-tuning; no test-set access. See `scripts/probe_pilot_features.py`.

### Probe results (real val split)

| Arm | Residential P | Residential R | Residential F1 | Macro-F1 |
|---|---|---|---|---|
| A — Real only | 0.3996 | 0.7019 | 0.5093 | 0.6486 |
| B — Real + 3,000 cVAE synthetic residential | **0.4340** | 0.6440 | **0.5185** | **0.6563** |
| C — Real + 3,000 real residential oversampled | 0.3967 | 0.6968 | 0.5056 | 0.6471 |
| D — Real only, class-weighted (identical to A in frozen-feature probe) | 0.3996 | 0.7019 | 0.5093 | 0.6486 |

Under the `balanced` class_weight setting, arm B had +0.0092 residential F1 over A and +0.0130 over C.

### Audit — the deltas are not robust

**Check 1 — class_weight sensitivity.** B − A on residential F1 under four class_weight settings:

| class_weight | B − A residential F1 |
|---|---|
| `balanced` | **+0.0092** |
| `none` | **−0.0099** |
| `{0:1, 1:1, 2:2}` | +0.0059 |
| `{0:1, 1:1, 2:4}` | **−0.0023** |

The delta flips sign depending on the class_weight setting. Under neutral weighting, arm B is worse than arm A by −0.01. The `balanced` improvement is a classifier interaction with the class_weight setting, not a robust synthetic-data effect.

**Check 2 — feature-space statistics (ResNet-50, 2048-d):**

- Synthetic → nearest real residential cosine: mean 0.4386, max 0.6612
- Synthetic → nearest real non-residential cosine: mean 0.4887
- `>= 0.95` (memorization): 0/3000 — synthetics are not memorized
- Diversity (pairwise cosine distance): 0.3653 — synthetics are diverse

**Check 3 — 5-fold CV on train:**
Run for all 5 folds under `balanced` weighting.

| Fold | A F1 | B F1 | Delta |
|---|---|---|---|
| 1 | 0.4989 | 0.5160 | +0.0170 |
| 2 | 0.4969 | 0.5084 | +0.0115 |
| 3 | 0.4945 | 0.5074 | +0.0129 |
| 4 | 0.5000 | 0.5100 | +0.0100 |
| 5 | 0.4876 | 0.5025 | +0.0149 |

A: mean 0.4956, std 0.0044. B: mean 0.5089, std 0.0043. B − A: mean +0.0133, std 0.0025, positive folds 5/5. Under `balanced` weighting, adding synthetic residential features improves residential F1 consistently. This is a within-distribution consistency result, not a robustness proof — the 5 folds share most training data.

### Final diagnostic — representation mismatch

See `scripts/diagnose_g1c.py`. Real-only logistic regression classifier (trained on real train features only, no synthetic) applied to 3,000 cVAE-generated samples:

| Predicted class | Count | Percentage |
|---|---|---|
| Highway | 2,700 | 90.0% |
| City street | 283 | 9.4% |
| **Residential** | **17** | **0.6%** |

**Feature-space centroid cosine:**

| Class | Cosine |
|---|---|
| Highway centroid | 0.2282 |
| Residential centroid | 0.2281 |
| City street centroid | 0.2195 |

The synthetics are equidistant from highway and residential centroids, and farther from city street. They are feature-space noise between the two larger classes.

**Visual inspection of `nn_grid.png`:** each row shows one synthetic followed by its 3 nearest real residential and 3 nearest real non-residential images. Every synthetic is a blurred brown-gray gradient blob with no lane structure, buildings, or scene content. Real neighbors are sharp road scenes. The synthetics are visually unrelated to the class they were conditioned on.

**Diagnosis: REPRESENTATION MISMATCH.** The cVAE's 128×128 latent + MSE reconstruction at 30 epochs produces samples that do not occupy the residential feature region. A real-only classifier assigns 0.6% of synthetics to residential, versus 90% to highway. This explains the frozen-feature probe's inconsistency — the samples are not residential, so adding them to a residential augmentation arm cannot produce a robust residential-recall improvement.

### What G1c establishes

1. The cVAE's synthetic residential samples are not feature-space residential. A real-only classifier assigns 0.6% to residential.
2. The apparent `balanced`-weighting improvement in the probe (+0.0092 residential F1) is not robust to class_weight choice.
3. The 5-fold consistency under `balanced` weighting is real but conditional on that setting. It does not justify a full fine-tune.
4. The synthetics are diverse and not memorized; the failure is representation, not mode collapse or copy-reproduction.

### Preserved baseline

`v0.1.0-g1b-cvae` remains the G1b milestone. No fine-tune was run. G1c is documented as a negative result with a specific, diagnosed cause.

### Files

- `scripts/probe_pilot_features.py` — frozen-feature probe
- `scripts/audit_probe.py` — class_weight sensitivity + feature-space + 5-fold CV
- `scripts/diagnose_g1c.py` — representation-mismatch diagnostic
- `scripts/train_pilot_arm.py`, `scripts/evaluate_pilot.py` — written but not run (kept for reference)
- `outputs/g1c_probe/probe_results.json`, `outputs/g1c_probe/audit_results.json`
- `outputs/g1c_diagnostic/diagnostic.json`, `outputs/g1c_diagnostic/nn_grid.png`


---

## 2026-11-06 — Step 0: residential label audit and Week 5 robustness report

Two diagnostic artifacts produced before any new generative or augmentation work. Neither trains a new model; both use existing infrastructure (Week 5 ResNet-50, frozen ResNet-50 features, the 128×128 cache, BDD100K label JSONs).

### Step 0a — residential label audit

`scripts/audit_residential_labels.py`. Frozen ImageNet ResNet-50 features (2048-d), 5-fold OOF logistic regression (class_weight="balanced").

**OOF per-class recall:**

| Class | Recall | n |
|---|---|---|
| City street | 0.6399 | 43,516 |
| Highway | 0.7579 | 17,379 |
| **Residential** | **0.7026** | 8,074 |

**Confusion matrix (rows=true, cols=pred):**

|  | pred city | pred hwy | pred res |
|---|---:|---:|---:|
| true city | 27,848 | 7,452 | 8,216 |
| true hwy | 2,694 | 13,171 | 1,514 |
| true res | 1,631 | 770 | 5,673 |

**Suspect rate** (model confidently disagrees with label): city 36.0%, highway 24.2%, residential 29.7%.

**Grids:** `outputs/label_audit/suspect_residential.png` (24 images the model is confident are NOT residential but are labeled residential) and `confident_residential.png` (24 images the model is confident ARE residential).

**Interpretation:** The residential label carries real signal (OOF recall 0.70) but ~30% of "residential" images are boundary-ambiguous with city street. The class is fuzzy at the BDD100K definition, not mislabeled. Not a label-cleaning problem.

### Step 0b — robustness report card

`scripts/robustness_report.py`. Week 5 ResNet-50 evaluated on 4,745 clean test images, broken down by BDD100K `weather` and `timeofday` attributes. Bootstrap 95% CIs on all metrics.

**Overall:** macro-F1 0.571 [0.553, 0.587]. City recall 0.955 [0.947, 0.962], highway 0.559 [0.532, 0.588], residential 0.124 [0.097, 0.149].

**By weather:**

| Weather | n | macro-F1 [95% CI] |
|---|---|---|
| clear | 2,627 | 0.577 [0.552, 0.601] |
| overcast | 582 | 0.614 [0.562, 0.661] |
| partly cloudy | 350 | 0.610 [0.548, 0.669] |
| **rainy** | **326** | **0.481 [0.406, 0.550]** |
| **snowy** | **366** | **0.526 [0.466, 0.583]** |
| undefined | 487 | 0.476 [0.415, 0.536] |
| foggy | 7 | 0.413 [0.111, 0.861] |

**By time of day:**

| Time | n | macro-F1 [95% CI] |
|---|---|---|
| dawn/dusk | 367 | 0.612 [0.547, 0.675] |
| daytime | 2,447 | 0.574 [0.551, 0.596] |
| **night** | **1,931** | **0.561 [0.526, 0.595]** |

**Cross-tab insights:** rainy has 69% city vs clear's 61%; night has 8% residential vs daytime's 16%. Both weather/subgroup class distributions differ from the overall. The rainy gap is therefore partly real (against the distribution); the night gap is negligible (against the distribution).

### What these artifacts establish

1. **The residential weakness dominates.** Residential recall is 0.124. No weather or time subgroup shows a comparable gap. Adversarial weather affects macro-F1 by ~0.10; the residential class affects it by ~0.30.
2. **Rainy weather is a real but secondary gap.** ~0.10 macro-F1 drop that is not fully explained by scene-class distribution. But rainy is only 326 of 4,745 test images (7%).
3. **Night is not a weakness.** ~0.01 macro-F1 drop, overlapping CIs, and the class distribution (fewer residential) actually favors night.
4. **The residential class is fuzzy.** OOF recall 0.70, but 30% boundary-ambiguous. Not a label-cleaning problem.

### Artifacts (durable, reusable)

- `docs/robustness_report.md` — human-readable subgroup report
- `outputs/robustness_report/report.json` — machine-readable
- `outputs/label_audit/` — OOF probs, features, suspect/confident grids, per-image analysis
- `scripts/audit_residential_labels.py`, `scripts/robustness_report.py`

### Decision

No new model trained. No generative experiment run. The diagnostics establish what's actually wrong with the Week 5 model and provide a reusable toolkit for evaluating future generative pipelines.

**The dominant finding — residential class boundary is fuzzy — is not addressable by generative augmentation.** Any future direction that assumes "more residential training data would help" should account for this: the label itself is partly city-street-like, so the task is ambiguous at the class definition level.

---


### Correction (2026-11-06) — pipeline mismatch invalidates the robustness report

The robustness report above (Step 0b) was generated using the 128×128
cache content strip upsampled to 224×224. Pipeline verification
(`scripts/verify_week5_pipeline.py`, `outputs/pipeline_verification/`)
established that this is a different pipeline from Week 5's training
pipeline, and it changes the model's behavior substantially.

**Three-condition verification (all on the clean 4,745-image test
subset; condition (a) also on the full 4,932-image split as a
reference check):**

| Condition | n | accuracy | macro-F1 | city | highway | residential |
|---|---|---|---|---|---|---|
| (a) native JPEG → Resize(256)→CenterCrop(224) [reference, full test] | 4,932 | **0.7800** | **0.6984** | 0.8825 | 0.7136 | 0.4121 |
| (a) native JPEG → Resize(256)→CenterCrop(224) [clean test] | 4,745 | 0.7819 | 0.6986 | 0.8839 | 0.7163 | 0.4078 |
| (b) native JPEG → resize full frame to 224×224 (squashed) | 4,745 | 0.7966 | 0.7283 | 0.9076 | 0.6625 | 0.5178 |
| (c) 128×128 cache content strip → 224×224 | 4,745 | 0.7503 | 0.5710 | 0.9545 | 0.5591 | 0.1235 |

**Reference check:** condition (a) on the full 4,932 test images
reproduces the published Week 5 card exactly (acc 0.7800, macro-F1
0.6984, residential recall 0.4121). The checkpoint, class mapping,
and preprocessing are verified.

**Clean subset effect:** (a) full vs (a) clean — macro-F1 delta +0.0002,
residential delta −0.0044. The excluded 187 images do not affect the
model's behavior.

**Pipeline effect:** condition (c) reduces macro-F1 by −0.128 and
residential recall by −0.284 relative to condition (a). The 0.12
residential recall reported in the robustness report was measuring the
wrong input pipeline.

**Exploratory finding (b):** the squashed-full-frame pipeline
outperforms the training pipeline by +0.030 macro-F1 and +0.110
residential recall on the test set. **This was observed on test, so
it is exploratory and does not justify any deployment change.**
A hypothesis (untested): Week 5's training transforms include
RandomResizedCrop, which stretches aspect ratios; the squashed frame
may sit closer to that training distribution. To adopt (b) for any
purpose, it must first be validated on the val split with a paired
bootstrap, then confirmed once on test.

**What is retracted:**
- "Residential recall = 0.124" — this was the model's performance on the
  wrong input pipeline. The correct value is 0.408 (condition a) on the
  clean test set. The 0.124 does **not** reflect the model's actual
  residential ability; it reflects a train-test preprocessing shift.
- "Residential weakness dominates" — overstated. At condition (a),
  city recall is 0.88, highway 0.72, residential 0.41. Residential is
  the weakest class, but the gap is 0.47 (city vs residential), not
  0.83 (0.95 vs 0.12).
- All weather/time subgroup conclusions (rainy real gap, night no gap,
  snowy moderate gap) — measured on the wrong pipeline. Must be
  rerun at condition (a).

**What is not retracted:**
- The label audit (Step 0a) — used frozen features, not Week 5's
  classifier. The 0.70 OOF residential recall, suspect rate, and grids
  are valid (with the caveat that the recall figure overstates signal
  because precision is 0.37).
- The G1c cVAE finding (0.6% synthetic residential in a real-only
  classifier) — real and synthetic images both went through the cache
  pipeline, so the comparison is internally consistent. The 0.6%
  reflects the cVAE's inability to produce class-faithful samples at
  128px; it does not establish that residential cues are absent at
  128px (the frozen-feature audit found ~0.70 OOF recall on cache
  features, so some signal exists).

**Corrected robustness report at condition (a) is next.**
The current `docs/robustness_report.md` is preserved with a SUPERSEDED
banner.

**Lesson:** the pipeline mismatch between training and evaluation is
the same class of bug as train-serve skew: silent, and only caught by a
reference check (evaluate the model on its training pipeline, verify
published numbers reproduce exactly). The reference check must be a
permanent test.

---

## 2026-11-08 — Generative Video Showcase (Wan 2.2 FastWan via WanGP)

**Status:** Complete.

**Objective:** Generate synthetic nighttime urban video to demonstrate feasibility of generative augmentation on a 6 GB GPU.

**Method:** Used WanGP v19.21 with FastWan2.2-TI2V-5B (3-step distilled model). Discovered the correct integration path via ``from shared.api import init`` after multiple failed attempts with standard ``diffusers`` and Gradio API wrappers.

**Results:**
- Successfully generated 720p, 9-second videos in ~1 minute each.
- Generated 5 showcase videos for GitHub.
- Documented the full pipeline in ``docs/reproducibility.md``.

**Learning:** WanGP is not a standard Gradio frontend. Its custom ``mmgp`` memory manager (block-swapping, INT8 quantization) is essential for running large models on low VRAM. Attempting to reimplement this with ``diffusers`` failed.

---

## 2026-11-09 — Method 2: CycleGAN Day-to-Night Translation

**Status:** ✅ Complete — Successful structural preservation.

**Objective:** Train a from-scratch CycleGAN to translate daytime BDD100K residential scenes to nighttime, preserving original scene geometry (unlike Method 1/Stable Diffusion which hallucinated).

**Approach:**
- 1,500 daytime + 1,500 nighttime residential images (zero-overlap holdout)
- 20 epochs total (5 pilot + 15 resume)
- LR decay from 2e-4 to 2e-5 in second half
- Replay buffer (50 images) for discriminator stability
- Identity loss (λ=5) to preserve scene colors

**Results:**
- Structure: Cars, buildings, and road signs preserved exactly
- Appearance: Convincing nighttime illumination, dark sky, streetlights
- Purple artifacts: Eliminated by epoch 20 (vs. epoch 5)
- Training time: 57 minutes on RTX 4050

**Metrics (holdout):**
- CLIP night score: ~0.95 (vs. real night calibration baseline ~0.70)
- SSIM: ~0.20 (structural preservation, expected for day-to-night translation)
- Road pixel fraction: preserved within ±0.10

**Finding:** Unlike Method 1 (SD 1.5 Img2Img, which hallucinated new objects and deleted highway signs), CycleGAN learns the exact day-to-night mapping and preserves scene identity.

**Artifacts:**
- Model: ``outputs/method_2_resume/checkpoints/epoch_20.pth``
- Comparisons: ``outputs/method_2_resume/comparisons/epoch20_*.jpg``
- Model card: ``docs/model_cards/bdd100k_cyclegan_v1.md``
- Full log: ``docs/method_2_cyclegan_log.md``

**Next step:** Generate 500 synthetic night images → Arm D training.

---

## 2026-11-10 — Method 2 synthetic generation (filter failure)

**Status:** Generation complete (282/500 accepted) but filters unreliable.

**Filter breakdown:**
- paved_surface (SegFormer): 208 rejections — 5/5 inspected were **false rejections** (good structural preservation)
- clip: 5 rejections — 3/5 inspected were **false rejections**
- luminance: 5 rejections — 4/5 inspected were **false rejections**
- 20 accepted images inspected — 3-5 showed structural failures (near-black, smeared)

**Conclusion:** No automated filter is reliable. Manual curation required.

**Next step:** Manual visual inspection of all 500 generated images. Build a curation manifest with good/bad labels. Use good images only for Arm D.

**Artifacts:**
- `outputs/method_2_synthetic/filter_log.csv` — all 500 with filter reasons
- `outputs/method_2_synthetic/rejected_inspection/` — 5 sample rejections
- `outputs/method_2_synthetic/accepted_inspection/` — 20 sample acceptances
- `outputs/method_2_synthetic/clip_lum_inspection/` — 10 clip/lum rejections

---

