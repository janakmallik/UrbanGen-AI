
# Findings — UrbanGen-AI

One-page summary of the project's negative findings, positive findings, and diagnostic contributions.

## The cVAE finding (negative)

A conditional VAE trained from scratch on BDD100K scene images at 128×128 (30 epochs, β=1, latent dim 128, MSE reconstruction):

- **Trains correctly:** converges by epoch 25 (val loss 273.9), 49% reduction in reconstruction error, KL stable at 76–88, 128/128 active latent dimensions, no posterior collapse.
- **Produces diverse, non-memorized samples:** pairwise cosine distance 0.365, 0/3000 above 0.95 nearest-neighbor similarity to real training images.
- **Fails to produce residential samples:** a real-trained logistic classifier assigns **0.6% of 3,000 synthetic residential samples to residential** (90% to highway, 9.4% to city street). Visual inspection shows each synthetic is a blurred brown-gray gradient; the nearest real neighbors are sharp road scenes.

**Diagnosis:** representation mismatch. The cVAE at 128×128 with 128-dim latent and MSE reconstruction produces samples that do not occupy the residential feature region. Not a training failure — a limit of the architecture and loss function at this resolution.

## Method 1 — Stable Diffusion Img2Img (negative)

Tested SD 1.5 Img2Img at strengths 0.25, 0.4, 0.6:
- At low strength: stayed daytime (no night transformation)
- At high strength: hallucinated houses, deleted highway signs, invented streetlights
- Never found a strength that preserved structure AND produced night

**Diagnosis:** Standard Img2Img cannot preserve scene geometry while changing lighting. The prompt overrides the source image at high strength.

## Method 2 — CycleGAN (positive)

Trained from scratch on 1,500 daytime + 1,500 nighttime residential images, 20 epochs, LR decay:
- **Preserves structure exactly:** Cars, buildings, road signs stay in place
- **Night appearance:** Dark sky, streetlights, headlights — genuine nighttime
- **Eliminates purple artifacts:** By epoch 20 (through LR decay)
- **Reproducible:** 57 minutes on RTX 4050, 2.1 GB peak VRAM

**Finding:** CycleGAN learns the day-to-night mapping while preserving scene identity. Generated images are structurally usable as synthetic training data (pending Arm D validation of downstream improvement).

## The pipeline-mismatch finding

The project's own robustness report initially reported residential recall 0.124 for Week 5. Pipeline verification showed this was measured on the wrong input pipeline (128×128 cache strip upsampled to 224×224). The corrected evaluation on native JPEGs (Resize(256)→CenterCrop(224)) reproduces published Week 5 exactly: **residential recall 0.4121**.

The pipeline mismatch reduced residential recall by 3.3× (0.41 → 0.12) and macro-F1 by 0.13 (0.70 → 0.57). It is the same class of bug as train-serve skew — silent, catches no errors on its own, requires a reference check to detect.

## The corrected robustness finding

On the correct pipeline, Week 5's robustness profile:

| Weather | macro-F1 | residential recall |
|---|---:|---:|
| clear | 0.685 | 0.372 |
| overcast | 0.738 | 0.526 |
| partly cloudy | 0.739 | 0.432 |
| **rainy** | **0.599** | **0.270** |
| snowy | 0.654 | 0.377 |
| undefined | 0.700 | 0.488 |

| Time | macro-F1 | residential recall |
|---|---:|---:|
| dawn/dusk | 0.722 | 0.532 |
| daytime | 0.725 | 0.476 |
| **night** | **0.610** | **0.184** |

**Within-class comparisons (unpaired bootstrap, exploratory):**

| Comparison | Class | Δ | 95% CI | n_A | n_B |
|---|---|---:|---|---:|---:|
| Clear → rainy | Highway | **−0.235** | [−0.368, −0.106] | 740 | 63 |
| Daytime → night | Highway | −0.136 | [−0.188, −0.079] | 614 | 506 |
| Daytime → night | **Residential** | **−0.292** | [−0.370, −0.212] | 397 | 147 |

**Night is the dominant robustness weakness.** Residential recall drops 0.29 (from 0.476 to 0.184) with class-specific n > 145. Highway drops 0.14. Both statistically supported.

**Rainy highway is a secondary weakness** — the −0.235 drop is real but has only 63 rainy highway images, so the estimate is uncertain.

## The Week 5 recipe improvement (Arm A)

Retraining ResNet-50 with a modified recipe (class weights + IMAGENET1K_V2 + best-epoch selection) achieved **night residential recall 0.3605** — nearly **2× the published baseline** of 0.184.

This is important context: the "0.184 baseline" that motivated the generative work was a weak recipe, not a fundamental limitation of the classifier. Any synthetic-data improvement should be measured against 0.3605, not 0.184.

## The WanGP Image-to-Video showcase

A working Image-to-Video generation pipeline was built using Wan 2.2 FastWan 5B via the WanGP engine (`shared.api`). This demonstrates that a consumer laptop (RTX 4050, 6 GB VRAM) can generate synthetic nighttime driving video from real BDD100K images.

**Results:**
- Successfully generated 720p, 9-second videos in ~1 minute each.
- The pipeline preserves the ``mmgp`` memory manager (block-swapping, INT8 quantization) which is essential for running a 5B model on 6 GB VRAM.
- 2 showcase videos are available in ``docs/assets/generated_examples/``.
- Full environment and reproduction commands are in ``docs/reproducibility.md``.

**Limitations:**
- Temporal drift on longer videos. Some structural morphing in the background.
- Not bit-for-bit reproducible (depends on GPU hardware, PyTorch version, and WanGP version).
- Generated videos are synthetic and must be labeled as AI-generated.

## The diagnostic toolkit

Three scripts for evaluating any generative model's downstream utility:

| Script | Purpose | Time |
|---|---|---|
| `scripts/probe_pilot_features.py` | Frozen ResNet-50 + logistic regression, four augmentation arms | ~10 min |
| `scripts/diagnose_g1c.py` | Real-only classifier on synthetic samples + NN grid + centroid distances | ~10 min |
| `scripts/verify_week5_pipeline.py` | Three-condition pipeline check + reference reproduction | ~15 min |

Plus four supporting diagnostics:
- `scripts/audit_probe.py` — class-weight sensitivity
- `scripts/audit_residential_labels.py` — OOF label consistency
- `scripts/robustness_report_v2.py` — per-weather/per-time subgroup analysis
- `scripts/diagnose_blur_aware_evaluator.py` — matched-distribution classifier

**Any future generative pipeline** can be tested with these in ~30 min before committing to a fine-tune.

## What the project is NOT

- **Not a claim that all generative approaches are useless.** The cVAE failed and SD 1.5 Img2Img failed, but **CycleGAN succeeded** (structural preservation). Depth ControlNet is untested.
- **Not a claim that CycleGAN solves the residential weakness.** It produces high-quality synthetic night images, but whether they improve downstream classification is Arm D's task (not yet run).
- **Not a claim that the residential class is unlearnable.** The label is consistent; the model is weak (residential recall 0.41 at native resolution). Arm A's recipe (0.3605) proves the class IS learnable.
- **Not a claim that Week 5 is broken.** It reproduces published numbers exactly. It has a night weakness and a residential weakness, both documented.

## What future work should test

1. **Arm D training.** Use Method 2's CycleGAN-generated night images to test whether synthetic augmentation improves Week 5's night residential recall beyond the Arm A baseline (0.3605).
2. **Method 3 (Depth ControlNet).** Test whether structurally-conditioned diffusion can beat CycleGAN's quality on the same source images.
3. **Night augmentation.** Simple tone-curve + ColorJitter augmentation, tested on the val split first. (UrbanMind-AI scope.)
4. **Residential redefinition.** A binary "urban scene (city+residential) vs highway" task might be well-posed where the three-class task is not. (UrbanMind-AI scope.)
5. **Object-level features.** Week 9a's segmentation + detection features improved residential recall from 0.41 to 0.67. (UrbanMind-AI scope.)
6. **DDPM at higher resolution.** 128×128 is a hard limit for a VAE. Diffusion at higher resolution may produce class-faithful samples. (Future project.)

---

## Visual evidence

### cVAE failure proof
8 synthetics (top) vs 7 nearest real residential neighbors (bottom):
![cVAE diagnostic](../assets/method_results/cvae_diagnostic/nn_grid.png)

### Residential label audit
Suspect and confident grids both show residential-looking scenes (label is consistent):
![Suspect residential](../assets/method_results/label_audit/suspect_residential.png)
![Confident residential](../assets/method_results/label_audit/confident_residential.png)

### Method 1 — SD Img2Img (failed)
Hallucinated houses and deleted signs (5 comparisons):
![Method 1 comparison 1](../assets/method_results/compare_01.jpg)
![Method 1 comparison 2](../assets/method_results/compare_02.jpg)

### Method 2 — CycleGAN (successful)
Epoch 20 comparisons — real day (left) vs generated night (right), structure preserved:
![Method 2 epoch 20 comparison 1](../assets/method_results/method_2/epoch20_holdout_1.jpg)
![Method 2 epoch 20 comparison 2](../assets/method_results/method_2/epoch20_holdout_2.jpg)
![Method 2 epoch 20 comparison 3](../assets/method_results/method_2/epoch20_holdout_3.jpg)
