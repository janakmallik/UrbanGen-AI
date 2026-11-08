# Findings — UrbanGen-AI

One-page summary of the project's negative findings and diagnostic contributions, after the pipeline-mismatch correction.

## The cVAE finding

A conditional VAE trained from scratch on BDD100K scene images at 128×128 (30 epochs, β=1, latent dim 128, MSE reconstruction):

- **Trains correctly:** converges by epoch 25 (val loss 273.9), 49% reduction in reconstruction error, KL stable at 76–88, 128/128 active latent dimensions, no posterior collapse.
- **Produces diverse, non-memorized samples:** pairwise cosine distance 0.365, 0/3000 above 0.95 nearest-neighbor similarity to real training images.
- **Fails to produce residential samples:** a real-trained logistic classifier assigns **0.6% of 3,000 synthetic residential samples to residential** (90% to highway, 9.4% to city street). Visual inspection shows each synthetic is a blurred brown-gray gradient; the nearest real neighbors are sharp road scenes.

**Diagnosis:** representation mismatch. The cVAE at 128×128 with 128-dim latent and MSE reconstruction produces samples that do not occupy the residential feature region. Not a training failure — a limit of the architecture and loss function at this resolution.

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

## The WanGP Image-to-Video showcase

A working Image-to-Video generation pipeline was built using Wan 2.2 FastWan 5B via the WanGP engine (`shared.api`). This demonstrates that a consumer laptop (RTX 4050, 6 GB VRAM) can generate synthetic nighttime driving video from real BDD100K images.

**Results:**
- Successfully generated 720p, 9-second videos in ~1 minute each.
- The pipeline preserves the ``mmgp`` memory manager (block-swapping, INT8 quantization) which is essential for running a 5B model on 6 GB VRAM.
- 5 showcase videos are available in ``docs/assets/generated_examples/``.
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

- **Not a generative success.** The cVAE failed to produce class-faithful samples.
- **Not a claim that all generative augmentation is useless.** DDPM, ControlNet, and image-to-image were not tested for downstream training (only the WanGP video showcase was tested as a demonstration of feasibility). The diagnostic toolkit is what they can be tested with.
- **Not a claim that the residential class is unlearnable.** The label is consistent; the model is weak (residential recall 0.41 at native resolution).
- **Not a claim that Week 5 is broken.** It reproduces published numbers exactly. It has a night weakness and a residential weakness, both documented.

## What future work should test

1. **Night augmentation.** Night is the dominant weakness. Simple tone-curve + ColorJitter augmentation, or a pretrained img2img model, could be tested on the val split first, then confirmed on test.
2. **Residential redefinition.** A binary "urban scene (city+residential) vs highway" task might be well-posed where the three-class task is not.
3. **Object-level features.** Week 9a's segmentation + detection features improved residential recall from 0.41 to 0.67. That's a non-generative path worth investigating.
4. **DDPM at higher resolution.** 128×128 is a hard limit for a VAE. Diffusion models at 128×128 or higher may produce class-faithful samples — testable with `scripts/diagnose_g1c.py` and `scripts/probe_pilot_features.py`.

None of these are in scope for UrbanGen-AI. The project closes with the negative finding, the diagnostic toolkit, and the WanGP I2V showcase.