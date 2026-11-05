# Model Card: BDD100K cVAE (G1b)

## What this is

A conditional variational autoencoder trained from scratch on BDD100K scene images (128×128, 3 classes: city street / highway / residential). Trained for 30 epochs, β=1.0, seed 42.

**This is a first-pass generative baseline.** It does not claim to be SOTA, and its class conditioning is weak (see "Known limitations" below). It exists to establish the training pipeline and to provide a comparison point for more capable generative models (G2 DDPM, G3 DCGAN).

## Architecture

- **Encoder:** 4 stride-2 conv blocks (32→64→128→256 channels), BatchNorm + ReLU. Flattened to 256×8×8 = 16,384 dims.
- **Class conditioning:** 16-dim class embedding concatenated with flattened encoder features (before μ/logσ² heads) and with latent z (before decoder).
- **Latent:** 128-dim, standard reparameterization.
- **Decoder:** transposed-conv path (256→128→64→32→3) with Sigmoid output.
- **Parameters:** 7,654,099.

## Training configuration (frozen)

| Parameter | Value |
|---|---|
| Image size | 128×128 (aspect-preserved via 128×72 content + black padding) |
| Loss | `recon_sse + β·kl`, β=1.0 |
| Recon loss | Sum of squared error over content pixels (rows 28:100), mean over batch |
| KL loss | Sum over 128 latent dims, mean over batch. Computed in FP32. |
| Optimizer | AdamW (lr=1e-3, wd=1e-4), cosine annealing over 30 epochs |
| Batch size | 32 |
| Precision | FP16 AMP with FP32 KL and gradient clipping at norm 1.0 |
| Seed | 42 |
| Epochs | 30 |

## Training results

| Metric | Epoch 1 | Epoch 25 (best) | Epoch 30 |
|---|---|---|---|
| Train recon_mse | 0.0152 | 0.0078 | 0.0078 |
| Train KL | 76.2 | 87.7 | 87.8 |
| Val recon_mse | 0.0094 | 0.0068 | 0.0068 |
| Val KL | 73.5 | 87.2 | 87.2 |
| Active dims | 128/128 | 128/128 | 128/128 |
| Best val loss | — | **273.9** | — |

**Convergence:** recon_mse plateaued after epoch 20. Additional training would produce marginal gains.

**Comparison to class-mean baseline:** 7.6× better (0.0068 vs 0.0592 per-pixel MSE). The latent is encoding real per-image information, not just class labels.

## Sample quality

Samples are **visibly blurry** — expected for a standard VAE with MSE reconstruction at 128×128. Sample grids at epochs 5/10/15/20/25/30 preserved in `outputs/cvae_beta1_seed42/samples/` (gitignored). Final grid in this repo at `docs/model_cards/cvae_samples_beta1_seed42.png`.

**Memorization analysis:**
- MSE to nearest real residential: mean 0.026, median 0.027 (approximately 2.3× closer than the class-mean baseline, but not near-duplicate).
- Perceptual nearest-neighbor (ResNet-50 features, content region only): mean cosine 0.624, **0/64 above 0.90** — no samples near-duplicate any real training image.

**Diversity:** Pairwise cosine distance within each class: 0.109 (city), 0.113 (highway), 0.121 (residential). Samples within a class are meaningfully different, not mode-collapsed.

**Class conditioning:** WEAK. The three rows in the sample grid are barely distinguishable. Highway shows faint horizontal band structure. **Residential shows no visible residential signal** (no greenery, no narrower roads).

## Known limitations

1. **Blur.** Samples are severely blurry. This is a property of standard VAE + MSE reconstruction, not a training bug. Perceptual losses or diffusion models produce sharper outputs.

2. **Weak class conditioning.** The 16-dim class embedding concatenated at only two points (encoder input, decoder input) is insufficient for strong 3-way class conditioning at this scale. Highway shows faint signal; residential shows none visibly. Possible improvements: larger class embedding, conditioning at each decoder block, auxiliary classifier loss on latent.

3. **Evaluation limitation.** The Week 5 ResNet-50 classifier used for the conditional consistency check achieves only **5.4% accuracy on real residential images** (diagnostic: `outputs/diagnostic/classifier_pipeline_diagnostic.json`). It is effectively a city-street detector (92.2% accuracy on real city street images). **It cannot be used to evaluate residential generation.** The 0% synthetic residential accuracy result is an artifact of classifier weakness, not evidence about the cVAE.

4. **Padding.** The 128×128 images contain 43.75% black padding (rows 0:28 and 100:128). The masked reconstruction loss excludes these rows. The generated images may produce arbitrary content there (visualization zeroes them).

5. **Not validated for downstream augmentation.** G1b does not establish that these synthetic samples improve downstream recognition. That is a separate experiment (G1c).

## What this model is NOT

- Not a claim that synthetic data improves downstream classification.
- Not a claim that the class conditioning is broken — only that it is weak.
- Not a photorealistic generator — blur is expected at this architecture and loss.
- Not validated against a classifier that can judge residential quality.

## Citation

Standard VAE architecture (Kingma & Welling, 2013), standard practice for both reconstruction and KL reduction. UrbanGen-AI's contribution is the training pipeline, the diagnostic methodology, and the documented finding that the Week 5 classifier cannot evaluate residential generation.

## Changelog

| Date | Notes |
|---|---|
| 2026-11-05 | Initial model card. 30-epoch training, β=1, seed 42. Samples documented with memorization, diversity, and classifier limitations. |

