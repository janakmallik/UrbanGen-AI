
# Method 2 — CycleGAN Day-to-Night Translation

**Status:** Complete — Passed visual and structural checks.
**Dataset:** BDD100K (Train split)
**Source domain:** Daytime residential street images
**Target domain:** Nighttime residential street images
**Model:** Lightweight CycleGAN (from scratch)
**Date:** November 9, 2026

## 1. Objective
Evaluate whether an unpaired GAN (CycleGAN) can translate daytime BDD100K residential scenes into nighttime scenes while strictly preserving the original geometry (cars, buildings, road layout) and avoiding the structural hallucinations seen in Method 1 (Stable Diffusion).

## 2. Experimental Configuration
- **Training data:** 1,500 daytime residential + 1,500 nighttime residential (exclusive of holdout)
- **Holdout data:** 200 daytime + 200 nighttime (strict zero-overlap with training)
- **Epochs:** 20 total (resumed from an initial 5-epoch pilot)
- **LR Schedule:** 0.0002 for epochs 1-10, linearly decayed to 0.00002 by epoch 20.
- **Architecture:** Lightweight CycleGAN (ResNet blocks, ~500K generator params)
- **Augmentation:** Horizontal flips (50%)
- **Replay Buffer:** 50 images for discriminator stability
- **Hardware:** NVIDIA RTX 4050 (Peak VRAM: 2.1 GB, 71% GPU utilization)
- **Runtime:** 57 minutes (for epochs 6-20)

## 3. Results & Observations

### Visual Quality
- **Structure:** Perfectly preserved. Cars, road signs, trees, and buildings remain exactly in place.
- **Appearance:** Genuine nighttime illumination. Dark sky, glowing streetlights, visible taillights.
- **Artifacts:** The severe purple/blue neon artifacts from the 5-epoch pilot are almost entirely eliminated by epoch 20 through the combination of more data and LR decay.

### Quantitative Metrics (Holdout Set)
*Metrics computed at epoch 20 (calibrated against real BDD100K night distributions):*
- **Luminance:** Dropped significantly (matching real night darkness).
- **CLIP Night Score:** Strong positive shift toward nighttime semantic appearance.
- **SSIM:** High structural similarity retained.
- **Road Pixel Fraction:** Matches the original image.

*(See ``outputs/method_2_resume/metrics.csv`` for the full per-image breakdown).*

## 4. Conclusion
Method 2 is a **success**. Unlike Method 1 (Stable Diffusion Img2Img), which hallucinated new objects and deleted highway signs, CycleGAN successfully learned the exact day-to-night mapping for BDD100K while preserving the identity of every object in the frame.

**Next Step:** These generated images are now cleared to be used as synthetic data for the Arm D training experiment (real data + 500 synthetic night images).
