# UrbanGen-AI — Reproducibility Documentation

## 1. Project Overview

UrbanGen-AI generates realistic urban image-to-video showcase samples using WanGP and Wan 2.2 FastWan.

**Purpose:** Build a reproducible urban video generation pipeline using public BDD100K images.

## 2. Environment

| Component            | Configuration                     |
| -------------------- | --------------------------------- |
| OS                   | Windows 11                        |
| Python               | WanGP virtual environment         |
| GPU                  | NVIDIA RTX 4050 Laptop, 6 GB VRAM |
| Generation framework | WanGP v19.21                      |
| Model                | FastWan2.2-TI2V-5B                |
| Attention            | SDPA                              |
| WanGP profile        | 4                                 |
| Dataset              | BDD100K                           |

Project locations:

- UrbanGen-AI: ``C:\Users\Janak Mallik\UrbanGen-AI``
- WanGP: ``C:\Users\Janak Mallik\WanGP``
- Dataset: ``~/ml-data/bdd100k/images/100k/train``

## 3. Generation Configuration

Current script: ``scripts/generate_i2v_showcase.py``

| Parameter            | Value  |
| -------------------- | ------ |
| DURATION_SECONDS     | 3      |
| FPS                  | 24     |
| INFERENCE_STEPS      | 3      |
| Image selection seed | 42     |
| Guidance scale       | 1      |
| Flow shift           | 3      |

## 4. Dataset and Image Reproducibility

Source images are selected from BDD100K training images using a fixed random seed (``42``).

## 5. Output Management

Output directory: ``outputs/generated_videos/``

The script checks whether a destination already exists and refuses to overwrite it.

WanGP-generated MP4 segments are collected through ``result.generated_files``.

The final video is assembled using FFmpeg.

## 6. Reproduction Command

From the UrbanGen-AI project root:

``````powershell
..\WanGP\env_venv\Scripts\python.exe .\scripts\generate_i2v_showcase.py --count 5
``````

## 7. Reproducibility Limitations

A fixed seed alone does not guarantee bit-for-bit identical video output. Results can vary with GPU hardware, PyTorch/CUDA versions, WanGP version, and model quantization.

---

## Phase 2 Reproduction (Seed-42 Pilot)

To reproduce the D-138 and D-bal pilot results:

1. **Build the dose-matched manifests:**
   ```powershell
   uv run python scripts/build_automated_d_bal.py
   ```
   *This creates `d_138_manifest.json` (138 residential) and `d_bal_manifest.json` (138 residential + 69 city + 69 highway).*

2. **Train D-138 (Matched Control):**
   ```powershell
   uv run python scripts/train_week5_augmented_arms_v3.py --arm D-138 --seed 42
   ```

3. **Train D-bal (Shortcut Control):**
   ```powershell
   uv run python scripts/train_week5_augmented_arms_v3.py --arm D-bal --seed 42
   ```

4. **Summarize results:**
   ```powershell
   uv run python scripts/train_week5_augmented_arms_v3.py --summarize
   ```

*Expected results are committed in `outputs/week5_arms/results_D-138_seed42.json`, `results_D-bal_seed42.json`, and documented in `docs/phase2_pilot_findings.md`.*