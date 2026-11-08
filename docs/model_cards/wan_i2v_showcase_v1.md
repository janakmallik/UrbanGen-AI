# Model Card: Wan 2.2 FastWan Image-to-Video Showcase

## What this covers
A synthetic video generation pipeline using Wan 2.2 FastWan (5B parameter model) via the WanGP engine. Generates nighttime urban dashcam footage from real BDD100K images.

## Architecture
- **Model:** FastWan2.2-TI2V-5B-Full-Diffusers
- **Parameters:** 5 billion
- **Pipeline:** Text + Image to Video (TI2V)
- **LoRA:** FastWan LoRA (3-step inference)
- **Memory manager:** WanGP ``mmgp`` (block-swapping, INT8 quantization)

## Training configuration
- **Dataset:** Pre-trained by the Wan team (no additional training done)
- **Inference steps:** 3 (distilled LoRA)
- **Guidance scale:** 1
- **Resolution:** 1280×720
- **Duration:** 9 seconds
- **FPS:** 24

## Results
- **Generation time:** ~1 minute per video on RTX 4050 6 GB.
- **Output quality:** Coherent nighttime residential street scenes with recognizable houses, streetlights, and vehicles.
- **Limitations:** Temporal drift on longer videos. Some structural morphing.

## Known limitations
- Not bit-for-bit reproducible (depends on GPU hardware, PyTorch version, WanGP version).
- Generated videos are synthetic and must be labeled as such.
- May not generalize to all BDD100K scenes (tested on nighttime residential).

## What this model is NOT
- Not a replacement for real training data.
- Not evaluated for downstream classifier improvement (that was the cVAE experiment).
- Not a general-purpose video generator.

## Reproduce
See ``docs/reproducibility.md`` for exact commands and environment.