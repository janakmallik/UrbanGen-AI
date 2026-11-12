# UrbanGen-AI: Final Research Report

## Executive Summary
UrbanGen-AI investigated whether synthetic urban-scene images could improve a downstream ResNet-50 classifier's recognition of underrepresented nighttime residential scenes. We tested five generative methods. The project concludes with one positive finding, three negative findings, and a documented daytime performance trade-off.

## The Findings
1. **CycleGAN (Method 2) works:** Training a from-scratch CycleGAN (1,500 day + 1,500 night) successfully learned day-to-night translation while preserving structural geometry.
2. **Synthetic Augmentation is valuable, but costly (Arm D):** Adding 276 curated CycleGAN images to the Week 5 training set improved nighttime residential recall from **0.3605 to 0.5170** (paired bootstrap 95% CI excludes zero).
3. **The Trade-off:** The Arm D model suffered a **-0.0241 daytime macro-F1 drop** relative to the baseline, violating the pre-registered 0.02 guardrail.
4. **cVAE Failed:** 128x128 VAE produced 0.6% residential fidelity (representation mismatch).
5. **SD Img2Img Failed:** Hallucinated buildings and destroyed structure.
6. **SD 1.5 + Depth ControlNet Failed:** While fitting in 6 GB VRAM, it generated cartoon/video-game priors at 512x288, failing the human inspection gate (0/30 acceptable).

## Method 2 vs Method 3
Method 3 (Depth ControlNet) was evaluated as a potential replacement for Method 2 (CycleGAN). It was deprioritized because pretrained diffusion models at low resolution cannot preserve photorealism without excessive prompt engineering. CycleGAN's from-scratch training on the specific dataset proved superior for this constraint.

## Reusable Artifacts
- `src/urbangen/data/preprocessing.py` (for UrbanMind-AI Week 12)
- `tests/unit/test_week5_pipeline.py` (Golden test)
- `docs/robustness_report_v2.md`

## Limitations
- Arm D results are based on a single training seed (42).
- Checkpoints were deleted post-training, preventing post-hoc daytime statistical audits.