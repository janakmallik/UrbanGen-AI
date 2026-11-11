
# Method 3 — Depth ControlNet (Plan + Placeholder)
**Status:** Not started — deferred. Metric calibration completed
(`outputs/controlnet_calibration/`); generation pipeline never run.
Method 3 was **deprioritized after Arm D-M2 (CycleGAN) succeeded** at seed 42.
See "Deferral reasoning" below. This document preserves the plan for future
work if the question is ever revisited.

## Objective
Test whether depth-conditioned diffusion can preserve urban scene structure better than CycleGAN (Method 2) or Img2Img (Method 1). The hypothesis: depth conditioning locks geometry, so the model can change lighting aggressively without hallucinating new content.

## Status update (2026-11-11)

See "Deferral reasoning" at the bottom of this document for the full
justification. In brief: the central research question was answered
with Method 2 (CycleGAN), running Method 3 would require a full second
pilot → curation → Arm D cycle, and the current single-seed finding is
defensible as-is.

## Architecture
- **Base model:** Stable Diffusion 1.5 (``stable-diffusion-v1-5/stable-diffusion-v1-5``)
- **Conditioning:** ControlNet Depth (``lllyasviel/control_v11f1p_sd15_depth``)
- **Depth extraction:** DPT-Hybrid-MiDaS (``Intel/dpt-hybrid-midas``)
- **Framework:** PyTorch + Hugging Face Diffusers
- **Precision:** FP16
- **Resolution:** 512×288

## Prerequisites status
| Component | Status |
|---|---|
| Metric calibration set | ✅ Complete (`outputs/controlnet_calibration/`) |
| SD 1.5 checkpoint | ✅ Cached |
| Depth ControlNet checkpoint | ✅ Available on HuggingFace (not yet downloaded) |
| DPT-Hybrid-MiDaS | ❌ Not cached (would need download, ~500 MB) |
| VRAM feasibility | ⏳ Not verified on 6 GB |

## Plan (updated 2026-11-08, refined from earlier discussion)

1. **Metric calibration** — 5 daytime + 5 nighttime reference images
   (`outputs/controlnet_calibration/`, completed).
2. **Environment audit** — verify cached models, verify VRAM fits.
3. **One-image pipeline smoke test** — verify generation works before scaling.
4. **30-output pilot** — 5 daytime sources × 2 strengths (0.60, 0.75) ×
   3 conditioning scales (0.6, 0.8, 1.0).
5. **Human inspection of 30 outputs** — do any produce night scenes with
   preserved structure? This is the decisive pilot gate.
6. **If successful → scale to ~500 outputs + filter + curate + Arm D-M3.**
7. **If unsuccessful → document failure mode, retain CycleGAN as alternative.**

Explicitly out of scope: YOLO-conditioned ControlNet (considered and rejected
in the Nov 8 discussion).

## Evaluation

Same metrics as Method 2, plus:
- SSIM (structural similarity)
- LPIPS (perceptual similarity)
- CLIP night score
- Luminance
- SegFormer paved surface fraction (with known caveat: unreliable at night)
- Segmentation mIoU (new)
- Downstream residential recall (for Arm D-M3 if pilot scales)
- Runtime and peak VRAM
- **Human inspection** — required for structural-preservation evidence

## Expected challenges
- **6 GB VRAM:** Depth ControlNet requires loading SD 1.5 (~2 GB) + ControlNet (~1.5 GB) + depth model (~500 MB) simultaneously. Peak VRAM may exceed 6 GB with attention slicing + CPU offload.
- **Segmentation filter failure repeated:** The SegFormer road detection issue from Method 2 will apply to Method 3 as well. Manual curation will be required.

## Reproducibility
- Model checkpoints are public (HuggingFace)
- Generation is deterministic given fixed seed
- No training involved (inference only)

## Next step
Method 3 is **deferred** as of 2026-11-11. If the project is ever resumed, the immediate next step is the environment audit and single-image smoke test (see Plan steps 2-3 above). No 30-output pilot until the smoke test passes.

## Deferral reasoning (2026-11-11)

Method 3 was deferred after Arm D-M2 (CycleGAN synthetic augmentation)
produced a qualified success at seed 42:

- Night residential recall: 0.3605 (A) → 0.4150 (C) → **0.5170 (D)**
- Paired bootstrap D − A: Δ=+0.158, 95% CI [+0.085, +0.229], excludes zero
- Trade-off: day macro-F1 drop of 0.0224 vs C, exceeding pre-registered 0.02 guardrail

The central research question ("do generative models produce useful synthetic
data?") was answered with Method 2. Running Method 3 would:

- Require a full second pilot → curation → Arm D-M3 cycle (~1-2 days)
- Produce a second result that could contradict or complicate the first
- Answer a *different* question: "is depth-conditioned diffusion better
  than CycleGAN for this specific use case?" — not "does synthetic augmentation work?"

The current single-seed finding is defensible and complete. Method 3 remains
available as a **separate, independent experiment** if the specific question
about depth-conditioning vs CycleGAN ever becomes important.

If resumed, the immediate next step is the 30-output pilot (steps 1-4 above).

## Related documents
- Method 2 log: ``docs/method_2_cyclegan_log.md``
- Findings (filter failure): ``docs/findings.md``
- Model comparison: ``docs/generative_image_methods_log.md``