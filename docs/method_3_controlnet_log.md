
# Method 3 — Depth ControlNet (Plan + Placeholder)

**Status:** Not started. This is a planning document, to be updated when Method 3 runs.

## Objective
Test whether depth-conditioned diffusion can preserve urban scene structure better than CycleGAN (Method 2) or Img2Img (Method 1). The hypothesis: depth conditioning locks geometry, so the model can change lighting aggressively without hallucinating new content.

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
| SD 1.5 checkpoint | ✅ Cached |
| Depth ControlNet checkpoint | ✅ Available on HuggingFace (not yet downloaded) |
| DPT-Hybrid-MiDaS | ❌ Not cached (would need download, ~500 MB) |
| VRAM feasibility | ⏳ Not verified on 6 GB |

## Plan
1. **Environment audit** (verify cached models, verify VRAM fits)
2. **Single-image smoke test** — one source image, verify pipeline loads and produces output
3. **5-image pilot** — 5 sources × 6 configs (2 strengths × 3 conditioning scales) = 30 generations
4. **Inspect 30 outputs** — do any produce night scenes with preserved structure?
5. **Scale up if successful** — 500 generations + filtering + Arm D training
6. **If unsuccessful** — document failure mode, move on

## Evaluation
Same metrics as Method 2:
- SSIM (structural similarity)
- LPIPS (perceptual similarity)
- CLIP night score
- Luminance
- SegFormer paved surface fraction (with known caveat: unreliable at night)

## Expected challenges
- **6 GB VRAM:** Depth ControlNet requires loading SD 1.5 (~2 GB) + ControlNet (~1.5 GB) + depth model (~500 MB) simultaneously. Peak VRAM may exceed 6 GB with attention slicing + CPU offload.
- **Segmentation filter failure repeated:** The SegFormer road detection issue from Method 2 will apply to Method 3 as well. Manual curation will be required.

## Reproducibility
- Model checkpoints are public (HuggingFace)
- Generation is deterministic given fixed seed
- No training involved (inference only)

## Next step
Implement environment audit + single-image smoke test. No 5-image pilot until smoke test passes.

## Related documents
- Method 2 log: ``docs/method_2_cyclegan_log.md``
- Findings (filter failure): ``docs/findings.md``
- Model comparison: ``docs/generative_image_methods_log.md``
