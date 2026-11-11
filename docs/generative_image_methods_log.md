
# Generative Image Methods Log

Master index of all four generative methods tested in UrbanGen-AI.

## Cross-method comparison

| # | Method | Status | Approach | Output | Key metric |
|---|---|---|---|---|---|
| 1 | **Stable Diffusion Img2Img** | ❌ Failed | Pretrained diffusion, prompt-driven | None | N/A (structural hallucinations) |
| 2 | **CycleGAN from scratch** | ✅ Success | Unpaired GAN, trained on BDD100K | 282 synthetic night images | Structural preservation confirmed |
| 3 | **Depth ControlNet** | ⏳ Pending | Pretrained SD 1.5 + depth conditioning | Not yet generated | Not yet measured |
| 4 | **WanGP I2V (Wan 2.2)** | ✅ Showcase only | Pretrained video diffusion via WanGP | 2 showcase videos | Video generation feasible on 6 GB |

## Method 1 — Stable Diffusion Img2Img

**Status:** Failed. Documented negative result.

**What we tried:** Standard SD 1.5 Img2Img at strengths 0.25, 0.4, 0.6 to translate daytime residential scenes to night.

**What happened:** 
- Low strength: image stayed daytime
- High strength: hallucinated houses, deleted highway signs, invented streetlights
- Never found a setting that preserved structure AND produced night

**Diagnosis:** Img2Img cannot preserve scene geometry while changing lighting. The prompt overrides the source image.

**Full log:** ``docs/method_1_day_to_night_log.md``

## Method 2 — CycleGAN from Scratch

**Status:** Success. Positive result.

**What we tried:** Lightweight CycleGAN trained on 1,500 day + 1,500 night residential images for 20 epochs with LR decay.

**What happened:**
- Structural preservation: cars, buildings, road signs stay in place
- Night appearance: dark sky, streetlights, headlights
- Purple artifacts eliminated by epoch 20 (through LR decay)
- 57 minutes on RTX 4050, 2.1 GB peak VRAM

**Filter failure discovered:** After generating 500 candidates, we found that the automated filters (CLIP, luminance, paved_surface/SegFormer) were unreliable in both directions. The 282 accepted images require manual curation before use.

**Full log:** ``docs/method_2_cyclegan_log.md``
**Model card:** ``docs/model_cards/bdd100k_cyclegan_v1.md``
**Synthetic dataset card:** ``docs/dataset_cards/bdd100k_synthetic_cyclegan.md``

## Method 3 — Depth ControlNet
**Status:** Deferred (2026-11-11). Plan documented; metric calibration completed. Generation pipeline not run. See `docs/method_3_controlnet_log.md` for full plan and deferral reasoning.

**Plan:** Stable Diffusion 1.5 + ``lllyasviel/control_v11f1p_sd15_depth`` + ``Intel/dpt-hybrid-midas`` depth extractor. 512×288 resolution. 5-image pilot first, then scale to 500 if successful.

**Hypothesis:** Depth conditioning preserves scene geometry better than CycleGAN (which preserves pixels but not semantics) or Img2Img (which fails at high strength).

**Prerequisites verified:**
- ✅ ``stable-diffusion-v1-5/stable-diffusion-v1-5`` cached
- ✅ ``lllyasviel/control_v11f1p_sd15_depth`` available on HuggingFace
- ❌ ``Intel/dpt-hybrid-midas`` NOT cached (would need to download)

**Full plan:** Included in the audit rounds; not yet written to a method log.

## Method 4 — WanGP I2V Showcase

**Status:** Showcase complete. Not usable for Arm D (video format incompatible with image classifier).

**What we built:** Working Image-to-Video pipeline using Wan 2.2 FastWan 5B via WanGP's ``shared.api``. Generates 9-second 720p videos in ~1 minute on 6 GB VRAM.

**Why not for Arm D:** 
- Output is video, not images
- Only 2 showcase videos (not enough for training)
- Frame extraction + filtering was considered but not pursued (WanGP integration consumed the session)

**Full log:** ``docs/reproducibility.md``
**Model card:** ``docs/model_cards/wan_i2v_showcase_v1.md``

## Key findings across methods

1. **Pretrained models are not automatically better than from-scratch models.** SD 1.5 Img2Img (pretrained) failed; CycleGAN (from scratch) succeeded for our specific task. The pretrained model's prior overrode the source image.

2. **Pixel-level structure preservation is achievable.** CycleGAN preserves cars, buildings, road signs. This is harder than it sounds — the model learned that a road is a road, a car is a car, even when lighting changes drastically.

3. **Semantic-level structure preservation is harder.** The road is preserved at the pixel level, but SegFormer's segmentation of the road changes. This may be a real limitation of CycleGAN (it changes appearance so much that semantic segmentation fails) or a SegFormer failure (daytime-trained model cannot recognize night roads). We don't know which.

4. **No automated filter is reliable for generative augmentation.** Every filter we tried (CLIP, luminance, SegFormer) had high false reject and/or false accept rates. Manual visual inspection is the only reliable gate.

5. **Downstream evaluation is the real test.** We do not know if CycleGAN's synthetic data improves the Week 5 classifier. Arm D will answer this.
