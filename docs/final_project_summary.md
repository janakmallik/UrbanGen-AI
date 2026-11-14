# UrbanGen-AI: Final Project Summary

**Status:** Closed as a single-seed research pilot.

## What the project accomplished
1. **Tested 4 generative methods** for urban scene synthesis (cVAE, SD Img2Img, CycleGAN, SD ControlNet).
2. **Only CycleGAN succeeded** in generating structurally valid nighttime residential images.
3. **Proved synthetic augmentation works** for a specific weakness: nighttime residential recall improved from **0.36 to 0.52** (276 images) and **0.58** (138 images).
4. **Documented the trade-off:** All synthetic arms violated the 0.02 daytime macro-F1 guardrail.
5. **Built a reusable diagnostic toolkit** (parity tests, golden tests, pipeline verifiers) that can be applied to any future generative pipeline.

## The core finding
> CycleGAN synthetic data is useful for nighttime residential recognition, but it costs daytime accuracy. The effect is dose-responsive (138 images gave the best pilot result), and mixing in non-residential synthetic data (D-bal) did not help.

## Limitations
- **Single seed:** No multi-seed variance was measured.
- **Confounded shortcut control:** D-bal used uncurated city/highway images.
- **No external validation:** Results are specific to BDD100K and ResNet-50.

## Where to look
- **Full details:** `docs/phase2_pilot_findings.md`
- **Main results:** `outputs/week5_arms/summary.json`
- **Generative method analysis:** `docs/findings.md`
- **Lessons learned:** `docs/lessons.md`
- **Reproduction commands:** `docs/reproducibility.md`