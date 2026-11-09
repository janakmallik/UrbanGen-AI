# Claims Ledger

Every quantitative claim in UrbanGen-AI, with evidence file and status.

**Status legend:**
- ✅ **Verified** — measured and reproducible from a committed script
- ⚠️ **Provisional** — measured but with caveats (small n, exploratory, unverified source)
- ❌ **Retracted** — previously claimed, now shown to be wrong or artifact

---

## Verified claims

| Claim | Evidence | Notes |
|---|---|---|
| Week 5 test macro-F1 = 0.6984 | `outputs/pipeline_verification/pipeline_verification.json` (condition a on 4,932 full test) | Reproduces exactly; verified against training script |
| Week 5 test accuracy = 0.7800 | Same | Reproduces exactly |
| Week 5 residential recall = 0.4121 | Same | Reproduces exactly |
| Class mapping: 0=city, 1=highway, 2=residential | `UrbanMind-AI/src/urbanintel/data/bdd100k.py` | `VIABLE_CLASSES` order verified |
| Week 5 eval transform: Resize(256)→CenterCrop(224)→ImageNet norm | `UrbanMind-AI/scripts/train_bdd100k_resnet50.py` lines 136-142 | Verified |
| cVAE samples are 0.6% residential in a real-trained classifier | `outputs/g1c_diagnostic/diagnostic.json` | 0/3000 above 0.95 (not memorized), but 90% classified as highway |
| cVAE samples are diverse, not memorized | `outputs/label_audit/` (per-image analysis) | Pairwise cosine 0.365, 0/3000 above 0.95 to real residential |
| cVAE trained correctly (no collapse) | `outputs/` training logs, `docs/model_cards/bdd100k_cvae_v1.md` | 128/128 active dims, KL stable 76→88, converged epoch 25 |
| Rainy highway recall drops 0.235 vs clear | `docs/robustness_report_v2.md` within-class comparison | n_A=740, n_B=63; ⚠️ small n on B |
| Night highway recall drops 0.136 vs daytime | Same | n_A=614, n_B=506; both > 145 |
| Night residential recall drops 0.292 vs daytime | Same | n_A=397, n_B=147; both > 145 |
| Rainy city street recall rises 0.042 vs clear | Same | n_A=1599, n_B=226 |
| Residential labels are mostly consistent (not noisy) | `outputs/label_audit/suspect_residential.png`, `confident_residential.png` | Visual inspection: both grids show residential-looking scenes |
| CycleGAN achieves structural preservation with night appearance on 5 holdout images | `outputs/method_2_resume/metrics.csv` | Epoch 20, 5 held-out images, CLIP night score > 0.93, SSIM ~0.2 |

---

## Provisional claims (measured with caveats)

| Claim | Evidence | Caveat |
|---|---|---|
| Residential OOF recall ≈ 0.70 | `outputs/label_audit/per_image_analysis.json` | OOF precision is only 0.37 (8,216 city images predicted as residential); 0.70 recall is inflated by high false-positive rate |
| Overcast outperforms clear (macro-F1 0.738 vs 0.685) | `docs/robustness_report_v2.md` | Within-class comparison not run; class-mix confound not ruled out |
| Squashed frame (condition b) outperforms training pipeline (condition a) by +0.030 macro-F1 / +0.110 residential | `outputs/pipeline_verification/pipeline_verification.json` | Test-set observed; **not validated on val split**; do not adopt without val-first validation |
| Rainy residential recall drop (−0.101) | `docs/robustness_report_v2.md` | n_B=37; CI includes zero; not statistically supported |
| Snowy macro-F1 (0.654) vs clear (0.685) | Same | Not tested within-class; class-mix confound likely |
| Daytime residential recall 0.476, night 0.184 | Same | Correct, but 0.476 is itself lower than expected for a "daytime" subset — the residency class remains the weakest class |
| CycleGAN-generated night images will improve Week 5 residential recall | Not yet tested (Arm D) | Structural + appearance valid, but classifier improvement requires empirical validation |

---

## Retracted claims

| Claim | Original evidence | Why retracted |
|---|---|---|
| Week 5 residential recall = 0.124 | Original `robustness_report.md` (Step 0b) | **Pipeline mismatch.** Measured on 128×128 cache upsampled to 224×224, not Week 5's training pipeline (Resize(256)→CenterCrop(224) on native JPEGs). Correct value is 0.408. |
| Night is not a weakness | Same report | **Wrong pipeline + macro-F1 hides per-class effects.** Corrected report shows residential recall drops 0.292 at night (CI excludes zero). Night is a major weakness. |
| Rainy is a real weather gap (macro-F1 based) | Same report | Partially retracted. The macro-F1 gap is real, but the **within-class analysis** shows the effect is concentrated in highway (−0.235, small n=63). Rainy residential is not significantly affected. Rainy city street is significantly *better*. |
| Snowy is a moderate gap (macro-F1 based) | Same report | Not tested within-class. The macro-F1 number exists but may be class-mix confound. Marked provisional. |
| Residential "class is fuzzy" | Pre-visual inspection | **Retracted after visual inspection.** The suspect and confident grids both show residential-looking scenes. The label is consistent. What's real is that OOF recall (0.70) is inflated by precision (0.37). |
| Video leakage in OOF CV inflated recall | Audit assumption | **Retracted.** BDD100K 100K set has one frame per video (10th second), so adjacent-frame leakage doesn't apply. Weaker issue: some rides share videos, so near-duplicates across folds are possible but rare. |

---

## Pending verification

| Claim | Not yet verified | What's needed |
|---|---|---|
| Squashed frame adoption | Val-split confirmation | Paired bootstrap on val, then test once |
| Overcast > clear | Within-class paired test | Same images impossible (different weathers); unpaired bootstrap by class |
| Residential class definition | Redefinition study | Merge residential + city → "urban street" vs highway, or split residential subclasses |
| Week 12 serving impact | Not applicable | UrbanMind-AI scope, not UrbanGen-AI |

---

## How this ledger is maintained

- Every commit that touches a metric or a claim updates this file
- Every AI-written document that asserts a number must reference a claim here
- Claims that lack a `outputs/` file or a `docs/` file reference are flagged and removed before the next commit
- A claim marked "Verified" is reproducible by running the named script


# Claims Ledger

Every quantitative claim in UrbanGen-AI, with evidence file and status.

**Status legend:**
- ✅ **Verified** — measured and reproducible from a committed script
- ⚠️ **Provisional** — measured but with caveats
- ❌ **Retracted** — previously claimed, now shown to be wrong or artifact

## Verified claims

| Claim | Evidence | Notes |
|---|---|---|
| Week 5 test macro-F1 = 0.6984 | ``outputs/pipeline_verification/pipeline_verification.json`` | Reproduces exactly on native JPEGs |
| Week 5 test accuracy = 0.7800 | Same | Reproduces exactly |
| Week 5 residential recall = 0.4121 | Same | Reproduces exactly |
| Class mapping: 0=city, 1=highway, 2=residential | ``src/urbangen/data/preprocessing.py`` | VIABLE_CLASSES order verified |
| cVAE samples are 0.6% residential | ``outputs/g1c_diagnostic/diagnostic.json`` | 90% classified as highway |
| cVAE samples are diverse, not memorized | ``outputs/label_audit/`` | 0/3000 above 0.95 cosine |
| Night residential recall drops 0.292 vs daytime | ``docs/robustness_report_v2.md`` | Both n > 145 |
| WanGP generates 720p 9-second video in ~1 min on 6 GB VRAM | ``docs/reproducibility.md`` | Verified via ``generate_i2v_showcase.py`` |

## Provisional claims

| Claim | Caveat |
|---|---|
| Residential labels are mostly correct | Visual inspection only; no numerical label-accuracy check was run |
| Squashed frame (condition b) outperforms training pipeline | Test-set observed; not validated on val split |

## Retracted claims

| Claim | Why retracted |
|---|---|
| Week 5 residential recall = 0.124 | Pipeline mismatch (128×128 cache). Correct value is 0.408. |
| Night is not a weakness | Wrong pipeline. Night residential recall drops 0.292. |
| Residential "class is fuzzy" | Retracted after visual inspection. The label is consistent. |