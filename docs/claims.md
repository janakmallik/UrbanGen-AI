# Claims Ledger

Every quantitative claim in UrbanGen-AI, with evidence file and status.

**Status legend:**
- ✅ **Verified** — measured and reproducible from a committed script, with the stated n and split
- ⚠️ **Provisional** — measured but with caveats (small n, exploratory, unverified source)
- ❌ **Retracted** — previously claimed, now shown to be wrong or artifact

**Note on split distinction:** Week 5 was evaluated on two different test splits. The **full test split** (n=4,932) is the complete `resnet_test` partition. The **clean test split** (n=4,745) excludes 187 images via `resnet_test_excluded.json`. Metrics are reported with the split named explicitly. Do not compare across splits without noting the difference.

---

## Verified claims

| Claim | Evidence | Notes |
|---|---|---|
| Week 5 full-test macro-F1 = 0.6984 | `outputs/pipeline_verification/pipeline_verification.json` (condition a, n=4,932) | Reproduces exactly; verified against training script |
| Week 5 full-test accuracy = 0.7800 | Same | Reproduces exactly |
| Week 5 full-test residential recall = 0.4121 | Same | Reproduces exactly |
| Week 5 clean-test residential recall = 0.4078 | Same (condition a, n=4,745) | Reproduces exactly; 0.408 is the rounded value used in older docs |
| Class mapping: 0=city, 1=highway, 2=residential | `UrbanMind-AI/src/urbanintel/data/bdd100k.py` | `VIABLE_CLASSES` order verified |
| Week 5 eval transform: Resize(256)→CenterCrop(224)→ImageNet norm | `UrbanMind-AI/scripts/train_bdd100k_resnet50.py` lines 136–142 | Verified |
| cVAE samples are 0.6% residential in a real-trained classifier | `outputs/g1c_diagnostic/diagnostic.json` | 90% classified as highway; 3,000 synthetic samples tested |
| cVAE trained correctly (no collapse) | `outputs/cvae_beta1_seed42/` training logs, `docs/model_cards/bdd100k_cvae_v1.md` | 128/128 active dims, KL stable 76→88, converged epoch 25 |
| Night highway recall drops 0.136 vs daytime | `docs/robustness_report_v2.md` within-class comparison | n_A=614, n_B=506; both > 145 |
| Night residential recall drops 0.292 vs daytime | Same | n_A=397, n_B=147; both > 145 |
| Rainy city street recall rises 0.042 vs clear | Same | n_A=1,599, n_B=226 |
| Method 2 generated 282 images at 56.4% survival from 500 sources | `outputs/method_2_synthetic/filter_log.csv` | Rejections: clip=5, luminance=5, paved_surface=208 |
| Method 2 source selection: 500 filenames, seed-42 reproducibility | `outputs/method_2_synthetic/source_selection.json`, `source_split_membership.json` | Ordered reconstruction matches recorded selection; zero overlap with val_clean, test_clean, or reserved splits |
| WanGP generates 720p ~3-second video in ~1 min on 6 GB VRAM | `docs/reproducibility.md`, `scripts/generate_i2v_showcase.py` | Reproducible given fixed environment |

---

## Provisional claims (measured with caveats)

| Claim | Evidence | Caveat |
|---|---|---|
| cVAE samples are diverse, not obviously memorized | `outputs/label_audit/` per-image analysis | 0/3,000 above 0.95 nearest-neighbor cosine; pairwise cosine distance 0.365. Evidence against memorization, **not** proof of meaningful diversity in every sense |
| Residential label consistency | `outputs/label_audit/suspect_residential.png`, `confident_residential.png` | Visual inspection only; no numerical label-accuracy check was run |
| Residential OOF recall ≈ 0.70 | `outputs/label_audit/per_image_analysis.json` | OOF precision is only 0.37 (8,216 city images predicted as residential); 0.70 recall is inflated by high false-positive rate |
| Rainy highway recall drops 0.235 vs clear | `docs/robustness_report_v2.md` | n_A=740, n_B=63; small n on comparison set |
| Overcast outperforms clear (macro-F1 0.738 vs 0.685) | Same | Within-class comparison not run; class-mix confound not ruled out |
| Snowy macro-F1 (0.654) vs clear (0.685) | Same | Not tested within-class; class-mix confound likely |
| Daytime residential recall 0.476, night 0.184 | Same | 0.476 is itself lower than expected for a daytime subset; residential remains the weakest class overall |
| Squashed frame (condition b) outperforms training pipeline by +0.030 macro-F1 / +0.110 residential recall | `outputs/pipeline_verification/pipeline_verification.json` | Test-set observed; **not validated on val split**; do not adopt without val-first validation |
| Rainy residential recall drop (−0.101) | `docs/robustness_report_v2.md` | n_B=37; CI includes zero; not statistically supported |
| CycleGAN achieves structural preservation with night appearance on 5 holdout images | `outputs/method_2_resume/metrics.csv` | Epoch 20, **n=5 held-out images**, CLIP night score > 0.93, SSIM ~0.2. Small sample; not established as robust structural preservation across the full distribution |
| SegFormer-B0 is unreliable as a nighttime structural filter | `outputs/method_2_synthetic/rejected_inspection/` (5 inspected paved_surface rejections) | All 5 inspected were visually good night transformations; **sample size 5**; this suggests unreliability, does not prove it |
| CLIP night + luminance filters over-reject good images | `outputs/method_2_synthetic/clip_lum_inspection/` (10 inspected) | 8/10 inspected clip/luminance rejections were visually good night images |
| The 282 accepted set contains some structural failures | Inspection of 20 accepted images (`outputs/method_2_synthetic/accepted_inspection/`) | 3–5 of 20 inspected showed near-black output or smeared structure. **Do not extrapolate 15–25% to the full 282**; full curation will determine the actual rate |
| No automated filter reliably separates good from bad synthetic night images | Combined evidence: paved_surface (208 rejections, 5/5 inspected false), clip/luminance (10 inspected, 8 false), accepted set (20 inspected, 3–5 false) | **Based on 35 inspected images total.** Suggestive, not proven. Full visual curation is the planned next step |
| CycleGAN-generated night images will improve Week 5 residential recall | Not yet tested (Arm D) | Structural + appearance valid at n=5; classifier improvement requires empirical validation |

---

## Retracted claims

| Claim | Original evidence | Why retracted |
|---|---|---|
| Week 5 residential recall = 0.124 | Original `robustness_report.md` (Step 0b) | **Pipeline mismatch.** Measured on 128×128 cache upsampled to 224×224, not Week 5's training pipeline. Correct full-test value is 0.4121; correct clean-test value is 0.4078 |
| Night is not a weakness | Same report | **Wrong pipeline + macro-F1 hides per-class effects.** Corrected report shows residential recall drops 0.292 at night (CI excludes zero). Night is a major weakness |
| Rainy is a real weather gap (macro-F1 based) | Same report | Partially retracted. The macro-F1 gap is real, but within-class analysis shows the effect is concentrated in highway (−0.235, small n=63). Rainy residential is not significantly affected. Rainy city street is significantly *better* |
| Snowy is a moderate gap (macro-F1 based) | Same report | Not tested within-class. The macro-F1 number exists but may be class-mix confound. Marked provisional above |
| Residential "class is fuzzy" | Pre-visual inspection | **Retracted after visual inspection.** Both grids show residential-looking scenes. The label is consistent. What's real is that OOF recall (0.70) is inflated by precision (0.37) |
| Video leakage in OOF CV inflated recall | Audit assumption | **Retracted.** BDD100K 100K set has one frame per video (10th second), so adjacent-frame leakage doesn't apply. Weaker issue: some rides share videos, so near-duplicates across folds are possible but rare |
| The 282 accepted images are clean synthetic data | Showed to contain structural failures (near-black, smeared) when inspecting 20-image sample | **Retracted.** Manual curation required. Sample was 20 images; actual failure rate to be determined by full 282-image inspection |

---

## Pending verification

| Claim | Not yet verified | What's needed |
|---|---|---|
| Squashed frame adoption | Val-split confirmation | Paired bootstrap on val, then test once |
| Overcast > clear | Within-class paired test | Same images impossible (different weathers); unpaired bootstrap by class |
| Residential class definition | Redefinition study | Merge residential + city → "urban street" vs highway, or split residential subclasses |
| Arm D downstream improvement | Arm D run | Pre-registered protocol, curated synthetic set, paired bootstrap vs Arm A (0.3605 recall, 0.6475 macro-F1) |
| Week 12 serving impact | Not applicable | UrbanMind-AI scope, not UrbanGen-AI |

---

## How this ledger is maintained

- Every commit that touches a metric or a claim updates this file
- Every AI-written document that asserts a number must reference a claim here
- Claims that lack an `outputs/` file or a `docs/` file reference are flagged and removed before the next commit
- A claim marked "Verified" is reproducible by running the named script, on the named split, with the stated n
- Claims based on visual inspection are Provisional unless the inspection covered the full population being claimed about
- "Provisional" does not mean "wrong" — it means "measured, but the measurement has documented limitations"