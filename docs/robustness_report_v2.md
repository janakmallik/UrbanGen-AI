# Week 5 ResNet-50 — Robustness Report (Corrected)

Corrected report using condition (a): native JPEGs with `Resize(256) → CenterCrop(224) → ImageNet norm`. Supersedes `robustness_report.md` which used the 128×128 cache pipeline (see `outputs/pipeline_verification/`).

## Verification and provenance

- **Pipeline verified** against `UrbanMind-AI/scripts/train_bdd100k_resnet50.py`: eval transform is `Resize(256) → CenterCrop(224) → ToTensor → Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`. Condition (a) matches this exactly.
- **Class mapping verified**: `VIABLE_CLASSES = ["city street", "highway", "residential"]` → `LABEL_TO_ID = {name: i for i, name in enumerate(VIABLE_CLASSES)}`, so 0=city, 1=highway, 2=residential.
- **Reference check ✅ PASS**: full test (n=4,932) accuracy=0.7800 (expected 0.7800), macro-F1=0.6984 (expected 0.6984), residential recall=0.4121 (expected 0.4121).
- **Seed**: 42
- **Bootstrap resamples**: 1000

## Data integrity

- Clean split manifest: 4,745 filenames
- Excluded split manifest: 187 filenames
- Full test (clean ∪ excluded): 4,932 filenames
- Missing images: 0
- Missing metadata: 0

## Overall (clean test set, 4,745 images)

- Accuracy: 0.7819
- Balanced accuracy: 0.6693
- Macro-F1: 0.6986
- city street — P=0.7992 R=0.8839 F1=0.8394 (n=2945)
- highway — P=0.7446 R=0.7163 F1=0.7302 (n=1209)
- residential — P=0.7415 R=0.4078 F1=0.5262 (n=591)

## By weather

| Weather | n | acc | macro-F1 | City R | Hwy R | Res P | Res R | Res F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clear | 2627 | 0.778 | 0.685 | 0.882 | 0.711 | 0.709 | 0.372 | 0.487 |
| foggy ⚠️ | 7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| overcast | 582 | 0.782 | 0.738 | 0.829 | 0.805 | 0.820 | 0.526 | 0.641 |
| partly cloudy | 350 | 0.800 | 0.739 | 0.843 | 0.867 | 0.826 | 0.432 | 0.567 |
| rainy | 326 | 0.764 | 0.599 | 0.925 | 0.476 | 0.667 | 0.270 | 0.385 |
| snowy | 366 | 0.770 | 0.654 | 0.908 | 0.574 | 0.719 | 0.377 | 0.495 |
| undefined | 487 | 0.807 | 0.700 | 0.922 | 0.590 | 0.755 | 0.488 | 0.593 |

⚠️ marks subgroups with total n < 30. Class-specific n may still be smaller; check the within-class comparisons below for class-specific sample sizes.

## By time of day

| Time | n | acc | macro-F1 | City R | Hwy R | Res P | Res R | Res F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dawn/dusk | 367 | 0.771 | 0.722 | 0.797 | 0.831 | 0.735 | 0.532 | 0.617 |
| daytime | 2447 | 0.781 | 0.725 | 0.871 | 0.769 | 0.747 | 0.476 | 0.582 |
| night | 1931 | 0.785 | 0.610 | 0.914 | 0.632 | 0.711 | 0.184 | 0.292 |

## Within-class comparisons (unpaired bootstrap)

Direction: each comparison reads **A → B** where the heading specifies A and B. Δ = metric(B) − metric(A). Positive Δ = B better; negative = A better.

⭐ = nominal 95% unpaired bootstrap CI excludes zero. **Comparisons are exploratory and not adjusted for multiple testing.** (Six comparisons total across two weather/time pairs × three classes.)

### Clear → rainy (Δ = rainy − clear)

| Class | A recall | B recall | Δ | 95% CI | n_A | n_B | Flags |
|---|---:|---:|---:|---|---:|---:|---|
| city street | 0.882 | 0.925 | +0.042 | [+0.006, +0.078] | 1599 | 226 | ⭐ |
| highway | 0.711 | 0.476 | -0.235 | [-0.368, -0.106] | 740 | 63 | ⭐ |
| residential | 0.372 | 0.270 | -0.101 | [-0.248, +0.051] | 288 | 37 | — |

### Daytime → night (Δ = night − daytime)

| Class | A recall | B recall | Δ | 95% CI | n_A | n_B | Flags |
|---|---:|---:|---:|---|---:|---:|---|
| city street | 0.871 | 0.914 | +0.043 | [+0.020, +0.066] | 1436 | 1278 | ⭐ |
| highway | 0.769 | 0.632 | -0.136 | [-0.188, -0.079] | 614 | 506 | ⭐ |
| residential | 0.476 | 0.184 | -0.292 | [-0.370, -0.212] | 397 | 147 | ⭐ |

## Cross-tabs

### Weather × scene class

| Weather | city street | highway | residential | Total |
|---|---|---|---|---|
| clear | 1599 | 740 | 288 | 2627 |
| undefined | 344 | 61 | 82 | 487 |
| rainy | 226 | 63 | 37 | 326 |
| overcast | 345 | 159 | 78 | 582 |
| snowy | 251 | 54 | 61 | 366 |
| partly cloudy | 178 | 128 | 44 | 350 |
| foggy | 2 | 4 | 1 | 7 |

### Time-of-day × scene class

| Time | city street | highway | residential | Total |
|---|---|---|---|---|
| night | 1278 | 506 | 147 | 1931 |
| daytime | 1436 | 614 | 397 | 2447 |
| dawn/dusk | 231 | 89 | 47 | 367 |

## Interpretation guide

- ⚠️ on a subgroup: total n < 30; treat as qualitative.
- ⚠️ small n on a comparison: class-specific n < 30 in A or B.
- ⭐: nominal 95% CI excludes zero (exploratory, unadjusted).
- Compare **within a class** to isolate weather/time effects from class-distribution differences.
- Recall alone can be inflated by predicting a class more often; check precision and F1 alongside recall.