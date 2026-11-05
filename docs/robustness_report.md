# Week 5 ResNet-50 — Robustness Report Card

Evaluates UrbanMind-AI's Week 5 ResNet-50 on the clean BDD100K test split, broken down by weather and time-of-day attributes from the BDD100K label JSONs.

- **Total test images:** 4745
- **Bootstrap resamples:** 1000
- **Random seed:** 42

## Overall metrics

- **Macro-F1:** 0.5710 [0.5531, 0.5870]
- **city street recall:** 0.9545 [0.9468, 0.9615]  (n=4745)
- **highway recall:** 0.5591 [0.5324, 0.5875]  (n=4745)
- **residential recall:** 0.1235 [0.0974, 0.1492]  (n=4745)

## By weather

| Weather | n | Macro-F1 [95% CI] | City recall | Highway recall | Residential recall |
|---|---|---|---|---|---|
| clear | 2627 | 0.577 [0.552, 0.601] | 0.952 (n=2627) | 0.549 (n=2627) | 0.142 (n=2627) |
| foggy | 7 | 0.413 [0.111, 0.861] | 1.000 (n=7) | 0.500 (n=7) | 0.000 (n=7) |
| overcast | 582 | 0.614 [0.562, 0.661] | 0.919 (n=582) | 0.717 (n=582) | 0.154 (n=582) |
| partly cloudy | 350 | 0.610 [0.548, 0.669] | 0.955 (n=350) | 0.695 (n=350) | 0.136 (n=350) |
| rainy | 326 | 0.481 [0.406, 0.550] | 0.982 (n=326) | 0.270 (n=326) | 0.108 (n=326) |
| snowy | 366 | 0.526 [0.466, 0.583] | 0.956 (n=366) | 0.537 (n=366) | 0.066 (n=366) |
| undefined | 487 | 0.476 [0.415, 0.536] | 0.983 (n=487) | 0.311 (n=487) | 0.073 (n=487) |

## By time of day

| Time | n | Macro-F1 [95% CI] | City recall | Highway recall | Residential recall |
|---|---|---|---|---|---|
| dawn/dusk | 367 | 0.612 [0.547, 0.675] | 0.931 (n=367) | 0.652 (n=367) | 0.170 (n=367) |
| daytime | 2447 | 0.574 [0.551, 0.596] | 0.948 (n=2447) | 0.622 (n=2447) | 0.113 (n=2447) |
| night | 1931 | 0.561 [0.526, 0.595] | 0.966 (n=1931) | 0.466 (n=1931) | 0.136 (n=1931) |

## Cross-tab: weather × scene class

Sample counts per (weather, scene class) cell. Small cells produce unstable metrics.

| Weather | city street | highway | residential | Total |
|---|---|---|---|---|
| clear | 1599 | 740 | 288 | 2627 |
| undefined | 344 | 61 | 82 | 487 |
| rainy | 226 | 63 | 37 | 326 |
| overcast | 345 | 159 | 78 | 582 |
| snowy | 251 | 54 | 61 | 366 |
| partly cloudy | 178 | 128 | 44 | 350 |
| foggy | 2 | 4 | 1 | 7 |

## Cross-tab: time-of-day × scene class

| Time | city street | highway | residential | Total |
|---|---|---|---|---|
| night | 1278 | 506 | 147 | 1931 |
| daytime | 1436 | 614 | 397 | 2447 |
| dawn/dusk | 231 | 89 | 47 | 367 |

## Interpretation guide

- **Weather or time subgroups with n < 30** should be treated as qualitative only; the bootstrap CIs will be very wide.
- **Confounding:** if a subgroup's scene-class distribution differs significantly from the overall distribution, per-subgroup F1 differences partly reflect class mix, not weather. Check the cross-tabs.
- **CI overlap:** if adverse-weather F1 CI overlaps the clear-weather F1 CI, there is no statistically-supported weather gap.
- **Within-class analysis:** the per-class recall columns show whether any gap is uniform across classes or concentrated in one.
