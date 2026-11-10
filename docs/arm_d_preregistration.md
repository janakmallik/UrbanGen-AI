# Arm D Pre-registration: Synthetic Night Data Augmentation

**Date:** 2026-11-10
**Status:** Locked before training

## Objective
Test whether adding manually curated CycleGAN-generated nighttime images to the ResNet-50 training set improves nighttime residential recall compared to the Arm A recipe baseline.

## Baseline (Arm A)
- **Model:** ResNet-50 (IMAGENET1K_V2)
- **Recipe:** Inverse-frequency class weights, 5 epochs, best-epoch selection, seed 42.
- **Baseline Metrics:** Night residential recall = **0.3605**, Night macro-F1 = **0.6475**
- **Artifact:** `outputs/week5_arms/results_A_seed42.json`

## Treatment (Arm D)
- **Data:** Baseline training data + 276 curated synthetic nighttime residential images.
- **Dataset Provenance:** `outputs/method_2_synthetic/curated_manifest.json`
- **Weighting:** Synthetic images are treated as regular training images (residential class). No special weighting applied initially.
- **Training config:** Identical to Arm A (same optimizer, epochs, class weights). Only the training data changes.

## Evaluation
- **Primary Metric:** Night residential recall (n=147).
- **Guardrail Metric:** Daytime macro-F1 (must not drop by more than 0.02 compared to Arm A).
- **Statistics:** Paired bootstrap (1,000 resamples) comparing Arm D predictions to Arm A predictions on the 147 nighttime residential test images.

## Pre-registered Success/Failure Criteria
1. **Success:** Arm D night residential recall improves by ≥ +0.03 over Arm A, AND the 95% bootstrap CI excludes zero, AND Day macro-F1 does not degrade by > 0.02.
2. **Partial/Neutral:** Arm D shows improvement, but CI includes zero, or improvement is < +0.03.
3. **Failure (Negative Result):** Arm D night residential recall drops by ≥ -0.02 with CI excluding zero.

*Decision rule will not move after results are seen.*