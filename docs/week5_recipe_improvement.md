
# Week 5 Recipe Improvement (Arm A)

## The Finding
The published Week 5 ResNet-50 baseline achieves a **night residential recall of 0.184**. 

However, when we retrained ResNet-50 using a slightly modified recipe (Arm A), we achieved a **night residential recall of 0.3605**. This nearly **doubles** the published baseline, entirely without synthetic data.

## The Recipe
The Arm A training script (`scripts/train_week5_augmented_arms_v2.py`) differs from the original Week 5 recipe in three ways:
1. **Class Weights:** We applied inverse-frequency class weights to the loss function, significantly upweighting the residential class (weight 2.85x) and downweighting the city street class.
2. **Pretrained Weights:** We initialized with `ResNet50_Weights.IMAGENET1K_V2` (the original baseline used V1).
3. **Best-Epoch Selection:** We trained for 5 epochs and selected the checkpoint with the highest validation macro-F1, rather than using the final epoch.

## The Results (Night Subset)

| Metric | Published Week 5 | Arm A Recipe | Delta |
| :--- | :--- | :--- | :--- |
| **Night Macro-F1** | 0.610 | **0.6475** | **+0.038** |
| **Night Residential Recall** | 0.184 | **0.3605** | **+0.177** |
| Night Residential Precision | ~0.37 | 0.541 | +0.17 |
| Day Macro-F1 | 0.725 | 0.7457 | +0.021 |
| Overall Test Macro-F1 | 0.6984 | 0.7183 | +0.020 |

## Why This Matters
This finding reveals that the "night residential weakness" (the primary motivation for the generative augmentation experiments) is largely a **recipe problem**, not a fundamental limitation of the architecture or the data.

Because our Arm A recipe gets 0.36 night residential recall, any synthetic data generated for the "Arm D" augmentation experiment has a much higher bar to clear. It must beat 0.36, not just 0.18.

## Reproducibility
To reproduce this finding: