# The complete guide to what's running right now

> **Note (2026-11-10):** The pre-registered decision rule in this document
> (arm C residential recall ≥ 0.35 → stop) was superseded by the macro-F1
> rule used in `train_week5_augmented_arms_v2.py` (arm C night macro-F1 ≥
> 0.68 → stop). See `docs/decision_rule_reconciliation.md` for the
> rationale. Arms A, C, D were run under the superseded rule. Arm B was
> not run.

You're running a controlled experiment with three training variants of the same neural network. Let me walk through every piece of it — from the data on disk, to the pixels being fed to the GPU, to how we'll interpret the final number.

---

## Part 1: The problem we're trying to solve

### 1.1 What is a "scene classifier"?

UrbanMind-AI (the parent project) has a model called **Week 5**. It's a ResNet-50 — a standard convolutional neural network architecture from 2015 that's still an industry workhorse. Its job is simple:

> **Given a photograph taken from a car's dashboard, decide which of three scenes it shows: "city street", "highway", or "residential street".**

That's it. That's the entire task. It doesn't detect cars or read license plates or estimate traffic. Just: three-way scene classification.

The dataset is **BDD100K**, a well-known benchmark of 100,000 driving-scene images collected from dashcams. BDD100K labels each image with a *scene type* (`city street` / `highway` / `residential`), a *time of day* (`daytime` / `night` / `dawn/dusk`), a *weather condition* (`clear` / `rainy` / `snowy` / etc.), and various other attributes.

UrbanMind-AI trained Week 5 on the ~78,833 labeled training images and evaluated it on the ~9,864 val + test images. It achieved about 78% overall accuracy and 70% macro-F1.

Those are decent numbers. But "78% overall accuracy" hides something interesting.

### 1.2 The hidden failure: residential at night

Let me show you what "78% overall accuracy" actually means for the different classes:

| Scene | Daytime accuracy | Night accuracy |
|---|---|---|
| City street | ~88% | ~91% |
| Highway | ~72% | ~63% |
| **Residential** | **~48%** | **~18%** |

The model is genuinely good at recognizing city streets — at night it actually gets *better* (probably because city streetlights and building lights form a distinctive visual pattern). It's okay at highways, slightly worse at night.

But residential streets? The model gets **48% right during the day and only 18% right at night.** That means at night, if you show the model a residential street, it gets it wrong **82% of the time**. Almost every residential street at night is classified as either "city street" or "highway."

**This is the weakness we're trying to fix.**

### 1.3 Why is it failing?

We don't know for sure yet. That's the point of the experiment. But there are a few hypotheses:

**Hypothesis A — Insufficient training data.** Maybe the training set just doesn't have enough nighttime residential images. If the model only sees 500 nighttime residential examples, it can't learn what they look like.

**Hypothesis B — Insufficient training variety.** Maybe the training data for residential streets is all in bright daylight. So the model learned "residential street = bright houses, bright trees, bright driveways" and simply hasn't seen the *appearance* of a residential street at night (dim lights, dark shapes, headlights illuminating nearby houses).

**Hypothesis C — Fundamental ambiguity.** Maybe nighttime residential streets *actually look* like something else at night — perhaps they're ambiguous between "city street with low traffic" and "highway with exits." Maybe the labels themselves are the issue.

**Hypothesis D — Architecture limits.** Maybe ResNet-50's capacity is the problem — it can't represent enough visual concepts to handle both lit and unlit residential scenes.

The three arms A/B/C distinguish between these hypotheses.

### 1.4 Why not just train a bigger model?

That would test Hypothesis D (architecture). But bigger models are more expensive (more GPU time, more memory), and the failure pattern (residential tanks at night, city *improves* at night) doesn't obviously suggest "the model isn't big enough." It suggests "the model didn't learn the right visual pattern." That's a data problem, not a capacity problem — usually.

Also: the project already committed to using Week 5's architecture (ResNet-50) because UrbanMind-AI uses it downstream. Changing the architecture would break compatibility with the parent project.

---

## Part 2: What is a "recall" exactly?

You'll see the number **"night residential recall: 0.184"** many times. Let's unpack it precisely.

### 2.1 The confusion matrix

For a classifier with 3 classes, evaluation produces a 3×3 table. Rows are "the true class," columns are "what the model predicted." Each cell counts images.

Here's a toy example with 100 residential nighttime test images:

| | Predicted city | Predicted highway | Predicted residential | **Row total** |
|---|---|---|---|---|
| **True city** | 0 | 0 | 0 | 0 |
| **True highway** | 0 | 0 | 0 | 0 |
| **True residential** | 43 | 39 | **18** | 100 |

The **bottom row** is what we care about for "night residential recall." 100 real residential-street nighttime images were shown to the model. It predicted:
- "residential" for 18 of them (correct)
- "city street" for 43 of them (wrong)
- "highway" for 39 of them (wrong)

**Recall = (correct predictions) / (total true positives) = 18 / 100 = 0.18.**

That's what "night residential recall: 0.184" means. **18.4% of nighttime residential street images were correctly classified.**

### 2.2 Why recall specifically, and not precision or accuracy?

Different metrics answer different questions. Let me define them all:

- **Recall** (aka sensitivity, true positive rate): "Of all the residential nighttime images, what fraction did we correctly identify?" — *This measures "can the model find residential streets?"*
- **Precision** (aka positive predictive value): "Of all the images we *predicted* as residential, what fraction actually were?" — *This measures "when the model says residential, is it right?"*
- **Accuracy**: "Of all the test images (across all classes), what fraction did we get right?" — *This is misleading when classes are imbalanced.*
- **F1**: harmonic mean of precision and recall — balances both.
- **Macro-F1**: F1 computed per class then averaged — treats each class equally regardless of how common it is.

**We track recall because that's the specific weakness.** The model *already* predicts "residential" for plenty of images — its precision for residential is around 0.37 (when it says "residential," it's right 37% of the time). It just doesn't *catch* enough actual residential images. That's a recall problem.

If we optimized for precision, we could trivially improve it by having the model predict "residential" less often (then when it does, it's more likely right). But that would make recall *worse* — we'd miss even more residential streets. So recall is the metric that captures "how well does the model find the things we care about."

### 2.3 Why the emphasis on night specifically?

The nighttime residential recall is only 18%. But the overall residential recall (day + night combined) is 41%. That means daytime residential recall is around 48% (higher because there are more daytime samples).

If we improved nighttime recall by making the model better on *all* residential streets, we'd want both to go up. But the failure is **concentrated at night** — the model fails dramatically more at night. So the specific target is: **improve nighttime residential recall without hurting daytime performance.**

That's important — we don't want to "cheat" by making the model predict "residential" for everything at night (that would tank precision and kill city/highway accuracy at night). We want it to genuinely recognize residential streets in low light.

---

## Part 3: What are we actually running — the three arms

Let me now describe each arm at the level of "what the code actually does."

### 3.1 Shared setup — the training recipe

All three arms share exactly the same recipe, differing only in what training data they see:

- **Model**: ResNet-50 architecture, initialized with ImageNet-pretrained weights (torchvision's `ResNet50_Weights.IMAGENET1K_V2`). This is a standard starting point — the model already knows how to recognize basic visual features (edges, textures, colors) from having been trained on 1.28M images.
- **Final layer**: the default ResNet-50 has a 1,000-way output (for ImageNet's 1,000 classes). We replace it with a 3-way output (city, highway, residential).
- **Optimizer**: AdamW with learning rate 1e-4 and weight decay 1e-4. This is a standard adaptive optimizer — it adjusts each parameter's effective learning rate based on its history.
- **Batch size**: 8. Eight images at a time. The GPU computes the loss for those 8 images together, then updates the weights.
- **Epochs**: 5. One epoch means "one pass through the entire training set." Five epochs means the model sees every training image five times.
- **Loss function**: Cross-entropy loss with class weights computed from the training set's label distribution. Cross-entropy penalizes the model heavily for confident wrong answers. The class weights help prevent the model from being dominated by the more common classes.
- **Best-epoch selection**: after each epoch, we evaluate the model on a validation set (4,727 held-out images that aren't in training and aren't in test). We keep the model that gets the highest macro-F1 on validation. This prevents overfitting — a model that memorizes training data perfectly but generalizes badly will be discarded.
- **Seed**: 42. This controls all randomness (weight initialization if we weren't using pretrained weights, data shuffling, augmentation sampling). Same seed → same results. Determinism is important for reproducibility.

**The recipe does NOT change across arms.** Only the training data differs.

### 3.2 Arm A — pure baseline

**What happens in code**: load all 68,969 training images (43,516 city + 17,379 highway + 8,074 residential), apply the eval transform (Resize(256) → CenterCrop(224) → ToTensor → Normalize), feed them to the model in batches of 8, compute cross-entropy loss, backpropagate, update weights.

Repeat 5 times (5 epochs). Evaluate after each epoch. Save the best.

**What this measures**: This is our **control**. It answers: "what does this exact recipe with this exact data achieve on night residential recall?" If it comes out at ~0.18, our setup matches the published baseline and everything else is comparable. If it comes out at 0.30, our recipe is somehow stronger than the original. If it comes out at 0.10, something's off.

Without arm A, **we wouldn't know whether improvements in arms B and C are real or artifacts of recipe differences**. Arm A is what makes B and C interpretable.

**What could go wrong here**: If arm A's night residential recall is very different from 0.184, we have a pipeline mismatch problem again. Possible causes:
- Different ImageNet weights (IMAGENET1K_V2 vs. V1) — different starting points
- Different data split (which images are in "train" vs "val")
- Different preprocessing (maybe the eval transform isn't matching)
- Different optimizer or epochs or batch size

If arm A's result is way off from 0.184, we diagnose before trusting arms B and C.

### 3.3 Arm B — night residential oversampling ×3

**What happens in code**: same as arm A, except we take the subset of training images that are **both "residential scene" and "nighttime"** and add them to the training set **two extra times**. So a training image that's nighttime residential appears three times per epoch instead of once.

Wait — does that actually change what the model learns? In some ways yes, in some ways no.

**Effect 1 — more frequent appearance**: the model sees nighttime residential images 3× as often. In any given epoch, roughly 3× as many of the batches will contain at least one nighttime residential image. The model has more chances to learn from them.

**Effect 2 — stronger gradient signal**: during backpropagation, every misclassification of a nighttime residential image contributes to the weight update 3× more often across the epoch (because the image appears 3× more often). So the model "feels" the penalty more strongly.

**Effect 3 — same total diversity**: oversampling doesn't create *new* nighttime residential examples. It's the same ~500 images seen more often. So if the model fails because it hasn't seen *enough variety* of nighttime residential scenes, oversampling doesn't help — it just shows the model the same few examples more.

**What this measures**: **Hypothesis A — Insufficient training data.** If night residential recall is low because the model just doesn't see enough nighttime residential examples, arm B should improve it. If it doesn't, we've learned that the problem isn't quantity — it's quality/variety.

**What we expect**: Probably a modest improvement. Oversampling is a standard, cheap intervention. In many domains it helps by 5-15 percentage points. But at 18% baseline, even +10 points to 28% leaves a big gap.

**A subtle trap**: oversampling changes the class balance. If we oversample nighttime residential 3× but don't account for it in class weights, we've skewed the effective loss function. The current script computes class weights from `full_train` (arm A's data), not the oversampled data, which is a deliberate choice — it means arm B's "effective class weight" for residential is actually 3× higher than arm A's. That's intentional (it amplifies the effect of oversampling) but it means arm B's total effect combines "more samples" + "implicitly higher class weight."

### 3.4 Arm C — photometric augmentation

**What happens in code**: same as arm A, except every training image is randomly transformed **on the fly** (each time it's loaded) with:

```python
transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
transforms.Lambda(lambda t: torch.clamp(t * (0.3 + 0.7 * torch.rand(1)), 0.0, 1.0))
```

Let me break this down precisely:

**`ColorJitter`** — a torchvision transform that randomly perturbs image appearance within a range:
- **brightness=0.4**: multiply pixel values by a random factor in [1−0.4, 1+0.4] = [0.6, 1.4]. So a pixel at 0.5 becomes anywhere from 0.3 to 0.7.
- **contrast=0.4**: adjust contrast by a factor in [0.6, 1.4]. High contrast → brighter brights, darker darks. Low contrast → compress toward gray.
- **saturation=0.2**: adjust color intensity by [0.8, 1.2]. Higher → more vivid colors. Lower → toward grayscale.
- **hue=0.1**: shift colors around the hue wheel by ±0.1 × 180° = ±18°. So reds become slightly orange, blues become slightly teal, etc.

**The Lambda transform** — multiplies all pixel values by a random scalar in [0.3, 1.0]:

```python
t * (0.3 + 0.7 * torch.rand(1))
```

This is a **gamma-like** brightness adjustment (though not true gamma, which would be `t ** gamma`). The point is: sometimes the image comes in very dim (30% brightness), sometimes near full (100%). Combined with ColorJitter's brightness, this dramatically expands the range of appearances seen during training.

**Effect**: the model sees each training image in ~infinitely many lighting conditions. A clear daytime residential street might appear:
- Epoch 1, batch 1: bright and warm (brightness=1.3, hue=+15°)
- Epoch 1, batch 20: dark and cool (multiplier=0.4, hue=−12°)
- Epoch 2, batch 5: medium brightness, high contrast (brightness=1.0, contrast=1.35)

The model learns "a residential street's *shape and structure* is what matters, not its specific brightness."

**What this measures**: **Hypothesis B — Insufficient training variety.** If the model learned "residential street = bright sunny scene" and fails on night because night residential looks different, arm C should help. It explicitly teaches the model that appearance varies.

**What we expect**: This is the interesting one. Photometric augmentation is one of the strongest standard augmentations in image classification. It's essentially free (a bit of CPU compute). If the model's failure is an appearance issue, arm C should improve night residential recall meaningfully.

**What could go wrong**: The augmentation might be too aggressive. If every image becomes a wildly different brightness, the model might learn "ignore brightness entirely," which is *not* what we want — we want it to still use brightness as a signal but not be confounded by it. There's a sweet spot. The current parameters (brightness=0.4, contrast=0.4) are on the strong side.

### 3.5 Arm D — synthetic augmentation (NOT RUNNING NOW)

**Why it's not running**: `outputs/method_4_wan/frames_clip_filtered/` doesn't exist yet. Arm D is skipped automatically.

**What it would do**: same as arm A, but add ~100-500 CLIP-filtered synthetic images (generated by some image model like Wan video or CycleGAN) as extra "residential" training examples.

**When we'll run it**: only if arm C's result is in the 0.20–0.34 range — i.e., conventional augmentation helps a bit but leaves a gap. Then we invest in the expensive pipeline.

**Why the filters matter (recall the audits)**:
- **Circular filter (bug)**: filter synthetic frames by "Week 5 predicts residential" — the model selects its own training data → false positive
- **CLIP filter (correct)**: use CLIP zero-shot to judge "does this look like a residential street" → independent judgment

And the training **must** be on all 3 classes (not just residential), because a 3-class model trained on only 1 class collapses to "predict that class for everything," scoring 100% recall — meaningless.

---

## Part 4: What happens inside one training step

Let me trace a single batch through the code, to demystify the loop.

```python
for x, y in train_loader:
    x = x.to(device)  # CPU tensor (8, 3, 224, 224) → GPU
    y = y.to(device)  # CPU tensor (8,) of ints [0, 1, 2] → GPU
    optimizer.zero_grad()  # clear previous gradient
    logits = model(x)  # forward pass: (8, 3) scores
    loss = criterion(logits, y)  # scalar loss value
    loss.backward()  # compute gradients
    optimizer.step()  # update weights
```

### 4.1 The batch

`x` is a float tensor of shape `(8, 3, 224, 224)`. That's 8 images, 3 color channels, 224 pixels tall, 224 pixels wide. Each pixel is a float in normalized space (ImageNet-normalized).

`y` is an integer tensor of shape `(8,)`. Each value is 0, 1, or 2 — the true class for the corresponding image.

### 4.2 The forward pass

`model(x)` runs the 8 images through ResNet-50's 50 layers. Output is shape `(8, 3)` — for each image, three scores (logits). Higher score = model thinks more likely.

Logits are raw scores, not probabilities. They can be any real number.

### 4.3 The loss

`criterion(logits, y)` computes cross-entropy loss. For each image:

$$\text{loss}_i = -\log\left(\frac{e^{\text{logit}_{y_i}}}{\sum_j e^{\text{logit}_j}}\right) \cdot w_{y_i}$$

That is: convert logits to probabilities via softmax, then look up the probability of the correct class, then take negative log, then multiply by the class weight.

- **If the model is confident and correct** (probability of true class near 1): loss ≈ 0.
- **If the model is confident and wrong** (probability of true class near 0): loss is huge (log of near-zero is very negative).
- **If the model is uncertain** (probability ~1/3): loss is moderate.

Cross-entropy *heavily penalizes* confident wrong answers. That's what drives learning.

The class weight $w_{y_i}$ multiplies the loss for samples of class $y_i$. Since residential (class 2) is only 8,074 / 68,969 ≈ 11.7% of the training data, its weight is high (roughly 2.85× compared to a balanced class weight of 1/3). This makes the model pay more attention to residential images.

### 4.4 The backward pass

`loss.backward()` computes the gradient of the loss with respect to every parameter in the model (about 23.5 million parameters in ResNet-50). The gradient is "how should each parameter change to make this batch's loss decrease?"

This is done via backpropagation — starting from the loss at the output, computing gradients layer by layer back to the input.

### 4.5 The update

`optimizer.step()` applies the gradients to update the weights. AdamW uses adaptive learning rates — parameters that have had consistently small gradients get larger updates, and vice versa. This is more robust than vanilla SGD.

### 4.6 What one batch teaches the model

Before this batch, the model has some weights. After this batch, the weights are slightly different — nudged to make the 8 images in this batch slightly better classified.

Over the whole epoch (8,621 batches), the model sees every training image once and updates weights after each batch. Over 5 epochs, it sees everything 5 times.

**Data augmentation (in arm C)** means that even though "the same image" is seen 5 times, the model sees a *different transformed version* each time. So it's not really learning "this exact pixel pattern is residential" — it's learning "the *concept* of residential street, invariant to brightness/color/contrast."

---

## Part 5: How the evaluation works

After each epoch of training, and after training finishes, we evaluate.

### 5.1 Validation evaluation (during training)

After each epoch, run the model on **4,727 validation images** (held out from training; not part of test either). Compute macro-F1. This tells us "how well is the model generalizing?"

We keep the model with the highest validation macro-F1 across the 5 epochs. This is **best-epoch selection**. It's a form of early stopping / model selection that prevents overfitting to training data.

### 5.2 Night residential test evaluation (the metric that matters)

The 147 nighttime residential test images are **held out from training AND validation**. They're from the actual test set (using UrbanMind-AI's clean split manifest, which excludes any images that appeared in training).

For each of the 147 images:
1. Load the JPEG
2. Apply the eval transform (Resize(256) → CenterCrop(224) → ToTensor → Normalize)
3. Run the model forward → get logits → argmax → predicted class
4. Compare to true class (always "residential," class 2)

**Recall = (predictions that are class 2) / 147**

If 27 images are predicted as class 2 (and correctly, since all 147 are actually class 2), recall = 27/147 = 0.184.

### 5.3 Paired bootstrap confidence intervals

A point estimate of "27 out of 147" is not very precise. If we re-ran the experiment with 147 different nighttime residential images, we'd get a slightly different number. The **bootstrap** estimates that variability.

**How it works** (in the paired-bootstrap case):

1. We have 147 images with predictions A[i] ∈ {0,1,2} for arm A, B[i] for arm B.
2. For 1,000 iterations:
   - Sample 147 indices *with replacement* from the 147 images
   - Compute recall for arm A on this resample
   - Compute recall for arm B on the same resample
   - Record the difference (B recall − A recall)
3. This gives 1,000 values of "how different is B from A under resampling."
4. Report the 2.5th percentile and 97.5th percentile — that's the 95% CI.

**Interpretation**:
- If the CI is [0.02, 0.10], we're fairly confident B beats A by 2-10 points.
- If the CI is [−0.05, 0.15], the effect could be negative or positive — inconclusive.
- **Paired bootstrap** is used because the same images are evaluated by both arms — this removes image-difficulty variability from the CI.

**Critical caveat**: With only **one seed** (like now, `--seeds 42`), the CI reflects *test-set variability*, not *training-run variability*. If you re-trained arm B with seed 123, you might get a meaningfully different result. This is why the plan calls for `--seeds 42 123 7` for the full comparison. Phase 1 just uses seed 42 to get a quick yes/no.

---

## Part 6: How we'll interpret the results

### 6.1 Arm A — the reference check

**First thing we look at**: what's arm A's night residential recall?

- **Near 0.184** → ✅ Pipeline matches the published baseline. Our arms B/C are comparable.
- **Much higher (e.g., 0.30)** → ⚠️ Our recipe is somehow stronger. Possible causes: different ImageNet weights, different data, different transform. We still trust B/C deltas, but can't directly compare arm A to the historical 0.184.
- **Much lower (e.g., 0.10)** → ⚠️ Something's wrong. Diagnose before trusting B/C.

This is the same discipline as the pipeline-mismatch episode. Before interpreting any "improvement," verify the baseline reproduces.

### 6.2 Arms B and C — the deltas

Assuming arm A is fine:

| Observation | Interpretation | Action |
|---|---|---|
| C−A CI **excludes zero, positive** | Photometric augmentation **helps** | Read the effect size |
| C−A CI **includes zero** | Photometric augmentation **doesn't clearly help** | Read the effect size anyway |
| C−A **negative** | Photometric augmentation **hurts** | Surprising — diagnose |
| B−A CI **excludes zero, positive** | Oversampling **helps** | Read the effect size |
| B−A CI **includes zero** | Oversampling **doesn't clearly help** | Read the effect size anyway |

**Note**: "CI includes zero" doesn't prove "no effect." With n=147, CIs are wide. A small effect (say +0.03) might not be detectable. We interpret the *magnitude* of the point estimate, not just whether the CI excludes zero.

### 6.3 The decision rule (locked in before the run)

**Arm C is the primary decision gate.** Here's the full decision tree, with the reasoning for each branch:

**Branch 1: Arm C's night residential recall ≥ 0.35**

Interpretation: **Photometric augmentation closes most of the gap.** The model's failure was largely about appearance. Just by making the training images vary in brightness/color/contrast, we taught the model to recognize residential streets in low light.

**Action**: **Stop the generative plan.** Don't install Wan2GP. Don't train CycleGAN. Don't spend 10-15 hours. Document that photometric augmentation solves this particular weakness.

**Caveats**: 
- Is arm C's improvement *real* or a fluke? Bootstrap CI answers partially, but with one seed we should re-run with seeds 42, 123, 7 to confirm.
- Does arm C hurt other metrics? Check macro-F1 — if it dropped by more than 0.01, that's a red flag. The model might be trading off day accuracy for night accuracy in a way that's not a genuine improvement.
- Does arm C hurt daytime residential? Should not — we want night improvement without day regression.

**Branch 2: Arm C's night residential recall 0.20–0.34**

Interpretation: **Photometric augmentation helps but leaves a gap.** The failure is partly about appearance. The other part is about something else — either data diversity, structural features, or something generative models might add.

**Action**: **Proceed to Phase 2.** Install Wan2GP, generate night videos with CLIP filtering, train arm D, compare.

**Why this justifies the cost**: A 10-15 hour investment in generative work is *proportionate* only if there's still a gap to close. Arm C narrowing the gap to ~0.28 means there's a 20+ point gap that arm D might partially close.

**Caveats**: Before committing, verify the improvement is real (multiple seeds). And check that any night improvement isn't costing daytime performance — if the model is just biased toward more residential predictions overall, both day and night might appear to improve while real precision is destroyed.

**Branch 3: Arm C's night residential recall < 0.20**

Interpretation: **Photometric augmentation doesn't meaningfully help.** The failure is **not** primarily about appearance variation. Generative models that produce synthetic nighttime images (Wan, CycleGAN) also operate by varying appearance — so they probably won't help either.

**Action**: **Stop. Document and close.** Photometric augmentation is the cheapest possible intervention. If it fails, the more expensive appearance-based interventions probably won't succeed. Worth documenting as a negative finding.

**Caveats**: This conclusion assumes generative models "only" change appearance. They also generate *new scene content* (new houses, new driveways, new trees). So a very small chance remains that content diversity helps even when appearance diversity doesn't. But that's a stretch — the default interpretation is the gap is structural, not visual.

### 6.4 What "we learned something" looks like in each branch

Every branch produces a **real finding**:

- **Branch 1**: "Photometric augmentation is sufficient for nighttime residential robustness in BDD100K; generative models are unnecessary."
- **Branch 2**: "Appearance augmentation helps partially; residual gap may be addressable by content-level augmentation (tested via arm D)."
- **Branch 3**: "Nighttime residential classification failure is not an appearance problem; suggests semantic or structural ambiguity in the task."

Any of these is a legitimate contribution. The point of running arms A/B/C is to **distinguish between these hypotheses**.

---

## Part 7: Why this methodology is worth the effort

You might wonder: why not just train the model and see if it gets better? Why all this arm structure, this bootstrap, this CI, this pre-registration?

Because **without this discipline, "it got better" is an illusion-prone claim.** Here are the traps we're avoiding:

### Trap 1: Confusing pipeline mismatch with improvement

Remember the pipeline-mismatch episode. The original robustness report said night residential recall was 0.124. The corrected value is 0.408 (day) / 0.184 (night). A **3.3× factor** came from simply using the *wrong input pipeline*. If we ran arms B/C without a reference check (arm A), we might report "arm C improved night residential from 0.124 to 0.30!" — when actually the baseline was 0.184 all along, and arm C just got back to where we should have started.

Arm A protects against this. If arm A doesn't reproduce 0.184, we stop and diagnose.

### Trap 2: Confusing augmentation with class collapse

A model that predicts "residential" for everything gets 100% night residential recall. Its F1 collapses to near-zero (since precision is ~12%, F1 ~21%). If we only measured recall, we'd celebrate a broken model.

The class weights, macro-F1 tracking, and evaluation on all 3 classes protect against this. A genuinely improved model improves recall **without** collapsing other metrics.

### Trap 3: Confusing noise with signal

A single run of arm C might come out at 0.25 while arm A comes out at 0.18. Great — but is the 0.07 improvement real, or just what one particular random initialization produced? Without multiple seeds and CIs, we can't tell.

Phase 1 uses one seed for speed, but the CI at least tells us the *test-set* uncertainty. If the CI includes zero, the point estimate is not distinguishable from noise on the test set alone. That's why the full comparison (Phase 3) uses 3 seeds.

### Trap 4: Confusing correlation with causation

Suppose arm B improves night residential recall by +0.08. Is it because of oversampling? Or because arm B happened to train on a slightly different random order of images that *coincidentally* learned a better representation? With one run, we can't distinguish.

Running arms A/B/C with **identical recipes** ensures the only difference is the training data. The improvement is attributable to the intervention.

### Trap 5: Confusing "held-out" with "held-out properly"

Some image-split bugs cause val/test images to leak into training. Then the model "succeeds" on val/test but fails on new data. The current setup uses UrbanMind-AI's verified split manifests (SHA-256-checked by the golden test), so any leakage would have failed the golden test before we got here.

Still — this discipline is why the pipeline is trustworthy. One clean mistake elsewhere and the whole experiment is worthless. The golden test's passing is a **precondition** for this experiment's validity.

### Trap 6: Confusing pre-registration with post-hoc storytelling

If we didn't pre-register the decision threshold, we might see arm C = 0.22, and reason "well 0.22 is pretty good, maybe we don't need generative" — then see it in a better run, get 0.30, and reason "see, arm C confirms!" That's not science; it's fitting the narrative to whatever happened.

The threshold was set at 0.35 (for "stop") / 0.20-0.34 (for "proceed cautiously") / <0.20 (for "document and close") **before running**. It doesn't move. Whatever result arm C produces, the decision is determined.

---

## Part 8: Time expectations and watching the run

### 8.1 Current run — what to expect

With `num_workers=0`, the current run will take **10–15 hours total** (Arm A + Arm B + Arm C, all with seed 42). That's too long — that's why I recommended restarting with `num_workers=4` to cut it to **5–7 hours**.

With `num_workers=4`, per-arm breakdown:
- **Arm A**: ~1.5–2.5 hours
- **Arm B**: ~1.5–2.5 hours (slightly slower — more data)
- **Arm C**: ~1.5–2.5 hours (same data as A, but each image requires augmentation compute on the CPU — small extra cost)

Total ~5–7 hours.

### 8.2 What you'll see as it runs

**During each arm**, after each epoch ends (5 epochs per arm):
```
=== Arm A (seed=42) ===
  epoch 1/5  val_macro_f1=0.XXXX
  epoch 2/5  val_macro_f1=0.XXXX
  ...
```

`val_macro_f1` is a scalar from 0 to 1. It should be:
- Epoch 1: ~0.55–0.65 (model is learning)
- Epoch 2: ~0.65–0.68
- Epoch 3: ~0.68–0.70
- Epoch 4: ~0.69–0.70
- Epoch 5: ~0.69–0.70 (plateau)

If val_macro_f1 doesn't improve or degrades, the model is overfitting or the learning rate is wrong. That's a signal to diagnose — but 5 epochs at lr=1e-4 with pretrained weights should be stable.

**At the end of each arm**:
```
  Best val macro-F1: 0.XXXX at epoch N
  Night residential recall: 0.XXXX
```

The "Night residential recall" is the metric that determines the outcome.

**At the very end**:
```
SUMMARY — night residential recall
  Arm A: 0.XXXX ± 0.000  (seeds=[0.XXXX])
  Arm B: 0.XXXX ± 0.000  (seeds=[0.XXXX])
  Arm C: 0.XXXX ± 0.000  (seeds=[0.XXXX])

  Bootstrap B - A: Δ=+0.XXX  CI [+0.XXX, +0.XXX]  CI excludes zero
  Bootstrap C - A: Δ=+0.XXX  CI [+0.XXX, +0.XXX]  CI excludes zero
```

The `±` values will be 0.000 with one seed — expected, ignore them. The bootstrap lines are what matter for interpretation.

### 8.3 What if it crashes mid-run?

If arm A finishes and arm B crashes:
- You'll have `outputs/week5_arms/results.json` with only arm A's data.
- The crash log tells you what failed.
- We diagnose and re-run, usually without losing arm A (the script would need to be modified to cache arm A's result — it currently doesn't, so a crash mid-run loses everything from that run).

This is a known limitation. For future runs with `--seeds 42 123 7`, this becomes more painful. A future improvement would be checkpoint-per-arm, but not blocking now.

---

## Part 9: What's actually going to happen over the next few days

Assuming you let arm A/B/C finish:

**Day 1 (2026-11-10)**: Complete the run. Read the summary. Apply the decision rule.

**If arm C ≥ 0.35 (Branch 1)** — the project essentially ends here for this weakness. You'd:
- Re-run with `--seeds 42 123 7` to confirm the finding isn't a seed fluke (3× cost = 15-20 hours)
- Document the result as a positive finding in `docs/experiments.md`
- Write a model card for the "robustness-enhanced" Week 5
- Possibly tag `v0.2.0-g2-night-robust` or similar
- Update README with the finding
- Project closes with a real contribution: "we showed photometric augmentation fixes night residential weakness without generative models"

**If arm C is 0.20–0.34 (Branch 2)** — proceed to Phase 2:
- Install Wan2GP (~10 GB download)
- Generate 5 test clips with nighttime residential prompts
- Inspect visually. If good, generate 20 more.
- CLIP filter the frames (real judgment, not circular)
- Re-run `train_week5_augmented_arms.py` with arm D active
- Compare arm D to best of B/C
- Decision: does generative augmentation beat conventional?

**If arm C < 0.20 (Branch 3)** — project closes here:
- Document that photometric augmentation doesn't help
- Interpret as "the failure is not appearance-based"
- Note that generative models (which also vary appearance) probably wouldn't help either
- Suggest future directions: architecture changes, task redefinition, or acceptance that the task is genuinely hard at night
- Project closes with a negative finding — still legitimate

---

## Part 10: The deeper point

What you're running is a **controlled scientific experiment**, not a "try something and see if it works."

- **Controlled**: arm A is a control, B and C are treatments differing in exactly one variable.
- **Scientific**: hypothesis-driven (three candidate explanations), pre-registered decision thresholds, quantitative metrics with confidence intervals.
- **Honest**: three possible outcomes, all of which produce legitimate knowledge. No outcome is "the project failed."
- **Reproducible**: fixed seed, fixed recipe, fixed dataset. Anyone can rerun and get the same number (within CI).
- **Interpretable**: if arm C fails, we've *learned* that the gap isn't appearance-based, which narrows the search for other causes.

This is what makes the project credible. The typical pattern in ML projects is: "we tried X, it seemed to help, we shipped it." Then six months later, someone reruns it with a different random seed and it doesn't reproduce, or the pipeline had a subtle bug, or the improvement was really just training-set leakage.

The UrbanGen-AI project is the alternative to that pattern. Every claim traces back to a script, a JSON output, and a pre-registered interpretation rule. Every "improvement" is guarded by a baseline. Every hypothesis is testable. That's the whole point.

The current run — arms A, B, C — is a small, careful piece of that larger discipline. It's answering one concrete question with one concrete design. If it works, the answer is definitive. If it doesn't, the answer is still definitive (just a different answer).

**That's what's running right now.** That's why it's worth 5-7 hours of GPU time. And that's what makes the eventual number mean something.