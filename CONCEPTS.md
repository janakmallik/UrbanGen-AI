# Start

**Cosine annealing** is a learning-rate scheduling technique that gradually decreases the learning rate following a cosine-shaped curve during training. Instead of keeping the learning rate constant, or reducing it abruptly at fixed intervals, it starts relatively high and smoothly decreases toward a minimum value. A higher learning rate helps the model make larger parameter updates early in training, while a lower learning rate helps it make smaller, more precise adjustments later. In deep learning, cosine annealing can help training converge more smoothly and reduce the need for manually designed learning-rate decay steps. Some implementations also use warm restarts, where the learning rate periodically rises again before decreasing.

**Mixed precision training** is a technique that uses different numerical precisions during neural-network training instead of performing every operation in standard 32-bit floating point (FP32). It commonly uses 16-bit floating point (FP16 or BF16) for suitable operations while retaining FP32 for operations that need greater numerical stability, such as certain loss calculations or parameter updates. This reduces memory consumption and can accelerate training, particularly on GPUs with specialized hardware for lower-precision computation. For example, on your RTX 4050, mixed precision can help fit larger batches or models into the available VRAM and potentially improve training speed. However, FP16 can introduce numerical underflow or overflow, so loss scaling may be used; BF16 generally has a wider numerical range.

**Reconstruction loss (MSE):** Measures how different the VAE's reconstructed output is from the original input. MSE (Mean Squared Error) calculates the average squared difference between corresponding pixel values. A lower MSE means the reconstruction is closer to the original image. During training, it encourages the decoder to generate images that preserve the input's visual information.

**KL divergence (KL):** Measures how far the encoder's learned latent distribution is from the desired standard normal distribution \\(N(0,I)\\). Minimizing KL encourages the latent space to remain smooth and organized, making it possible to sample meaningful latent vectors and generate new images. The \\(\beta\\) coefficient controls how strongly this regularization influences training: higher \\(\beta\\) encourages a more structured latent space but can reduce reconstruction quality.

**AdamW:** An optimization algorithm used to update a neural network's weights during training. It combines adaptive learning rates with decoupled weight decay, which helps regularize the model and can reduce overfitting. Unlike basic SGD, AdamW adapts the update size for individual parameters based on their gradients, often making training easier to tune.

**MSE encourages accurate reconstruction, KL encourages a useful latent space, and AdamW updates the model parameters to minimize the combined loss.**

## Distributions in our models

This is the most important part for your project. Distributions are not just statistical concepts that we study separately. Models use them to represent data, make predictions, introduce randomness, and generate new samples.

| Model or component      | How distributions are used                                                |
| ----------------------- | ------------------------------------------------------------------------- |
| LightGBM classification | Produces class probabilities, often normalized across classes             |
| ResNet classification   | Produces categorical class probabilities through softmax                  |
| VAE                     | Learns Gaussian latent distributions and samples latent vectors           |
| cVAE                    | Learns Gaussian latent distributions conditioned on a class               |
| DDPM                    | Adds and removes Gaussian noise through a sequence of steps               |
| DCGAN                   | Often samples random latent vectors from a normal or uniform prior        |
| Model calibration       | Checks whether predicted probabilities correspond to observed frequencies |
| Bootstrap evaluation    | Uses repeated resampling to estimate uncertainty in metrics               |


**How the distributions work in our three generative models?**
**In a VAE,** the encoder receives an image and predicts two things: the mean and variance of a Gaussian distribution in latent space. We sample a vector from that distribution and send it to the decoder, which reconstructs the image. During training, reconstruction loss encourages the output to resemble the original image, while KL divergence encourages the learned latent distribution to remain close to a standard normal distribution. After training, we can sample from the standard normal prior and pass those vectors through the decoder to generate new images.

**In a cVAE,** we add a class condition. For example, we tell the model that we want a residential street image. The encoder and decoder use that class information, so the latent distribution is learned with the condition in mind. At generation time, we provide the residential class label and sample a latent vector from the prior. The decoder uses both to generate a residential-style image. This is how class-conditional generation works: the distribution and generation process are guided by the requested class.

**In a DDPM,** we use Gaussian noise in a different way. During training, we gradually add small amounts of Gaussian noise to real images (or, in our planned latent DDPM, to VAE latent representations). The model learns to predict the noise added at each step. During generation, we start with random Gaussian noise and repeatedly remove predicted noise until a sample emerges. The class condition can guide this denoising process toward residential, highway, or city-street samples.

**In a DCGAN,** we usually start with a random latent vector sampled from a simple prior distribution, often a standard normal or uniform distribution. The generator transforms this random vector into an image. The discriminator tries to distinguish generated images from real ones, and the generator learns to produce more realistic images. In a conditional DCGAN, class information is also supplied to guide the generated image toward a requested category.

## We actually getting from distributions

Think of distributions as giving our models a way to represent uncertainty and variation rather than treating every input or output as one fixed value.

For our generative project:

- VAE: A structured latent space from which we can sample new images.
- cVAE: A structured latent space that also responds to a requested class.
- DDPM: A process that transforms random noise into realistic, condition-guided samples.
- DCGAN: A generator that transforms random latent inputs into images.

This is why we need distributions: they provide the mathematical foundation for randomness, sampling, and generation.

## One distinction we should remember

There are three different things that are easy to confuse:

**1. Data distribution:** How the real dataset is distributed. For example, one dataset has more city-street images than residential images.
**2. Model distribution:** A probability distribution learned or predicted by a model. For example, a VAE encoder's Gaussian latent distribution.
**3. Sampling distribution:** The distribution from which we draw random values. For example, the standard normal prior used to generate latent vectors.

They can interact, but they are not the same thing. A generative model tries to learn a useful representation of the real data distribution, then uses sampling to produce new examples.

The key connection to UrbanGen-AI: We don't just want to generate images that look convincing. We want to generate diverse, class-correct synthetic images whose addition to real training data improves residential-scene recognition on our frozen evaluation protocol. The distributions make generation possible; our downstream evaluation tells us whether that generation is actually useful.

# Sampling
Sampling connects probability distributions, randomness, training, data preparation, model evaluation, and image generation.

All three generative models (cVAE, latent DDPM, and conditional DCGAN) depend on sampling in different ways. We also use sampling when creating batches, splitting data, estimating evaluation uncertainty, and selecting synthetic images.

**Sampling means selecting or drawing one or more values from a population, dataset, or probability distribution.**

Imagine our BDD100K dataset contains 68,969 training images. If we select 100 images from those 68,969, we have taken a sample of the dataset.

**There are two broad situations:**

- Dataset sampling: Selecting actual observations from an existing dataset.
- Probability sampling: Drawing values from a probability distribution, including values that may not correspond to any previously observed data point.

For example, selecting 100 residential images from BDD100K is dataset sampling. Drawing a random vector from a Gaussian distribution for a DCGAN generator is probability sampling.

**The distinction is essential: sampling existing images does not create new data, but sampling a latent distribution can provide the random input from which a generative model creates a new image.**


## Different ways to sample from a dataset

These are the sampling methods most relevant to our work.

**Random sampling: Every observation has an equal chance of being selected.**
This is useful when we want a representative sample without deliberately favoring any class. However, random sampling does not guarantee that every class will be represented proportionally, particularly when the sample is small.

**Stratified sampling: We divide the dataset into groups, called strata, and sample from each group separately.**

For our three BDD100K classes, the strata would be:

- City street
- Highway
- Residential

Suppose we want a sample of 1,000 images that preserves the original class proportions. We would select approximately 631 city-street, 252 highway, and 117 residential images.

This is useful when we want a sample that reflects the original dataset's class distribution.



**Balanced sampling: Balanced sampling deliberately selects approximately the same number of observations from each class.**

For example, we might select:

- 300 city-street images
- 300 highway images
- 300 residential images

This differs from stratified sampling: stratified sampling preserves the original proportions, while balanced sampling changes them.

Balanced sampling can be useful when a model struggles with minority classes. However, it changes the effective training distribution, so we should not assume that its predictions are automatically calibrated to the original real-world class frequencies.

**Sampling with replacement: After selecting an observation, we put it back into the available pool before selecting again. Therefore, the same observation can be selected more than once.**

For example, if we draw 100 images with replacement, some images may appear multiple times while others may not appear at all.

This is used in bootstrap evaluation and some training strategies.

**Sampling without replacement: Once an observation has been selected, it cannot be selected again during that sampling operation.**

For example, selecting 1,000 unique images from our training dataset.

This is useful for dataset splits, validation samples, and creating non-duplicated subsets.


**Weighted sampling: Different observations are assigned different probabilities of being selected.**

For example, we could assign residential images higher sampling weights because they represent only 11.71% of our training data.

The model would encounter residential images more frequently during training, without necessarily physically duplicating the image files.

This can help address class imbalance, but it can also cause the model to learn a training distribution that differs from the real data distribution.

Important: Weighted sampling, class-weighted loss, and oversampling are three different techniques. They can have similar goals but operate differently.


## Sampling from probability distributions

Now we move from selecting existing images to drawing random numerical values.

This is what generative models use.

Suppose we have a normal distribution:

\\[ Z \sim \mathcal{N}(0,1) \\]

We can draw random values from it:

\\[ -0.73,\quad 0.21,\quad 1.35,\quad -0.14 \\]

Each time we sample, we may get different values. The values tend to concentrate around zero, but larger positive or negative values are also possible.

If we sample 1,000 values, their histogram should approximately resemble the normal distribution, although the exact shape will vary because sampling is random.

The important distinction is that the distribution describes the probabilities, while sampling actually draws the values.

## Important sampling methods for probability distributions

Uniform sampling: Draws values from a range where every value has equal probability density. For example, sampling a random number between -1 and 1. This is commonly used for initialization, random transformations, and some latent-space priors.

Gaussian sampling: Draws values from a normal distribution. For example, sampling from a standard normal distribution with mean zero and variance one. This is central to VAE latent sampling and DDPM noise generation.

Categorical sampling: Selects one category according to specified probabilities. Suppose a classifier predicts:

\\[ P(\text{city})=0.6,\quad P(\text{highway})=0.3,\quad P(\text{residential})=0.1 \\]

A categorical sample could select city street, highway, or residential according to those probabilities. This is different from simply selecting the category with the highest probability (argmax).

Bernoulli sampling: Draws a binary outcome, either 0 or 1, according to a probability. It is useful for binary random decisions and some probabilistic models.

Importance sampling: Samples observations more frequently when they are considered more important, often correcting for the altered sampling probabilities through mathematical weights. It is useful when some regions of a probability distribution are more important to estimate accurately than others.

Monte Carlo sampling: Uses repeated random samples to estimate a quantity that is difficult to calculate exactly. For example, we can generate many random latent vectors and estimate the average quality or class consistency of generated images.

These are not all mutually exclusive categories. Monte Carlo is a general estimation approach that can use Gaussian, uniform, or other sampling methods.

## Where sampling happens in our cVAE

This is one of the most important parts of our project.

A conventional autoencoder encodes an image into a fixed latent vector. A VAE instead learns a probability distribution over possible latent representations.

For each input image, the encoder produces:

- A mean vector \\(\mu\\)
- A variance vector \\(\sigma^2\\), commonly represented internally using log-variance

The model samples a latent vector from that Gaussian distribution:

\\[ z = \mu + \sigma \odot \epsilon \\]

where:

\\[ \epsilon \sim \mathcal{N}(0,I) \\]

This is called the reparameterization trick.

**Why do we need it? We want the model to sample randomly while still allowing gradients to flow through the encoder during backpropagation. Instead of sampling \\(z\\) directly as an opaque operation, we sample the independent noise \\(\epsilon\\) and transform it using the encoder's mean and standard deviation.**


For our cVAE, the class label is also supplied. For example, the model receives a residential image and its residential label. The encoder learns a class-conditioned latent distribution, and the decoder learns to reconstruct an image using the sampled latent vector and class condition.

After training, we do not need a real image to generate a new image. We can sample from the prior:

\\[ z \sim \mathcal{N}(0,I) \\]

Then provide the residential label to the decoder:

\\[ \text{Image} = Decoder(z,\text{residential}) \\]

This is how the cVAE produces new residential images.

One important detail: The encoder's distribution for a particular training image and the standard normal prior used for generation are related, but they are not the same object. KL divergence encourages the former to remain close to the latter.

## Where sampling happens in our latent DDPM

DDPM uses sampling in a different way.

During training, we take a real image's VAE latent representation and progressively add Gaussian noise. The model learns to predict the noise at different time steps.

During generation, we start with random Gaussian noise:

\\[ z_T \sim \mathcal{N}(0,I) \\]

Then the trained diffusion model repeatedly predicts and removes noise, moving from a noisy latent representation toward a structured one.

Eventually, the VAE decoder converts the resulting latent representation into an image.

So, in the diffusion model, sampling is not just one operation. It is part of the entire generation process. We sample the initial noise and then repeatedly sample or calculate updates according to the reverse diffusion process.

The sampling algorithm used during inference matters: the number of denoising steps and the chosen sampler can affect generation speed and image quality.

## Where sampling happens in our DCGAN

A DCGAN generator normally takes a random latent vector:

\\[ z \sim \mathcal{N}(0,I) \\]

or sometimes:

\\[ z \sim U(-1,1) \\]

The generator transforms that vector into an image.

In a conditional DCGAN, we also provide a class label. For example, we sample a latent vector and provide the residential label. The generator learns to produce an image corresponding to that condition.


Unlike a VAE, the DCGAN does not generally use an encoder to estimate a Gaussian distribution for each input image. Its generator starts directly from a simple random prior.

## Sampling in training, validation, and testing

Sampling is also important outside generative models.

During training, we usually shuffle the training dataset and form mini-batches. This helps prevent the model from learning an unintended ordering pattern in the data. We generally use random shuffling without replacement within an epoch, so every training observation is processed once per epoch.

During validation, we usually evaluate on the complete fixed validation set rather than randomly selecting a new subset every epoch. This makes comparisons between checkpoints more reliable.

During final testing, we also evaluate on the complete frozen test set. We do not repeatedly select favorable test subsets or tune model parameters based on test results.

For UrbanGen-AI, this means the training, validation, and test manifests have distinct roles. Generative models are trained only on the training images. Validation data can support model selection and generation-quality checks. The frozen test set is reserved for final downstream evaluation.

## Bootstrap sampling: how we measure whether our results are reliable

This is especially relevant to the evaluation methodology we want for UrbanGen-AI.

Suppose our real-only ResNet model gets a residential recall of 0.40, while our cVAE-augmented model gets 0.44.

Is the improvement real, or could it be due to which particular images happened to be in the test set?

We can use bootstrap sampling to estimate uncertainty.

We repeatedly draw samples from the test predictions with replacement, calculate the metric for each resampled dataset, and examine the distribution of those metric values.

For example:

- Draw a bootstrap sample of 4,745 test observations.
- Calculate residential recall.
- Repeat this process thousands of times.
- Examine the resulting distribution of recall values.

We can then estimate a confidence interval for the metric or, more importantly, for the difference between two models.

For our comparison, we should use paired bootstrap sampling: resample the same test observations for both models, then calculate the difference in their metrics. This preserves the fact that both models were evaluated on the same images.

If multiple images come from the same video or sequence, we should consider grouped bootstrap sampling at the video or sequence level to avoid treating highly related images as independent observations.

Bootstrap does not create new independent test evidence. It estimates uncertainty conditional on the test sample we already have.

## How do we choose the right sampling method?

Here is a practical reference for our project.

| Our task                                     | Appropriate sampling                       |
| -------------------------------------------- | ------------------------------------------ |
| Create a random training subset              | Random sampling without replacement        |
| Preserve class proportions                   | Stratified sampling                        |
| Give minority classes more training exposure | Weighted sampling or oversampling          |
| Form shuffled mini-batches                   | Random permutation / shuffled batching     |
| VAE latent generation                        | Gaussian sampling                          |
| cVAE class-controlled generation             | Gaussian latent sampling + class condition |
| DDPM initial noise                           | Gaussian sampling                          |
| DDPM training noise                          | Gaussian sampling                          |
| DCGAN generation                             | Gaussian or uniform latent sampling        |
| Select one predicted class randomly          | Categorical sampling                       |
| Estimate metric uncertainty                  | Bootstrap sampling                         |
| Estimate expected generation quality         | Monte Carlo sampling                       |
| Select images for manual review              | Stratified or uncertainty-based sampling   |

## Important problems and mistakes to avoid

There are several issues that matter in actual engineering work.

Randomness and reproducibility: Random sampling produces different results on different runs. We use random seeds to make experiments reproducible. However, setting a seed does not guarantee identical results across every GPU, library version, or hardware configuration.

Sampling bias: If we select only easy or visually clean images, our sample may not represent the full dataset. A model trained on such a sample may perform poorly on more difficult examples.

Data leakage: We must never allow test observations to enter training, synthetic-data selection, or model tuning. For UrbanGen-AI, test images must remain excluded from generative training and synthetic-image filtering.

Duplicates: Sampling with replacement intentionally creates repeated observations, but accidental duplicates can also arise through dataset preparation. We should check for duplicates across splits and avoid using repeated copies of the same image to inflate the apparent dataset size.

Distribution mismatch: If we train with a balanced class distribution but deploy in an environment where city-street images are much more common, predicted probabilities may not reflect real-world frequencies. We need to distinguish classification performance from probability calibration.

Poor latent sampling: A VAE may produce blurry or unrealistic images if its learned latent space does not support good generation. Randomly sampling from the standard normal prior does not guarantee high-quality output. We need to evaluate the generated images.

Too few generated samples: A small sample of synthetic images may make a generative model look better or worse by chance. Use consistent generation budgets and multiple seeds when comparing models.

## The complete picture: where sampling fits into UrbanGen-AI

A. Dataset preparation

BDD100K training images → stratified / random / weighted sampling → training batches

B. Generative model training

cVAE

Gaussian latent distribution → sample z → decoder

DDPM

Gaussian noise → iterative denoising

DCGAN

Random latent vector → generator

C. Synthetic data creation

Generate residential, highway, and city-street images using the same controlled generation budget.

D. Synthetic-data evaluation

Evaluate condition fidelity, diversity, memorization, image quality, and computational cost. Use validation data for model selection.

E. Downstream utility

Compare real-only, conventional augmentation, cVAE, DDPM, and DCGAN training arms using the same frozen test set.

F. Statistical confidence

Paired bootstrap sampling → confidence intervals → determine whether observed improvements are convincing.

## What you should remember

There are three major roles sampling plays in our work:

1. Selecting data: Choosing which real images are used for training, validation, or analysis.
2. Creating randomness: Drawing latent vectors and noise so generative models can create new images.
3. Measuring uncertainty: Repeatedly resampling evaluation results to understand how reliable our metrics are.

And one important conceptual distinction:

- The distribution tells us what values are likely.
- Sampling draws values according to that distribution.
- The model learns how to use those values to make predictions or generate outputs.
- Evaluation tells us whether those outputs are useful.

For UrbanGen-AI, this is the full chain: we sample real training data, learn generative models, sample latent vectors or noise to generate synthetic images, and finally use statistical resampling to judge whether those images genuinely improve downstream recognition.

That is why sampling is not a minor implementation detail. It is part of the experimental design and the mathematical foundation of our generative pipeline.


---
# Condition fidelity
**Fidelity in generative AI** means how closely generated samples resemble real data in terms of their characteristics and quality. For UrbanGen-AI, it means whether images generated by our cVAE, DDPM, or DCGAN look like plausible real urban scenes. For example, a high-fidelity residential image should have realistic buildings, roads, vehicles, sidewalks, lighting, and other visual details. Fidelity is not simply image sharpness or resolution; an image can look sharp but still contain unrealistic structures or incorrect objects. We evaluate fidelity using visual inspection and quantitative metrics such as FID (Fréchet Inception Distance) and KID (Kernel Inception Distance), while recognizing that these metrics have limitations.

**Condition fidelity** means how accurately a generated sample follows the specific condition or instruction given to the generative model. In our conditional models, the condition is the requested scene class: city street, highway, or residential. For example, if we provide the residential label, the model should generate an image that genuinely represents a residential scene. If it generates a highway image instead, the image might look realistic and therefore have good visual fidelity, but it has poor condition fidelity. Conversely, an image might clearly represent a residential street but contain distorted buildings and unrealistic roads: it has good condition fidelity but poor visual fidelity.

**The distinction for our experiments:** Fidelity asks, "Does this generated image look like a plausible real urban scene?" Condition fidelity asks, "Does it actually represent the class we requested?" We need both because our ultimate goal is not merely to generate realistic-looking images; it is to generate realistic, diverse, correctly labeled synthetic images that improve downstream residential-scene recognition.

---

nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -l 2
UrbanMind-AI
UrbanGen-AI