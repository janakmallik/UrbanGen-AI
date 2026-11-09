
# Model Card — BDD100K CycleGAN v1 (Day-to-Night)

## What this covers
A CycleGAN model trained from scratch to translate daytime urban scenes into nighttime scenes while preserving the original scene geometry.

## Architecture
- **Model:** Lightweight CycleGAN (ResNet-based generator, PatchGAN discriminator)
- **Generator parameters:** ~500K
- **Discriminator parameters:** ~2.7M
- **Input resolution:** 512×288
- **Channels:** RGB (3)
- **Normalization:** InstanceNorm
- **Activation:** ReLU (encoder), Tanh (output)

## Training configuration
- **Dataset:** BDD100K (train split only)
  - 1,500 daytime residential images
  - 1,500 nighttime residential images
  - 200 day / 200 night held-out (zero overlap)
- **Losses:** Adversarial (LSGAN), cycle-consistency (L1, λ=10), identity (L1, λ=5)
- **Optimizer:** Adam (lr=2e-4, β1=0.5)
- **LR schedule:** Constant 2e-4 for epochs 1-10, linear decay to 2e-5 by epoch 20
- **Batch size:** 2
- **Epochs:** 20 total (5 pilot + 15 resume)
- **Data augmentation:** Horizontal flip (p=0.5)
- **Replay buffer:** 50 images
- **Seed:** 42

## Results
- **Training time:** 57 minutes (epochs 6-20 on RTX 4050)
- **Peak VRAM:** 2.1 GB / 6.0 GB
- **GPU utilization:** ~71%
- **Output quality:** Preserves cars, buildings, road signs, and scene layout while producing convincing nighttime illumination.
- **Purple/blue artifacts:** Nearly eliminated by epoch 20 through LR decay.

## Known limitations
- Generates at 512×288 (not full 1280×720). Would need upscaling for some downstream tasks.
- Small model. Bigger CycleGAN architectures would produce higher-quality images but exceed 6 GB VRAM budget.
- Not bit-for-bit reproducible (depends on GPU, PyTorch, CUDA versions).
- Trained only on residential scenes. Does not handle city street or highway well.

## What this model is NOT
- Not a replacement for real BDD100K nighttime images. It generates *synthetic* night images from real day images.
- Not a general-purpose day-to-night model. It is specific to BDD100K dashcam residential scenes.
- Not evaluated for downstream classifier improvement yet (that is Arm D's task).

## Reproduce
```powershell
uv run python scripts\method_2_cyclegan_resume.py
```

Checkpoints: ``outputs/method_2_resume/checkpoints/epoch_20.pth``
