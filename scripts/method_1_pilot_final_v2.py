import json
import random
import shutil
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageDraw

# Paths
DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")
TRAIN_LABEL = DATA_ROOT / "labels" / "bdd100k_labels_images_train.json"
IMAGES_DIR = DATA_ROOT / "images" / "100k" / "train"

OUTPUT_DIR = Path(r"C:\Users\Janak Mallik\UrbanGen-AI\outputs\method_1_pilot_final_v2")
ORIGINALS_DIR = OUTPUT_DIR / "originals"
COMPARISONS_DIR = OUTPUT_DIR / "comparisons"

for directory in (OUTPUT_DIR, ORIGINALS_DIR, COMPARISONS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Configuration
COUNT = 5
SEED = 42

STRENGTH = 0.40
STEPS = 30
GUIDANCE_SCALE = 6.0

# Preserve 16:9 aspect ratio
TARGET_W = 512
TARGET_H = 288

DISPLAY_W = 1280
DISPLAY_H = 720

PROMPT = (
    "photorealistic nighttime residential street, "
    "dark night sky, dim ambient lighting, "
    "warm illuminated streetlights, "
    "visible car headlights, glowing building windows, "
    "realistic dark road, natural nighttime photography, "
    "original street layout and camera perspective"
)

NEGATIVE_PROMPT = (
    "daylight, daytime, sunlight, sunny sky, "
    "overexposed, washed out, "
    "distorted architecture, deformed vehicles, "
    "duplicated objects, blurry, cartoon, "
    "painting, illustration, oversaturated"
)


def select_training_images(count):
    with TRAIN_LABEL.open("r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []

    for rec in data:
        attrs = rec.get("attributes", {})

        if attrs.get("scene") == "residential" and attrs.get("timeofday") == "daytime":
            img_path = IMAGES_DIR / rec["name"]

            if img_path.exists():
                candidates.append(img_path)

    candidates.sort(key=lambda p: p.name)

    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible images found; {count} required.")

    return random.Random(SEED).sample(candidates, count)


def make_comparison(orig_path, gen_path, out_path):
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    with Image.open(orig_path) as img:
        orig = img.convert("RGB").resize((DISPLAY_W, DISPLAY_H), resampling)

    with Image.open(gen_path) as img:
        gen = img.convert("RGB").resize((DISPLAY_W, DISPLAY_H), resampling)

    canvas = Image.new(
        "RGB",
        (DISPLAY_W * 2, DISPLAY_H + 50),
        (30, 30, 30),
    )

    canvas.paste(orig, (0, 50))
    canvas.paste(gen, (DISPLAY_W, 50))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 15), "ORIGINAL (DAY)", fill="white")
    draw.text(
        (DISPLAY_W + 10, 15),
        "GENERATED (NIGHT)",
        fill="white",
    )

    canvas.save(out_path, quality=95)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("Loading SD 1.5 Img2Img...")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()

    selected = select_training_images(COUNT)

    print(f"\nSelected images: {len(selected)}")
    print(f"Resolution: {TARGET_W}x{TARGET_H}")
    print(f"Strength: {STRENGTH}")
    print(f"Steps: {STEPS}")
    print(f"Guidance: {GUIDANCE_SCALE}")

    for i, img_path in enumerate(selected, 1):
        print(f"\n[{i}/{COUNT}] {img_path.name}")

        orig_copy = ORIGINALS_DIR / (f"original_{i:02d}_{img_path.name}")

        shutil.copy2(img_path, orig_copy)

        with Image.open(img_path) as img:
            init_image = img.convert("RGB").resize(
                (TARGET_W, TARGET_H),
                getattr(Image, "Resampling", Image).LANCZOS,
            )

        generator = torch.Generator(device="cuda").manual_seed(SEED + i)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()

        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=init_image,
            strength=STRENGTH,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=STEPS,
            generator=generator,
        ).images[0]

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

        out_path = OUTPUT_DIR / f"night_{i:02d}.jpg"
        comp_path = COMPARISONS_DIR / f"compare_{i:02d}.jpg"

        result.save(out_path, quality=95)

        make_comparison(
            orig_copy,
            out_path,
            comp_path,
        )

        print(f"Saved: {out_path.name}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Peak allocated VRAM: {peak_allocated:.2f} GB")
        print(f"Peak reserved VRAM: {peak_reserved:.2f} GB")

    print("\nGeneration complete.")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Comparisons: {COMPARISONS_DIR}")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
