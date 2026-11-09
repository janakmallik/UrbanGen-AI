"""Calibrate metrics on real BDD100K images before running ControlNet pilot.

Computes SegFormer class proportions, luminance, and CLIP similarity for:
  - 5 real daytime residential images (train split, seed 999)
  - 5 real nighttime residential images (train split, seed 999)

IMPORTANT INTERPRETATION NOTES:
  - CLIP scores are SOFTMAX OVER THE 4 PROMPTS. They are comparative
    ("which prompt fits this image best?"), NOT absolute probabilities of
    being nighttime or residential. A score of 0.55 means "prompt 2 is
    55% of the softmax mass over these 4 prompts", not "55% chance this
    is a nighttime residential photo".
  - SegFormer proportions are PREDICTED class fractions (not IoU against
    ground truth). We do not have per-image ground-truth masks for these
    images (they are not in the BDD100K 7K seg subset).
  - No thresholds are set here. This script only measures. Human + audit
    review sets thresholds after inspecting distributions.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")
TRAIN_LABEL = DATA_ROOT / "labels" / "bdd100k_labels_images_train.json"
IMAGES_DIR = DATA_ROOT / "images" / "100k" / "train"


def build_image_index():
    """Recursively index all training images by filename.

    BDD100K nests training images under train/trainA/ and train/trainB/,
    but the label JSON references them by bare filename. Building a
    filename -> path index once resolves the label without ambiguity.
    """
    print(f"Building image index from {IMAGES_DIR} (recursive)...")
    index = {}
    for p in IMAGES_DIR.rglob("*.jpg"):
        if p.name in index:
            raise RuntimeError(f"Duplicate filename detected: {p.name} at {p} and {index[p.name]}")
        index[p.name] = p
    print(f"Indexed {len(index):,} images")
    return index


IMAGE_INDEX = build_image_index()
OUTPUT_DIR = Path(r"C:\Users\Janak Mallik\UrbanGen-AI\outputs\controlnet_calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNT_PER_CLASS = 5
SEED = 999  # Different from Method 3 source images (seed 42)

ADE_GROUPS = {
    "road": [6, 11],  # road + sidewalk
    "vehicle": [20, 80, 83, 102],  # car, bus, truck, van
    "sky": [2],
    "building": [1, 25, 48],  # building, house, skyscraper
    "vegetation": [4, 9, 17, 72],  # tree, grass, plant, palm
    "person": [12],
    "other": None,
}

CLIP_PROMPTS = [
    "a photo of a daytime residential street",
    "a photo of a nighttime residential street",
    "a photo of a daytime urban scene",
    "a photo of a nighttime urban scene",
]


def load_models(device):
    print("Loading SegFormer-B0...")
    seg_processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    seg_model = (
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        .to(device)
        .eval()
    )

    print("Loading CLIP (may download ~600 MB on first run)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return seg_processor, seg_model, clip_processor, clip_model


def compute_segformer_proportions(image, seg_processor, seg_model, device):
    inputs = seg_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1).squeeze(0)
    total = pred.numel()

    proportions = {}
    used = 0
    for group, class_ids in ADE_GROUPS.items():
        if class_ids is None:
            continue
        mask = torch.zeros_like(pred, dtype=torch.bool)
        for cid in class_ids:
            mask |= pred == cid
        proportions[group] = float(mask.sum().item() / total)
        used += mask.sum().item()
    proportions["other"] = float((total - used) / total)
    return proportions


def compute_luminance(image):
    arr = np.array(image.convert("RGB")).astype(float)
    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return float(luminance.mean() / 255.0)


def compute_clip_scores(image, clip_processor, clip_model, device):
    inputs = clip_processor(text=CLIP_PROMPTS, images=image, return_tensors="pt", padding=True).to(
        device
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
    # Softmax over the 4 prompts: RELATIVE preference, not absolute probability
    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)
    return {
        "daytime_residential": float(probs[0]),
        "nighttime_residential": float(probs[1]),
        "daytime_urban": float(probs[2]),
        "nighttime_urban": float(probs[3]),
    }


def select_images():
    with TRAIN_LABEL.open("r", encoding="utf-8") as f:
        data = json.load(f)

    day_candidates = []
    night_candidates = []
    missing = 0
    for rec in data:
        attrs = rec.get("attributes", {})
        if attrs.get("scene") != "residential":
            continue
        tod = attrs.get("timeofday")
        if tod not in ("daytime", "night"):
            continue
        path = IMAGE_INDEX.get(rec["name"])
        if path is None:
            missing += 1
            continue
        if tod == "daytime":
            day_candidates.append(path)
        elif tod == "night":
            night_candidates.append(path)

    if missing:
        print(f"WARNING: {missing} labeled images were not found in the index")

    day_candidates.sort(key=lambda p: p.name)
    night_candidates.sort(key=lambda p: p.name)

    print(f"Daytime pool: {len(day_candidates):,} images")
    print(f"Nighttime pool: {len(night_candidates):,} images")

    if len(day_candidates) < COUNT_PER_CLASS:
        sys.exit(f"ERROR: only {len(day_candidates)} day images, need {COUNT_PER_CLASS}")
    if len(night_candidates) < COUNT_PER_CLASS:
        sys.exit(f"ERROR: only {len(night_candidates)} night images, need {COUNT_PER_CLASS}")

    rng = random.Random(SEED)
    day_selected = rng.sample(day_candidates, COUNT_PER_CLASS)
    night_selected = rng.sample(night_candidates, COUNT_PER_CLASS)
    return day_selected, night_selected


def _stats(values):
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "mean": round(mean, 4),
        "std": round(var**0.5, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": n,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    seg_processor, seg_model, clip_processor, clip_model = load_models(device)

    day_images, night_images = select_images()

    manifest = {
        "config": {
            "segformer": "nvidia/segformer-b0-finetuned-ade-512-512",
            "clip": "openai/clip-vit-base-patch32",
            "seed": SEED,
            "count_per_class": COUNT_PER_CLASS,
            "clip_prompts": CLIP_PROMPTS,
            "notes": (
                "CLIP scores are softmax over the 4 prompts (comparative, not absolute). "
                "SegFormer proportions are predicted class fractions (not IoU)."
            ),
        },
        "daytime": [],
        "nighttime": [],
    }

    for split_name, images in [("daytime", day_images), ("nighttime", night_images)]:
        print(f"\n--- {split_name.capitalize()} ---")
        for i, img_path in enumerate(images, 1):
            print(f"  [{i}/{len(images)}] {img_path.name}")
            with Image.open(img_path) as img:
                pil = img.convert("RGB")

            seg = compute_segformer_proportions(pil, seg_processor, seg_model, device)
            lum = compute_luminance(pil)
            clip = compute_clip_scores(pil, clip_processor, clip_model, device)

            manifest[split_name].append(
                {
                    "path": str(img_path),
                    "luminance": round(lum, 4),
                    "segformer_proportions": {k: round(v, 4) for k, v in seg.items()},
                    "clip_scores": {k: round(v, 4) for k, v in clip.items()},
                }
            )
            print(f"    luminance={lum:.3f}  clip_night_res={clip['nighttime_residential']:.3f}")

    summary = {}
    for key in ["daytime", "nighttime"]:
        records = manifest[key]
        summary[key] = {
            "luminance": _stats([r["luminance"] for r in records]),
            "clip_nighttime_residential": _stats(
                [r["clip_scores"]["nighttime_residential"] for r in records]
            ),
            "clip_daytime_residential": _stats(
                [r["clip_scores"]["daytime_residential"] for r in records]
            ),
            "road_proportion": _stats([r["segformer_proportions"]["road"] for r in records]),
            "vehicle_proportion": _stats([r["segformer_proportions"]["vehicle"] for r in records]),
            "sky_proportion": _stats([r["segformer_proportions"]["sky"] for r in records]),
            "building_proportion": _stats(
                [r["segformer_proportions"]["building"] for r in records]
            ),
        }
    manifest["summary"] = summary

    out_path = OUTPUT_DIR / "calibration.json"
    out_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Calibration saved to: {out_path}")
    print(f"{'=' * 60}")
    print("\nDistributions (for setting thresholds):")
    for key in ["daytime", "nighttime"]:
        print(f"\n{key.upper()}:")
        for metric, stats in summary[key].items():
            print(
                f"  {metric:30s}  mean={stats['mean']}  std={stats['std']}  "
                f"range=[{stats['min']}, {stats['max']}]  n={stats['n']}"
            )


if __name__ == "__main__":
    main()
