"""Method 2: Generate synthetic night images from CycleGAN.

NOTE: The filters in this script (CLIP night, luminance, paved_surface/SegFormer)
were later found to be unreliable. See docs/findings.md "The filter failure"
section. The 282 accepted images require manual curation before use in Arm D
training.

Modes:
  --dry-run   Verify all provenance, manifest, and source selection.
              Does NOT generate images. Prints a full report.

  (default)   Run full 500-image generation after verification passes.

Provenance chain:
  Method 2 original 500 → reconstruct via seed-42 sample of original pool
  Resume script expanded to 1,500 with same seed
  Verified: original 500 ⊂ day_train, day_train[500:] ∩ original_500 = ∅
"""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn
# from PIL import Image
# from skimage.metrics import structural_similarity as ssim
# from torchvision import transforms
# from transformers import (
#     CLIPModel,
#     CLIPProcessor,
#     SegformerForSemanticSegmentation,
#     SegformerImageProcessor,
# )

# ruff: noqa: N803, N806, SIM115

DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")
TRAIN_LABEL = DATA_ROOT / "labels" / "bdd100k_labels_images_train.json"
IMAGES_DIR = DATA_ROOT / "images" / "100k" / "train"
IMAGE_INDEX = {p.name: p for p in IMAGES_DIR.rglob("*.jpg")}
CKPT_PATH = Path(
    r"C:\Users\Janak Mallik\UrbanGen-AI\outputs\method_2_resume\checkpoints\epoch_20.pth"
)
MANIFEST_PATH = Path(
    r"C:\Users\Janak Mallik\UrbanGen-AI\outputs\method_2_resume\split_manifest.json"
)
OUTPUT_DIR = Path(r"C:\Users\Janak Mallik\UrbanGen-AI\outputs\method_2_synthetic")

IMG_H, IMG_W = 288, 512
SEED = 42
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_NIGHT_MIN = 0.50
LUMINANCE_MAX = 0.20
PAVED_SURFACE_TOLERANCE = 0.10
ADE_PAVED_SURFACE_CLASSES = [6, 11]

CLIP_PROMPTS = [
    "a photo of a daytime residential street",
    "a photo of a nighttime residential street",
]

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)


# ============================================================
# Prerequisites
# ============================================================
print("Verifying prerequisite files...")
for path, name in [(CKPT_PATH, "checkpoint"), (TRAIN_LABEL, "labels"), (MANIFEST_PATH, "manifest")]:
    if not path.exists():
        sys.exit(f"ERROR: {name} not found at {path}")
    print(f"  {name}: OK")

# ============================================================
# Manifest validation (strengthened)
# ============================================================
manifest = json.loads(MANIFEST_PATH.read_text())
print("\nValidating manifest...")

expected_counts = {
    "day_train": 1500,
    "night_train": 1500,
    "day_holdout": 200,
    "night_holdout": 200,
}
for key, expected in expected_counts.items():
    if key not in manifest:
        sys.exit(f"ERROR: missing key '{key}'")
    if len(manifest[key]) != expected:
        sys.exit(f"ERROR: {key}: expected {expected}, got {len(manifest[key])}")
    for i, entry in enumerate(manifest[key]):
        if not isinstance(entry, str) or not entry.strip():
            sys.exit(f"ERROR: {key}[{i}] = {entry!r} not a non-empty string")
    if len(set(manifest[key])) != len(manifest[key]):
        sys.exit(f"ERROR: {key} has duplicate entries")
    print(f"  {key}: {len(manifest[key])} (matches expected)")

# Six pairwise overlap checks
splits = {
    "day_train": set(manifest["day_train"]),
    "night_train": set(manifest["night_train"]),
    "day_holdout": set(manifest["day_holdout"]),
    "night_holdout": set(manifest["night_holdout"]),
}
pairs = [
    ("day_train", "day_holdout"),
    ("night_train", "night_holdout"),
    ("day_train", "night_train"),
    ("day_holdout", "night_holdout"),
    ("day_train", "night_holdout"),
    ("day_holdout", "night_train"),
]
for a, b in pairs:
    overlap = splits[a] & splits[b]
    if overlap:
        sys.exit(f"ERROR: overlap {a} & {b}: {len(overlap)} images")
    print(f"  {a} & {b}: 0 (verified)")

# ============================================================
# Provenance verification (no fallback)
# ============================================================
print("\nVerifying Method 2 original 500 provenance...")

# Reconstruct original 500 from original pool (before resume expansion)
# The original pool was: all daytime residential images from train, sorted by name
with TRAIN_LABEL.open("r", encoding="utf-8") as f:
    all_records = json.load(f)


# Build original pool (before any exclusion) — matches what method_2_cyclegan_pilot.py used
def build_original_day_pool():
    pool = []
    seen = set()
    missing = 0
    for rec in all_records:
        name = rec.get("name")
        if name in seen:
            continue
        seen.add(name)
        attrs = rec.get("attributes", {})
        if attrs.get("scene") != "residential":
            continue
        if attrs.get("timeofday") != "daytime":
            continue
        path = IMAGE_INDEX.get(name)
        if path is None:
            missing += 1
            continue
        pool.append(path)
    pool.sort(key=lambda p: p.name)
    return pool, missing


original_day_pool, missing_count = build_original_day_pool()
print(f"  Original day pool size: {len(original_day_pool)}")
print(f"  Missing labeled images: {missing_count}")

# Reconstruct the original 500 as method_2_cyclegan_pilot.py did:
# rng = random.Random(42); day_s = rng.sample(day_imgs, 550); original_train = day_s[:500]
recon_rng = random.Random(SEED)
recon_sample = recon_rng.sample(original_day_pool, 550)
method_2_original_500 = {p.name for p in recon_sample[:500]}
method_2_original_holdout = {p.name for p in recon_sample[500:]}

# Assertions
assert len(method_2_original_500) == 500, (
    f"reconstructed 500 has {len(method_2_original_500)} entries"
)
assert method_2_original_500.issubset(splits["day_train"]), (
    "reconstructed 500 not subset of manifest day_train"
)
assert method_2_original_holdout.issubset(splits["day_holdout"]), (
    "reconstructed 50 not subset of manifest day_holdout"
)
assert not (set(manifest["day_train"][500:]) & method_2_original_500), (
    "day_train[500:] overlaps with reconstructed original 500 — resume order was not preserved"
)
print("  Reconstructed original 500: verified ⊂ day_train")
print("  Reconstructed original 50 holdout: verified ⊂ day_holdout")
print("  day_train[500:] has zero overlap with original 500: verified")

# Reserved set: ALL FOUR splits
method_2_reserved = (
    splits["day_train"] | splits["day_holdout"] | splits["night_train"] | splits["night_holdout"]
)
print(f"  Total reserved (all 4 splits): {len(method_2_reserved)} images")


# ============================================================
# Source selection
# ============================================================
def get_500_day_sources():
    eligible = []
    skipped_reserved = 0
    for p in original_day_pool:
        if p.name in method_2_reserved:
            skipped_reserved += 1
            continue
        eligible.append(p)

    print("\nSource selection:")
    print(f"  Total original day pool: {len(original_day_pool)}")
    print(f"  Skipped (reserved): {skipped_reserved}")
    print(f"  Eligible: {len(eligible)}")

    if len(eligible) < 500:
        sys.exit(f"ERROR: only {len(eligible)} eligible, need 500")

    eligible.sort(key=lambda p: p.name)
    rng = random.Random(SEED)
    selected = rng.sample(eligible, 500)

    selected_names = [p.name for p in selected]
    assert len(set(selected_names)) == 500, "duplicate names in selection"
    assert not (set(selected_names) & method_2_reserved), "selected overlaps reserved"

    return selected, len(eligible)


sources, total_eligible = get_500_day_sources()
print("  Selected 500 (verified unique, verified no reserved overlap)")

# ============================================================
# Dry-run mode: stop here
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

if args.dry_run:
    print("\n" + "=" * 60)
    print("DRY RUN — verification passed, generation skipped")
    print("=" * 60)
    print("\nProvenance verified:")
    print("  - Method 2 original 500 reconstructed and confirmed ⊂ day_train")
    print("  - Original 50 holdout confirmed ⊂ day_holdout")
    print("  - day_train[500:] has zero overlap with original 500")
    print("  - All 6 split overlap checks passed")
    print("  - Manifest has expected counts (1500/1500/200/200)")
    print("\nSource selection:")
    print(f"  - Total original day pool: {len(original_day_pool)}")
    print(f"  - Reserved (excluded): {len(method_2_reserved)}")
    print(f"  - Eligible after exclusion: {total_eligible}")
    print("  - Selected: 500 (unique, no reserved overlap)")
    print("\nFirst 5 selected source names:")
    for p in sources[:5]:
        print(f"  {p.name}")
    print("\nLast 5 selected source names:")
    for p in sources[-5:]:
        print(f"  {p.name}")
    print("\nTo run full generation: remove --dry-run")
    sys.exit(0)

# After the dry-run sys.exit(0):

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.metrics import structural_similarity as ssim  # noqa: E402
from torchvision import transforms  # noqa: E402
from transformers import (  # noqa: E402
    CLIPModel,
    CLIPProcessor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

if not torch.cuda.is_available():
    sys.exit("CUDA required")

DEVICE = "cuda"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Load models (only after verification passes)
# ============================================================
torch.cuda.empty_cache()
torch.cuda.synchronize()
vram_baseline = torch.cuda.memory_allocated() / (1024**3)

print("\nLoading models...")
print(f"  VRAM baseline: {vram_baseline:.3f} GB")

ckpt_hash = hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
if not isinstance(ckpt, dict):
    sys.exit("ERROR: Invalid checkpoint structure")
if ckpt.get("epoch") != 20:
    sys.exit(f"ERROR: Expected epoch 20, got {ckpt.get('epoch')}")
if not isinstance(ckpt.get("G_AB"), dict):
    sys.exit("ERROR: Missing or invalid G_AB state dictionary")


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(128) for _ in range(4)])
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.dec(self.res(self.enc(x)))


clip_mod = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
seg_proc = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
seg_mod = (
    SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    .to(DEVICE)
    .eval()
)

G_AB = Generator().to(DEVICE)
G_AB.load_state_dict(ckpt["G_AB"], strict=True)
G_AB.eval()
del ckpt

torch.cuda.synchronize()
vram_after_load = torch.cuda.memory_allocated() / (1024**3)
print(f"  VRAM after loading: {vram_after_load:.3f} GB (+{vram_after_load - vram_baseline:.3f})")

if OUTPUT_DIR.exists() and list(OUTPUT_DIR.iterdir()):
    sys.exit(f"ERROR: {OUTPUT_DIR} is non-empty.")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

source_manifest = {
    "checkpoint": str(CKPT_PATH),
    "checkpoint_sha256": ckpt_hash,
    "checkpoint_epoch": 20,
    "seed": SEED,
    "source_selection_method": "sorted eligible filenames + random.Random(42).sample",
    "source_domain": "daytime residential",
    "total_eligible": total_eligible,
    "selected_count": len(sources),
    "selected_sources": [p.name for p in sources],
    "excluded_splits": [
        "day_train",
        "day_holdout",
        "night_train",
        "night_holdout",
    ],
    "provenance_verified": True,
    "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
    "missing_labeled_images": missing_count,
    "reserved_unique_count": len(method_2_reserved),
    "reserved_split_counts": {key: len(splits[key]) for key in splits},
    "image_index_behavior": "recursive dictionary indexing; last path wins",
    "filters": {
        "clip_night_min": CLIP_NIGHT_MIN,
        "clip_status": "PROVISIONAL — relative softmax, requires human inspection",
        "luminance_max": LUMINANCE_MAX,
        "paved_surface_tolerance": PAVED_SURFACE_TOLERANCE,
        "ssim_used_as_filter": False,
    },
    "vram_gb": {
        "baseline": round(vram_baseline, 3),
        "after_load": round(vram_after_load, 3),
    },
}
(OUTPUT_DIR / "source_selection.json").write_text(json.dumps(source_manifest, indent=2))
print("Saved source_selection.json")

# ============================================================
# Generation loop
# ============================================================
transform = transforms.Compose(
    [
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

log_path = OUTPUT_DIR / "filter_log.csv"
with open(log_path, "w", newline="") as log_file:
    writer = csv.writer(log_file)
    writer.writerow(
        [
            "index",
            "source",
            "output",
            "kept",
            "clip_night",
            "luminance",
            "paved_frac_orig",
            "paved_frac_fake",
            "ssim_diag",
            "reason",
        ]
    )

    kept = 0
    rejections = {"clip": 0, "luminance": 0, "paved_surface": 0}
    start = time.time()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for i, src_path in enumerate(sources, 1):
        with Image.open(src_path) as img:
            orig_pil = img.convert("RGB").resize((IMG_W, IMG_H), Image.LANCZOS)
        tensor = transform(orig_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            fake_tensor = G_AB(tensor)
        fake_pil = Image.fromarray(
            (
                (fake_tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1).numpy().transpose(1, 2, 0)
                * 255
            ).astype(np.uint8)
        )

        inputs = clip_proc(
            text=CLIP_PROMPTS, images=fake_pil, return_tensors="pt", padding=True
        ).to(DEVICE)
        with torch.no_grad():
            clip_out = clip_mod(**inputs)
        clip_n = float(clip_out.logits_per_image.softmax(dim=1).squeeze(0)[1])

        arr = np.array(fake_pil).astype(float)
        lum = float(
            (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).mean() / 255.0
        )

        def paved_frac(img_pil):
            inp = seg_proc(images=img_pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = seg_mod(**inp)
            up = torch.nn.functional.interpolate(
                out.logits, size=img_pil.size[::-1], mode="bilinear", align_corners=False
            )
            pred = up.argmax(dim=1).squeeze(0)
            mask = torch.zeros_like(pred, dtype=torch.bool)
            for cid in ADE_PAVED_SURFACE_CLASSES:
                mask |= pred == cid
            return float(mask.sum().item() / mask.numel())

        rf_orig = paved_frac(orig_pil)
        rf_fake = paved_frac(fake_pil)

        a = np.array(orig_pil).astype(float)
        b = np.array(fake_pil).astype(float)
        s_diag = float(ssim(a, b, channel_axis=2, data_range=255))

        reason = "accepted"
        if clip_n < CLIP_NIGHT_MIN:
            reason = "clip"
            rejections["clip"] += 1
        elif lum > LUMINANCE_MAX:
            reason = "luminance"
            rejections["luminance"] += 1
        elif abs(rf_orig - rf_fake) > PAVED_SURFACE_TOLERANCE:
            reason = "paved_surface"
            rejections["paved_surface"] += 1

        out_name = ""
        if reason == "accepted":
            out_name = f"synth_{i:04d}_{src_path.stem}.jpg"
            fake_pil.save(OUTPUT_DIR / out_name, quality=95)
            kept += 1

        writer.writerow(
            [
                i,
                src_path.name,
                out_name,
                int(reason == "accepted"),
                f"{clip_n:.3f}",
                f"{lum:.3f}",
                f"{rf_orig:.3f}",
                f"{rf_fake:.3f}",
                f"{s_diag:.3f}",
                reason,
            ]
        )

        if i % 50 == 0:
            print(f"  [{i}/500] kept={kept}  elapsed={time.time() - start:.0f}s")

torch.cuda.synchronize()
elapsed = time.time() - start
survival = kept / len(sources)
generation_peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

final_manifest = {
    **source_manifest,
    "results": {
        "kept": kept,
        "survival_rate": round(survival, 3),
        "rejections": rejections,
        "elapsed_seconds": round(elapsed, 1),
        "generation_peak_vram_gb": round(generation_peak_vram, 2),
    },
}
(OUTPUT_DIR / "generation_manifest.json").write_text(json.dumps(final_manifest, indent=2))

print(f"\n{'=' * 60}")
print(f"Generated {kept}/{len(sources)} ({survival:.1%}) in {elapsed / 60:.1f} min")
print(f"Rejections: {rejections}")
print(
    f"VRAM: baseline={vram_baseline:.2f}, after_load={vram_after_load:.2f}, generation_peak={generation_peak_vram:.2f} GB"
)
print(f"Output: {OUTPUT_DIR}")
print("\nNEXT: Human inspection of 20 accepted + 20 rejected before Arm D")
print(f"{'=' * 60}")
