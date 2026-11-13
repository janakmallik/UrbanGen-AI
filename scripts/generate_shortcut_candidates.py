"""Generate City + Highway synthetic night images for the D-bal control arm.

Gates:
  G3 — Source pool verification (400+ per class, no overlap with residential/reserved/val/test)
  G4 — One-image smoke test (--smoke-test)
  G5 — 20+20 pilot (--pilot)
  G6 — Hash provenance + per-image failure records

Usage:
  python scripts/generate_shortcut_candidates.py --smoke-test
  python scripts/generate_shortcut_candidates.py --pilot
  python scripts/generate_shortcut_candidates.py  # full 400+400
"""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import (
    CLIPModel,
    CLIPProcessor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from shortcut_control_common import (
    CLIP_NIGHT_MIN,
    DATA_ROOT,
    IMG_H,
    IMG_W,
    LUMINANCE_MAX,
    PAVED_SURFACE_TOLERANCE,
    SEED,
    checkpoint_sha256,
    compute_clip_night_for_class,
    compute_luminance,
    compute_paved_frac,
    load_generator,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = _REPO_ROOT / "outputs" / "method_2_resume" / "split_manifest.json"
RESIDENTIAL_LOG = _REPO_ROOT / "outputs" / "method_2_synthetic" / "filter_log.csv"
BASE_OUT = _REPO_ROOT / "outputs" / "method_2_synthetic" / "shortcut_control"
URBANMIND_ENV = __import__("os").environ.get("URBANMIND_REPO")
URBANMIND = Path(URBANMIND_ENV) if URBANMIND_ENV else None

CANDIDATES_PER_CLASS = 400
SMOKE_COUNT = 1
PILOT_COUNT = 20


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def verify_reserved_manifest() -> set[str]:
    """G3: Verify all 4 reserved splits have expected counts + no overlap."""
    m2 = json.loads(MANIFEST_PATH.read_text())
    expected = {"day_train": 1500, "day_holdout": 200, "night_train": 1500, "night_holdout": 200}
    for key, n in expected.items():
        if key not in m2:
            raise RuntimeError(f"Reserved manifest missing key: {key}")
        if len(m2[key]) != n:
            raise RuntimeError(f"{key} has {len(m2[key])} entries, expected {n}")
    # Pairwise disjointness
    keys = list(expected.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            overlap = set(m2[a]) & set(m2[b])
            if overlap:
                raise RuntimeError(f"Reserved splits {a}/{b} overlap: {len(overlap)}")
    return set().union(*m2.values())


def verify_residential_sources(image_index: dict) -> set[str]:
    """G3: Verify all 500 residential sources present AND exist in image index."""
    with open(RESIDENTIAL_LOG) as f:
        rows = list(csv.DictReader(f))
    sources = set()
    for r in rows:
        name = r["source"].strip()
        if not name:
            raise RuntimeError("Residential log contains blank source name")
        if name in sources:
            raise RuntimeError(f"Residential log contains duplicate source: {name}")
        sources.add(name)
    if len(sources) != 500:
        raise RuntimeError(f"Residential log has {len(sources)} unique sources, expected 500")
    missing = sources - set(image_index.keys())
    if missing:
        raise RuntimeError(
            f"{len(missing)} residential sources not in image index: {list(missing)[:5]}"
        )
    return sources


def verify_val_test_no_overlap(candidates: set[str]) -> None:
    """G3: Verify candidates don't appear in val_clean or test_clean. Fails closed."""
    if URBANMIND is None:
        raise RuntimeError(
            "URBANMIND_REPO not set — cannot verify val/test overlap. "
            "Set URBANMIND_REPO or accept that this gate is skipped (not allowed)."
        )
    val_path = URBANMIND / "data/splits/resnet_val_clean.json"
    test_path = URBANMIND / "data/splits/resnet_test_clean.json"
    if not val_path.exists() or not test_path.exists():
        raise RuntimeError(f"Missing val/test split files: {val_path}, {test_path}")
    val = set(json.loads(val_path.read_text()))
    test = set(json.loads(test_path.read_text()))
    for name in candidates:
        if name in val:
            raise RuntimeError(f"Source {name} appears in val_clean")
        if name in test:
            raise RuntimeError(f"Source {name} appears in test_clean")


def select_sources(scene: str, n: int, excluded: set, image_index: dict, seed: int) -> list[Path]:
    records = json.loads((DATA_ROOT / "labels" / "bdd100k_labels_images_train.json").read_text())
    pool = []
    for rec in records:
        attrs = rec.get("attributes", {})
        if attrs.get("scene") == scene and attrs.get("timeofday") == "daytime":
            name = rec["name"]
            if name not in excluded and name in image_index:
                pool.append(image_index[name])
    pool.sort(key=lambda p: p.name)
    if len(pool) < n:
        raise RuntimeError(f"Only {len(pool)} eligible {scene} images, need {n}")
    print(f"  {scene} pool: {len(pool)}, selecting {n}")
    return random.Random(seed).sample(pool, n)


def generate_class(
    scene_label: str,
    sources: list,
    out_dir: Path,
    generator,
    clip_mod,
    clip_proc,
    seg_mod,
    seg_proc,
    device,
    transform,
    checkpoint_hash: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "filter_log.csv"
    accepted, rejected, failed = [], [], []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "source",
                "source_sha256",
                "output",
                "output_sha256",
                "kept",
                "clip_night",
                "luminance",
                "paved_frac_orig",
                "paved_frac_fake",
                "reason",
            ]
        )

        for i, src_path in enumerate(sources, 1):
            try:
                src_hash = hashlib.sha256(src_path.read_bytes()).hexdigest()
                with Image.open(src_path) as img:
                    orig_pil = img.convert("RGB").resize((IMG_W, IMG_H), Image.LANCZOS)
                tensor = transform(orig_pil).unsqueeze(0).to(device)
                with torch.inference_mode():
                    fake_tensor = generator(tensor)
                fake_pil = Image.fromarray(
                    (
                        (fake_tensor.squeeze(0).cpu() * 0.5 + 0.5)
                        .clamp(0, 1)
                        .numpy()
                        .transpose(1, 2, 0)
                        * 255
                    ).astype(np.uint8)
                )

                clip_n = compute_clip_night_for_class(
                    clip_mod, clip_proc, fake_pil, device, scene_label
                )
                lum = compute_luminance(fake_pil)
                rf_orig = compute_paved_frac(seg_mod, seg_proc, orig_pil, device)
                rf_fake = compute_paved_frac(seg_mod, seg_proc, fake_pil, device)

                reason = "accepted"
                if clip_n < CLIP_NIGHT_MIN:
                    reason = "clip"
                elif lum > LUMINANCE_MAX:
                    reason = "luminance"
                elif abs(rf_orig - rf_fake) > PAVED_SURFACE_TOLERANCE:
                    reason = "paved_surface"

                out_name = f"{scene_label}_{i:04d}_{src_path.stem}.jpg"
                fake_pil.save(out_dir / out_name, quality=95)
                out_hash = hashlib.sha256((out_dir / out_name).read_bytes()).hexdigest()

                if reason == "accepted":
                    accepted.append(out_name)
                else:
                    rejected.append(out_name)

                writer.writerow(
                    [
                        i,
                        src_path.name,
                        src_hash,
                        out_name,
                        out_hash,
                        int(reason == "accepted"),
                        f"{clip_n:.4f}",
                        f"{lum:.4f}",
                        f"{rf_orig:.4f}",
                        f"{rf_fake:.4f}",
                        reason,
                    ]
                )
            except Exception as e:
                failed.append(src_path.name)
                writer.writerow(
                    [
                        i,
                        src_path.name,
                        "",
                        "",
                        "",
                        0,
                        "",
                        "",
                        "",
                        "",
                        f"error: {type(e).__name__}: {e}",
                    ]
                )
                print(f"  [{scene_label}] FAILED on {src_path.name}: {e}")

            if i % 50 == 0:
                print(f"  [{scene_label} {i}/{len(sources)}] accepted={len(accepted)}")

    manifest = {
        "class": scene_label,
        "attempted": len(sources),
        "accepted": len(accepted),
        "rejected_by_filter": len(rejected),
        "failed": len(failed),
        "survival_rate": round(len(accepted) / len(sources), 3),
        "accepted_filenames": accepted,
        "rejected_filenames": rejected,
        "failed_sources": failed,
        "source_selection_seed": SEED if scene_label == "city" else SEED + 1,
        "filter_thresholds": {
            "clip_night_min": CLIP_NIGHT_MIN,
            "luminance_max": LUMINANCE_MAX,
            "paved_surface_tolerance": PAVED_SURFACE_TOLERANCE,
        },
        "checkpoint_sha256": checkpoint_hash,
        "git_sha": git_sha(),
        "generation_timestamp": __import__("datetime").datetime.now().isoformat(),
    }
    (out_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2))
    return len(accepted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Generate 1 city + 1 highway")
    parser.add_argument("--pilot", action="store_true", help="Generate 20 city + 20 highway")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = "cuda"

    # Determine run mode
    if args.smoke_test:
        n_per_class = SMOKE_COUNT
        run_suffix = "smoke"
    elif args.pilot:
        n_per_class = PILOT_COUNT
        run_suffix = "pilot"
    else:
        n_per_class = CANDIDATES_PER_CLASS
        run_suffix = "full"

    print(f"=== Run mode: {run_suffix.upper()} ({n_per_class} per class) ===")

    # Pre-flight: refuse to overwrite the EXACT output directories this run will write to
    out_suffix = "candidates" if run_suffix == "full" else f"candidates_{run_suffix}"
    for label in ("city", "highway"):
        p = BASE_OUT / label / out_suffix
        if p.exists() and any(p.iterdir()):
            sys.exit(f"ERROR: {p} is not empty. Move or delete before retrying.")

    # G3: Build image index FIRST
    print("\n[G3] Building train image index...")
    image_index = {}
    duplicates = []
    for p in (DATA_ROOT / "images" / "100k" / "train").rglob("*.jpg"):
        if p.name in image_index:
            duplicates.append(p.name)
        else:
            image_index[p.name] = p
    if duplicates:
        raise RuntimeError(
            f"Duplicate filenames in train image tree: {len(duplicates)} (e.g. {duplicates[:5]})"
        )
    print(f"  Train image index: {len(image_index)} unique filenames")

    # G3: Verify reserved manifest + residential sources
    print("\n[G3] Verifying source pools...")
    reserved = verify_reserved_manifest()
    print("  Reserved splits: 3,400 images (verified)")
    missing_reserved = reserved - set(image_index.keys())
    if missing_reserved:
        raise RuntimeError(
            f"{len(missing_reserved)} reserved images missing from train index "
            f"(e.g. {list(missing_reserved)[:5]})"
        )
    print("  All 3,400 reserved images present in train index")
    residential = verify_residential_sources(image_index)
    print("  Residential sources: 500 (verified, all exist in index)")
    overlap_res = reserved & residential
    if overlap_res:
        raise RuntimeError(
            f"Residential sources overlap reserved splits: {len(overlap_res)} "
            f"(e.g. {list(overlap_res)[:5]})"
        )
    excluded = reserved | residential

    # Select sources: ALWAYS sample 400 from the full pool, then slice.
    # This guarantees smoke ⊂ pilot ⊂ full.
    city_sources_full = select_sources(
        "city street", CANDIDATES_PER_CLASS, excluded, image_index, seed=SEED
    )
    hwy_sources_full = select_sources(
        "highway", CANDIDATES_PER_CLASS, excluded, image_index, seed=SEED + 1
    )
    city_sources = city_sources_full[:n_per_class]
    hwy_sources = hwy_sources_full[:n_per_class]

    if {p.name for p in city_sources} & {p.name for p in hwy_sources}:
        raise RuntimeError("City and highway source sets overlap")
    verify_val_test_no_overlap({p.name for p in city_sources + hwy_sources})
    print(f"  Selected: {len(city_sources)} city, {len(hwy_sources)} highway")

    # Load models
    print("\nLoading models...")
    checkpoint_hash = checkpoint_sha256()
    print(f"  Checkpoint SHA-256: {checkpoint_hash[:16]}...")

    free_vram = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"  Free VRAM: {free_vram:.2f} GB")
    if free_vram < 3.0:
        raise RuntimeError(f"Only {free_vram:.2f} GB free VRAM, need ≥3.0 GB")

    generator = load_generator(device)
    clip_mod = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    seg_proc = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    seg_mod = (
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        .to(device)
        .eval()
    )

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Generate
    print("\nGenerating City synthetic images...")
    city_kept = generate_class(
        "city",
        city_sources,
        BASE_OUT / "city" / out_suffix,
        generator,
        clip_mod,
        clip_proc,
        seg_mod,
        seg_proc,
        device,
        transform,
        checkpoint_hash,
    )

    print("\nGenerating Highway synthetic images...")
    hwy_kept = generate_class(
        "highway",
        hwy_sources,
        BASE_OUT / "highway" / out_suffix,
        generator,
        clip_mod,
        clip_proc,
        seg_mod,
        seg_proc,
        device,
        transform,
        checkpoint_hash,
    )

    print(f"\n=== SURVIVAL ({run_suffix}) ===")
    print(f"City:    {city_kept}/{n_per_class} = {city_kept / n_per_class:.1%}")
    print(f"Highway: {hwy_kept}/{n_per_class} = {hwy_kept / n_per_class:.1%}")
    print("Residential (reference): 282/500 = 56.4%")


if __name__ == "__main__":
    main()
