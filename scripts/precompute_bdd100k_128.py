"""Pre-compute 128x128 uint8 image cache for generative modeling.

BDD100K JPEGs are 1280x720. Decoding and resizing 68,969 images every
epoch is the throughput bottleneck. This script decodes each image once,
applies a fixed preprocessing policy, and stores the result as a
memory-mapped uint8 .npy per split.

Preprocessing policy (frozen, applied identically to all splits):
    - BDD100K is 16:9 (1280x720).
    - Resize preserving aspect ratio to fit inside 128x128:
      resize to (128, 72), then center-pad to (128, 128) with 0.
    - This preserves scene geometry (buildings, roads) instead of
      distorting it with a direct square resize.
    - The same policy is used for cVAE training, sample evaluation, and
      downstream classifier inputs.

Cache layout (data/bdd100k_128/):
    train_images.npy  (68969, 128, 128, 3) uint8  ~3.16 GiB
    train_labels.npy  (68969,)             int64
    val_images.npy    (4727, 128, 128, 3)  uint8  ~226 MiB
    val_labels.npy    (4727,)              int64
    test_images.npy   (4745, 128, 128, 3)  uint8  ~227 MiB
    test_labels.npy   (4745,)              int64
    manifest.json     provenance record

The test cache is written but marked 'read_only_downstream' in the
manifest. G1 training scripts must refuse to load it.

Memory discipline:
    - Files are created with np.lib.format.open_memmap and filled in
      place. Peak RAM is one worker's batch of decoded images, not the
      full split.
    - Multiprocessing is bounded: at most MAX_IN_FLIGHT futures at once.

Strict integrity:
    - Any missing image raises. Any duplicate filename in the image
      tree raises. Any split manifest hash mismatch raises.
    - The previous G0 audit baseline (label JSON hashes, split manifest
      hashes) is preserved in the manifest.

Run:
    uv run python scripts/precompute_bdd100k_128.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.config import data_root, urbanmind_root  # noqa: E402

CLASS_NAMES = ("city street", "highway", "residential")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
TARGET_SIZE = 128
MAX_IN_FLIGHT = 64  # bounded futures to keep RAM low

# --- Frozen provenance (must match G0 audit) ---
EXPECTED_TRAIN_JSON_SHA = "c1998792e385fd79213187576ca07d5249563d0ab83ae976688ad60668baca9e"
EXPECTED_VAL_JSON_SHA = "c835be652f0c002897be51e74deedd16a1dcee0b7230f19c0456fba772f694af"
EXPECTED_MANIFEST_SHAS = {
    "resnet_val_clean.json": "9ee1b6c69f4e58e83edb704fbd8d42b3972a9b0410aa838a939143205f924682",
    "resnet_val_excluded.json": "05d54fda00e07083d73f2114607636a6c4056802b5bfd5f5614de2c79d0c4760",
    "resnet_test_clean.json": "ca47e16a4c6148520b2a0cc3de74b78db20392507c93ea4d5a4039c695d03979",
    "resnet_test_excluded.json": "ec48bc0166145448e726bb277a7e62084deb727a457424f38f4184e1a6d9daea",
}
EXPECTED_SPLIT_SIZES = {"train": 68_969, "val": 4_727, "test": 4_745}


class CacheError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise CacheError(msg)


# --- Preprocessing ---
def _preprocess_one(path_str: str) -> bytes:
    """Decode one image, apply frozen preprocessing, return raw bytes (128*128*3)."""
    with Image.open(path_str) as img:
        img = img.convert("RGB")
        # Resize preserving aspect ratio to fit inside 128x128
        # 1280x720 -> scale to (128, 72)
        img = img.resize((TARGET_SIZE, 72), Image.BILINEAR)
        # Center-pad to 128x128 with black
        padded = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
        pad_top = (TARGET_SIZE - 72) // 2
        padded.paste(img, (0, pad_top))
        return padded.tobytes()


def _build_split_memmap(
    split_name: str,
    filenames: list[str],
    name_to_label: dict[str, int],
    image_index: dict[str, Path],
    out_path: Path,
    labels_path: Path,
    workers: int,
) -> dict:
    n = len(filenames)
    expected = EXPECTED_SPLIT_SIZES[split_name]
    _require(
        n == expected,
        f"{split_name}: expected {expected} filenames, got {n}",
    )

    # Verify all filenames exist and have labels — fail loudly
    missing_images: list[str] = []
    missing_labels: list[str] = []
    tasks: list[tuple[str, int]] = []
    labels = np.empty(n, dtype=np.int64)
    for i, fn in enumerate(filenames):
        p = image_index.get(fn)
        if p is None:
            missing_images.append(fn)
            continue
        lab = name_to_label.get(fn)
        if lab is None:
            missing_labels.append(fn)
            continue
        tasks.append((str(p), lab))
        labels[i] = lab

    _require(not missing_images, f"{split_name}: {len(missing_images)} images missing from disk")
    _require(not missing_labels, f"{split_name}: {len(missing_labels)} images missing labels")

    # Create the memmap file, write incrementally
    images = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(n, TARGET_SIZE, TARGET_SIZE, 3)
    )
    np.save(labels_path, labels)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        # Submit in bounded windows
        pending: dict = {}
        next_task = 0
        done = 0
        pbar = tqdm(total=n, desc=f"  {split_name}", unit="img")

        def fill_window() -> None:
            nonlocal next_task
            while next_task < n and len(pending) < MAX_IN_FLIGHT:
                tasks_i = next_task
                # tasks is 1:1 with filenames order since no missing
                fut = ex.submit(_preprocess_one, tasks[tasks_i][0])
                pending[fut] = tasks_i
                next_task += 1

        fill_window()
        while pending:
            done_futs, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done_futs:
                idx = pending.pop(fut)
                raw = fut.result()
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                images[idx] = arr
                done += 1
                pbar.update(1)
            fill_window()
        pbar.close()

    _require(done == n, f"{split_name}: processed {done} of {n}")
    images.flush()
    del images  # close memmap

    # Compute class counts
    class_counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(len(CLASS_NAMES))}

    return {
        "images_path": str(out_path.relative_to(_REPO_ROOT)),
        "labels_path": str(labels_path.relative_to(_REPO_ROOT)),
        "n": n,
        "images_sha256": sha256_file(out_path),
        "labels_sha256": sha256_file(labels_path),
        "class_counts": class_counts,
        "read_only_downstream": split_name == "test",
    }


def _load_manifest_filenames(path: Path, expected_sha: str) -> list[str]:
    actual = sha256_file(path)
    _require(actual == expected_sha, f"{path.name} SHA-256 mismatch: {actual}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_name_to_label() -> dict[str, int]:
    label_dir = data_root() / "labels"
    train_path = label_dir / "bdd100k_labels_images_train.json"
    val_path = label_dir / "bdd100k_labels_images_val.json"

    _require(train_path.exists(), f"missing {train_path}")
    _require(val_path.exists(), f"missing {val_path}")
    _require(sha256_file(train_path) == EXPECTED_TRAIN_JSON_SHA, "train label JSON SHA mismatch")
    _require(sha256_file(val_path) == EXPECTED_VAL_JSON_SHA, "val label JSON SHA mismatch")

    mapping: dict[str, int] = {}
    train_recs = json.loads(train_path.read_text(encoding="utf-8"))
    for rec in train_recs:
        scene = rec.get("attributes", {}).get("scene")
        if scene in CLASS_TO_ID:
            mapping[rec["name"]] = CLASS_TO_ID[scene]

    # Validation images use the val label JSON
    val_recs = json.loads(val_path.read_text(encoding="utf-8"))
    for rec in val_recs:
        scene = rec.get("attributes", {}).get("scene")
        if scene in CLASS_TO_ID:
            mapping[rec["name"]] = CLASS_TO_ID[scene]

    return mapping


def _load_train_filenames() -> list[str]:
    """Filenames from the train label JSON with valid 3-class scenes."""
    train_path = data_root() / "labels" / "bdd100k_labels_images_train.json"
    data = json.loads(train_path.read_text(encoding="utf-8"))
    return [rec["name"] for rec in data if rec.get("attributes", {}).get("scene") in CLASS_TO_ID]


def main() -> int:
    try:
        workers = min(8, os.cpu_count() or 4)
        out_dir = _REPO_ROOT / "data" / "bdd100k_128"
        out_dir.mkdir(parents=True, exist_ok=True)

        urbanmind = urbanmind_root()
        print(f"UrbanMind checkout: {urbanmind}")

        # Load labels
        print("\nLoading label mappings...")
        name_to_label = _load_name_to_label()
        print(f"  {len(name_to_label):,} labeled 3-class images in train+val JSONs")

        # Index all images; fail on duplicate filenames
        print("\nIndexing images...")
        images_root = data_root() / "images"
        _require(images_root.exists(), f"missing {images_root}")
        image_index: dict[str, Path] = {}
        duplicates: list[str] = []
        for jpg in images_root.rglob("*.jpg"):
            if jpg.name in image_index:
                duplicates.append(jpg.name)
            else:
                image_index[jpg.name] = jpg
        _require(not duplicates, f"{len(duplicates)} duplicate filenames: {duplicates[:5]}")
        print(f"  {len(image_index):,} unique images indexed")

        # Load split manifests with hash verification
        print("\nLoading split manifests...")
        splits_dir = urbanmind / "data" / "splits"
        val_clean = _load_manifest_filenames(
            splits_dir / "resnet_val_clean.json",
            EXPECTED_MANIFEST_SHAS["resnet_val_clean.json"],
        )
        test_clean = _load_manifest_filenames(
            splits_dir / "resnet_test_clean.json",
            EXPECTED_MANIFEST_SHAS["resnet_test_clean.json"],
        )
        # Also verify the excluded manifests, though we don't consume them
        for nm in ("resnet_val_excluded.json", "resnet_test_excluded.json"):
            _load_manifest_filenames(splits_dir / nm, EXPECTED_MANIFEST_SHAS[nm])
        print(f"  val_clean:  {len(val_clean):,}")
        print(f"  test_clean: {len(test_clean):,}")

        # Train filenames from label JSON
        train_filenames = _load_train_filenames()
        print(f"  train:      {len(train_filenames):,}")

        # Manifest
        manifest: dict = {
            "policy": {
                "target_size": TARGET_SIZE,
                "aspect_ratio_preserving": True,
                "original_aspect": "16:9 (1280x720)",
                "resize_step": "Resize to (128, 72) then center-pad to (128, 128) with black",
                "resample_filter": "PIL.Image.BILINEAR",
            },
            "class_names": list(CLASS_NAMES),
            "source": {
                "urbanmind_commit": "dfcfaf7",
                "urbanmind_tag": "v0.10.0-week10-calibration",
                "label_json_sha256": {
                    "train": EXPECTED_TRAIN_JSON_SHA,
                    "val": EXPECTED_VAL_JSON_SHA,
                },
                "split_manifest_sha256": EXPECTED_MANIFEST_SHAS,
            },
            "splits": {},
        }

        # Process splits
        for split_name, filenames in [
            ("train", train_filenames),
            ("val", val_clean),
            ("test", test_clean),
        ]:
            print(f"\nProcessing {split_name} ({len(filenames):,} images)...")
            info = _build_split_memmap(
                split_name,
                filenames,
                name_to_label,
                image_index,
                out_dir / f"{split_name}_images.npy",
                out_dir / f"{split_name}_labels.npy",
                workers,
            )
            manifest["splits"][split_name] = info
            print(f"  saved {info['images_path']}: {info['n']:,} images")
            print(f"  SHA: {info['images_sha256'][:16]}...")
            print(f"  class counts: {info['class_counts']}")
            print(f"  read_only_downstream: {info['read_only_downstream']}")

        # Write manifest
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Disk usage report
        total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.npy"))
        print(f"\n{'=' * 60}")
        print(f"Cache complete: {total_bytes / (1024**3):.2f} GiB total")
        print(f"Manifest: {manifest_path}")
        print(f"{'=' * 60}")
        return 0

    except CacheError as e:
        print(f"\nCACHE FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
