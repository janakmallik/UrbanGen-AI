"""Unit tests for BDD100K128Dataset.

Two groups:
- Fixture-based tests: build a tiny synthetic cache, run on any machine.
- Integration tests: use the real G1-pre cache, skip on CI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from urbangen.data.bdd100k_generative import BDD100K128Dataset

_CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "bdd100k_128"


def _make_tiny_cache(root: Path, n_per_split: dict[str, int] | None = None) -> dict:
    """Create a minimal synthetic cache and return its manifest dict."""
    n_per_split = n_per_split or {"train": 6, "val": 6, "test": 6}
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "class_names": ["city street", "highway", "residential"],
        "splits": {},
    }
    for split, n in n_per_split.items():
        labels = np.array([i % 3 for i in range(n)], dtype=np.int64)
        rng = np.random.default_rng(0)
        images = rng.integers(0, 256, size=(n, 128, 128, 3), dtype=np.uint8)
        img_path = root / f"{split}_images.npy"
        lbl_path = root / f"{split}_labels.npy"
        np.save(img_path, images)
        np.save(lbl_path, labels)
        _class_names = ["city street", "highway", "residential"]
        class_counts = {_class_names[i]: int((labels == i).sum()) for i in range(3)}
        manifest["splits"][split] = {
            "images_path": str(img_path.relative_to(root.parent)),
            "labels_path": str(lbl_path.relative_to(root.parent)),
            "n": n,
            "images_sha256": hashlib.sha256(img_path.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(lbl_path.read_bytes()).hexdigest(),
            "class_counts": class_counts,
            "read_only_downstream": split == "test",
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------- Fixture-based tests (run on CI) ----------------


def test_tiny_cache_loads(tmp_path):
    _make_tiny_cache(tmp_path / "cache")
    ds = BDD100K128Dataset(tmp_path / "cache", "train")
    assert len(ds) == 6


def test_test_split_refused_without_flag(tmp_path):
    _make_tiny_cache(tmp_path / "cache")
    with pytest.raises(PermissionError):
        BDD100K128Dataset(tmp_path / "cache", "test")


def test_test_split_loads_with_flag(tmp_path):
    _make_tiny_cache(tmp_path / "cache")
    ds = BDD100K128Dataset(tmp_path / "cache", "test", allow_test=True)
    assert len(ds) == 6


def test_item_shape_and_dtype(tmp_path):
    _make_tiny_cache(tmp_path / "cache")
    ds = BDD100K128Dataset(tmp_path / "cache", "train")
    img, label = ds[0]
    assert img.shape == (3, 128, 128)
    assert img.dtype == torch.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert isinstance(label, int)
    assert label in (0, 1, 2)


def test_writable_tensor(tmp_path):
    """The returned tensor must be writable (no memmap view)."""
    _make_tiny_cache(tmp_path / "cache")
    ds = BDD100K128Dataset(tmp_path / "cache", "train")
    img, _ = ds[0]
    img.add_(1.0)  # would fail or warn on a memmap view


def test_sha_mismatch_raises(tmp_path):
    _make_tiny_cache(tmp_path / "cache")
    manifest_path = tmp_path / "cache" / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["splits"]["train"]["images_sha256"] = "0" * 64  # corrupt
    manifest_path.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        BDD100K128Dataset(tmp_path / "cache", "train")


def test_bad_dtype_raises(tmp_path):
    _make_tiny_cache(tmp_path / "cache")
    # Overwrite images with float32, update SHA so it passes verification
    float_images = np.zeros((6, 128, 128, 3), dtype=np.float32)
    img_path = tmp_path / "cache" / "train_images.npy"
    np.save(img_path, float_images)
    manifest_path = tmp_path / "cache" / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["splits"]["train"]["images_sha256"] = hashlib.sha256(img_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="dtype"):
        BDD100K128Dataset(tmp_path / "cache", "train")


def test_windows_style_manifest_path(tmp_path):
    """The committed manifest was written on Windows (backslashes). The
    loader must resolve it correctly on any platform."""
    import json

    cache = tmp_path / "cache"
    cache.mkdir()
    # Simulate a Windows-style manifest path
    n = 3
    labels = np.array([0, 1, 2], dtype=np.int64)
    images = np.zeros((n, 128, 128, 3), dtype=np.uint8)
    np.save(cache / "train_images.npy", images)
    np.save(cache / "train_labels.npy", labels)
    manifest = {
        "class_names": ["city street", "highway", "residential"],
        "splits": {
            "train": {
                "images_path": r"data\bdd100k_128\train_images.npy",  # Windows style
                "labels_path": r"data\bdd100k_128\train_labels.npy",
                "n": n,
                "images_sha256": hashlib.sha256(
                    (cache / "train_images.npy").read_bytes()
                ).hexdigest(),
                "labels_sha256": hashlib.sha256(
                    (cache / "train_labels.npy").read_bytes()
                ).hexdigest(),
                "class_counts": {"city street": 1, "highway": 1, "residential": 1},
                "read_only_downstream": False,
            }
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    ds = BDD100K128Dataset(cache, "train")
    assert len(ds) == 3


# ---------------- Integration tests (skip if real cache missing) ----------------

# The manifest is committed (provenance record), but the .npy caches
# (~3.59 GiB) are gitignored. On CI the manifest exists but the .npy
# files do not, so the skipif must test for every cache file the loader
# actually opens — images and labels, not just the manifest.
_REQUIRED_CACHE_FILES = (
    "manifest.json",
    "train_images.npy",
    "train_labels.npy",
    "val_images.npy",
    "val_labels.npy",
    "test_images.npy",
    "test_labels.npy",
)

_cache_available = all((_CACHE_ROOT / name).is_file() for name in _REQUIRED_CACHE_FILES)

_skip_if_no_cache = pytest.mark.skipif(
    not _cache_available,
    reason=("Real BDD100K 128x128 cache files not present; run scripts/precompute_bdd100k_128.py"),
)


@_skip_if_no_cache
def test_real_train_split_loads():
    ds = BDD100K128Dataset(_CACHE_ROOT, "train")
    assert len(ds) == 68_969


@_skip_if_no_cache
def test_real_val_split_loads():
    ds = BDD100K128Dataset(_CACHE_ROOT, "val")
    assert len(ds) == 4_727


@_skip_if_no_cache
def test_real_test_split_loads_with_flag():
    ds = BDD100K128Dataset(_CACHE_ROOT, "test", allow_test=True)
    assert len(ds) == 4_745
