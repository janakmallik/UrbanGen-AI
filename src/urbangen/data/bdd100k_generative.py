"""BDD100K 128x128 generative dataset.

Loads from the memory-mapped cache produced by
scripts/precompute_bdd100k_128.py. The cache is uint8 memmap; this
Dataset converts to float32 in [0, 1] and returns (image, class_label).

Refuses to load the test split unless explicitly overridden — test is
read_only_downstream in the cache manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from urbangen.config import data_root  # noqa: F401  (unused here but keeps import symmetry)


def _resolve_cache_path(cache_root: Path, stored: str) -> Path:
    """Resolve a manifest-stored path across platforms.

    The manifest may store Windows-style paths (backslashes) or POSIX-style
    paths (forward slashes), depending on where it was committed. Normalize
    to forward slashes, take the basename, join to cache_root.
    """
    name = stored.replace("\\", "/").split("/")[-1]
    return cache_root / name


class BDD100K128Dataset(Dataset):
    """One (image, label) per item. Image: float32 (3, 128, 128) in [0, 1]."""

    SPLITS = ("train", "val", "test")

    def __init__(self, cache_root: Path, split: str, allow_test: bool = False):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")
        if split == "test" and not allow_test:
            raise PermissionError(
                "The test split is read_only_downstream. Pass allow_test=True "
                "only from the final evaluation script, and only after the "
                "experiment protocol is frozen in docs/experiments.md."
            )

        self.cache_root = Path(cache_root)
        manifest_path = self.cache_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cache manifest not found at {manifest_path}. "
                f"Run scripts/precompute_bdd100k_128.py first."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest = manifest
        info = manifest["splits"][split]

        self.images_path = _resolve_cache_path(self.cache_root, info["images_path"])
        self.labels_path = _resolve_cache_path(self.cache_root, info["labels_path"])

        # Verify cache files exist and match their recorded SHA
        self._verify(self.images_path, info["images_sha256"], "images")
        self._verify(self.labels_path, info["labels_sha256"], "labels")

        # Load memmap (read-only)
        self.images = np.load(self.images_path, mmap_mode="r")
        self.labels = np.load(self.labels_path, mmap_mode="r")
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"{split}: images ({len(self.images)}) and labels ({len(self.labels)}) disagree"
            )

        # After loading memmaps, validate shapes
        expected_shape = (128, 128, 3)
        if self.images.ndim != 4 or self.images.shape[1:] != expected_shape:
            raise ValueError(
                f"{split}: images shape {self.images.shape} does not match "
                f"(N, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]})"
            )
        if self.images.dtype != np.uint8:
            raise ValueError(f"{split}: images dtype {self.images.dtype}, expected uint8")
        if self.labels.dtype != np.int64:
            raise ValueError(f"{split}: labels dtype {self.labels.dtype}, expected int64")
        if not np.array_equal(np.unique(self.labels), [0, 1, 2]):
            raise ValueError(f"{split}: label values not {0, 1, 2}")

        # Validate class counts against manifest.
        # Manifest stores class_counts keyed by class NAME ('city street', ...);
        # actual counts are keyed by class ID (0, 1, 2). Convert manifest keys
        # to IDs using the manifest's class_names ordering.
        class_names = manifest["class_names"]
        expected_counts = {i: int(info["class_counts"][name]) for i, name in enumerate(class_names)}
        actual_counts = {i: int((self.labels == i).sum()) for i in range(len(class_names))}
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"{split}: class counts {actual_counts} do not match manifest {expected_counts}"
            )

    @staticmethod
    def _verify(path: Path, expected_sha: str, what: str) -> None:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected_sha:
            raise RuntimeError(
                f"Cache {what} SHA-256 mismatch at {path}: "
                f"got {actual[:16]}..., expected {expected_sha[:16]}..."
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = np.array(self.images[idx], copy=True)  # (128, 128, 3) uint8
        label = int(self.labels[idx])
        # (H, W, C) uint8 [0, 255] -> (C, H, W) float32 [0, 1]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, label
