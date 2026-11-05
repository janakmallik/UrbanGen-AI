"""Verify class_id -> class_name mapping in the cache."""

import json
from pathlib import Path

import numpy as np

cache = Path("data/bdd100k_128")
labels = np.load(cache / "train_labels.npy")
manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))

print("manifest class_names:", manifest["class_names"])
print()
for i, name in enumerate(manifest["class_names"]):
    count = int((labels == i).sum())
    print(f"  class_id {i}: {name:12s} -> {count:,} train images")

# Also verify against val and test
for split in ("val", "test"):
    labels_split = np.load(cache / f"{split}_labels.npy")
    print(f"\n{split} split:")
    for i, name in enumerate(manifest["class_names"]):
        count = int((labels_split == i).sum())
        print(f"  class_id {i}: {name:12s} -> {count:,} images")
