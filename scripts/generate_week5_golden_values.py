"""One-off: generate golden values for test_week5_pipeline.py.

Run once locally, then paste the output into GOLDEN_VALUES in
tests/unit/test_week5_pipeline.py. Not committed (it's a generator).

Run:
    uv run python scripts/generate_week5_golden_values.py

Output: prints Python-literal lists suitable for pasting.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env", override=False)

from urbangen.data.preprocessing import (  # noqa: E402
    WEEK5_CLASS_TO_ID,
    week5_eval_transform,
)

_URBANMIND = Path(os.environ["URBANMIND_REPO"])
_DATA_ROOT = Path(os.environ["URBANINTEL_DATA_ROOT"])
_WEEK5_CKPT = _URBANMIND / "bdd_resnet50_best.pt"
_SPLIT = _URBANMIND / "data" / "splits" / "resnet_test_clean.json"
_LABEL = _DATA_ROOT / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"
_IMAGES = _DATA_ROOT / "bdd100k" / "images"


def main() -> int:
    print("Loading labels...")
    label_data = json.loads(_LABEL.read_text(encoding="utf-8"))
    name_to_class: dict[str, int] = {}
    for rec in label_data:
        scene = rec.get("attributes", {}).get("scene")
        if scene in WEEK5_CLASS_TO_ID:
            name_to_class[rec["name"]] = WEEK5_CLASS_TO_ID[scene]

    print("Loading clean test split...")
    clean = json.loads(_SPLIT.read_text(encoding="utf-8"))

    print("Indexing images...")
    image_index: dict[str, Path] = {}
    for p in _IMAGES.rglob("*.jpg"):
        image_index.setdefault(p.name, p)

    # Pick 4 city / 3 highway / 3 residential (deterministic, sorted)
    by_class: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for name in clean:
        if name in name_to_class and name in image_index:
            by_class[name_to_class[name]].append(name)
    for c in by_class:
        by_class[c].sort()
    chosen = by_class[0][:4] + by_class[1][:3] + by_class[2][:3]

    print(
        f"Chosen {len(chosen)} files "
        f"({len(by_class[0][:4])}/{len(by_class[1][:3])}/{len(by_class[2][:3])})"
    )

    print("Loading Week 5...")
    model = models.resnet50(num_classes=3)
    ckpt = torch.load(_WEEK5_CKPT, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()

    print("Predicting...")
    predictions: list[int] = []
    with torch.no_grad():
        for name in chosen:
            with Image.open(image_index[name]) as img:
                x = week5_eval_transform()(img.convert("RGB")).unsqueeze(0)
            pred = int(model(x).argmax(dim=1).item())
            predictions.append(pred)
            print(f"  {name} → class {pred} (true {name_to_class[name]})")

    correct = sum(1 for i, n in enumerate(chosen) if predictions[i] == name_to_class[n])
    acc = correct / len(chosen)

    print("\n--- Paste into GOLDEN_VALUES ---")
    print('"filenames": [')
    for n in chosen:
        print(f'    "{n}",')
    print("],")
    print('"expected_class_ids": [')
    for p in predictions:
        print(f"    {p},")
    print("],")
    print(f'"expected_accuracy": {acc:.1f},')

    return 0


if __name__ == "__main__":
    sys.exit(main())
