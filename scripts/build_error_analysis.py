"""Build a contact sheet of Arm A vs Arm D nighttime errors."""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()
_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "outputs" / "week5_arms"
_DATA_ROOT_ENV = os.environ.get("URBANINTEL_DATA_ROOT")
if not _DATA_ROOT_ENV:
    raise RuntimeError("URBANINTEL_DATA_ROOT environment variable is not set. See .env.example")
_DATA_ROOT = Path(_DATA_ROOT_ENV) / "bdd100k"
_VAL_IMAGES = _DATA_ROOT / "images" / "100k" / "val"
_URBANMIND_ENV = os.environ.get("URBANMIND_REPO")
if not _URBANMIND_ENV:
    raise RuntimeError("URBANMIND_REPO environment variable is not set. See .env.example")
_URBANMIND = Path(_URBANMIND_ENV)
_CONTACT_DIR = _REPO_ROOT / "outputs" / "error_analysis"
_CONTACT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    a = json.loads((_OUT_DIR / "results_A_seed42.json").read_text())
    d = json.loads((_OUT_DIR / "results_D_seed42.json").read_text())

    labels = np.array(a["night"]["labels"])
    preds_a = np.array(a["night"]["predictions"])
    preds_d = np.array(d["night"]["predictions"])

    # We need the filenames of the night test set in order.
    val_records = json.loads((_DATA_ROOT / "labels" / "bdd100k_labels_images_val.json").read_text())
    test_clean = set(
        json.loads((_URBANMIND / "data" / "splits" / "resnet_test_clean.json").read_text())
    )

    night_names = []
    for rec in val_records:
        if rec["name"] in test_clean and rec["attributes"].get("timeofday") == "night":
            night_names.append(rec["name"])

    if len(night_names) != len(labels):
        raise RuntimeError(f"Label mismatch: {len(night_names)} vs {len(labels)}")

    # Verify that Arm A and D have identical labels and ordering
    c = json.loads((_OUT_DIR / "results_C_seed42.json").read_text())
    if a["night"]["labels"] != d["night"]["labels"]:
        raise RuntimeError("Arm A and Arm D night labels differ. Cannot compare.")
    if a["night"]["labels"] != c["night"]["labels"]:
        raise RuntimeError("Arm A and Arm C night labels differ. Cannot compare.")
    print("✅ Verified: Arm A, C, D have identical night labels and ordering.")

    # Find cases where A failed and D succeeded (True label is 2: residential)
    win_indices = [
        i for i in range(len(labels)) if labels[i] == 2 and preds_a[i] != 2 and preds_d[i] == 2
    ]
    fail_indices = [
        i for i in range(len(labels)) if labels[i] == 2 and preds_a[i] == 2 and preds_d[i] != 2
    ]

    import csv

    csv_path = _CONTACT_DIR / "night_errors.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ground_truth", "arm_a_pred", "arm_d_pred"])
        for idx in win_indices + fail_indices:
            writer.writerow([night_names[idx], labels[idx], preds_a[idx], preds_d[idx]])
    print(f"Saved machine-readable CSV: {csv_path}")
    print(f"Nighttime Residential Wins (A failed, D succeeded): {len(win_indices)}")
    print(f"Nighttime Residential Losses (A succeeded, D failed): {len(fail_indices)}")

    # Render contact sheet
    w, h = 256, 144
    cols = 4
    rows = 3
    canvas = Image.new("RGB", (cols * w + 20, rows * h + 40), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Nighttime Residential: A fails, D succeeds", fill=(255, 255, 255))

    image_index = {p.name: p for p in _VAL_IMAGES.rglob("*.jpg")}

    for i, idx in enumerate(win_indices[: cols * rows]):
        row = i // cols
        col = i % cols
        name = night_names[idx]
        path = image_index.get(name)
        if path:
            img = Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)
            canvas.paste(img, (col * w + 10, row * h + 40))
            draw.text(
                (col * w + 10, row * h + 25),
                f"A={preds_a[idx]}, D={preds_d[idx]}",
                fill=(255, 100, 100),
            )

    out_path = _CONTACT_DIR / "night_wins.jpg"
    canvas.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
