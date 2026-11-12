"""Recover daytime predictions for Arm C and Arm D to enable paired bootstrap."""

import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from urbangen.data.preprocessing import WEEK5_CLASS_TO_ID, week5_eval_transform  # noqa: E402

load_dotenv()
DATA_ROOT = Path(os.environ["URBANINTEL_DATA_ROOT"]) / "bdd100k"
URBANMIND = Path(os.environ["URBANMIND_REPO"])
_OUT_DIR = _REPO_ROOT / "outputs" / "week5_arms"


class ClassificationDataset(Dataset):
    def __init__(self, records, transform):
        self.samples = [(r["path"], r["label"]) for r in records]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        return x, label


def load_records(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels and build day subset
    val_records = load_records(DATA_ROOT / "labels" / "bdd100k_labels_images_val.json")
    test_clean_names = set(
        json.loads((URBANMIND / "data" / "splits" / "resnet_test_clean.json").read_text())
    )

    image_index = {}
    for p in (DATA_ROOT / "images").rglob("*.jpg"):
        image_index.setdefault(p.name, p)

    day_records = []
    for rec in val_records:
        attrs = rec.get("attributes", {})
        if rec["name"] in test_clean_names and attrs.get("timeofday") == "daytime":
            path = image_index.get(rec["name"])
            if path:
                day_records.append(
                    {"path": path, "label": WEEK5_CLASS_TO_ID[attrs["scene"]], "name": rec["name"]}
                )

    print(f"Day subset size: {len(day_records)}")
    loader = DataLoader(
        ClassificationDataset(day_records, week5_eval_transform()), batch_size=64, shuffle=False
    )

    # Evaluate both checkpoints
    for arm in ["C", "D"]:
        ckpt_path = _OUT_DIR / f"checkpoint_{arm}_seed42.pt"
        if not ckpt_path.exists():
            print(f"Missing checkpoint: {ckpt_path}")
            continue

        print(f"Evaluating Arm {arm}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(2048, 3)
        model.load_state_dict(ckpt["best_state_dict"])
        model.to(device).eval()

        preds, labels, names = [], [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(device)
                preds.extend(model(x).argmax(dim=1).cpu().numpy())
                labels.extend(y.numpy())
                # Store names in order
                names.extend([r["name"] for r in day_records[i * 64 : (i + 1) * 64]])

        # Save to a new json
        out = {
            "arm": arm,
            "seed": 42,
            "day": {"labels": labels, "predictions": preds, "names": names},
        }
        out_path = _OUT_DIR / f"daytime_preds_{arm}_seed42.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
