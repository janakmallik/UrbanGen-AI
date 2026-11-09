import json
from pathlib import Path

DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")
TRAIN_LABEL = DATA_ROOT / "labels" / "bdd100k_labels_images_train.json"
IMAGES_DIR = DATA_ROOT / "images" / "100k" / "train"

with TRAIN_LABEL.open("r", encoding="utf-8") as f:
    data = json.load(f)

counts = {
    "daytime": {"total": 0, "exists": 0},
    "night": {"total": 0, "exists": 0},
}
missing_examples = []

for rec in data:
    attrs = rec.get("attributes", {})
    if attrs.get("scene") != "residential":
        continue

    tod = attrs.get("timeofday")
    if tod not in counts:
        continue

    counts[tod]["total"] += 1
    path = IMAGES_DIR / rec["name"]

    if path.exists():
        counts[tod]["exists"] += 1
    elif len(missing_examples) < 10:
        missing_examples.append(str(path))

for name, c in counts.items():
    print(
        f"{name}: {c['total']:,} labeled, {c['exists']:,} on disk, "
        f"{c['total'] - c['exists']:,} missing"
    )

print("\nMissing examples:")
for path in missing_examples:
    print(path)
