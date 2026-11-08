import json
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")
TRAIN_LABEL = DATA_ROOT / "labels" / "bdd100k_labels_images_train.json"
SEG_IMAGES = DATA_ROOT / "seg" / "images" / "train"
SEG_LABELS = DATA_ROOT / "seg" / "labels" / "train"

# Load labels
with TRAIN_LABEL.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Get night residential images
night_res = [
    rec["name"]
    for rec in data
    if rec.get("attributes", {}).get("scene") == "residential"
    and rec.get("attributes", {}).get("timeofday") == "night"
]
print(f"Night residential images in train split: {len(night_res):,}")

# Index segmentation image stems
seg_image_stems = {p.stem for p in SEG_IMAGES.glob("*") if p.is_file()}
seg_label_stems = {p.stem for p in SEG_LABELS.glob("*") if p.is_file()}
print(f"Segmentation images in train: {len(seg_image_stems):,}")
print(f"Segmentation labels in train: {len(seg_label_stems):,}")

# Check overlap
night_res_stems = {Path(n).stem for n in night_res}
matched_images = night_res_stems & seg_image_stems
matched_labels = night_res_stems & seg_label_stems

print(f"\n--- Pairing ---")
print(f"Night residential images WITH segmentation image: {len(matched_images):,}")
print(f"Night residential images WITH segmentation label: {len(matched_labels):,}")

if len(matched_labels) >= 500:
    print(
        f"\n✅ Method 3 FEASIBLE: {len(matched_labels)} night residential images have matching seg labels"
    )
elif len(matched_labels) >= 100:
    print(
        f"\n⚠️ Method 3 PARTIAL: Only {len(matched_labels)} night residential images have seg labels (target: 500)"
    )
else:
    print(f"\n❌ Method 3 INFEASIBLE: Only {len(matched_labels)} matched pairs")
