import json
from collections import Counter
from pathlib import Path

# Paths
DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")
TRAIN_LABEL = DATA_ROOT / "labels" / "bdd100k_labels_images_train.json"


def main():
    print(f"Reading {TRAIN_LABEL}...")
    if not TRAIN_LABEL.exists():
        print(f"❌ Error: Label file not found at {TRAIN_LABEL}")
        return

    with TRAIN_LABEL.open("r", encoding="utf-8") as f:
        data = json.load(f)

    residential_counts = Counter()
    all_tod_counts = Counter()
    filename_prefixes = set()

    for rec in data:
        attrs = rec.get("attributes", {})
        scene = attrs.get("scene", "unknown")
        tod = attrs.get("timeofday", "unknown")

        all_tod_counts[tod] += 1

        if scene == "residential":
            residential_counts[tod] += 1

        # Extract prefix (not claiming this is a verified video ID yet)
        name = rec.get("name", "")
        if "-" in name:
            filename_prefixes.add(name.split("-")[0])

    print("\n--- Data Audit: BDD100K Train Split (Images Only) ---")
    print(f"Total images: {len(data):,}")
    print(f"Unique filename prefixes: {len(filename_prefixes):,}")

    print("\nTime of Day Distribution (All Scenes):")
    for tod, count in all_tod_counts.most_common():
        print(f"  {tod}: {count:,}")

    print("\nResidential Scene Distribution:")
    total_res = sum(residential_counts.values())
    print(f"  Total residential images: {total_res:,}")
    for tod, count in residential_counts.most_common():
        print(f"    {tod}: {count:,}")

    print("\n--- Feasibility Check (Images Only) ---")
    day_res = residential_counts.get("daytime", 0)
    night_res = residential_counts.get("night", 0)

    if day_res >= 500:
        print(
            f"✅ Method 1/2 (Day->Night): We have {day_res} daytime residential images. (Target: 500) -> FEASIBLE"
        )
    else:
        print(
            f"⚠️ Method 1/2 (Day->Night): Only {day_res} daytime residential images. (Target: 500) -> LIMITED"
        )

    if night_res >= 500:
        print(
            f"✅ Method 3 (Pix2Pix) IMAGE DATA: We have {night_res} nighttime residential images."
        )
        print("   ⚠️ NOTE: This does NOT confirm segmentation masks exist for these images.")
        print("   A separate segmentation-mask audit is required before approving Method 3.")
    else:
        print(
            f"⚠️ Method 3 (Pix2Pix) IMAGE DATA: Only {night_res} nighttime residential images. (Target: 500) -> LIMITED"
        )


if __name__ == "__main__":
    main()
