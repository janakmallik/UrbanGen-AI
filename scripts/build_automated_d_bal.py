import csv
import json
import random
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_OUT = _REPO_ROOT / "outputs" / "method_2_synthetic"
RESIDENTIAL_MANIFEST = BASE_OUT / "curated_manifest.json"
CITY_LOG = BASE_OUT / "shortcut_control" / "city" / "candidates" / "filter_log.csv"
HWY_LOG = BASE_OUT / "shortcut_control" / "highway" / "candidates" / "filter_log.csv"

DOSE = 138


def load_kept_from_csv(csv_path):
    kept = []
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["kept"] == "1"]
        rows.sort(key=lambda r: r["output"])
        kept = [r["output"] for r in rows]
    return kept


def main():
    # 1. Load all automated kept images
    city_kept = load_kept_from_csv(CITY_LOG)
    hwy_kept = load_kept_from_csv(HWY_LOG)

    res_manifest = json.loads(RESIDENTIAL_MANIFEST.read_text())
    res_accepted = load_kept_from_csv(BASE_OUT / "filter_log.csv")
    # final_keep_labels are 1-based indices into the accepted list
    res_kept = [res_accepted[i - 1] for i in res_manifest["final_keep_labels"]]

    print("Available kept images (automated):")
    print(f"  Residential: {len(res_kept)}")
    print(f"  City: {len(city_kept)}")
    print(f"  Highway: {len(hwy_kept)}")

    if len(city_kept) < DOSE or len(hwy_kept) < DOSE:
        print("Not enough images to hit target dose.")

    # 2. Randomly sample DOSE from each
    rng = random.Random(42)

    # D-bal: 138 residential + 69 city + 69 highway = 276 total (dose-matched to Arm D)
    # D-138: 138 residential only
    selected_res = rng.sample(res_kept, 138)
    selected_city = rng.sample(city_kept, 69)
    selected_hwy = rng.sample(hwy_kept, 69)

    # 3. Build D-bal and D-138 manifests
    d_bal_manifest = {
        "dose_residential": 138,
        "dose_city": 69,
        "dose_highway": 69,
        "total_images": 276,
        "residential": selected_res,
        "city": selected_city,
        "highway": selected_hwy,
    }

    d_138_manifest = {"dose": 138, "residential": selected_res}

    out_dir = BASE_OUT / "shortcut_control"
    (out_dir / "d_bal_manifest.json").write_text(json.dumps(d_bal_manifest, indent=2))
    (out_dir / "d_138_manifest.json").write_text(json.dumps(d_138_manifest, indent=2))

    print("\nSaved D-bal manifest: 138 res + 69 city + 69 hwy = 276 total")
    print("Saved D-138 manifest: 138 res")


if __name__ == "__main__":
    main()
