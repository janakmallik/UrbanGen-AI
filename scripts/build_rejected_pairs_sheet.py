"""Build contact sheet comparing rejected source/generated pairs.

For the 10 pilot rows rejected by the paved_surface filter, show source
on left, generated on right, with the paved-frac delta labeled.
"""

import csv
from pathlib import Path

from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_OUT = _REPO_ROOT / "outputs" / "method_2_synthetic" / "shortcut_control"
DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")

# Rows to inspect: (scene, row_index) — these are pilot rows rejected by paved_surface
TARGETS = [
    ("city", 2),
    ("city", 3),
    ("city", 8),
    ("city", 16),
    ("city", 17),
    ("highway", 3),
    ("highway", 4),
    ("highway", 8),
    ("highway", 10),
    ("highway", 20),
]

THUMB_W, THUMB_H = 320, 180
PADDING = 8
HEADER = 30
OUT = BASE_OUT / "rejected_pairs_review.jpg"


def main():
    # Build image index once (recursively)
    image_index = {}
    for p in (DATA_ROOT / "images" / "100k" / "train").rglob("*.jpg"):
        image_index[p.name] = p
    print(f"Built index of {len(image_index)} images")
    rows = len(TARGETS)
    canvas_w = THUMB_W * 2 + PADDING * 3
    canvas_h = (THUMB_H + HEADER) * rows + PADDING * (rows + 1) + 40
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Rejected pairs: SOURCE (left) vs GENERATED (right)", fill=(255, 255, 255))

    for i, (scene, idx) in enumerate(TARGETS):
        candidates_dir = BASE_OUT / scene / "candidates_pilot"
        log_path = candidates_dir / "filter_log.csv"

        with open(log_path) as f:
            rows_data = list(csv.DictReader(f))

        row = next((r for r in rows_data if int(r["index"]) == idx), None)
        if row is None:
            print(f"Missing row {idx} in {scene}")
            continue

        src_name = row["source"]
        gen_name = row["output"]
        src_path = image_index.get(src_name)
        gen_path = candidates_dir / gen_name

        y = PADDING + 40 + i * (THUMB_H + HEADER)
        delta = float(row["paved_frac_fake"]) - float(row["paved_frac_orig"])
        label = f"{scene} row {idx} | orig={row['paved_frac_orig']} fake={row['paved_frac_fake']} Δ={delta:+.3f}"
        draw.text((PADDING, y - 20), label, fill=(255, 220, 100))

        if src_path is not None and src_path.exists():
            with Image.open(src_path) as img:
                thumb = img.convert("RGB").resize((THUMB_W, THUMB_H), Image.LANCZOS)
            canvas.paste(thumb, (PADDING, y))
        else:
            print(f"Missing source: {src_path}")

        if gen_path.exists():
            with Image.open(gen_path) as img:
                thumb = img.convert("RGB").resize((THUMB_W, THUMB_H), Image.LANCZOS)
            canvas.paste(thumb, (PADDING * 2 + THUMB_W, y))
        else:
            print(f"Missing generated: {gen_path}")

    canvas.save(OUT, quality=92)
    print(f"\nSaved: {OUT}")
    print(f"Open with: Invoke-Item {OUT}")


if __name__ == "__main__":
    main()
