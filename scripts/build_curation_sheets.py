"""Build contact sheets of ACCEPTED city + highway candidates for manual curation.

Corrections applied:
1. Uses 'candidates' directory (not candidates_full)
2. Derives accepted filenames from filter_log.csv (verified source of truth)
3. Saves index-to-filename manifest alongside sheets
4. Validates expected counts (235 city, 162 highway)
5. Fails fast on missing columns, duplicates, or corrupted images
6. Sorts filenames deterministically for stable indices
7. Records unreadable images (skips them, but logs)
"""

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_OUT = _REPO_ROOT / "outputs" / "method_2_synthetic" / "shortcut_control"

EXPECTED_ACCEPTED = {"city": 235, "highway": 162}
REQUIRED_COLUMNS = {"kept", "source", "output", "output_sha256"}

THUMB_W, THUMB_H = 256, 144
COLS, ROWS = 5, 5
PADDING = 6
HEADER = 22
PER_PAGE = COLS * ROWS


def load_accepted(csv_path: Path, expected: int) -> list[dict]:
    """Load accepted rows from filter_log.csv. Fails fast on any issue."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise RuntimeError(
                f"{csv_path.name} missing required columns. "
                f"Found: {reader.fieldnames}. Need: {REQUIRED_COLUMNS}"
            )
        rows = list(reader)

    accepted = [r for r in rows if r["kept"] == "1"]

    # Deterministic sort by filename
    accepted.sort(key=lambda r: r["output"])

    # Validate count
    if len(accepted) != expected:
        raise RuntimeError(
            f"{csv_path.parent.parent.name}: expected {expected} accepted, got {len(accepted)}"
        )

    # Check for duplicates
    names = [r["output"] for r in accepted]
    if len(set(names)) != len(names):
        raise RuntimeError(f"Duplicate filenames in {csv_path}")

    return accepted


def build_class_sheets(scene: str):
    csv_path = BASE_OUT / scene / "candidates" / "filter_log.csv"
    accepted = load_accepted(csv_path, EXPECTED_ACCEPTED[scene])

    candidates_dir = BASE_OUT / scene / "candidates"
    out_dir = BASE_OUT / scene / "curation_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build index manifest
    index_manifest = {
        "scene": scene,
        "total_accepted": len(accepted),
        "csv_source": str(csv_path),
        "images": [],
    }

    page_w = COLS * (THUMB_W + PADDING) + PADDING
    page_h = ROWS * (THUMB_H + HEADER + PADDING) + PADDING + 30

    unreadable = []

    for page_idx in range(0, len(accepted), PER_PAGE):
        batch = accepted[page_idx : page_idx + PER_PAGE]
        page_num = page_idx // PER_PAGE + 1

        canvas = Image.new("RGB", (page_w, page_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 8),
            f"{scene.upper()} — Page {page_num} — "
            f"Images {page_idx + 1}–{page_idx + len(batch)} of {len(accepted)}",
            fill=(255, 255, 255),
        )

        for i, row in enumerate(batch):
            row_pos = i // COLS
            col = i % COLS
            x = PADDING + col * (THUMB_W + PADDING)
            y = 30 + PADDING + row_pos * (THUMB_H + HEADER + PADDING)
            idx = page_idx + i + 1
            fname = row["output"]

            # Record in index manifest
            index_manifest["images"].append(
                {
                    "index": idx,
                    "filename": fname,
                    "sha256": row["output_sha256"],
                    "source": row["source"],
                }
            )

            # Label above thumb
            draw.text((x, y), f"#{idx}: {fname[:40]}", fill=(255, 220, 100))

            img_path = candidates_dir / fname
            if not img_path.exists():
                unreadable.append((idx, fname, "missing"))
                continue

            actual_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()
            if actual_hash.lower() != row["output_sha256"].lower():
                raise RuntimeError(
                    f"Hash mismatch for index {idx}: {img_path}\n"
                    f"  CSV: {row['output_sha256']}\n"
                    f"  Disk: {actual_hash}"
                )

            try:
                with Image.open(img_path) as img:
                    thumb = img.convert("RGB").resize((THUMB_W, THUMB_H), Image.LANCZOS)
                canvas.paste(thumb, (x, y + HEADER - 8))
            except Exception as e:
                unreadable.append((idx, fname, str(e)))

        out_path = out_dir / f"page_{page_num:02d}.jpg"
        canvas.save(out_path, quality=88)
        print(f"Saved {out_path}")

    # Save index manifest
    manifest_path = out_dir / "index_manifest.json"
    manifest_path.write_text(json.dumps(index_manifest, indent=2))

    if unreadable:
        raise RuntimeError(
            f"{scene}: {len(unreadable)} unreadable images. "
            f"Resolve before manual curation. First: {unreadable[0]}"
        )
    print(f"\n✅ All {len(accepted)} {scene} images readable")

    n_pages = (len(accepted) + PER_PAGE - 1) // PER_PAGE
    print(f"\n{scene}: {len(accepted)} images → {n_pages} pages")
    print(f"Index manifest: {manifest_path}")
    print(f"Open: {out_dir}")


def main():
    for scene in ("city", "highway"):
        build_class_sheets(scene)


if __name__ == "__main__":
    main()
