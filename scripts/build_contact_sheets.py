"""Build contact sheets for manual curation of 282 accepted synthetic images.

Layout: 15 pages, 20 pairs per page (4 cols × 5 rows).
Each pair: source (day, 160×90) on left, generated (night, 160×90) on right.
Labels: index + generated filename above each pair, "SOURCE" / "GENERATED" above halves.

Validates CSV integrity, cleans stale pages, fails on missing images or duplicates.

Output: outputs/method_2_synthetic/contact_sheets/page_XX.jpg
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

THUMB_W = 160
THUMB_H = 90
PER_PAGE = 20  # 4 cols × 5 rows
COLS = 4
ROWS = 5
HEADER = 30

PAIR_W = THUMB_W * 2 + 2
PAIR_H = THUMB_H + HEADER

OUT_DIR = Path("outputs/method_2_synthetic/contact_sheets")
GEN_ROOT = Path("outputs/method_2_synthetic")
FILTER_LOG = GEN_ROOT / "filter_log.csv"
SOURCE_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k\images\100k\train")

# --- Safety: only clear files inside expected parent ---
EXPECTED_PARENT = Path("outputs/method_2_synthetic")
if OUT_DIR.parent != EXPECTED_PARENT:
    sys.exit(f"Refusing to operate: OUT_DIR.parent = {OUT_DIR.parent}")

# --- CSV integrity check ---
if not FILTER_LOG.exists():
    sys.exit(f"ERROR: {FILTER_LOG} not found")

with open(FILTER_LOG) as f:
    reader = csv.DictReader(f)
    required_cols = {"kept", "source", "output"}
    if not required_cols.issubset(reader.fieldnames or []):
        sys.exit(
            f"ERROR: filter_log.csv missing required columns. "
            f"Found: {reader.fieldnames}. Need: {required_cols}"
        )
    log = list(reader)

accepted = sorted(
    [r for r in log if r["kept"] == "1"],
    key=lambda r: r["output"],
)
print(f"Total accepted: {len(accepted)}")
if len(accepted) == 0:
    sys.exit("ERROR: no accepted images in filter log")
if len(accepted) != 282:
    print(f"⚠️  WARNING: expected 282 accepted images, found {len(accepted)}")

outputs = [r["output"] for r in accepted]
if len(set(outputs)) != len(outputs):
    sys.exit(f"ERROR: {len(outputs) - len(set(outputs))} duplicate generated filenames")

# Safety: generated filenames must be plain basenames (no path traversal)
if any(not name or Path(name).name != name or "/" in name or "\\" in name for name in outputs):
    sys.exit("ERROR: generated filenames must be plain basenames")

# Safety: source filenames must also be plain basenames
source_names_for_check = [r["source"] for r in accepted]
if any(
    not name or Path(name).name != name or "/" in name or "\\" in name
    for name in source_names_for_check
):
    sys.exit("ERROR: source filenames must be plain basenames")

pairs = [(r["source"], r["output"]) for r in accepted]
if len(set(pairs)) != len(pairs):
    sys.exit("ERROR: duplicate source→output mappings")

# Verify 282 distinct source images (not repeated samples)
sources = [r["source"] for r in accepted]
if len(set(sources)) != len(sources):
    sys.exit(f"ERROR: {len(sources) - len(set(sources))} duplicate source images in accepted set")

print("CSV integrity check: PASSED")

# --- Build source index, fail on duplicates ---
source_index = {}
duplicates = []
for p in SOURCE_ROOT.rglob("*.jpg"):
    if p.name in source_index:
        duplicates.append(p.name)
    else:
        source_index[p.name] = p
print(f"Source index: {len(source_index)} unique files")

if duplicates:
    print(f"\n❌ DUPLICATE source filenames ({len(duplicates)}):")
    for d in duplicates[:10]:
        print(f"  {d}")
    sys.exit(f"Duplicate basenames: {len(duplicates)}. Cannot build reliable contact sheets.")

# --- Pre-flight: verify all images present ---
missing_sources = []
missing_generated = []
for src_name, gen_name in pairs:
    src = source_index.get(src_name)
    if not src or not src.exists():
        missing_sources.append(src_name)
    gen = GEN_ROOT / gen_name
    if not gen.exists():
        missing_generated.append(gen_name)

if missing_sources or missing_generated:
    print("\n❌ Missing files:")
    if missing_sources:
        print(f"  Missing sources ({len(missing_sources)}):")
        for s in missing_sources[:10]:
            print(f"    {s}")
    if missing_generated:
        print(f"  Missing generated ({len(missing_generated)}):")
        for g in missing_generated[:10]:
            print(f"    {g}")
    sys.exit("Fix missing files before curation.")

print("Pre-flight check: all images present")

# --- Clean stale pages ---
if OUT_DIR.exists():
    stale = list(OUT_DIR.glob("page_*.jpg"))
    if stale:
        print(f"\nRemoving {len(stale)} stale contact sheet files")
        for f in stale:
            f.unlink()

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Build pages ---
render_failures = []
page_num = 1
for start in range(0, len(accepted), PER_PAGE):
    batch = accepted[start : start + PER_PAGE]

    page_w = COLS * PAIR_W + (COLS + 1) * 5
    page_h = ROWS * PAIR_H + (ROWS + 1) * 5 + 30  # +30 for page title
    page = Image.new("RGB", (page_w, page_h), (30, 30, 30))
    draw = ImageDraw.Draw(page)

    draw.text(
        (10, 8),
        f"Page {page_num} — Images {start + 1}–{start + len(batch)} of {len(accepted)}",
        fill=(255, 255, 255),
    )

    for i, r in enumerate(batch):
        row = i // COLS
        col = i % COLS
        x = col * (PAIR_W + 5) + 5
        y = row * (PAIR_H + 5) + HEADER + 5 + 30

        idx = start + i + 1
        src_name = r["source"]
        gen_name = r["output"]

        # Filename label above pair
        draw.text((x, y - 30), f"#{idx} | {gen_name}", fill=(255, 255, 255))
        # SOURCE / GENERATED labels
        draw.text((x, y - 15), "SOURCE", fill=(255, 200, 100))
        draw.text((x + THUMB_W + 2, y - 15), "GENERATED", fill=(100, 200, 255))

        # Source (left half) — letterboxed to preserve full scene
        src_path = source_index.get(src_name)
        if src_path and src_path.exists():
            try:
                with Image.open(src_path) as img:
                    src = ImageOps.contain(
                        img.convert("RGB"),
                        (THUMB_W, THUMB_H),
                        method=Image.LANCZOS,
                    )
                # Center in the 160x90 slot (letterbox)
                src_x = x + (THUMB_W - src.width) // 2
                src_y = y + (THUMB_H - src.height) // 2
                page.paste(src, (src_x, src_y))
            except Exception as e:
                render_failures.append(f"source {src_name}: {e}")

        # Generated (right half) — letterboxed
        gen_path = GEN_ROOT / gen_name
        if gen_path.exists():
            try:
                with Image.open(gen_path) as img:
                    gen = ImageOps.contain(
                        img.convert("RGB"),
                        (THUMB_W, THUMB_H),
                        method=Image.LANCZOS,
                    )
                gen_x = x + THUMB_W + 2 + (THUMB_W - gen.width) // 2
                gen_y = y + (THUMB_H - gen.height) // 2
                page.paste(gen, (gen_x, gen_y))
            except Exception as e:
                render_failures.append(f"generated {gen_name}: {e}")

    out_file = OUT_DIR / f"page_{page_num:02d}.jpg"
    page.save(out_file, quality=90)
    print(f"Saved {out_file.name} ({len(batch)} pairs)")
    page_num += 1

print(f"\n✅ {page_num - 1} contact sheets saved to {OUT_DIR}")
if render_failures:
    print(f"\n❌ {len(render_failures)} rendering failures:")
    for f in render_failures[:10]:
        print(f"  {f}")
    if len(render_failures) > 10:
        print(f"  ... and {len(render_failures) - 10} more")
    sys.exit(
        f"Contact sheets incomplete: {len(render_failures)} pairs failed to render. "
        f"Fix data before curation."
    )
print(f"\n✅ All {len(accepted)} pairs rendered without errors")
print("\nNext steps:")
print(f"  1. Open {OUT_DIR}")
print("  2. For each page, record: reject=[indices], uncertain=[indices]")
print("  3. Paste the per-page labels")
print("  4. Run verify_source_split_membership.py")
