"""Build a contact sheet of the 30 Phase 3 pilot outputs."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "outputs" / "method_3_pilot"
_GEN_DIR = _OUT_DIR / "generated"
_DEPTH_DIR = _OUT_DIR / "depth"
_MANIFEST = _OUT_DIR / "pilot_manifest.json"
_CONTACT = _OUT_DIR / "contact_sheet.jpg"

THUMB_W, THUMB_H = 256, 144
HEADER = 30
PADDING = 5


def main():
    manifest = json.loads(_MANIFEST.read_text())
    sources = manifest["source_selection"]["pilot_sources"]
    strengths = manifest["config"]["strengths"]
    scales = manifest["config"]["scales"]

    # Layout: 6 columns (2 strengths × 3 scales), 5 rows (one per source)
    cols = len(strengths) * len(scales)
    rows = len(sources)

    pair_w = THUMB_W + PADDING
    pair_h = THUMB_H + HEADER + PADDING
    canvas_w = cols * pair_w + PADDING
    canvas_h = rows * pair_h + PADDING + 30  # extra for title

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 8),
        "Method 3 Phase 3: 30-output pilot (rows=sources, cols=strength×scale)",
        fill=(255, 255, 255),
    )

    # Column headers
    col = 0
    for strength in strengths:
        for scale in scales:
            x = PADDING + col * pair_w
            draw.text((x + 4, 30), f"s={strength} c={scale}", fill=(180, 200, 255))
            col += 1

    # Build each cell
    for row, src_name in enumerate(sources):
        src_stem = Path(src_name).stem
        y = PADDING + 30 + row * pair_h

        # Row label
        draw.text((PADDING, y - 2), src_stem[:20], fill=(255, 200, 100))

        col = 0
        for strength in strengths:
            for scale in scales:
                gen_name = f"{src_stem}_s{int(strength * 100)}_c{int(scale * 100)}.png"
                gen_path = _GEN_DIR / gen_name
                if not gen_path.exists():
                    raise RuntimeError(f"Missing: {gen_path}")

                with Image.open(gen_path) as img:
                    thumb = img.convert("RGB").resize((THUMB_W, THUMB_H), Image.LANCZOS)

                x = PADDING + col * pair_w
                canvas.paste(thumb, (x, y + HEADER - 15))

                # Small label
                draw.text(
                    (x + 2, y + HEADER - 15 - 12), f"#{row * 6 + col + 1}", fill=(255, 255, 255)
                )
                col += 1

    canvas.save(_CONTACT, quality=92)
    print(f"✅ Contact sheet saved: {_CONTACT}")
    print(f"   {rows} sources × {cols} configs = {rows * cols} images")
    print(f"   Open with: Invoke-Item {_CONTACT}")


if __name__ == "__main__":
    main()
