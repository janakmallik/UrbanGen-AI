import json
import shutil
from pathlib import Path

CALIB = Path("outputs/controlnet_calibration/calibration.json")
OUT = Path("outputs/controlnet_calibration/inspect")
OUT.mkdir(parents=True, exist_ok=True)

with CALIB.open("r", encoding="utf-8") as f:
    data = json.load(f)

for split in ["daytime", "nighttime"]:
    for i, rec in enumerate(data[split], 1):
        src = Path(rec["path"])
        dst = OUT / f"{split}_{i:02d}_{src.name}"
        shutil.copy2(src, dst)
        print(f"Copied: {dst.name}")
