import csv
import json
from pathlib import Path

base = Path("outputs/method_2_synthetic")
d138 = json.loads((base / "shortcut_control/d_138_manifest.json").read_text())
dbal = json.loads((base / "shortcut_control/d_bal_manifest.json").read_text())
curated = json.loads((base / "curated_manifest.json").read_text())

with open(base / "filter_log.csv") as f:
    rows = [r for r in csv.DictReader(f) if r["kept"] == "1"]
    accepted = sorted(rows, key=lambda r: r["output"])
    d_full = {accepted[i - 1]["output"] for i in curated["final_keep_labels"]}

d138_set = set(d138["residential"])
dbal_res_set = set(dbal["residential"])

print("D-138 is subset of D:", d138_set.issubset(d_full))
print("D-bal residential matches D-138:", d138_set == dbal_res_set)
print("D-bal city count:", len(dbal["city"]), "(expected 69)")
print("D-bal highway count:", len(dbal["highway"]), "(expected 69)")
