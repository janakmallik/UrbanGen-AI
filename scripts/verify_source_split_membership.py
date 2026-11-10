"""Source split membership verification (partial leakage check).

Checks:
  1. All source images are in BDD100K train split (daytime residential)
  2. Manifest schema is strictly valid (4 keys, lists, no duplicates)
  3. Manifest splits are mutually disjoint (all 6 pairs)
  4. Reconstructed 500 == source_selection.json recorded 500 (reproducibility)
  5. Filter log sources == source_selection.json recorded 500 (recording consistency)
  6. The 282 accepted sources are a subset of the recorded 500
  7. No source overlaps val_clean, test_clean, or Method 2's reserved splits
  8. No duplicate basenames in image tree
  9. No duplicate generated filenames in accepted set
 10. All generated files exist on disk
 11. Generated filename encodes the source stem (lineage check)

This is NOT a comprehensive leakage verification. Limitations:
  - Filename-level checks only, not content hashes
  - Does NOT prove source_selection.json was written by the generation script
    at runtime and not modified afterward (outputs/ is gitignored).
    The dry-run reproducibility check (Check 4) is the strongest mitigation
    available without a git-tracked run artifact.
  - Does NOT verify generated images against evaluation sets directly
    (generated images are day→night translations of train sources, so
    their "content" is derived from train, which is the intended use)

Output: outputs/method_2_synthetic/source_split_membership.json
"""

import csv
import json
import random
import sys
from pathlib import Path

DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")
TRAIN_LABEL = DATA_ROOT / "labels" / "bdd100k_labels_images_train.json"
IMAGES_DIR = DATA_ROOT / "images" / "100k" / "train"
URBANMIND = Path(r"C:\Users\Janak Mallik\UrbanMind-AI")
VAL_CLEAN = URBANMIND / "data" / "splits" / "resnet_val_clean.json"
TEST_CLEAN = URBANMIND / "data" / "splits" / "resnet_test_clean.json"

GEN_ROOT = Path("outputs/method_2_synthetic")
FILTER_LOG = GEN_ROOT / "filter_log.csv"
SOURCE_SELECTION = GEN_ROOT / "source_selection.json"  # [ADDED]
OUT_PATH = GEN_ROOT / "source_split_membership.json"

SEED = 42  # [ADDED] must match method_2_generate_synthetic.py


def load_manifest_filenames(path):
    """Load manifest and return set of filenames, strict about schema."""
    if not path.exists():
        sys.exit(f"ERROR: manifest not found: {path}")
    data = json.loads(path.read_text())

    result = None
    schema_desc = None

    if isinstance(data, list):
        if not data:
            raise ValueError(f"{path.name}: empty list")
        if all(isinstance(x, str) for x in data):
            pre_dedup_len = len(data)
            result = set(data)
            schema_desc = f"list[str] (n={len(data)})"
        elif all(isinstance(x, dict) and "name" in x and isinstance(x["name"], str) for x in data):
            pre_dedup_len = len(data)
            result = {x["name"] for x in data}
            schema_desc = f"list[dict] with 'name' key (n={len(data)})"
        else:
            raise ValueError(
                f"{path.name}: unsupported list schema. First item type={type(data[0]).__name__}"
            )
    elif isinstance(data, dict):
        for key in ("filenames", "files", "names"):
            if (
                key in data
                and isinstance(data[key], list)
                and all(isinstance(x, str) for x in data[key])
            ):
                pre_dedup_len = len(data[key])
                result = set(data[key])
                schema_desc = f"dict['{key}'] = list[str] (n={len(data[key])})"
                break
        if result is None:
            raise ValueError(
                f"{path.name}: unsupported dict schema. "
                f"Top-level keys: {list(data.keys())}. "
                f"Expected one of: filenames, files, names — as list[str]."
            )
    else:
        raise ValueError(f"{path.name}: unsupported top-level type {type(data).__name__}")

    if not result:
        raise ValueError(f"{path.name}: extracted empty filename set")

    if pre_dedup_len != len(result):
        raise ValueError(f"{path.name}: contains {pre_dedup_len - len(result)} duplicate entries")

    normalized = {Path(name).name for name in result}
    if len(normalized) != len(result):
        raise ValueError(
            f"{path.name}: {len(result) - len(normalized)} entries collapse to same basename"
        )

    return normalized, schema_desc


def main():
    print("=" * 60)
    print("Leakage verification")
    print("=" * 60)

    # --- Load accepted synthetic outputs ---
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

    # [ADDED] Strict CSV validation — kept must be "0"/"1", accepted must have output
    for i, row in enumerate(log):
        # 'source' and 'kept' must be non-empty for every row.
        # 'output' is empty for rejected rows by design (see method_2_generate_synthetic.py),
        # so it is only required for accepted rows.
        for col in ("source", "kept"):
            if not row.get(col):
                sys.exit(f"ERROR: filter_log.csv row {i + 2} has empty/missing '{col}'")
        if row["kept"] not in ("0", "1"):
            sys.exit(
                f"ERROR: filter_log.csv row {i + 2} has invalid kept value "
                f"{row['kept']!r} — must be '0' or '1'"
            )
        if row["kept"] == "1" and not row["output"].strip():
            sys.exit(f"ERROR: filter_log.csv row {i + 2} is accepted (kept=1) but has empty output")

    # [ADDED] Strict filename safety for both source and output fields
    for i, row in enumerate(log):
        fields = ("source",) if row["kept"] != "1" else ("source", "output")
        for field in fields:
            name = row[field]
            if (
                not name
                or name != name.strip()
                or Path(name).name != name
                or "/" in name
                or "\\" in name
            ):
                sys.exit(
                    f"ERROR: filter_log.csv row {i + 2} has invalid "
                    f"{field} filename: {row[field]!r}"
                )
    print("✅ Filename safety: all source and output fields are plain basenames")

    accepted = [r for r in log if r["kept"] == "1"]
    source_list = [r["source"] for r in accepted]
    if len(source_list) != len(set(source_list)):
        sys.exit(f"ERROR: duplicate source filenames: {len(source_list) - len(set(source_list))}")
    source_names = set(source_list)
    generated_names = {r["output"] for r in accepted}
    print(f"\nAccepted synthetic outputs: {len(accepted)}")
    print(f"Unique sources: {len(source_names)}")
    print(f"Unique generated: {len(generated_names)}")

    if len(generated_names) != len(accepted):
        sys.exit(f"ERROR: duplicate generated filenames: {len(accepted) - len(generated_names)}")

    # [ADDED] Lineage check — generated filename encodes the source stem
    # (method_2_generate_synthetic.py names outputs as synth_{i:04d}_{source_stem}.jpg)
    lineage_failures = []
    for r in accepted:
        src_stem = Path(r["source"]).stem
        output_stem = Path(r["output"]).stem
        if not output_stem.endswith(f"_{src_stem}"):
            lineage_failures.append((r["source"], r["output"]))
    if lineage_failures:
        print(
            f"\n❌ FAIL: {len(lineage_failures)} generated filenames don't encode their source stem:"
        )
        for s, g in lineage_failures[:10]:
            print(f"  source={s}  output={g}")
        sys.exit(1)
    print(f"✅ Lineage: all {len(accepted)} generated filenames encode their source stem")

    # Verify all generated files actually exist on disk
    missing_generated = []
    for r in accepted:
        gen_path = GEN_ROOT / r["output"]
        if not gen_path.exists():
            missing_generated.append(r["output"])
    if missing_generated:
        print(f"\n❌ FAIL: {len(missing_generated)} generated files missing from disk:")
        for g in missing_generated[:10]:
            print(f"  {g}")
        sys.exit(1)
    print(f"✅ All {len(generated_names)} generated files exist on disk")

    # --- Load BDD100K train label JSON ---
    print("\nLoading BDD100K train labels...")
    train_records = json.loads(TRAIN_LABEL.read_text())
    train_names = set()
    for rec in train_records:
        scene = rec.get("attributes", {}).get("scene")
        tod = rec.get("attributes", {}).get("timeofday")
        if scene == "residential" and tod == "daytime":
            train_names.add(rec["name"])
    print(f"  Train split daytime residential: {len(train_names)}")

    # --- Load manifests ---
    print("\nLoading Week 5 manifests...")
    val_clean, val_schema = load_manifest_filenames(VAL_CLEAN)
    test_clean, test_schema = load_manifest_filenames(TEST_CLEAN)
    print(f"  val_clean: {len(val_clean)} filenames — schema: {val_schema}")
    print(f"  test_clean: {len(test_clean)} filenames — schema: {test_schema}")

    # --- Check 1: All sources in train split ---
    not_in_train = source_names - train_names
    if not_in_train:
        print(f"\n❌ FAIL: {len(not_in_train)} sources NOT in train split:")
        for n in list(not_in_train)[:10]:
            print(f"  {n}")
        sys.exit(1)
    print(f"\n✅ All {len(source_names)} sources present in train split")

    # --- Load Method 2 manifest ---
    manifest_path = GEN_ROOT.parent / "method_2_resume" / "split_manifest.json"
    if not manifest_path.exists():
        sys.exit(
            f"ERROR: Method 2 manifest not found at {manifest_path}. "
            f"Cannot verify sources don't overlap Method 2 training."
        )

    m2_manifest = json.loads(manifest_path.read_text())

    # [ADDED] Top-level type check
    if not isinstance(m2_manifest, dict):
        sys.exit("ERROR: split_manifest.json must contain a JSON object")

    # [ADDED] Strict schema validation
    required_m2_keys = {"day_train", "day_holdout", "night_train", "night_holdout"}
    if not required_m2_keys.issubset(m2_manifest.keys()):
        sys.exit(
            f"ERROR: split_manifest.json missing required keys. "
            f"Found: {list(m2_manifest.keys())}. Need: {required_m2_keys}"
        )
    for key in required_m2_keys:
        if not isinstance(m2_manifest[key], list):
            sys.exit(f"ERROR: split_manifest.json['{key}'] is not a list")
        if any(not isinstance(x, str) or not x.strip() for x in m2_manifest[key]):
            sys.exit(f"ERROR: split_manifest.json['{key}'] contains non-string or empty entries")
        if len(set(m2_manifest[key])) != len(m2_manifest[key]):
            dups = len(m2_manifest[key]) - len(set(m2_manifest[key]))
            sys.exit(f"ERROR: split_manifest.json['{key}'] has {dups} duplicate entries")
    print("✅ split_manifest.json schema validated (4 keys, no duplicates)")

    # [ADDED] Cross-split disjointness — all 6 pairs
    # Normalize to basenames BEFORE disjointness check; reject post-normalization dups
    split_sets = {}
    for key in required_m2_keys:
        normalized = [Path(n).name for n in m2_manifest[key]]
        if len(set(normalized)) != len(normalized):
            sys.exit(
                f"ERROR: split_manifest.json['{key}'] has "
                f"{len(normalized) - len(set(normalized))} duplicate basenames after normalization"
            )
        split_sets[key] = set(normalized)
    split_pairs = [
        ("day_train", "day_holdout"),
        ("night_train", "night_holdout"),
        ("day_train", "night_train"),
        ("day_holdout", "night_holdout"),
        ("day_train", "night_holdout"),
        ("day_holdout", "night_train"),
    ]
    for a, b in split_pairs:
        overlap = split_sets[a] & split_sets[b]
        if overlap:
            sys.exit(f"ERROR: split_manifest.json has {len(overlap)} overlap between {a} and {b}")
    print("✅ split_manifest.json splits are mutually disjoint (6/6 pairs)")

    m2_reserved_basenames = {
        Path(n).name
        for n in (
            set(m2_manifest["day_train"])
            | set(m2_manifest["day_holdout"])
            | set(m2_manifest["night_train"])
            | set(m2_manifest["night_holdout"])
        )
    }

    source_overlap = source_names & m2_reserved_basenames
    if source_overlap:
        print(f"\n❌ FAIL: {len(source_overlap)} sources overlap with Method 2 reserved splits:")
        for n in list(source_overlap)[:10]:
            print(f"  {n}")
        sys.exit(1)
    print("✅ Zero source overlap with Method 2 reserved splits")

    # --- Check 4 + 5: Reproducibility + recording consistency ---
    # Recompute the eligible pool the way method_2_generate_synthetic.py does:
    #   original_day_pool = all daytime residential training images, sorted by name
    #   eligible = [p for p in original_day_pool if p.name not in method_2_reserved]
    #   eligible.sort(key=lambda p: p.name)
    #   selected = random.Random(42).sample(eligible, 500)
    #
    # Note: the generation script's IMAGE_INDEX is built from IMAGES_DIR.rglob("*.jpg"),
    # and original_day_pool filters records whose name is in IMAGE_INDEX. So reconstructing
    # from rglob and filtering by train_names is equivalent, given no duplicate basenames
    # (which we check separately below).
    # Fix 5 — verify no duplicate basenames BEFORE assuming rglob() is unambiguous
    _basename_counts = {}
    for _p in IMAGES_DIR.rglob("*.jpg"):
        _basename_counts[_p.name] = _basename_counts.get(_p.name, 0) + 1
    _dups = {n: c for n, c in _basename_counts.items() if c > 1}
    if _dups:
        sys.exit(
            f"ERROR: {len(_dups)} duplicate basenames in image tree — "
            f"rglob() index is ambiguous, cannot reconstruct pool"
        )

    all_eligible_sorted = sorted(
        p.name
        for p in IMAGES_DIR.rglob("*.jpg")
        if p.name in train_names and p.name not in m2_reserved_basenames
    )
    print(f"\nReconstructed eligible pool: {len(all_eligible_sorted)}")

    if len(all_eligible_sorted) < 500:
        sys.exit(
            f"ERROR: reconstructed eligible pool has {len(all_eligible_sorted)} entries, need ≥500"
        )

    recon_rng = random.Random(SEED)
    recon_500_list = recon_rng.sample(all_eligible_sorted, 500)
    recon_500 = set(recon_500_list)
    print(f"Reconstructed 500 (seed={SEED}): {len(recon_500)} unique")

    # Load recorded selection
    if not SOURCE_SELECTION.exists():
        sys.exit(
            f"ERROR: {SOURCE_SELECTION} not found. "
            f"Cannot verify sources match the recorded original selection."
        )
    selection_data = json.loads(SOURCE_SELECTION.read_text())
    if not isinstance(selection_data, dict):
        sys.exit("ERROR: source_selection.json must contain a JSON object")
    if "selected_sources" not in selection_data:
        sys.exit("ERROR: source_selection.json missing 'selected_sources' key")
    recorded_list = selection_data["selected_sources"]
    if not isinstance(recorded_list, list):
        sys.exit("ERROR: source_selection.json['selected_sources'] is not a list")
    if len(recorded_list) != 500:
        sys.exit(
            f"ERROR: source_selection.json has {len(recorded_list)} selected sources, expected 500"
        )
    # Fix 2 — validate every recorded entry before normalizing
    for i, name in enumerate(recorded_list):
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or Path(name).name != name
            or "/" in name
            or "\\" in name
        ):
            sys.exit(f"ERROR: invalid selected_sources entry at index {i}: {name!r}")
    if len(set(recorded_list)) != 500:
        sys.exit(
            f"ERROR: source_selection.json contains "
            f"{500 - len(set(recorded_list))} duplicate filenames"
        )
    recorded_set = set(recorded_list)

    # Fix 1 — ordered comparison (auditor: "ordering needs verification")
    if recon_500_list != recorded_list:
        for i, (a, b) in enumerate(zip(recon_500_list, recorded_list, strict=True)):
            if a != b:
                print(f"\n❌ First order mismatch at index {i}: recon={a!r}, recorded={b!r}")
                break
        sys.exit("ERROR: reconstructed selection order does not match source_selection.json")
    print("✅ Reconstructed 500 == recorded selection (ordered)")

    # Check 4 — recomputed 500 must equal recorded 500
    if recon_500 != recorded_set:
        missing_from_recon = recorded_set - recon_500
        extra_in_recon = recon_500 - recorded_set
        print("\n❌ FAIL: recomputed 500 does not match recorded source_selection.json")
        if missing_from_recon:
            print(f"  {len(missing_from_recon)} recorded sources not in recomputation:")
            for n in list(missing_from_recon)[:10]:
                print(f"    {n}")
        if extra_in_recon:
            print(f"  {len(extra_in_recon)} recomputed sources not in recorded:")
            for n in list(extra_in_recon)[:10]:
                print(f"    {n}")
        sys.exit(1)
    print("✅ Reconstructed 500 == source_selection.json selected_sources (reproducible)")

    # Check 5 — filter log sources must equal recorded 500
    all_filter_sources = {r["source"] for r in log if r["source"]}
    if all_filter_sources != recorded_set:
        missing_from_log = recorded_set - all_filter_sources
        extra_in_log = all_filter_sources - recorded_set
        print("\n❌ FAIL: filter log sources do not match recorded selection:")
        if missing_from_log:
            print(f"  {len(missing_from_log)} recorded sources missing from log:")
            for n in list(missing_from_log)[:10]:
                print(f"    {n}")
        if extra_in_log:
            print(f"  {len(extra_in_log)} log sources not in recorded selection:")
            for n in list(extra_in_log)[:10]:
                print(f"    {n}")
        sys.exit(1)
    print(f"✅ Filter log sources == recorded selection ({len(recorded_set)} sources)")

    # Check 6 — accepted must be subset of recorded 500
    accepted_not_recorded = source_names - recorded_set
    if accepted_not_recorded:
        print(
            f"\n❌ FAIL: {len(accepted_not_recorded)} accepted sources not in recorded selection:"
        )
        for n in list(accepted_not_recorded)[:10]:
            print(f"  {n}")
        sys.exit(1)
    print(f"✅ All {len(source_names)} accepted sources are within recorded 500")

    # --- Check 7: No overlap with val_clean ---
    overlap_val = source_names & val_clean
    if overlap_val:
        print(f"\n❌ FAIL: {len(overlap_val)} sources appear in val_clean:")
        for n in list(overlap_val)[:10]:
            print(f"  {n}")
        sys.exit(1)
    print("✅ Zero source overlap with val_clean")

    # --- Check 8: No overlap with test_clean ---
    overlap_test = source_names & test_clean
    if overlap_test:
        print(f"\n❌ FAIL: {len(overlap_test)} sources appear in test_clean:")
        for n in list(overlap_test)[:10]:
            print(f"  {n}")
        sys.exit(1)
    print("✅ Zero source overlap with test_clean")

    # Check 9 (duplicate basenames) already verified before pool reconstruction.
    print(f"✅ No duplicate basenames ({len(_basename_counts)} unique filenames)")

    # --- Save verification record ---
    import datetime

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "check_scope": "source_filename_membership_only",
        "note": (
            "Does NOT verify image content hashes; does NOT check generated images "
            "against eval sets. Reproducibility check recomputes seed-42 selection "
            "from documented procedure and compares against recorded source_selection.json."
        ),
        "accepted_synthetic_count": len(accepted),
        "unique_sources": len(source_names),
        "unique_generated": len(generated_names),
        "reconstructed_500_matches_recorded": True,
        "filter_log_matches_recorded": True,
        "sources_in_train_split": True,
        "sources_in_val_clean": 0,
        "sources_in_test_clean": 0,
        "generated_files_on_disk": len(generated_names),
        "sources_overlap_method_2_reserved": 0,
        "duplicate_basenames": 0,
        "manifest_schemas_observed": {
            "val_clean": val_schema,
            "test_clean": test_schema,
        },
        "passed": True,
    }
    OUT_PATH.write_text(json.dumps(record, indent=2))

    print("\n" + "=" * 60)
    print("✅ Source split membership PASSED")
    print("=" * 60)
    print("NOTE: This is a filename-level check, not content-level.")
    print("Reproducibility: recomputed seed-42 == recorded source_selection.json.")
    print("Limitation: source_selection.json is not git-tracked, so no proof")
    print("            it wasn't modified after the run.")
    print(f"Saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
