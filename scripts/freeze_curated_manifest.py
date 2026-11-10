"""Freeze the curated dataset by recording manual visual decisions.

Indices refer to contact-sheet labels (#1 - #282), which are sorted
by output filename to match `build_contact_sheets.py`.

Usage:
    uv run python scripts/freeze_curated_manifest.py `
      --reject 6 7 36 --reason "too_dark" `
      --uncertain 15 --reason "borderline"
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

FILTER_LOG = Path("outputs/method_2_synthetic/filter_log.csv")
OUT_MANIFEST = Path("outputs/method_2_synthetic/curated_manifest.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reject",
        nargs="+",
        type=int,
        action="append",
        help="Contact-sheet labels to reject (#1-#282)",
    )
    parser.add_argument(
        "--uncertain",
        nargs="+",
        type=int,
        action="append",
        help="Contact-sheet labels to mark uncertain (#1-#282)",
    )
    parser.add_argument(
        "--reason",
        type=str,
        action="append",
        help="Reason for the preceding batch (must match count of --reject/--uncertain)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifest")
    args = parser.parse_args()

    if OUT_MANIFEST.exists() and not args.force:
        sys.exit(f"ERROR: {OUT_MANIFEST} already exists. Use --force to overwrite.")

    # 1. Parse batches and reasons
    batches = []
    if args.reject:
        batches.extend(
            [
                (
                    "reject",
                    sub,
                    args.reason[i] if args.reason and i < len(args.reason) else "unspecified",
                )
                for i, sub in enumerate(args.reject)
            ]
        )
    if args.uncertain:
        offset = len(args.reject) if args.reject else 0
        batches.extend(
            [
                (
                    "uncertain",
                    sub,
                    args.reason[offset + i]
                    if args.reason and (offset + i) < len(args.reason)
                    else "unspecified",
                )
                for i, sub in enumerate(args.uncertain)
            ]
        )

    if not batches:
        sys.exit("ERROR: You must provide at least one --reject or --uncertain list.")

    # 2. Load filter log and build contact-sheet mapping
    with open(FILTER_LOG) as f:
        log = list(csv.DictReader(f))

    # Reproduce the exact sort used by build_contact_sheets.py
    accepted_sorted = sorted(
        [r for r in log if r["kept"] == "1"],
        key=lambda r: r["output"],
    )
    if len(accepted_sorted) != 282:
        sys.exit(f"ERROR: expected 282 accepted, found {len(accepted_sorted)}")

    # Contact-sheet label N (1-based) -> accepted_sorted[N-1]
    valid_labels = {i + 1: row for i, row in enumerate(accepted_sorted)}
    filter_log_hash = sha256_file(FILTER_LOG)

    decisions = {}
    for status, indices, reason in batches:
        for label in indices:
            if label not in valid_labels:
                sys.exit(
                    f"ERROR: Contact-sheet label #{label} is not in valid range "
                    f"(1-282). Check your contact sheets."
                )
            if label in decisions:
                sys.exit(f"ERROR: Label #{label} was assigned multiple decisions.")

            row = valid_labels[label]
            decisions[str(label)] = {
                "status": status,
                "reason": reason,
                "source": row["source"],
                "output": row["output"],
                "original_filter_index": int(row["index"]),
            }

    # 3. Calculate final counts
    all_labels = sorted(valid_labels.keys())
    final_keep_labels = [i for i in all_labels if str(i) not in decisions]
    uncertain_labels = sorted(int(i) for i, d in decisions.items() if d["status"] == "uncertain")
    rejected_labels = sorted(int(i) for i, d in decisions.items() if d["status"] == "reject")

    manifest = {
        "source_filter_log": str(FILTER_LOG),
        "source_filter_log_sha256": filter_log_hash,
        "indexing_scheme": "contact_sheet_label (1-282, sorted by output filename)",
        "total_accepted_by_filter": len(all_labels),
        "final_count": len(final_keep_labels),
        "final_keep_labels": final_keep_labels,
        "uncertain_labels": uncertain_labels,
        "rejected_labels": rejected_labels,
        "decisions": decisions,
    }

    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print("✅ Created curated_manifest.json")
    print(f"   Filter log hash: {filter_log_hash[:16]}...")
    print(f"   Kept: {len(final_keep_labels)} / {len(all_labels)}")
    print(f"   Rejected: {len(rejected_labels)}")
    print(f"   Uncertain: {len(uncertain_labels)}")
    print("\n   Decisions:")
    for label in sorted([int(k) for k in decisions]):
        d = decisions[str(label)]
        print(f"     #{label}: {d['status']} ({d['reason']}) -> {d['output']}")


if __name__ == "__main__":
    main()
