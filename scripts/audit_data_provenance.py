"""G0 data and provenance audit for UrbanGen-AI.

Verifies that UrbanGen-AI's frozen inputs are internally consistent and
match their claimed provenance. Fatal on any mismatch.

This audit does NOT reconstruct UrbanMind-AI's historical Week 5
train_test_split(seed=42) output. The four resnet_* manifests are treated
as frozen inputs; the audit verifies that they partition the source
validation label JSON into four disjoint sets and that all entries are
present in that JSON. Reproduction of the historical seed-42 split is
documented as out of scope.

Environment variables:
    URBANINTEL_DATA_ROOT   parent directory containing `bdd100k/`
    URBANMIND_REPO         path to a local UrbanMind-AI checkout

Both are required. Configure via `.env` (gitignored); see `.env.example`.

Run:
    uv run python scripts/audit_data_provenance.py

Ruff:
    uv run ruff check scripts/audit_data_provenance.py

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import hashlib
import json
import sys

# Force UTF-8 stdout on Windows to avoid cp1252 encode errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from urbangen.config import data_root, urbanmind_root  # noqa: E402

# --- Frozen provenance ---
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = data_root()
_URBANMIND_ROOT = urbanmind_root()
_LABEL_DIR = _DATA_ROOT / "labels"

EXPECTED_TRAIN_JSON_SHA = "c1998792e385fd79213187576ca07d5249563d0ab83ae976688ad60668baca9e"
EXPECTED_VAL_JSON_SHA = "c835be652f0c002897be51e74deedd16a1dcee0b7230f19c0456fba772f694af"
EXPECTED_CHECKPOINT_SHA = "f760b94c4bb334c95c5d6c3c7884534fe6f51ce397641249610aab1a98447f84"
EXPECTED_CHECKPOINT_SIZE = 94_367_514
EXPECTED_TRAIN_COUNT = 68_969
EXPECTED_VAL_POOL_COUNT = 9_864
EXPECTED_TRAIN_CLASS_COUNTS = {
    "city street": 43_516,
    "highway": 17_379,
    "residential": 8_074,
}
EXPECTED_VAL_CLASS_COUNTS = {
    "city street": 6_112,
    "highway": 2_499,
    "residential": 1_253,
}
VIABLE_CLASSES = ("city street", "highway", "residential")

MANIFEST_NAMES = (
    "resnet_val_clean.json",
    "resnet_val_excluded.json",
    "resnet_test_clean.json",
    "resnet_test_excluded.json",
)
EXPECTED_MANIFEST_LENGTHS = {
    "resnet_val_clean.json": 4_727,
    "resnet_val_excluded.json": 205,
    "resnet_test_clean.json": 4_745,
    "resnet_test_excluded.json": 187,
}


class AuditError(Exception):
    """Raised when a fatal check fails."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AuditError(msg)


# --- Section 1: data root ---
def check_data_root() -> None:
    print("=== 1. Data root ===")
    _require(_DATA_ROOT.exists(), f"Data root not found: {_DATA_ROOT}")
    _require((_DATA_ROOT / "images").exists(), "images/ missing")
    _require((_DATA_ROOT / "labels").exists(), "labels/ missing")
    print("  images/: OK")
    print("  labels/: OK")
    print(f"  path: {_DATA_ROOT}")


# --- Section 2: label JSON hashes ---
def check_label_jsons() -> None:
    print("\n=== 2. Label JSON SHA-256 ===")
    pairs = [
        ("bdd100k_labels_images_train.json", EXPECTED_TRAIN_JSON_SHA),
        ("bdd100k_labels_images_val.json", EXPECTED_VAL_JSON_SHA),
    ]
    for name, expected in pairs:
        p = _LABEL_DIR / name
        _require(p.exists(), f"Missing label JSON: {p}")
        actual = sha256_file(p)
        _require(actual == expected, f"{name} SHA-256 mismatch: {actual}")
        print(f"  {name}:")
        print(f"    size: {p.stat().st_size:,}")
        print(f"    sha256: {actual}")


# --- Section 3: duplicate filenames in image tree ---
def check_no_duplicate_images() -> None:
    print("\n=== 3. Image filename uniqueness ===")
    # _index_images() in UrbanMind's loader indexes by bare filename and
    # silently overwrites collisions. Detect any such collision here.
    seen: dict[str, Path] = {}
    duplicates: list[tuple[str, Path, Path]] = []
    images_root = _DATA_ROOT / "images"
    for jpg in images_root.rglob("*.jpg"):
        name = jpg.name
        if name in seen:
            duplicates.append((name, seen[name], jpg))
        else:
            seen[name] = jpg
    print(f"  unique filenames: {len(seen):,}")
    if duplicates:
        print(f"  DUPLICATES FOUND: {len(duplicates)}")
        for name, a, b in duplicates[:5]:
            print(f"    {name}: {a} vs {b}")
        raise AuditError(f"{len(duplicates)} duplicate image filenames in {images_root}")
    print("  duplicates: 0 (OK)")


# --- Section 4: manifests exist, hashes match source, schema valid ---
def check_manifests() -> dict[str, list[str]]:
    print("\n=== 4. Split manifests ===")
    result: dict[str, list[str]] = {}
    for name in MANIFEST_NAMES:
        local = _REPO_ROOT / "data" / "splits" / name
        source = _URBANMIND_ROOT / "data" / "splits" / name
        _require(local.exists(), f"Local manifest missing: {local}")
        _require(source.exists(), f"Source manifest missing: {source}")
        local_sha = sha256_file(local)
        source_sha = sha256_file(source)
        _require(
            local_sha == source_sha,
            f"{name}: local and source SHA-256 differ",
        )
        data = json.loads(local.read_text(encoding="utf-8"))
        _require(isinstance(data, list), f"{name}: expected JSON list")
        _require(
            all(isinstance(x, str) for x in data),
            f"{name}: non-string entries in manifest",
        )
        _require(
            len(set(data)) == len(data),
            f"{name}: internal duplicates",
        )
        expected_len = EXPECTED_MANIFEST_LENGTHS[name]
        _require(
            len(data) == expected_len,
            f"{name}: expected {expected_len} entries, got {len(data)}",
        )
        print(f"  {name}:")
        print(f"    entries: {len(data):,}")
        print(f"    sha256: {local_sha}")
        result[name] = data
    return result


# --- Section 5: partition integrity ---
def check_partition(manifests: dict[str, list[str]]) -> None:
    print("\n=== 5. Manifest partition integrity ===")
    val_clean = set(manifests["resnet_val_clean.json"])
    val_excl = set(manifests["resnet_val_excluded.json"])
    test_clean = set(manifests["resnet_test_clean.json"])
    test_excl = set(manifests["resnet_test_excluded.json"])

    checks = [
        ("val_clean ∩ val_excluded", val_clean & val_excl),
        ("test_clean ∩ test_excluded", test_clean & test_excl),
        ("val_clean ∩ test_clean", val_clean & test_clean),
        (
            "(val_clean ∪ val_excluded) ∩ (test_clean ∪ test_excluded)",
            (val_clean | val_excl) & (test_clean | test_excl),
        ),
    ]
    for label, overlap in checks:
        _require(len(overlap) == 0, f"{label} = {len(overlap)} (expected 0)")
        print(f"  {label} = 0")

    _require(
        len(val_clean | val_excl) == 4_932,
        f"|val_clean ∪ val_excluded| = {len(val_clean | val_excl)} (expected 4932)",
    )
    _require(
        len(test_clean | test_excl) == 4_932,
        f"|test_clean ∪ test_excluded| = {len(test_clean | test_excl)} (expected 4932)",
    )
    total = len(val_clean | val_excl | test_clean | test_excl)
    _require(total == EXPECTED_VAL_POOL_COUNT, f"total manifest union = {total}")
    print(f"  |val_clean ∪ val_excluded| = {len(val_clean | val_excl)}")
    print(f"  |test_clean ∪ test_excluded| = {len(test_clean | test_excl)}")
    print(f"  |all four| = {total}")


# --- Section 6: manifest entries present in val label JSON ---
def check_manifest_membership(manifests: dict[str, list[str]]) -> None:
    print("\n=== 6. Manifest entries present in val label JSON ===")
    val_json = json.loads(
        (_LABEL_DIR / "bdd100k_labels_images_val.json").read_text(encoding="utf-8")
    )
    all_val_names = {rec["name"] for rec in val_json}
    for name, entries in manifests.items():
        missing = set(entries) - all_val_names
        _require(
            not missing,
            f"{name}: {len(missing)} entries not in val label JSON",
        )
        print(f"  {name}: all {len(entries)} entries present")


# --- Section 7: class counts ---
def _count_classes(label_json: Path) -> dict[str, int]:
    data = json.loads(label_json.read_text(encoding="utf-8"))
    counts = dict.fromkeys(VIABLE_CLASSES, 0)
    for rec in data:
        scene = rec.get("attributes", {}).get("scene")
        if scene in counts:
            counts[scene] += 1
    return counts


def check_class_counts() -> None:
    print("\n=== 7. Class counts (3-class filtered) ===")
    train_counts = _count_classes(_LABEL_DIR / "bdd100k_labels_images_train.json")
    val_counts = _count_classes(_LABEL_DIR / "bdd100k_labels_images_val.json")

    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())
    _require(
        train_total == EXPECTED_TRAIN_COUNT,
        f"train total = {train_total} (expected {EXPECTED_TRAIN_COUNT})",
    )
    _require(
        val_total == EXPECTED_VAL_POOL_COUNT,
        f"val total = {val_total} (expected {EXPECTED_VAL_POOL_COUNT})",
    )

    for cls in VIABLE_CLASSES:
        _require(
            train_counts[cls] == EXPECTED_TRAIN_CLASS_COUNTS[cls],
            f"train {cls} = {train_counts[cls]} (expected {EXPECTED_TRAIN_CLASS_COUNTS[cls]})",
        )
        _require(
            val_counts[cls] == EXPECTED_VAL_CLASS_COUNTS[cls],
            f"val {cls} = {val_counts[cls]} (expected {EXPECTED_VAL_CLASS_COUNTS[cls]})",
        )

    print("  train (68,969 total):")
    for cls in VIABLE_CLASSES:
        print(f"    {cls:12s}: {train_counts[cls]:6,}")
    print("  val pool (9,864 total):")
    for cls in VIABLE_CLASSES:
        print(f"    {cls:12s}: {val_counts[cls]:6,}")


# --- Section 8: checkpoint structure + SHA + load ---
def check_checkpoint() -> None:
    print("\n=== 8. Week 5 ResNet-50 checkpoint ===")
    ckpt_path = _URBANMIND_ROOT / "bdd_resnet50_best.pt"
    _require(ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}")
    size = ckpt_path.stat().st_size
    _require(
        size == EXPECTED_CHECKPOINT_SIZE,
        f"Checkpoint size = {size} (expected {EXPECTED_CHECKPOINT_SIZE})",
    )
    actual_sha = sha256_file(ckpt_path)
    _require(
        actual_sha == EXPECTED_CHECKPOINT_SHA,
        f"Checkpoint SHA-256 mismatch: {actual_sha}",
    )
    print(f"  path: {ckpt_path}")
    print(f"  size: {size:,}")
    print(f"  sha256: {actual_sha}")

    import torch
    from torchvision.models import resnet50

    # Inspect structure first; don't assume a plain state_dict
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  top-level type: {type(raw).__name__}")
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        print("  unwrapped: 'state_dict' key")
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        print("  unwrapped: 'model_state_dict' key")
    elif isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        state_dict = raw
        print("  format: plain state_dict (all values are tensors)")
    else:
        raise AuditError(f"Unrecognized checkpoint structure: {type(raw).__name__}")

    _require(
        isinstance(state_dict, dict),
        "Unwrapped checkpoint is not a dict",
    )
    print(f"  parameter tensors: {len(state_dict)}")

    model = resnet50(num_classes=3)
    model.load_state_dict(state_dict)
    print("  load_state_dict: OK")

    fc_w_shape = tuple(model.fc.weight.shape)
    fc_b_shape = tuple(model.fc.bias.shape)
    _require(fc_w_shape == (3, 2048), f"fc.weight shape = {fc_w_shape}")
    _require(fc_b_shape == (3,), f"fc.bias shape = {fc_b_shape}")
    print(f"  fc.weight shape: {fc_w_shape}")
    print(f"  fc.bias shape:   {fc_b_shape}")

    n_params = sum(p.numel() for p in model.parameters())
    _require(n_params == 23_514_179, f"params = {n_params}")
    print(f"  total parameters: {n_params:,}")


# --- main ---
def main() -> int:
    try:
        check_data_root()
        check_label_jsons()
        check_no_duplicate_images()
        manifests = check_manifests()
        check_partition(manifests)
        check_manifest_membership(manifests)
        check_class_counts()
        check_checkpoint()
    except AuditError as e:
        print(f"\nAUDIT FAILED: {e}")
        return 1

    print("\n" + "=" * 60)
    print("AUDIT PASS — all checks green")
    print("=" * 60)
    print("\nNext: paste this output into docs/dataset_cards/")
    print("bdd100k_generative.md and docs/experiments.md, then commit G0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
