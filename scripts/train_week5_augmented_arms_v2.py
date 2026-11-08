"""Train Week 5 ResNet-50 under four augmentation arms, per-arm per-seed.

Usage:
    uv run python scripts/train_week5_augmented_arms_v2.py --arm A --seed 42
    uv run python scripts/train_week5_augmented_arms_v2.py --arm B --seed 42
    uv run python scripts/train_week5_augmented_arms_v2.py --arm C --seed 42
    uv run python scripts/train_week5_augmented_arms_v2.py --arm A --seed 42 --resume
    uv run python scripts/train_week5_augmented_arms_v2.py --summarize

Arms (all include Week 5 base augmentation: RRC + HFlip + ColorJitter 0.2):
    A: baseline (base aug only)
    B: base aug + night residential oversampled 3x (also tests class-weight combo)
    C: RRC + HFlip + ColorJitter(0.5,0.5,0.3,0.15) + RandomBrighten[0.5,1.5]
    D: base aug + CLIP-filtered synthetic frames (skipped if none exist)

Primary metric: night macro-F1 (all classes, timeofday=night, n~1931)
Secondary:      night residential recall (n~147)
Guardrail:      day macro-F1 must not drop > 0.02 vs arm A

Decision rule (pre-registered):
    arm C night macro-F1 >= 0.68  -> stop generative plan
    0.63 <= arm C < 0.68          -> proceed to Phase 2 (arm D)
    arm C < 0.63                  -> stop, document negative finding
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault("URBANMIND_REPO", str(_REPO_ROOT.parent / "UrbanMind-AI"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.config import data_root, urbanmind_root  # noqa: E402
from urbangen.data.preprocessing import (  # noqa: E402
    WEEK5_CLASS_NAMES,
    WEEK5_CLASS_TO_ID,
    week5_eval_transform,
)

# Paths (data_root returns .../ml-data/bdd100k — no doubling)
_URBANMIND = urbanmind_root()
_DATA = data_root()
_TRAIN_LABEL = _DATA / "labels" / "bdd100k_labels_images_train.json"
_VAL_LABEL = _DATA / "labels" / "bdd100k_labels_images_val.json"
_IMAGES = _DATA / "images"
_VAL_CLEAN = _URBANMIND / "data" / "splits" / "resnet_val_clean.json"
_TEST_CLEAN = _URBANMIND / "data" / "splits" / "resnet_test_clean.json"
_SYNTHETIC = _REPO_ROOT / "outputs" / "method_4_wan" / "frames_clip_filtered"
_OUT_DIR = _REPO_ROOT / "outputs" / "week5_arms"

# Recipe (identical across arms)
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
N_BOOTSTRAP = 1000
OVERSAMPLE_FACTOR = 3
NUM_WORKERS = 4

# Sanity checks on manifest size (from golden test / G0 audit)
EXPECTED_VAL_CLEAN = 4727
EXPECTED_TEST_CLEAN = 4745

# Pre-registered decision thresholds
C_STOP = 0.68
C_PROCEED = 0.63
DAY_GUARDRAIL = 0.02

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomBrighten:
    """Multiply tensor by a random scalar in [lo, hi]. Named for pickling."""

    def __init__(self, lo: float = 0.5, hi: float = 1.5):
        self.lo = lo
        self.hi = hi

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        scale = self.lo + (self.hi - self.lo) * torch.rand(1)
        return torch.clamp(t * scale, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"RandomBrighten({self.lo}, {self.hi})"


def base_augmentation() -> transforms.Compose:
    """Week 5 base recipe: RRC + HFlip + moderate ColorJitter."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def strong_augmentation() -> transforms.Compose:
    """Arm C: base + stronger photometric, symmetric brightness."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.15),
            transforms.ToTensor(),
            RandomBrighten(0.5, 1.5),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def index_images() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in _IMAGES.rglob("*.jpg"):
        idx.setdefault(p.name, p)
    return idx


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_split(
    records: list[dict],
    names_filter: set[str] | None,
    image_index: dict[str, Path],
) -> list[dict]:
    out = []
    for rec in records:
        scene = rec.get("attributes", {}).get("scene")
        if scene not in WEEK5_CLASS_TO_ID:
            continue
        name = rec["name"]
        if names_filter is not None and name not in names_filter:
            continue
        path = image_index.get(name)
        if path is None:
            continue
        out.append(
            {
                "path": path,
                "label": WEEK5_CLASS_TO_ID[scene],
                "timeofday": rec.get("attributes", {}).get("timeofday", "unknown"),
                "name": name,
            }
        )
    return out


class ClassificationDataset(Dataset):
    def __init__(self, records, transform, extra_synthetic=None):
        self.samples = [(r["path"], r["label"]) for r in records]
        if extra_synthetic:
            for p in extra_synthetic:
                self.samples.append((p, WEEK5_CLASS_TO_ID["residential"]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        return x, label


def build_train_loader(arm, full_train, synthetic, seed):
    if arm == "A":
        records = full_train
        ds = ClassificationDataset(records, base_augmentation())
    elif arm == "B":
        night_res = [r for r in full_train if r["label"] == 2 and r["timeofday"] == "night"]
        extra = night_res * (OVERSAMPLE_FACTOR - 1)
        records = full_train + extra
        ds = ClassificationDataset(records, base_augmentation())
        print(
            f"  Arm B: {len(night_res)} night-res ×{OVERSAMPLE_FACTOR} "
            f"= +{len(extra)} extra → total {len(records):,}"
        )
    elif arm == "C":
        records = full_train
        ds = ClassificationDataset(records, strong_augmentation())
    elif arm == "D":
        if not synthetic:
            raise ValueError(f"Arm D requires synthetic frames in {_SYNTHETIC}")
        records = full_train
        ds = ClassificationDataset(records, base_augmentation(), extra_synthetic=synthetic)
        print(f"  Arm D: +{len(synthetic)} synthetic → total {len(records) + len(synthetic):,}")
    else:
        raise ValueError(f"unknown arm: {arm}")

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        generator=g,
    )


def build_eval_loader(records):
    ds = ClassificationDataset(records, week5_eval_transform())
    return DataLoader(
        ds, batch_size=64, shuffle=False,
        num_workers=0,  # eval loaders: no workers — avoids Windows paging OOM
        pin_memory=True,
    )


def build_restricted(val_records, name_filter, image_index, timeofday):
    """Build a loader from val labels filtered by test_clean names (+ optional timeofday)."""
    records = []
    for rec in val_records:
        scene = rec.get("attributes", {}).get("scene")
        if scene not in WEEK5_CLASS_TO_ID:
            continue
        if rec["name"] not in name_filter:
            continue
        if timeofday is not None and rec.get("attributes", {}).get("timeofday") != timeofday:
            continue
        path = image_index.get(rec["name"])
        if path is None:
            continue
        records.append(
            {
                "path": path,
                "label": WEEK5_CLASS_TO_ID[scene],
                "timeofday": rec.get("attributes", {}).get("timeofday", "unknown"),
                "name": rec["name"],
            }
        )
    return build_eval_loader(records), len(records)


def fresh_model(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, 3)
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        preds.append(model(x).argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def compute_metrics(preds, labels):
    pc_recall = recall_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    pc_prec = precision_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    pc_f1 = f1_score(labels, preds, labels=[0, 1, 2], average=None, zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    return {
        "n": int(len(labels)),
        "macro_f1": float(macro_f1),
        "per_class": {
            WEEK5_CLASS_NAMES[i]: {
                "recall": float(pc_recall[i]),
                "precision": float(pc_prec[i]),
                "f1": float(pc_f1[i]),
                "support": int((labels == i).sum()),
            }
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
    }


def train_arm(
    args, full_train, val_clean, night_loader, day_loader, test_loader, synthetic, device
):
    result_path = _OUT_DIR / f"results_{args.arm}_seed{args.seed}.json"
    ckpt_path = _OUT_DIR / f"checkpoint_{args.arm}_seed{args.seed}.pt"

    if result_path.exists() and not args.force:
        print(f"  Result already exists: {result_path}")
        print("  Use --force to re-run.")
        return json.loads(result_path.read_text())

    print(f"\n=== Arm {args.arm} (seed={args.seed}) ===")
    train_loader = build_train_loader(args.arm, full_train, synthetic, args.seed)
    val_loader = build_eval_loader(val_clean)
    print(f"  train: {len(train_loader.dataset):,}  val: {len(val_clean):,}")

    ckpt = None
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"  Resuming from epoch {ckpt['epoch_completed'] + 1}/{EPOCHS}")

    model = fresh_model(args.seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch_completed"] + 1
        best_val_f1 = ckpt["best_val_macro_f1"]
        best_state = ckpt["best_state_dict"]
        best_epoch = ckpt["best_epoch"]
    else:
        start_epoch = 0
        best_val_f1 = -1.0
        best_state = None
        best_epoch = -1

    labels_arr = np.array([r["label"] for r in full_train])
    counts = np.bincount(labels_arr, minlength=3)
    class_weights = torch.tensor(len(labels_arr) / (3.0 * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  class counts: {counts.tolist()}  weights: {class_weights.tolist()}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        for n_batches, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            if n_batches % 500 == 0:
                print(f"    epoch {epoch + 1} batch {n_batches}/{len(train_loader)}", flush=True)

        val_preds, val_labels = evaluate(model, val_loader, device)
        val_f1 = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))
        print(
            f"  epoch {epoch + 1}/{EPOCHS}  "
            f"train_loss={total_loss / len(train_loader.dataset):.4f}  "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        torch.save(
            {
                "arm": args.arm,
                "seed": args.seed,
                "epoch_completed": epoch,
                "model_state_dict": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_macro_f1": best_val_f1,
                "best_epoch": best_epoch,
                "best_state_dict": best_state,
            },
            ckpt_path,
        )

    print(f"  Loading best state (epoch {best_epoch + 1}, val_macro_f1={best_val_f1:.4f})")
    model.load_state_dict(best_state)
    model.eval()

    night_preds, night_labels = evaluate(model, night_loader, device)
    day_preds, day_labels = evaluate(model, day_loader, device)
    test_preds, test_labels = evaluate(model, test_loader, device)

    night = compute_metrics(night_preds, night_labels)
    day = compute_metrics(day_preds, day_labels)
    test = compute_metrics(test_preds, test_labels)

    result = {
        "arm": args.arm,
        "seed": args.seed,
        "epochs_run": EPOCHS,
        "best_epoch": best_epoch + 1,
        "best_val_macro_f1": best_val_f1,
        "night": {**night, "predictions": night_preds.tolist(), "labels": night_labels.tolist()},
        "day": day,
        "test": test,
        "night_macro_f1": night["macro_f1"],
        "night_residential_recall": night["per_class"]["residential"]["recall"],
        "day_macro_f1": day["macro_f1"],
        "test_macro_f1": test["macro_f1"],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  Saved: {result_path}")

    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"\n  SUMMARY (arm {args.arm}, seed {args.seed}):")
    print(f"    night macro-F1:            {night['macro_f1']:.4f}  (primary)")
    print(f"    night residential recall:  {result['night_residential_recall']:.4f}  (secondary)")
    print(f"    day macro-F1:              {day['macro_f1']:.4f}  (guardrail)")
    print(f"    test macro-F1:             {test['macro_f1']:.4f}")
    for cls in WEEK5_CLASS_NAMES:
        m = night["per_class"][cls]
        print(
            f"    night {cls:14s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  n={m['support']}"
        )

    return result


def paired_bootstrap(labels, preds_a, preds_b, metric_fn, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        va = metric_fn(labels[idx], preds_a[idx])
        vb = metric_fn(labels[idx], preds_b[idx])
        diffs[i] = vb - va
    return {
        "mean": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def _night_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _night_res_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=2, zero_division=0)


def summarize():
    result_files = sorted(_OUT_DIR.glob("results_*.json"))
    if not result_files:
        print(f"No results in {_OUT_DIR}")
        return 1

    results: dict = {}
    for f in result_files:
        d = json.loads(f.read_text())
        results.setdefault(d["arm"], {})[d["seed"]] = d

    arms = sorted(results.keys())
    print(f"\n{'=' * 100}\nSUMMARY — arms {arms}\n{'=' * 100}")

    for arm in arms:
        print(f"\nArm {arm} (seeds: {sorted(results[arm].keys())})")
        for seed in sorted(results[arm].keys()):
            r = results[arm][seed]
            print(f"  seed {seed}:")
            print(f"    night macro-F1:            {r['night_macro_f1']:.4f}  ← PRIMARY")
            print(f"    night residential recall:  {r['night_residential_recall']:.4f}")
            print(f"    day macro-F1:              {r['day_macro_f1']:.4f}")
            print(f"    test macro-F1:             {r['test_macro_f1']:.4f}")
            for cls in WEEK5_CLASS_NAMES:
                m = r["night"]["per_class"][cls]
                print(
                    f"    night {cls:14s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                    f"F1={m['f1']:.3f}  n={m['support']}"
                )

    if len(arms) < 2:
        print("\nOnly 1 arm available. Run another arm, then re-summarize.")
        return 0

    if "A" not in results:
        print("\nArm A missing — cannot bootstrap comparisons.")
        return 0

    seed_a = sorted(results["A"].keys())[0]
    labels = np.array(results["A"][seed_a]["night"]["labels"])
    preds_a = np.array(results["A"][seed_a]["night"]["predictions"])

    # Sanity: all arms must use same night labels (same image set, same order)
    for arm in arms:
        for seed in results[arm]:
            arm_labels = np.array(results[arm][seed]["night"]["labels"])
            if not np.array_equal(arm_labels, labels):
                print(
                    f"⚠️  WARNING: {arm} seed {seed} night labels differ from A. "
                    f"Bootstrap results will be invalid."
                )

    print(f"\n{'=' * 100}")
    print(f"Paired bootstrap (seed {seed_a}, night set, {N_BOOTSTRAP} resamples)")
    print(f"{'=' * 100}")

    for arm in arms:
        if arm == "A":
            continue
        seed = sorted(results[arm].keys())[0]
        preds_b = np.array(results[arm][seed]["night"]["predictions"])
        if len(preds_b) != len(labels):
            print(f"  Arm {arm}: prediction count {len(preds_b)} != {len(labels)}, skipping")
            continue
        mf1 = paired_bootstrap(labels, preds_a, preds_b, _night_macro_f1)
        rec = paired_bootstrap(labels, preds_a, preds_b, _night_res_recall)
        flag_mf1 = "EXCLUDES ZERO" if (mf1["ci_low"] > 0 or mf1["ci_high"] < 0) else "includes zero"
        flag_rec = "EXCLUDES ZERO" if (rec["ci_low"] > 0 or rec["ci_high"] < 0) else "includes zero"
        print(
            f"\n  {arm} − A (night macro-F1): Δ={mf1['mean']:+.4f}  "
            f"CI [{mf1['ci_low']:+.4f}, {mf1['ci_high']:+.4f}]  {flag_mf1}"
        )
        print(
            f"  {arm} − A (night residential recall): Δ={rec['mean']:+.4f}  "
            f"CI [{rec['ci_low']:+.4f}, {rec['ci_high']:+.4f}]  {flag_rec}"
        )

    if "C" in results:
        seed_c = sorted(results["C"].keys())[0]
        c_night = results["C"][seed_c]["night_macro_f1"]
        a_day = results["A"][seed_a]["day_macro_f1"]
        c_day = results["C"][seed_c]["day_macro_f1"]

        print(f"\n{'=' * 100}")
        print("DECISION RULE (pre-registered)")
        print(f"{'=' * 100}")
        print(f"  Arm C night macro-F1: {c_night:.4f}")
        print(f"  Thresholds: ≥{C_STOP} stop / [{C_PROCEED}, {C_STOP}) proceed / <{C_PROCEED} stop")
        if c_night >= C_STOP:
            print(f"\n  → ARM C ≥ {C_STOP}: photometric closes the gap.")
            print("    ACTION: Stop generative plan. Document. No Phase 2.")
        elif c_night >= C_PROCEED:
            print(f"\n  → ARM C in [{C_PROCEED}, {C_STOP}): partial, gap remains.")
            print("    ACTION: Proceed to Phase 2 (Wan + CLIP + arm D).")
        else:
            print(f"\n  → ARM C < {C_PROCEED}: photometric doesn't meaningfully help.")
            print("    ACTION: Stop. Document negative finding.")

        day_drop = a_day - c_day
        print(
            f"\n  GUARDRAIL: day macro-F1 drop (A − C) = {day_drop:+.4f}  "
            f"(threshold ≤ {DAY_GUARDRAIL})"
        )
        if day_drop > DAY_GUARDRAIL:
            print(
                f"    ⚠️  day macro-F1 dropped by {day_drop:.4f} — "
                f"arm C may be trading day for night"
            )

    (_OUT_DIR / "summary.json").write_text(
        json.dumps({a: {str(s): r for s, r in results[a].items()} for a in arms}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {_OUT_DIR / 'summary.json'}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "B", "C", "D"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        return summarize()
    if not args.arm:
        parser.error("--arm required unless --summarize")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA — training will be extremely slow")

    print("Loading image index...")
    image_index = index_images()
    print(f"  {len(image_index):,} images indexed")
    if not image_index:
        print(f"ERROR: no images under {_IMAGES}")
        return 1

    print("Loading labels...")
    train_records = load_records(_TRAIN_LABEL)
    val_records = load_records(_VAL_LABEL)
    val_clean_names = set(json.loads(_VAL_CLEAN.read_text(encoding="utf-8")))
    test_clean_names = set(json.loads(_TEST_CLEAN.read_text(encoding="utf-8")))

    assert len(val_clean_names) == EXPECTED_VAL_CLEAN, (
        f"val_clean manifest has {len(val_clean_names)} entries, expected {EXPECTED_VAL_CLEAN}"
    )
    assert len(test_clean_names) == EXPECTED_TEST_CLEAN, (
        f"test_clean manifest has {len(test_clean_names)} entries, expected {EXPECTED_TEST_CLEAN}"
    )

    full_train = select_split(train_records, None, image_index)
    val_clean = select_split(val_records, val_clean_names, image_index)
    assert len(val_clean) == EXPECTED_VAL_CLEAN, (
        f"after filtering, val_clean = {len(val_clean)}, expected {EXPECTED_VAL_CLEAN}. "
        f"Some images may be missing on disk."
    )
    print(f"  train: {len(full_train):,}  val: {len(val_clean):,}")

    night_loader, night_n = build_restricted(val_records, test_clean_names, image_index, "night")
    day_loader, day_n = build_restricted(val_records, test_clean_names, image_index, "daytime")
    test_loader, test_n = build_restricted(val_records, test_clean_names, image_index, None)
    assert test_n == EXPECTED_TEST_CLEAN, f"full test = {test_n}, expected {EXPECTED_TEST_CLEAN}"
    print(f"  night: {night_n}  day: {day_n}  full test: {test_n}")

    synthetic = sorted(_SYNTHETIC.glob("*.png")) if _SYNTHETIC.exists() else []
    print(f"  synthetic frames: {len(synthetic)}")

    result = train_arm(
        args=args,
        full_train=full_train,
        val_clean=val_clean,
        night_loader=night_loader,
        day_loader=day_loader,
        test_loader=test_loader,
        synthetic=synthetic,
        device=device,
    )
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
