"""Audit the 276 curated CycleGAN images: Luminance, diversity, and class content."""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GEN_DIR = _REPO_ROOT / "outputs" / "method_2_synthetic"
_DATA_ROOT_ENV = os.environ.get("URBANINTEL_DATA_ROOT")
if not _DATA_ROOT_ENV:
    raise RuntimeError("URBANINTEL_DATA_ROOT environment variable is not set. See .env.example")
_DATA_ROOT = Path(_DATA_ROOT_ENV) / "bdd100k"
_MANIFEST = _GEN_DIR / "curated_manifest.json"
_FILTER_LOG = _GEN_DIR / "filter_log.csv"


def main():
    manifest = json.loads(_MANIFEST.read_text())
    keep_labels = set(manifest["final_keep_labels"])

    import csv

    with open(_FILTER_LOG) as f:
        log = list(csv.DictReader(f))
    accepted_sorted = sorted([r for r in log if r["kept"] == "1"], key=lambda r: r["output"])

    filenames = []
    for label in sorted(keep_labels):
        # label is 1-based index into the accepted_sorted list
        filenames.append(accepted_sorted[label - 1]["output"])

    # Verify integrity
    assert len(filenames) == 276, f"Expected 276 filenames, got {len(filenames)}"
    assert len(set(filenames)) == 276, "Duplicate filenames in manifest"
    for f in filenames:
        assert (_GEN_DIR / f).exists(), f"Missing file: {f}"

    print(f"Auditing {len(filenames)} curated images.")
    device = "cpu"  # Explicitly CPU-only for audit

    # 1. Luminance Check
    print("\n[1/3] Checking luminance...")
    lums = []
    for fname in filenames:
        img = Image.open(_GEN_DIR / fname).convert("L")
        arr = np.array(img).astype(float) / 255.0
        lums.append(arr.mean())

    print(
        f"  Luminance: min={min(lums):.3f} max={max(lums):.3f} mean={sum(lums) / len(lums):.3f} std={np.std(lums):.3f}"
    )

    # 2. Diversity and Memorization Check
    print("\n[2/3] Checking visual diversity & memorization...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()
    resnet.eval().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load synthetic
    syn_feats = []
    with torch.no_grad():
        for fname in filenames:
            img = Image.open(_GEN_DIR / fname).convert("RGB")
            syn_feats.append(resnet(transform(img).unsqueeze(0).to(device)).cpu().numpy().flatten())
    syn_feats = np.array(syn_feats)

    # Load 50 real nighttime residential training images for comparison
    # (Using the Method 2 split manifest to find real night images)
    m2_manifest = json.loads(
        (_REPO_ROOT / "outputs" / "method_2_resume" / "split_manifest.json").read_text()
    )
    real_night_names = m2_manifest["night_train"][:50]

    real_feats = []
    with torch.no_grad():
        for name in real_night_names:
            p = list(_DATA_ROOT.rglob(name))[0]
            img = Image.open(p).convert("RGB")
            real_feats.append(
                resnet(transform(img).unsqueeze(0).to(device)).cpu().numpy().flatten()
            )
    real_feats = np.array(real_feats)

    sim_syn = cosine_similarity(syn_feats)
    np.fill_diagonal(sim_syn, 0)
    sim_cross = cosine_similarity(syn_feats, real_feats)

    print(f"  Synthetic-to-Synthetic Cosine Mean: {sim_syn.mean():.3f}")
    print(
        f"  Synthetic-to-Real Cosine Mean:      {sim_cross.mean():.3f} (max={sim_cross.max():.3f})"
    )
    if sim_cross.max() > 0.95:
        print("  ⚠️ WARNING: Potential memorization detected (max cross-sim > 0.95)")
    else:
        print("  ✅ No extreme memorization detected.")

    # 3. SegFormer Content Check
    print("\n[3/3] Checking structural content (SegFormer)...")
    seg_proc = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    seg_mod = (
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        .to(device)
        .eval()
    )

    # ADE classes: road=6, building=1, sky=2, vegetation=4
    class_counts = {"road": 0, "building": 0, "sky": 0, "veg": 0, "total_px": 0}

    with torch.no_grad():
        for fname in filenames:
            img = Image.open(_GEN_DIR / fname).convert("RGB")
            inp = seg_proc(images=img, return_tensors="pt").to(device)
            out = seg_mod(**inp).logits
            up = torch.nn.functional.interpolate(
                out, size=(288, 512), mode="bilinear", align_corners=False
            )
            pred = up.argmax(dim=1).squeeze(0).cpu().numpy()

            class_counts["total_px"] += pred.size
            class_counts["road"] += (pred == 6).sum()
            class_counts["building"] += (pred == 1).sum()
            class_counts["sky"] += (pred == 2).sum()
            class_counts["veg"] += (pred == 4).sum()

    for k in ["road", "building", "sky", "veg"]:
        print(f"  {k:8s}: {100 * class_counts[k] / class_counts['total_px']:.1f}% of pixels")


if __name__ == "__main__":
    main()
