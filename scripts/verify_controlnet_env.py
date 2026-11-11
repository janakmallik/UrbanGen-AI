"""Phase 1: Verify ControlNet Depth + DPT-MiDaS load and run on 6 GB VRAM.

Does NOT generate a full image yet — only verifies:
  - ControlNet weights download and load
  - DPT-MiDaS weights download and load
  - Depth extraction runs on a real image
  - Peak VRAM stays under 6 GB
"""

import hashlib
import json
import random
import subprocess
from pathlib import Path

import diffusers
import numpy as np
import torch
import transformers
from diffusers import ControlNetModel
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor

# ---------------------------------------------------------------------------
# Fix 1 — Source image selection (verified daytime residential)
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")
IMAGES = DATA_ROOT / "images" / "100k" / "train"

with open(DATA_ROOT / "labels" / "bdd100k_labels_images_train.json") as f:
    records = json.load(f)

# Build image index FIRST (needed by exclusion checks below)
image_index = {}
_dup_names = []
for p in IMAGES.rglob("*.jpg"):
    if p.name in image_index:
        _dup_names.append(p.name)
    else:
        image_index[p.name] = p
assert not _dup_names, (
    f"Duplicate image filenames in tree: {len(_dup_names)} (e.g. {_dup_names[:5]})"
)
print(f"Image index: {len(image_index)} unique filenames")

# Build excluded set (Method 2 splits + evaluation holdouts)
URBANMIND = Path(r"C:\Users\Janak Mallik\UrbanMind-AI")
_REPO_ROOT = Path(__file__).resolve().parents[1]
m2_manifest = json.loads(
    (_REPO_ROOT / "outputs/method_2_resume" / "split_manifest.json").read_text()
)
# Validate schema FIRST, before accessing keys
assert isinstance(m2_manifest, dict), "Method 2 manifest is not a dict"
assert set(m2_manifest.keys()) == {"day_train", "day_holdout", "night_train", "night_holdout"}, (
    f"Unexpected Method 2 keys: {m2_manifest.keys()}"
)
for key in m2_manifest:
    assert isinstance(m2_manifest[key], list), f"{key} is not a list"
    assert len(set(m2_manifest[key])) == len(m2_manifest[key]), f"{key} has duplicate filenames"

# Now safe to access keys
m2_excluded = set()
for key in ["day_train", "day_holdout", "night_train", "night_holdout"]:
    m2_excluded.update(m2_manifest[key])

m2_union = set().union(*m2_manifest.values())
print(f"Method 2 manifest: 4 keys, total unique filenames = {len(m2_union)}")
# Note: len(m2_union) may be < 3400 if the four lists overlap. Verify before
# asserting 3400. If overlap exists, the "3,400 reserved" claim is wrong.

# Verify the four Method 2 splits are mutually disjoint
_m2_keys = list(m2_manifest.keys())
for _i, _ka in enumerate(_m2_keys):
    for _kb in _m2_keys[_i + 1 :]:
        _overlap = set(m2_manifest[_ka]) & set(m2_manifest[_kb])
        assert not _overlap, f"Method 2 split overlap {_ka}/{_kb}: {len(_overlap)} images"
print("Method 2 splits: mutually disjoint (verified)")

# Exclude validation and test holdouts (frozen spec requirement)
_val_raw = json.loads((URBANMIND / "data" / "splits" / "resnet_val_clean.json").read_text())
_test_raw = json.loads((URBANMIND / "data" / "splits" / "resnet_test_clean.json").read_text())
# Schema verification: both should be flat lists of strings
assert isinstance(_val_raw, list), f"val_clean is {type(_val_raw).__name__}, expected list"
assert isinstance(_test_raw, list), f"test_clean is {type(_test_raw).__name__}, expected list"
assert all(isinstance(x, str) for x in _val_raw), "val_clean contains non-string entries"
assert all(isinstance(x, str) for x in _test_raw), "test_clean contains non-string entries"
val_clean = set(_val_raw)
test_clean = set(_test_raw)
print(
    f"Schema OK: val_clean list[str] n={len(val_clean)}, test_clean list[str] n={len(test_clean)}"
)
excluded = m2_excluded | val_clean | test_clean
# Verify all manifest entries are bare filenames, not relative paths
for _label, _names in [
    ("Method 2", m2_excluded),
    ("validation", val_clean),
    ("test", test_clean),
]:
    assert all(Path(name).name == name for name in _names), (
        f"{_label} manifest contains paths rather than bare filenames"
    )
print("Manifest filename format: verified (all bare filenames)")
# Load all four UrbanMind split files for complete coverage
_val_excluded_raw = json.loads(
    (URBANMIND / "data" / "splits" / "resnet_val_excluded.json").read_text()
)
_test_excluded_raw = json.loads(
    (URBANMIND / "data" / "splits" / "resnet_test_excluded.json").read_text()
)
val_excluded = set(_val_excluded_raw)
test_excluded = set(_test_excluded_raw)
print(
    f"Full manifests: val_clean={len(val_clean)}, val_excluded={len(val_excluded)}, "
    f"test_clean={len(test_clean)}, test_excluded={len(test_excluded)}"
)

# Method 2 filenames must be in the train inventory (they're train-split samples)
_unknown_m2 = m2_excluded - set(image_index.keys())
if _unknown_m2:
    raise RuntimeError(
        f"Method 2 manifest contains {len(_unknown_m2)} filenames not in BDD100K train inventory "
        f"(e.g. {list(_unknown_m2)[:5]})"
    )

# All four holdout manifests must be in BDD100K val inventory
VAL_IMAGES = DATA_ROOT / "images" / "100k" / "val"
val_inventory = {p.name for p in VAL_IMAGES.rglob("*.jpg")}
if not val_inventory:
    raise RuntimeError(f"BDD100K val inventory is empty at {VAL_IMAGES}")
print(f"BDD100K val inventory: {len(val_inventory)} images")

for _label, _names in [
    ("val_clean", val_clean),
    ("val_excluded", val_excluded),
    ("test_clean", test_clean),
    ("test_excluded", test_excluded),
]:
    _unknown = _names - val_inventory
    if _unknown:
        raise RuntimeError(
            f"{_label} manifest contains {len(_unknown)} filenames not in BDD100K val inventory "
            f"(e.g. {list(_unknown)[:5]})"
        )

# Verify the full 10K val inventory is entirely disjoint from the 70K train inventory
_full_val_overlap = val_inventory & set(image_index.keys())
if _full_val_overlap:
    raise RuntimeError(
        f"LEAKAGE: {len(_full_val_overlap)} BDD100K val filenames also appear in train inventory "
        f"(e.g. {list(_full_val_overlap)[:5]})"
    )
print(
    "Full BDD100K val inventory (10K) verified entirely disjoint from BDD100K train inventory"
)
print(f"Excluded pool: {len(m2_excluded)} Method 2 + {len(val_clean)} val + {len(test_clean)} test")

# Verify all 6 split groups are mutually disjoint
_all_splits = {
    **m2_manifest,
    "validation_clean": list(val_clean),
    "validation_excluded": list(val_excluded),
    "test_clean": list(test_clean),
    "test_excluded": list(test_excluded),
}
_split_names = list(_all_splits)
for _i, _a in enumerate(_split_names):
    for _b in _split_names[_i + 1 :]:
        _leak = set(_all_splits[_a]) & set(_all_splits[_b])
        assert not _leak, f"Split leakage {_a}/{_b}: {len(_leak)} images"
print("All 6 split groups mutually disjoint: verified")

# Filter to daytime residential, exclude Method 2
eligible = []
for rec in records:
    attrs = rec.get("attributes", {})
    if (
        attrs.get("scene") == "residential"
        and attrs.get("timeofday") == "daytime"
        and rec["name"] not in excluded
        and rec["name"] in image_index
    ):
        eligible.append(rec["name"])

eligible.sort()
if not eligible:
    raise RuntimeError(
        f"No eligible daytime residential images found. "
        f"Checked {len(records)} records, {len(m2_excluded)} excluded from Method 2, "
        f"{len(val_clean)} from val, {len(test_clean)} from test."
    )
print(f"Eligible pool: {len(eligible)} daytime residential images")
rng = random.Random(999)
sample_name = rng.choice(eligible)
sample = image_index[sample_name]
assert sample.is_file(), f"Selected sample does not exist: {sample}"
assert sample_name not in excluded, f"Selected sample is in excluded set: {sample_name}"
# Persist selected filename BEFORE loading models so retry-after-failure uses same source
output_dir = _REPO_ROOT / "outputs" / "method_3_pilot"
output_dir.mkdir(parents=True, exist_ok=True)


_source_sha256 = hashlib.sha256(sample.read_bytes()).hexdigest()
selection_record = {
    "seed": 999,
    "eligible_pool_size": len(eligible),
    "selected_filename": sample_name,
    "selected_sha256": _source_sha256,
    "excluded_method2": len(m2_excluded),
    "excluded_val": len(val_clean),
    "excluded_test": len(test_clean),
}
(output_dir / "phase1_source_selection.json").write_text(json.dumps(selection_record, indent=2))
print(f"Source selection saved: {output_dir / 'phase1_source_selection.json'}")
# ---------------------------------------------------------------------------
# Fix 2 — Load DPT, run depth, release before loading ControlNet
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Cannot perform GPU verification.")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

dpt_baseline = torch.cuda.memory_allocated() / 1024**3
_free_vram_before_dpt = torch.cuda.mem_get_info()[0] / 1024**3
torch.cuda.reset_peak_memory_stats()
print(f"  Free VRAM before DPT: {_free_vram_before_dpt:.2f} GB")

print("\n[1/3] Loading DPT-Hybrid-MiDaS...")
depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(
    "cuda", dtype=torch.float16
)
dpt_commit_sha = getattr(depth_model.config, "_commit_hash", None) or "unresolved"
print(f"  Depth model loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

print("\n[2/3] Running depth extraction...")
img = Image.open(sample).convert("RGB").resize((512, 288), Image.LANCZOS)
inputs = depth_processor(images=img, return_tensors="pt").to("cuda", dtype=torch.float16)

with torch.no_grad():
    depth_output = depth_model(**inputs)
torch.cuda.synchronize()

# DPT peak measured AFTER inference (includes activations)
dpt_peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
dpt_peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
print(
    f"  DPT peak after inference: allocated={dpt_peak_alloc:.2f} GB, reserved={dpt_peak_reserved:.2f} GB"
)
print(f"  DPT incremental: {dpt_peak_alloc - dpt_baseline:.2f} GB")

depth = depth_output.predicted_depth

# ---------------------------------------------------------------------------
# Fix 3 — Raw depth range (before normalization)
# Fix 4 — Depth map polarity (save source alongside)
# ---------------------------------------------------------------------------
depth_np_raw = depth.squeeze().cpu().float().numpy()
depth_min = float(depth_np_raw.min())
depth_max = float(depth_np_raw.max())
if not np.isfinite(depth_np_raw).all():
    raise RuntimeError("Depth output contains NaN or infinite values.")
if depth_max - depth_min < 1e-8:
    raise RuntimeError("Depth output is effectively constant.")
depth_np = (depth_np_raw - depth_min) / (depth_max - depth_min)
# Resize predicted depth visualization to the target output dimensions.
depth_img = Image.fromarray((depth_np * 255).astype(np.uint8)).resize((512, 288), Image.LANCZOS)

output_dir = _REPO_ROOT / "outputs" / "method_3_pilot"
output_dir.mkdir(parents=True, exist_ok=True)

# MiDaS predicts INVERSE depth: brighter pixels = NEAR objects, darker = FAR.
# Compare side-by-side with the source image to verify polarity is correct.
img.save(output_dir / "smoke_test_source.png")
depth_img.save(output_dir / "smoke_test_depth.png")

# Diagnostic on RAW values (before normalization)
print(
    f"✅ Raw depth: min={depth_np_raw.min():.4f}, max={depth_np_raw.max():.4f}, "
    f"mean={depth_np_raw.mean():.4f}"
)
print("✅ Normalized to [0, 1] for visualization")
print(f"✅ Source + depth saved to {output_dir}/")

# Release DPT: explicit del of all GPU-held tensors
del depth_model, depth_output, depth, inputs
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"  DPT released. VRAM after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Safeguard B: reset peak stats so ControlNet measurement is isolated
cn_baseline = torch.cuda.memory_allocated() / 1024**3
_free_vram_before_cn = torch.cuda.mem_get_info()[0] / 1024**3
torch.cuda.reset_peak_memory_stats()
print(f"  Free VRAM before ControlNet: {_free_vram_before_cn:.2f} GB")

# ---------------------------------------------------------------------------
# Fix 5 — No controlnet.enable_attention_slicing() anywhere (pipeline method)
# ---------------------------------------------------------------------------
print("\n[3/3] Loading ControlNet Depth...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16,
).to("cuda")
cn_commit_sha = getattr(controlnet.config, "_commit_hash", None) or "unresolved"
torch.cuda.synchronize()
cn_peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
cn_peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
print(f"  ControlNet loaded. allocated={cn_peak_alloc:.2f} GB, reserved={cn_peak_reserved:.2f} GB")
print(f"  ControlNet incremental: {cn_peak_alloc - cn_baseline:.2f} GB")

print("\nPeak VRAM per component:")
print(f"  DPT:          {dpt_peak_alloc:.2f} GB allocated, {dpt_peak_reserved:.2f} GB reserved")
print(f"  ControlNet:   {cn_peak_alloc:.2f} GB allocated, {cn_peak_reserved:.2f} GB reserved")

# Release ControlNet
del controlnet
torch.cuda.empty_cache()
torch.cuda.synchronize()
_free_after_cleanup = torch.cuda.mem_get_info()[0] / 1024**3
print(f"  Free VRAM after full cleanup: {_free_after_cleanup:.2f} GB (should approach {_free_vram_before_dpt:.2f})")

# Available VRAM pass/fail gate — allocated AND reserved must both fit.
# Allocated = tensors actively in use. Reserved = PyTorch's memory pool (includes fragmentation).
# Neither equals total GPU usage (display + other processes take headroom).
_total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
_safe_limit_dpt = _free_vram_before_dpt * 0.95
_safe_limit_cn = _free_vram_before_cn * 0.95
print("\nVRAM gate thresholds (5% headroom):")
print(f"  DPT:        free_before={_free_vram_before_dpt:.2f} GB, gate={_safe_limit_dpt:.2f} GB")
print(f"  ControlNet: free_before={_free_vram_before_cn:.2f} GB, gate={_safe_limit_cn:.2f} GB")

_dpt_ok = dpt_peak_alloc < _safe_limit_dpt and dpt_peak_reserved < _safe_limit_dpt
_cn_ok = cn_peak_alloc < _safe_limit_cn and cn_peak_reserved < _safe_limit_cn
print("\nAvailable VRAM gate:")
print(
    f"  DPT:        allocated={dpt_peak_alloc:.2f} GB  reserved={dpt_peak_reserved:.2f} GB  "
    f"{'PASS' if _dpt_ok else 'FAIL'}"
)
print(
    f"  ControlNet: allocated={cn_peak_alloc:.2f} GB  reserved={cn_peak_reserved:.2f} GB  "
    f"{'PASS' if _cn_ok else 'FAIL'}"
)


try:
    _git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
    ).strip()
except Exception:
    _git_sha = "unknown"

report = {
    "phase": 1,
    "git_sha": _git_sha,
    "environment": {
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    },
    "models": {
        "depth": {
            "id": "Intel/dpt-hybrid-midas",
            "commit_sha": dpt_commit_sha,
        },
        "controlnet": {
            "id": "lllyasviel/control_v11f1p_sd15_depth",
            "commit_sha": cn_commit_sha,
        },
    },
    "source_selection": {
        "seed": 999,
        "sample_filename": sample_name,
        "sample_sha256": _source_sha256,
        "eligible_pool_size": len(eligible),
        "excluded_method2": len(m2_excluded),
        "excluded_val": len(val_clean),
        "excluded_test": len(test_clean),
    },
    "vram_gb": {
        "note": (
            "allocated = active tensors; reserved = PyTorch pool (includes fragmentation); "
            "neither equals total GPU usage (display + other processes)"
        ),
        "total_vram_gb": round(_total_vram, 2),
        "free_vram_before_dpt_gb": round(_free_vram_before_dpt, 2),
        "free_vram_before_controlnet_gb": round(_free_vram_before_cn, 2),
        "safe_limit_dpt_gb": round(_safe_limit_dpt, 2),
        "safe_limit_controlnet_gb": round(_safe_limit_cn, 2),
        "dpt_peak_allocated": round(dpt_peak_alloc, 2),
        "dpt_peak_reserved": round(dpt_peak_reserved, 2),
        "dpt_incremental": round(dpt_peak_alloc - dpt_baseline, 2),
        "dpt_gate_pass": _dpt_ok,
        "controlnet_peak_allocated": round(cn_peak_alloc, 2),
        "controlnet_peak_reserved": round(cn_peak_reserved, 2),
        "controlnet_incremental": round(cn_peak_alloc - cn_baseline, 2),
        "controlnet_gate_pass": _cn_ok,
    },
    "artifacts": {
        "source_image": str(output_dir / "smoke_test_source.png"),
        "depth_map": str(output_dir / "smoke_test_depth.png"),
    },
}
report_path = output_dir / "phase1_report.json"
report_path.write_text(json.dumps(report, indent=2))
print(f"\n✅ Provenance report saved: {report_path}")

# Assert AFTER report is saved so failures leave evidence on disk
assert _dpt_ok, (
    f"DPT exceeded safe VRAM limit (allocated={dpt_peak_alloc:.2f}, "
    f"reserved={dpt_peak_reserved:.2f}, limit={_safe_limit_dpt:.2f})"
)
assert _cn_ok, (
    f"ControlNet exceeded safe VRAM limit (allocated={cn_peak_alloc:.2f}, "
    f"reserved={cn_peak_reserved:.2f}, limit={_safe_limit_cn:.2f})"
)

print("\nComponent verification PASSED.")
print("Proceed to Phase 2 only after reviewing the depth map.")
print("NOTE: This does NOT verify that SD 1.5 + ControlNet + DPT can run")
print("      together within 6 GB. That is Phase 2's job.")
