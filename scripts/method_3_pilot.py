"""Method 3 Phase 3: The 30-output generation pilot.

Loads 5 frozen source images + generates depth maps,
then runs the frozen pilot grid: 5 sources × 2 strengths × 3 scales = 30 outputs.

Uses CUDA-only mode (Phase 2 proved VRAM feasibility).
Protects Phase 1 and Phase 2 evidence.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import diffusers
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)
from dotenv import load_dotenv
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "outputs" / "method_3_pilot"
_GEN_DIR = _OUT_DIR / "generated"
_DEPTH_DIR = _OUT_DIR / "depth"

# Frozen config
PROMPT = (
    "nighttime residential street, dark sky, streetlights, headlights, "
    "dashcam view, original scene geometry preserved"
)
NEGATIVE_PROMPT = "daytime, sunny, hallucinated objects, distorted structure, blurry, cartoon"
STEPS = 30
GUIDANCE = 6.0
SEED = 42
RESOLUTION = (512, 288)

# Frozen pilot grid
SOURCES = 5
STRENGTHS = [0.60, 0.75]
SCALES = [0.6, 0.8, 1.0]

# Data paths (from environment, matching src/urbangen/config.py)


load_dotenv()
_DATA_ROOT_ENV = os.environ.get("URBANINTEL_DATA_ROOT")
_URBANMIND_ENV = os.environ.get("URBANMIND_REPO")
if not _DATA_ROOT_ENV or not _URBANMIND_ENV:
    raise RuntimeError("URBANINTEL_DATA_ROOT and URBANMIND_REPO must be set (see .env.example)")

DATA_ROOT = Path(_DATA_ROOT_ENV) / "bdd100k"
IMAGES = DATA_ROOT / "images" / "100k" / "train"
URBANMIND = Path(_URBANMIND_ENV)


def _protect_artifacts() -> None:
    """Refuse to overwrite existing Phase 1/2/3 artifacts."""
    for p in (
        _OUT_DIR / "smoke_test_source.png",
        _OUT_DIR / "smoke_test_depth.png",
        _OUT_DIR / "phase1_source_selection.json",
        _OUT_DIR / "phase1_report.json",
        _OUT_DIR / "smoke_generation.png",
        _OUT_DIR / "smoke_comparison.png",
        _OUT_DIR / "smoke_generation_report.json",
    ):
        if not p.exists():
            raise RuntimeError(f"Required artifact missing: {p}")

    manifest_path = _OUT_DIR / "pilot_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(
            f"Phase 3 pilot already exists at {manifest_path}. "
            f"Move or delete {_GEN_DIR}, {_DEPTH_DIR}, and {manifest_path} before retrying."
        )


def _select_sources() -> list[Path]:
    """Select 5 pilot sources matching Phase 1's procedure, excluding the Phase 1 source."""
    import random

    # Read Phase 1's actual source filename from frozen artifact
    phase1_selection = json.loads((_OUT_DIR / "phase1_source_selection.json").read_text())
    phase1_source = phase1_selection["selected_filename"]
    phase1_seed = phase1_selection["seed"]
    if phase1_seed != 999:
        raise RuntimeError(f"Phase 1 seed changed unexpectedly: {phase1_seed}")

    with open(DATA_ROOT / "labels" / "bdd100k_labels_images_train.json") as f:
        records = json.load(f)

    m2_manifest = json.loads(
        (_REPO_ROOT / "outputs" / "method_2_resume" / "split_manifest.json").read_text()
    )
    m2_excluded = set()
    for key in ["day_train", "day_holdout", "night_train", "night_holdout"]:
        m2_excluded.update(m2_manifest[key])

    val_clean = set(
        json.loads((URBANMIND / "data" / "splits" / "resnet_val_clean.json").read_text())
    )
    test_clean = set(
        json.loads((URBANMIND / "data" / "splits" / "resnet_test_clean.json").read_text())
    )
    excluded = m2_excluded | val_clean | test_clean | {phase1_source}

    image_index = {}
    for p in IMAGES.rglob("*.jpg"):
        if p.name in image_index:
            raise RuntimeError(f"Duplicate filename: {p.name}")
        image_index[p.name] = p

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
    rng = random.Random(phase1_seed + 1)  # deterministic but distinct stream
    pilot_names = rng.sample(eligible, 5)
    print(f"Phase 1 source (excluded): {phase1_source}")
    print(f"Pilot sources (5): {pilot_names}")
    return [image_index[name] for name in pilot_names]


def _generate_depth(image_path: Path, out_path: Path, processor, model) -> Image.Image:
    """Run DPT on one source image."""
    img = Image.open(image_path).convert("RGB").resize(RESOLUTION, Image.LANCZOS)
    inputs = processor(images=img, return_tensors="pt").to("cuda", dtype=torch.float16)
    with torch.no_grad():
        depth_out = model(**inputs)
    depth = depth_out.predicted_depth.squeeze().cpu().float().numpy()
    d_min, d_max = float(depth.min()), float(depth.max())
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_img = Image.fromarray((depth_norm * 255).astype("uint8")).resize(
        RESOLUTION, Image.LANCZOS
    )
    depth_img.save(out_path)
    return depth_img


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Phase 3")

    _protect_artifacts()

    # Check output dirs are empty before starting
    for d in (_GEN_DIR, _DEPTH_DIR):
        if d.exists() and any(d.iterdir()):
            raise RuntimeError(f"Phase 3 output directory is not empty: {d}. Clean before running.")

    _GEN_DIR.mkdir(parents=True, exist_ok=True)
    _DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Free VRAM at start: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    _phase1 = json.loads((_OUT_DIR / "phase1_source_selection.json").read_text())
    phase1_source_name = _phase1["selected_filename"]

    # Step 1: Select sources
    sources = _select_sources()
    if len(sources) != SOURCES:
        raise RuntimeError(f"Expected {SOURCES} sources, got {len(sources)}")

    # Step 2: Generate depth maps
    print("\n[1/3] Loading DPT-MiDaS for depth extraction...")
    dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(
        "cuda", dtype=torch.float16
    )

    depth_maps = {}
    for i, src_path in enumerate(sources, 1):
        depth_path = _DEPTH_DIR / f"{src_path.stem}_depth.png"
        print(f"  [{i}/5] Extracting depth: {src_path.name}")
        depth_maps[src_path.stem] = _generate_depth(src_path, depth_path, dpt_processor, dpt_model)

    del dpt_model, dpt_processor
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  DPT released. Free VRAM: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    # Step 3: Load SD 1.5 + ControlNet
    print("\n[2/3] Loading SD 1.5 + ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
        revision="539f99181d33db39cf1af2e517cd8056785f0a87",
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        revision="451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Step 4: Run the 30-output grid
    print("\n[3/3] Running 30-output pilot grid...")
    manifest = {
        "phase": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _get_git_sha(),
        "gpu": torch.cuda.get_device_name(0),
        "environment": {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "cuda": torch.version.cuda,
        },
        "models": {
            "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "base_model_commit": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            "controlnet": "lllyasviel/control_v11f1p_sd15_depth",
            "controlnet_commit": "539f99181d33db39cf1af2e517cd8056785f0a87",
        },
        "config": {
            "steps": STEPS,
            "guidance": GUIDANCE,
            "seed": SEED,
            "resolution": list(RESOLUTION),
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "strengths": STRENGTHS,
            "scales": SCALES,
        },
        "source_selection": {
            "phase1_source_excluded": phase1_source_name,
            "pilot_sources": [s.name for s in sources],
            "seed": 999,
            "rng_seed": 1000,
            "procedure": "Phase 1 exclusion + independently seeded deterministic pilot selection (random.Random(1000).sample(5) over sorted eligible)",
        },
        "sources": [],
        "outputs": [],
        "status": "in_progress",
        "completed_count": 0,
        "total_count": SOURCES * len(STRENGTHS) * len(SCALES),
    }

    output_count = 0
    total_count = SOURCES * len(STRENGTHS) * len(SCALES)

    for src_path in sources:
        src_img = Image.open(src_path).convert("RGB").resize(RESOLUTION, Image.LANCZOS)
        depth_rgb = depth_maps[src_path.stem].convert("RGB")
        manifest["sources"].append(
            {
                "filename": src_path.name,
                "sha256": hashlib.sha256(src_path.read_bytes()).hexdigest(),
                "depth_path": str(_DEPTH_DIR / f"{src_path.stem}_depth.png"),
            }
        )

        for strength in STRENGTHS:
            for scale in SCALES:
                output_count += 1
                out_name = f"{src_path.stem}_s{int(strength * 100)}_c{int(scale * 100)}.png"
                out_path = _GEN_DIR / out_name

                print(
                    f"  [{output_count}/{total_count}] {src_path.stem}  "
                    f"strength={strength}  scale={scale}  →  {out_name}",
                    flush=True,
                )

                torch.cuda.reset_peak_memory_stats()
                generator = torch.Generator(device="cuda").manual_seed(SEED)
                t0 = time.time()

                with torch.inference_mode():
                    result = pipe(
                        prompt=PROMPT,
                        negative_prompt=NEGATIVE_PROMPT,
                        image=src_img,
                        control_image=depth_rgb,
                        strength=strength,
                        controlnet_conditioning_scale=scale,
                        num_inference_steps=STEPS,
                        guidance_scale=GUIDANCE,
                        generator=generator,
                    )

                torch.cuda.synchronize()
                elapsed = round(time.time() - t0, 2)
                peak_alloc = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                peak_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 2)

                result.images[0].save(out_path)
                manifest["outputs"].append(
                    {
                        "filename": out_name,
                        "source": src_path.name,
                        "source_sha256": manifest["sources"][-1]["sha256"]
                        if manifest["sources"][-1]["filename"] == src_path.name
                        else next(
                            s["sha256"]
                            for s in manifest["sources"]
                            if s["filename"] == src_path.name
                        ),
                        "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
                        "strength": strength,
                        "scale": scale,
                        "executed_steps": int(STEPS * strength),
                        "runtime_seconds": elapsed,
                        "peak_allocated_gb": peak_alloc,
                        "peak_reserved_gb": peak_reserved,
                    }
                )
                manifest["completed_count"] = len(manifest["outputs"])
                # Save manifest after every generation (crash-recovery safe)
                (_OUT_DIR / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Final integrity validation
    print("\n[Validation] Verifying pilot completeness...")
    if len(manifest["outputs"]) != total_count:
        raise RuntimeError(
            f"Expected {total_count} outputs, manifest has {len(manifest['outputs'])}"
        )
    if manifest["completed_count"] != total_count:
        raise RuntimeError(f"completed_count={manifest['completed_count']} != {total_count}")
    on_disk = sorted(p.name for p in _GEN_DIR.glob("*.png"))
    in_manifest = sorted(o["filename"] for o in manifest["outputs"])
    if on_disk != in_manifest:
        missing = set(in_manifest) - set(on_disk)
        extra = set(on_disk) - set(in_manifest)
        raise RuntimeError(f"Disk/manifest mismatch. Missing: {missing}, Extra: {extra}")
    print(f"  ✓ {total_count} outputs on disk, all manifest-registered")

    # Clean up
    del pipe, controlnet
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    manifest["status"] = "complete"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest_path = _OUT_DIR / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    peak_max = max(o["peak_reserved_gb"] for o in manifest["outputs"])
    print(f"\n✅ Phase 3 pilot complete: {output_count}/{total_count} outputs generated")
    print(f"  Max peak reserved VRAM: {peak_max:.2f} GB")
    print(f"  Manifest: {manifest_path}")
    print(f"  Generated images: {_GEN_DIR}")
    print(f"  Depth maps: {_DEPTH_DIR}")
    print("\nNext: inspect the 30 outputs and apply the human-inspection gate")
    print("  (5 of 30 must be plausible night scenes, from at least 3 sources)")


if __name__ == "__main__":
    main()
