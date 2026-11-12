"""Method 3 Phase 2: Single-image smoke test.

Loads Phase 1's frozen source + depth map, generates ONE nighttime image
with SD 1.5 + Depth ControlNet, measures peak VRAM, and saves evidence.

Fallback chain (Q1):
  1. Normal CUDA inference
  2. On CUDA OOM → recreate pipeline, enable attention slicing, retry
  3. On CUDA OOM → recreate pipeline, enable sequential CPU offload, retry
  4. On CUDA OOM → FAIL, write report, preserve error

Protects Phase 1 artifacts: refuses to overwrite smoke_test_source.png,
smoke_test_depth.png, phase1_source_selection.json, phase1_report.json.

Records which execution mode succeeded in smoke_generation_report.json.
"""

import hashlib
import json
import time
from pathlib import Path

import diffusers
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "outputs" / "method_3_pilot"

# Frozen Phase 1 artifacts (do not overwrite)
_SOURCE = _OUT_DIR / "smoke_test_source.png"
_DEPTH = _OUT_DIR / "smoke_test_depth.png"
_PHASE1_SELECTION = _OUT_DIR / "phase1_source_selection.json"
_PHASE1_REPORT = _OUT_DIR / "phase1_report.json"

# Phase 2 outputs (new)
_GENERATED = _OUT_DIR / "smoke_generation.png"
_COMPARISON = _OUT_DIR / "smoke_comparison.png"
_REPORT = _OUT_DIR / "smoke_generation_report.json"

# Frozen config (from docs/method_3_experiment_spec.md)
PROMPT = (
    "nighttime residential street, dark sky, streetlights, headlights, "
    "dashcam view, original scene geometry preserved"
)
NEGATIVE_PROMPT = "daytime, sunny, hallucinated objects, distorted structure, blurry, cartoon"
STRENGTH = 0.75
CONTROLNET_SCALE = 1.0
STEPS = 30
GUIDANCE = 6.0
SEED = 42
RESOLUTION = (512, 288)

_SAFE_LIMIT_GB = 5.7  # leave 0.3 GB headroom below 6 GB physical


def _protect_phase1_artifacts() -> None:
    """Refuse to overwrite Phase 1 evidence."""
    for p in (_SOURCE, _DEPTH, _PHASE1_SELECTION, _PHASE1_REPORT):
        if not p.exists():
            raise RuntimeError(f"Phase 1 artifact missing: {p}")
        if p.stat().st_size == 0:
            raise RuntimeError(f"Phase 1 artifact is empty: {p}")
    if _GENERATED.exists() or _COMPARISON.exists() or _REPORT.exists():
        raise RuntimeError(
            "Phase 2 output already exists. Move or delete previous run before retrying."
        )


def _build_pipeline(controlnet: ControlNetModel, mode: str):
    """Construct pipeline with the requested memory strategy."""
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if mode == "cuda":
        pipe = pipe.to("cuda")
    elif mode == "attention_slicing":
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    elif mode == "sequential_cpu_offload":
        # MUST be configured BEFORE moving anything to CUDA
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    pipe.set_progress_bar_config(disable=False)
    return pipe


def _attempt_generation(mode: str, source: Image.Image, depth: Image.Image):
    """Try generation in one mode. Returns (image, peak_alloc, peak_reserved)."""
    pipe = None
    controlnet = None
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print(f"\n[Attempt] mode={mode}")
        print(
            f"  VRAM before load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, "
            f"{torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB free"
        )

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=torch.float16,
        )
        pipe = _build_pipeline(controlnet, mode)

        print(
            f"  Pipeline ready ({mode}). VRAM: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated"
        )

        generator = torch.Generator(device="cuda").manual_seed(SEED)

        with torch.inference_mode():
            result = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=source,
                control_image=depth,
                strength=STRENGTH,
                controlnet_conditioning_scale=CONTROLNET_SCALE,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=generator,
            )

        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        return result.images[0], peak_alloc, peak_reserved
    finally:
        if pipe is not None:
            del pipe
        if controlnet is not None:
            del controlnet
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _make_comparison(source: Image.Image, depth_rgb: Image.Image, generated: Image.Image):
    """3-panel side-by-side comparison with labels."""
    from PIL import ImageDraw

    w, h = source.size
    header = 28
    canvas = Image.new("RGB", (w * 3 + 20, h + header), (30, 30, 30))
    canvas.paste(source, (0, header))
    canvas.paste(depth_rgb, (w + 10, header))
    canvas.paste(generated, (w * 2 + 20, header))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), "SOURCE", fill=(255, 255, 255))
    draw.text((w + 18, 6), "DEPTH", fill=(255, 255, 255))
    draw.text((w * 2 + 28, 6), "GENERATED NIGHT", fill=(255, 255, 255))
    return canvas


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Phase 2")

    _protect_phase1_artifacts()

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Free VRAM at start: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    source = Image.open(_SOURCE).convert("RGB").resize(RESOLUTION, Image.Resampling.LANCZOS)
    depth_grayscale = Image.open(_DEPTH).convert("L").resize(RESOLUTION, Image.Resampling.LANCZOS)
    depth_rgb = Image.merge("RGB", [depth_grayscale, depth_grayscale, depth_grayscale])
    assert source.size == depth_rgb.size == RESOLUTION, (
        f"Source/depth/expected dimensions mismatch: "
        f"source={source.size}, depth={depth_rgb.size}, expected={RESOLUTION}"
    )

    # Attempt chain: cuda → attention_slicing → sequential_cpu_offload → fail
    modes = ["cuda", "attention_slicing", "sequential_cpu_offload"]
    last_error = None
    success_mode = None
    generated = None
    peak_alloc = None
    peak_reserved = None
    runtime = None

    attempt_history = []

    # Build report

    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _resolve_commit_sha(model_id: str) -> str:
        try:
            from huggingface_hub import scan_cache_dir

            for repo in scan_cache_dir().repos:
                if repo.repo_id == model_id:
                    for rev in repo.revisions:
                        return rev.commit_hash
        except Exception:
            pass
        return "unresolved"

    report = {
        "phase": 2,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "environment": {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "cuda": torch.version.cuda,
        },
        "models": {
            "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "base_model_commit": _resolve_commit_sha("stable-diffusion-v1-5/stable-diffusion-v1-5"),
            "controlnet": "lllyasviel/control_v11f1p_sd15_depth",
            "controlnet_commit": _resolve_commit_sha("lllyasviel/control_v11f1p_sd15_depth"),
        },
        "inputs": {
            "source_path": str(_SOURCE),
            "source_sha256": _sha256(_SOURCE),
            "source_original_dimensions": list(Image.open(_SOURCE).size),
            "source_resized_dimensions": list(source.size),
            "depth_path": str(_DEPTH),
            "depth_sha256": _sha256(_DEPTH),
            "depth_original_dimensions": list(Image.open(_DEPTH).size),
            "depth_resized_dimensions": list(depth_rgb.size),
        },
        "config": {
            "strength": STRENGTH,
            "controlnet_conditioning_scale": CONTROLNET_SCALE,
            "num_inference_steps": STEPS,
            "guidance_scale": GUIDANCE,
            "seed": SEED,
            "resolution": list(RESOLUTION),
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
        },
        "attempt_chain": modes,
        "attempts": attempt_history,
    }

    try:
        for mode in modes:
            t0 = time.time()
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                generated, peak_alloc, peak_reserved = _attempt_generation(mode, source, depth_rgb)
                runtime = round(time.time() - t0, 2)
                success_mode = mode
                attempt_history.append(
                    {
                        "mode": mode,
                        "status": "PASS",
                        "runtime_seconds": runtime,
                        "peak_allocated_gb": round(peak_alloc, 2),
                        "peak_reserved_gb": round(peak_reserved, 2),
                    }
                )
                break
            except torch.cuda.OutOfMemoryError as e:
                last_error = str(e)
                attempt_history.append(
                    {
                        "mode": mode,
                        "status": "OOM",
                        "error": str(e)[:500],
                        "runtime_seconds": round(time.time() - t0, 2),
                    }
                )
                print(f"  OOM in mode={mode}")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except RuntimeError as e:
                # Only treat as OOM if the message clearly says CUDA allocation failed.
                # Host (CPU) OOM or generic "out of memory" in non-CUDA context is NOT recoverable here.
                msg = str(e).lower()
                is_cuda_oom = (
                    "cuda out of memory" in msg
                    or "cuda oom" in msg
                    or ("cuda" in msg and "out of memory" in msg)
                )
                if is_cuda_oom:
                    last_error = str(e)
                    attempt_history.append({
                        "mode": mode,
                        "status": "OOM",
                        "error": str(e)[:500],
                        "runtime_seconds": round(time.time() - t0, 2),
                        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
                    })
                    print(f"  OOM (RuntimeError) in mode={mode}")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                else:
                    # Non-OOM error: let the outer except handler deal with it
                    raise

    except Exception as e:
        report["attempts"] = attempt_history
        report["status"] = "FAIL"
        report["error"] = str(e)[:500]
        _REPORT.write_text(json.dumps(report, indent=2))
        print(f"\n❌ Unexpected error: {e}")
        print(f"Report: {_REPORT}")
        raise

    # All three modes failed — write failure report, then raise
    if success_mode is None:
        report["attempts"] = attempt_history
        report["status"] = "FAIL"
        report["generation_status"] = "FAIL"
        report["resource_gate"] = "NOT_EVALUATED"
        report["error"] = last_error or "All execution modes OOM"
        report["all_modes_failed"] = True
        _REPORT.write_text(json.dumps(report, indent=2))
        print("\n❌ Phase 2 smoke test FAILED — all fallback modes OOM")
        print(f"Report: {_REPORT}")
        raise RuntimeError(f"All VRAM modes failed. Last error: {last_error}")

    # Success — save artifacts
    generated.save(_GENERATED)
    comparison = _make_comparison(source, depth_rgb, generated)
    comparison.save(_COMPARISON)

    generation_passed = generated is not None
    resource_passed = peak_reserved < _SAFE_LIMIT_GB

    report["generation_status"] = "PASS" if generation_passed else "FAIL"
    report["resource_gate"] = "PASS" if resource_passed else "FAIL"
    report["status"] = "PASS" if (generation_passed and resource_passed) else "FAIL"
    report["execution_mode"] = success_mode
    report["runtime_seconds"] = runtime
    report["peak_allocated_gb"] = round(peak_alloc, 2)
    report["peak_reserved_gb"] = round(peak_reserved, 2)
    report["safe_limit_gb"] = _SAFE_LIMIT_GB
    report["artifacts"] = {
        "generated": str(_GENERATED),
        "comparison": str(_COMPARISON),
    }
    _REPORT.write_text(json.dumps(report, indent=2))

    print(f"\n  Generation status: {report['generation_status']}")
    print(
        f"  Resource gate:     {report['resource_gate']} "
        f"(peak reserved {peak_reserved:.2f} GB, limit {_SAFE_LIMIT_GB} GB)"
    )
    print(f"  Execution mode:    {success_mode}")
    print(f"  Runtime:           {runtime}s")
    print(f"  Generated:         {_GENERATED}")
    print(f"  Comparison:        {_COMPARISON}")
    print(f"  Report:            {_REPORT}")

    if report["status"] != "PASS":
        print(f"\n❌ Phase 2 smoke test status: {report['status']}")
        raise RuntimeError(
            f"Phase 2 smoke test status={report['status']} "
            f"(generation={report['generation_status']}, resource={report['resource_gate']})"
        )
    print("\n✅ Phase 2 smoke test PASSED")


if __name__ == "__main__":
    main()
