from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WANGP_ROOT = PROJECT_ROOT.parent / "WanGP"

sys.path.insert(0, str(WANGP_ROOT))

from shared.api import init  # noqa: E402

IMAGE_DIR = Path.home() / "ml-data" / "bdd100k" / "images" / "100k" / "train"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "generated_videos"
USED_IMAGES_FILE = OUTPUT_DIR / "used_images.txt"

VIDEO_COUNT = 2
FIRST_INDEX = 4
DURATION_SECONDS = 9
FPS = 24
INFERENCE_STEPS = 3
SEED = 42

PREVIOUS_IMAGE = "1eb002c1-ffb893a6.jpg"

PROMPTS = [
    "Realistic urban traffic documentary footage. Cars move naturally "
    "through a city intersection. Pedestrians walk along sidewalks. "
    "Natural daylight and realistic camera motion.",
    "A realistic urban avenue with buses, cars and motorcycles moving "
    "through traffic. Smooth forward camera movement, documentary style.",
    "A busy city street with pedestrians crossing and vehicles moving "
    "naturally. Realistic environmental motion and daylight.",
    "Realistic dashcam footage of an urban road. Cars drive through "
    "the street, buildings and roadside trees remain detailed.",
    "A cinematic urban intersection with flowing traffic, pedestrians "
    "and realistic street movement. Stable documentary camera.",
]


def get_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")

    if executable:
        return executable

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError as err:
        raise RuntimeError("FFmpeg executable not found.") from err


def stitch_videos(parts: list[Path], destination: Path) -> Path:
    # Remove duplicate paths.
    parts = list(dict.fromkeys(p.resolve() for p in parts if p.is_file()))

    if not parts:
        raise RuntimeError("No video segments to combine.")

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {destination}")

    if len(parts) == 1:
        shutil.copy2(parts[0], destination)
    else:
        ffmpeg = get_ffmpeg()
        concat_file = OUTPUT_DIR / "_concat_list.txt"

        concat_file.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in parts),
            encoding="utf-8",
        )

        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(destination),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            concat_file.unlink(missing_ok=True)

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Final video creation failed.")

    # Remove temporary WanGP segments only after successful stitching.
    for part in parts:
        if part != destination.resolve():
            part.unlink(missing_ok=True)

    return destination


def select_images(count: int) -> list[Path]:

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(IMAGE_DIR)

    used = set()

    if USED_IMAGES_FILE.exists():
        used.update(
            line.strip()
            for line in USED_IMAGES_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    used.add(PREVIOUS_IMAGE)

    images = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name not in used
    )

    if len(images) < count:
        raise RuntimeError("Not enough unused BDD100K images.")

    return random.Random(SEED).sample(images, count)


def generate(count: int = VIDEO_COUNT):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing WanGP...", flush=True)

    session = init(
        root=WANGP_ROOT,
        output_dir=OUTPUT_DIR,
        cli_args=[
            "--attention",
            "sdpa",
            "--profile",
            "4",
        ],
    )

    completed = []

    try:
        base_settings = session.get_default_settings("ti2v_2_2_fastwan")

        images = select_images(count)

        for offset, image_path in enumerate(images):
            index = FIRST_INDEX + offset

            print(
                f"\n[{offset + 1}/{count}] Generating {image_path.name}",
                flush=True,
            )

            settings = dict(base_settings)

            settings.update(
                {
                    "prompt": PROMPTS[(index - 1) % len(PROMPTS)],
                    "image_start": str(image_path.resolve()),
                    "video_length": f"{DURATION_SECONDS}s",
                    "force_fps": FPS,
                    "num_inference_steps": INFERENCE_STEPS,
                    "guidance_scale": 1,
                    "flow_shift": 3,
                    "seed": SEED + index,
                }
            )

            job = session.submit_task(settings)

            result = job.result()

            if not result.success:
                print("[FAILED] Generation error.", flush=True)
                continue

            segments = [
                Path(p)
                for p in result.generated_files
                if Path(p).is_file() and Path(p).suffix.lower() == ".mp4"
            ]

            print("[DEBUG] WanGP returned files:", flush=True)

            for p in segments:
                print(f"  {p.name} | {p.stat().st_size:,} bytes", flush=True)

            if not segments:
                print("[FAILED] No MP4 segments.", flush=True)
                continue

            destination = OUTPUT_DIR / f"urban_{index:02d}.mp4"

            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing video: {destination}")

            stitch_videos(segments, destination)

            completed.append(destination)
            USED_IMAGES_FILE.parent.mkdir(parents=True, exist_ok=True)

            with USED_IMAGES_FILE.open("a", encoding="utf-8") as f:
                f.write(image_path.name + "\n")

            print(
                f"[SUCCESS] {destination}",
                flush=True,
            )

    finally:
        session.close()

    print(
        f"\nGenerated {len(completed)}/{count} additional videos.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=VIDEO_COUNT,
    )

    args = parser.parse_args()

    generate(args.count)
