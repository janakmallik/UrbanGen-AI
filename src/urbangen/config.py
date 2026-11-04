"""Environment configuration for UrbanGen-AI.

Reads .env (gitignored) at import time. Required variables:
    URBANINTEL_DATA_ROOT  parent directory containing `bdd100k/`
    URBANMIND_REPO        path to a local UrbanMind-AI checkout

Raises RuntimeError with actionable guidance if a required variable is
missing.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise RuntimeError(
            f"Environment variable {name!r} is not set.\n"
            f"  Copy .env.example to .env and fill in your local paths.\n"
            f"  See README.md for setup instructions."
        )
    return Path(raw).expanduser().resolve()


def data_root() -> Path:
    """Path to the BDD100K dataset root, e.g. <ml-data>/bdd100k."""
    return _require_env("URBANINTEL_DATA_ROOT") / "bdd100k"


def urbanmind_root() -> Path:
    """Path to a local UrbanMind-AI checkout."""
    return _require_env("URBANMIND_REPO")
