from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "indexes"
EVAL_DIR = ROOT / "eval"

load_dotenv(ROOT.parent / ".env")
load_dotenv(ROOT / ".env")

CLAUDE_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None
