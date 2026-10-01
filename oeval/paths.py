"""Evaluation-stage path helpers."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
VICTIM_MODELS = CONFIG_DIR / "eval" / "victim_models.json"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval"
DATA_ROOT = PROJECT_ROOT / "data"
DATA_IMAGES_DIR = DATA_ROOT / "images"
DATA_TEXT_DIR = DATA_ROOT / "text"
DEFAULT_TARGET_ROOT = DATA_IMAGES_DIR / "target_images"
