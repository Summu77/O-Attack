"""Scoring-stage path helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
JUDGE_MODELS = CONFIG_DIR / "score" / "judge_models.json"
SCORE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "score"


def load_judge_registry() -> Dict[str, Any]:
    return json.loads(JUDGE_MODELS.read_text(encoding="utf-8"))


def resolve_judge(label: str | None = None) -> Dict[str, Any]:
    registry = load_judge_registry()
    key = label or registry.get("default", "")
    judges = registry.get("judges", {})
    if key not in judges:
        valid = ", ".join(sorted(judges))
        raise KeyError(f"Unknown judge '{key}'. Valid: {valid}")
    cfg = dict(judges[key])
    cfg["label"] = key
    return cfg
