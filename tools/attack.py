#!/usr/bin/env python3
"""Launch O-Attack adversarial sample generation."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml

from oattack.paths import ATTACK_CONFIG, ATTACK_OUTPUT_DIR, PROJECT_ROOT, resolve_path
from oattack.runtime import python_command


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must load as a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate O-Attack adversarial samples.")
    parser.add_argument("--config", type=Path, default=ATTACK_CONFIG)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sample-start", type=int, default=None)
    parser.add_argument("--clean-root", type=Path, default=None)
    parser.add_argument("--target-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epsilon", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_yaml(config_path)

    env = os.environ.copy()

    num_samples = args.num_samples if args.num_samples is not None else cfg.get("data", {}).get("num_samples", 100)
    sample_start = args.sample_start if args.sample_start is not None else cfg.get("data", {}).get("sample_start", 0)
    device = args.device or cfg.get("model", {}).get("device", "cuda:0")
    output_root = args.output_root or Path(cfg.get("output", {}).get("root", ATTACK_OUTPUT_DIR / "default"))

    overrides: List[str] = [
        f"data.num_samples={num_samples}",
        f"data.sample_start={sample_start}",
        f"model.device={device}",
        f"output.root={resolve_path(output_root)}",
    ]
    if args.clean_root:
        overrides.append(f"data.clean_root={resolve_path(args.clean_root)}")
    if args.target_root:
        overrides.append(f"data.target_root={resolve_path(args.target_root)}")
    if args.steps is not None:
        overrides.append(f"optim.steps={args.steps}")
    if args.alpha is not None:
        overrides.append(f"optim.alpha={args.alpha}")
    if args.epsilon is not None:
        overrides.append(f"optim.epsilon={args.epsilon}")
    overrides.extend(args.set)

    cmd = [*python_command(), "-m", "oattack.run", "--config", str(config_path), "--override", *overrides]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
