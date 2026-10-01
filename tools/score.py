#!/usr/bin/env python3
"""Score caption similarity with a configurable judge model."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from oeval.paths import VICTIM_MODELS
from oscore.paths import PROJECT_ROOT, SCORE_OUTPUT_DIR, resolve_judge
from oattack.runtime import python_command

RUN_SUFFIX = "short_sentence_v1"


def load_victim_registry() -> Dict[str, Dict]:
    return json.loads(VICTIM_MODELS.read_text(encoding="utf-8"))


def sanitize_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())


def parse_models(spec: str, registry: Dict[str, Dict]) -> List[str]:
    if spec.strip().lower() == "all":
        return list(registry.keys())
    names = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown model labels: {unknown}")
    return names


def find_prediction(runs_roots: List[Path], run_name: str) -> Optional[Path]:
    for root in runs_roots:
        candidate = root / run_name / "predictions.jsonl"
        if candidate.exists():
            return candidate
    return None


def write_compact_summary(summary_csv: Path, registry: Dict[str, Dict], compact_csv: Path, judge_cfg: Dict) -> None:
    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    with compact_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "display_name", "paper_model_id", "judge", "judge_model_id", "num_pairs", "num_success", "num_failed", "avg_similarity", "asr_gt_0_5"])
        for row in rows:
            label = row.get("model", "")
            cfg = registry.get(label, {})
            writer.writerow([label, cfg.get("display_name", label), cfg.get("paper_model_id", ""), judge_cfg.get("label", ""), judge_cfg.get("model_id", ""), row.get("num_pairs", ""), row.get("num_success", ""), row.get("num_failed", ""), row.get("mean", row.get("avg", "")), row.get("asr_gt_0_5", "")])


def resolve_api_key(judge_cfg: Dict, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    env_name = judge_cfg.get("api_key_env", "OPENROUTER_API_KEY")
    return os.getenv(env_name) or os.getenv("OPENROUTER_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score adversarial caption transfer with a judge model.")
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--attack-tag", type=str, required=True)
    parser.add_argument("--target-tag", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to outputs/score/<attack-tag>/")
    parser.add_argument("--models", type=str, default="all", help="Victim models from configs/eval/victim_models.json")
    parser.add_argument("--judge", type=str, default=None, help="Judge label from configs/score/judge_models.json")
    parser.add_argument("--api-url", type=str, default=None, help="Override judge API URL")
    parser.add_argument("--judge-model", type=str, default=None, help="Override judge model id")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--trust-env-proxy", action="store_true", default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    victim_registry = load_victim_registry()
    selected_models = parse_models(args.models, victim_registry)
    judge_cfg = dict(resolve_judge(args.judge))
    api_url = args.api_url or judge_cfg["api_url"]
    judge_model = args.judge_model or judge_cfg["model_id"]
    judge_cfg.update(api_url=api_url, model_id=judge_model)
    sleep = args.sleep if args.sleep is not None else float(judge_cfg.get("sleep", 0.1))
    trust_env_proxy = args.trust_env_proxy if args.trust_env_proxy is not None else bool(judge_cfg.get("trust_env_proxy", False))

    runs_roots = [path.resolve() for path in args.runs_root]
    attack_tag = sanitize_tag(args.attack_tag)
    output_dir = (args.output_dir or (SCORE_OUTPUT_DIR / attack_tag)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_tag = sanitize_tag(args.target_tag or f"target_for_{attack_tag}")
    run_tag = sanitize_tag(f"{attack_tag}_{judge_cfg['label']}")

    cmd: List[str] = [*python_command(), "-m", "oscore.run_similarity", "--output-dir", str(output_dir), "--run-tag", run_tag, "--api-url", api_url, "--judge-model", judge_model, "--include-avg", "--resume", "--sleep", str(sleep)]
    if trust_env_proxy:
        cmd.append("--trust-env-proxy")
    if judge_cfg.get("disable_reasoning"):
        cmd.append("--disable-reasoning")
    api_key = resolve_api_key(judge_cfg, args.api_key)
    env = os.environ.copy()
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])

    missing: List[str] = []
    added: List[str] = []
    for model_label in selected_models:
        adv_name = f"{model_label}_{attack_tag}_{RUN_SUFFIX}"
        tgt_name = f"{model_label}_target_{target_tag}_{RUN_SUFFIX}"
        adv_pred = find_prediction(runs_roots, adv_name)
        tgt_pred = find_prediction(runs_roots, tgt_name)
        if adv_pred and tgt_pred:
            cmd.extend(["--model", model_label, str(adv_pred), str(tgt_pred)])
            added.append(model_label)
        else:
            missing.append(model_label)
    if missing and args.strict:
        raise RuntimeError(f"Missing predictions for models: {missing}")
    if not added:
        raise RuntimeError("No valid model prediction pairs were found to score.")

    print(f"[score] judge={judge_cfg['label']} ({judge_model})")
    print("[score] models:", ", ".join(added))
    if missing:
        print("[skip] missing models:", ", ".join(missing))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)

    summary_csv = output_dir / f"summary_{run_tag}_qwen_api.csv"
    compact_csv = output_dir / f"compact_{run_tag}.csv"
    if summary_csv.exists():
        write_compact_summary(summary_csv, victim_registry, compact_csv, judge_cfg)
        print(f"[done] wrote compact summary: {compact_csv}")


if __name__ == "__main__":
    main()
