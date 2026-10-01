#!/usr/bin/env python3
"""Run VLM caption evaluation on adversarial images."""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from oeval.paths import EVAL_OUTPUT_DIR, PROJECT_ROOT, VICTIM_MODELS
from oattack.runtime import python_command
from oeval.artifacts import input_fingerprint

RUN_SUFFIX = "short_sentence_v1"
DEFAULT_TARGET_ROOT = PROJECT_ROOT / "data" / "images" / "target_images"
DEFAULT_PROMPT = "Please describe this image in a short sentence."


def load_registry() -> Dict[str, Dict]:
    return json.loads(VICTIM_MODELS.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adversarial images with registered victim VLMs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to outputs/eval/<attack-tag>/")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--attack-tag", type=str, default=None)
    parser.add_argument("--target-tag", type=str, default=None)
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--max-local-workers", type=int, default=None)
    parser.add_argument("--free-gpu-max-memory-mb", type=int, default=1500)
    parser.add_argument("--free-gpu-max-util", type=int, default=15)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--openrouter-proxy-url", type=str, default=None)
    return parser.parse_args()


def sanitize_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())


def default_attack_tag(input_root: Path) -> str:
    return sanitize_tag(f"{input_root.parent.name or 'input'}_{input_root.name or 'images'}")


def parse_models(spec: str, registry: Dict[str, Dict]) -> List[str]:
    if spec.strip().lower() == "all":
        return list(registry.keys())
    names = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown model labels: {unknown}")
    return names


def ensure_target_subset(adv_root: Path, target_root: Path, output_dir: Path) -> None:
    cmd = [*python_command(), "-m", "oeval.build_target_subset", "--adv-root", str(adv_root), "--target-root", str(target_root), "--output-dir", str(output_dir), "--overwrite"]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def summary_complete(summary_path: Path, expected: Dict | None = None) -> bool:
    if not summary_path.exists():
        return False
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    count = int(data.get("num_images", 0))
    complete = count > 0 and int(data.get("num_success", 0)) == count and int(data.get("num_failed", 0)) == 0
    return complete and (expected is None or all(data.get(key) == value for key, value in expected.items()))


def build_run_name(label: str, suffix_tag: str) -> str:
    return f"{label}_{suffix_tag}_{RUN_SUFFIX}"


def find_free_gpus(max_memory_mb: int, max_util: int) -> List[int]:
    proc = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    free: List[int] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        idx_str, mem_str, util_str = [part.strip() for part in line.split(",")]
        if int(float(mem_str)) <= max_memory_mb and int(float(util_str)) <= max_util:
            free.append(int(idx_str))
    return free


def build_inference_cmd(model_cfg: Dict, input_root: Path, output_root: Path, run_name: str, prompt: str, max_new_tokens: int) -> List[str]:
    cmd = [*python_command(model_cfg.get("conda_env")), "-m", "oeval.run_inference", "--input-root", str(input_root), "--output-root", str(output_root), "--run-name", run_name, "--model-key", model_cfg["model_key"], "--model-name", model_cfg["model_name"], "--model-dtype", model_cfg.get("model_dtype", "auto"), "--model-device-map", model_cfg.get("model_device_map", "auto"), "--prompt", prompt, "--max-new-tokens", str(max_new_tokens)]
    if model_cfg.get("local_files_only", False):
        cmd.append("--local-files-only")
    return cmd


def run_inference_job(model_label: str, model_cfg: Dict, input_root: Path, output_root: Path, run_name: str, prompt: str, max_new_tokens: int, gpu: int | None, proxy_url: str | None) -> None:
    summary_path = output_root / run_name / "summary.json"
    expected = {"model_key": model_cfg["model_key"], "model_name": model_cfg["model_name"],
                "prompt": prompt, "max_new_tokens": max_new_tokens,
                "input_fingerprint": input_fingerprint(input_root)}
    if summary_complete(summary_path, expected):
        print(f"[skip] completed: {run_name}")
        return
    env = os.environ.copy()
    if proxy_url:
        env["OPENROUTER_PROXY_URL"] = proxy_url
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if model_cfg["model_key"] != "deepseek_vl2_tiny":
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = build_inference_cmd(model_cfg, input_root, output_root, run_name, prompt, max_new_tokens)
    print(f"[run][{model_label}] gpu={gpu if gpu is not None else 'api'} run_name={run_name}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
    if not summary_complete(summary_path, expected):
        raise RuntimeError(f"Inference is incomplete: {summary_path}")


def run_local_model_pair(model_label: str, model_cfg: Dict, target_subset: Path, adv_root: Path, output_root: Path, target_tag: str, attack_tag: str, prompt: str, max_new_tokens: int, gpu_queue: queue.Queue, proxy_url: str | None) -> None:
    gpu = gpu_queue.get()
    try:
        run_inference_job(model_label, model_cfg, target_subset, output_root, build_run_name(model_label, f"target_{target_tag}"), prompt, max_new_tokens, gpu, proxy_url)
        run_inference_job(model_label, model_cfg, adv_root, output_root, build_run_name(model_label, attack_tag), prompt, max_new_tokens, gpu, proxy_url)
    finally:
        gpu_queue.put(gpu)


def main() -> None:
    args = parse_args()
    registry = load_registry()
    selected_models = parse_models(args.models, registry)
    adv_root = args.input_root.resolve()
    attack_tag = sanitize_tag(args.attack_tag or default_attack_tag(adv_root))
    output_root = (args.output_root or (EVAL_OUTPUT_DIR / attack_tag)).resolve()
    target_root = args.target_root.resolve()
    target_tag = sanitize_tag(args.target_tag or f"target_for_{attack_tag}")
    target_subset = output_root / f"_target_subset_{target_tag}"
    if not adv_root.exists():
        raise FileNotFoundError(f"Adversarial input root does not exist: {adv_root}")
    if not target_root.exists():
        raise FileNotFoundError(f"Target root does not exist: {target_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_target_subset(adv_root, target_root, target_subset)
    local_models = [name for name in selected_models if registry[name]["source"] == "local"]
    api_models = [name for name in selected_models if registry[name]["source"] == "api"]
    gpus = []
    if local_models:
        gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()] if args.gpus else find_free_gpus(args.free_gpu_max_memory_mb, args.free_gpu_max_util)
    if local_models and not gpus:
        raise RuntimeError("No free GPUs were detected for local models. Pass --gpus explicitly.")
    if local_models:
        gpu_queue: queue.Queue[int] = queue.Queue()
        for gpu in gpus:
            gpu_queue.put(gpu)
        max_workers = args.max_local_workers or len(gpus)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(local_models), len(gpus))) as executor:
            futures = [executor.submit(run_local_model_pair, model_label, registry[model_label], target_subset, adv_root, output_root, target_tag, attack_tag, args.prompt, args.max_new_tokens, gpu_queue, args.openrouter_proxy_url) for model_label in local_models]
            for future in as_completed(futures):
                future.result()
    for model_label in api_models:
        model_cfg = registry[model_label]
        run_inference_job(model_label, model_cfg, target_subset, output_root, build_run_name(model_label, f"target_{target_tag}"), args.prompt, args.max_new_tokens, None, args.openrouter_proxy_url)
        run_inference_job(model_label, model_cfg, adv_root, output_root, build_run_name(model_label, attack_tag), args.prompt, args.max_new_tokens, None, args.openrouter_proxy_url)
    print("[done] evaluation finished.")
    print(json.dumps({"attack_tag": attack_tag, "target_tag": target_tag, "output_root": str(output_root), "models": selected_models}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
