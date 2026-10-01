import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pil_to_tensor_255(pic) -> torch.Tensor:
    """Convert PIL image to CHW float tensor in [0, 255]."""
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


def hash_run_config(cfg: Any) -> str:
    """Create a deterministic hash for output folder naming."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    tracked = {
        "data": {
            "batch_size": cfg_dict["data"]["batch_size"],
            "sample_start": cfg_dict["data"].get("sample_start", 0),
            "num_samples": cfg_dict["data"]["num_samples"],
            "clean_root": cfg_dict["data"]["clean_root"],
            "target_root": cfg_dict["data"].get("target_root"),
            "dataset_type": cfg_dict["data"]["dataset_type"],
            "fixed_target_image": cfg_dict["data"].get("fixed_target_image"),
        },
        "model": {
            "ensemble": cfg_dict["model"]["ensemble"],
            "backbones": cfg_dict["model"]["backbones"],
            "device": cfg_dict["model"]["device"],
            "multi_device_strategy": cfg_dict["model"].get("multi_device_strategy", "model_parallel"),
            "input_res": cfg_dict["model"]["input_res"],
            "cls_last_k": cfg_dict["model"].get("cls_last_k", 1),
            "stochastic_proxy": cfg_dict["model"].get("stochastic_proxy", True),
            "attention_dropout": cfg_dict["model"].get("attention_dropout", 0.1),
            "ffn_dropout": cfg_dict["model"].get("ffn_dropout", 0.1),
            "attention_dropout_min": cfg_dict["model"].get("attention_dropout_min", 0.0),
            "ffn_dropout_min": cfg_dict["model"].get("ffn_dropout_min", 0.0),
            "attention_dropout_max": cfg_dict["model"].get("attention_dropout_max", 0.2),
            "ffn_dropout_max": cfg_dict["model"].get("ffn_dropout_max", 0.2),
            "dropout_schedule": cfg_dict["model"].get("dropout_schedule", False),
            "dropout_sample_per_forward": cfg_dict["model"].get("dropout_sample_per_forward", True),
            "backbone_overrides": cfg_dict["model"].get("backbone_overrides", {}),
        },
        "optim": {
            "attack": cfg_dict["optim"]["attack"],
            "steps": cfg_dict["optim"]["steps"],
            "alpha": cfg_dict["optim"]["alpha"],
            "epsilon": cfg_dict["optim"]["epsilon"],
            "target_type": cfg_dict["optim"].get("target_type", "image"),
            "source_aug_crops": cfg_dict["optim"].get("source_aug_crops", 1),
            "source_aug_aggregation": cfg_dict["optim"].get("source_aug_aggregation", "mean"),
            "source_aug_execution": cfg_dict["optim"].get("source_aug_execution", "strict"),
            "grad_mask_alpha": cfg_dict["optim"].get("grad_mask_alpha", 0.0),
        },
        "augmentation": cfg_dict.get("augmentation", {}),
        "text_objective": cfg_dict.get("text_objective", {}),        "oattack_objective": cfg_dict.get("oattack_objective", {}),
        "text_objective_tracked": {
            "enabled": cfg_dict.get("text_objective", {}).get("enabled", False),
            "alpha": cfg_dict.get("text_objective", {}).get("alpha", 0.0),
            "text_last_k": cfg_dict.get("text_objective", {}).get("text_last_k", 1),
            "mix_beta_min": cfg_dict.get("text_objective", {}).get("mix_beta_min", 0.0),
            "mix_beta_max": cfg_dict.get("text_objective", {}).get("mix_beta_max", 1.0),
            "attention_dropout_min": cfg_dict.get("text_objective", {}).get("attention_dropout_min", 0.0),
            "attention_dropout_max": cfg_dict.get("text_objective", {}).get("attention_dropout_max", 0.1),
            "ffn_dropout_min": cfg_dict.get("text_objective", {}).get("ffn_dropout_min", 0.0),
            "ffn_dropout_max": cfg_dict.get("text_objective", {}).get("ffn_dropout_max", 0.1),
        },
        "variance_objective_tracked": {
            "enabled": cfg_dict.get("variance_objective", {}).get("enabled", False),
            "visual_weight": cfg_dict.get("variance_objective", {}).get("visual_weight", 0.0),
            "text_weight": cfg_dict.get("variance_objective", {}).get("text_weight", 0.0),
        },
        "oattack_objective_tracked": {
            "visual_variance_weight": cfg_dict.get("oattack_objective", {}).get("visual_variance_weight", 1.0),
            "text_variance_weight": cfg_dict.get("oattack_objective", {}).get("text_variance_weight", 1.0),
            "aggregation_mode": cfg_dict.get("oattack_objective", {}).get("aggregation_mode", "per_model_layeravg_textpooled"),
            "variance_mode": cfg_dict.get("oattack_objective", {}).get("variance_mode", "subtract"),
        },
    }
    payload = json.dumps(tracked, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


@dataclass
class LocalRunLogger:
    run_dir: str

    def __post_init__(self) -> None:
        ensure_dir(self.run_dir)
        self.metrics_file = os.path.join(self.run_dir, "metrics.jsonl")
        self.sample_times_file = os.path.join(self.run_dir, "sample_times.jsonl")
        self.sample_time_summary_file = os.path.join(self.run_dir, "sample_time_summary.json")
        self._sample_elapsed_seconds: List[float] = []

    def log(self, metrics: Dict[str, Any]) -> None:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "metrics": metrics,
        }
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self) -> None:
        if self._sample_elapsed_seconds:
            summary = {
                "count": len(self._sample_elapsed_seconds),
                "average_seconds": float(sum(self._sample_elapsed_seconds) / len(self._sample_elapsed_seconds)),
                "min_seconds": float(min(self._sample_elapsed_seconds)),
                "max_seconds": float(max(self._sample_elapsed_seconds)),
                "total_seconds": float(sum(self._sample_elapsed_seconds)),
            }
            with open(self.sample_time_summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        return

    def log_sample_time(self, record: Dict[str, Any]) -> None:
        per_sample_elapsed = float(record["elapsed_seconds"])
        self._sample_elapsed_seconds.append(per_sample_elapsed)
        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            **record,
        }
        with open(self.sample_times_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def create_logger(cfg: Any, run_hash: str) -> LocalRunLogger:
    run_dir = os.path.join(cfg.output.root, "logs", run_hash)
    ensure_dir(run_dir)

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_hash": run_hash,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "artifacts": {
            "metrics": os.path.join(run_dir, "metrics.jsonl"),
            "sample_times": os.path.join(run_dir, "sample_times.jsonl"),
            "sample_time_summary": os.path.join(run_dir, "sample_time_summary.json"),
        },
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    return LocalRunLogger(run_dir=run_dir)
