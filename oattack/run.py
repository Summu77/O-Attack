import argparse
import json
import os
import time
from typing import Dict, List

import torch
import torchvision
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image

from oattack.attacks import get_attack
from oattack.augmentations import build_pipeline
from oattack.data import build_clean_dataset, build_pair_datasets
from oattack.models import EnsembleFeatureExtractor, EnsembleFeatureLoss, create_model
from oattack.paths import resolve_path
from oattack.utils import create_logger, ensure_dir, hash_run_config, pil_to_tensor_255, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run O-Attack v2.1 adversarial sample generation")
    parser.add_argument("--config", type=str, default="configs/attack/default.yaml")
    parser.add_argument("--override", nargs="*", default=[], help="OmegaConf dotlist overrides")
    return parser.parse_args()


def load_config(config_path: str, overrides: List[str]):
    cfg = OmegaConf.load(resolve_path(config_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return normalize_config_paths(cfg)




def normalize_config_paths(cfg):
    cfg.output.root = str(resolve_path(cfg.output.root))
    cfg.data.clean_root = str(resolve_path(cfg.data.clean_root))
    cfg.data.target_root = str(resolve_path(cfg.data.target_root))
    if cfg.data.get("fixed_target_image"):
        cfg.data.fixed_target_image = str(resolve_path(cfg.data.fixed_target_image))
    if bool(getattr(cfg.text_objective, "enabled", False)):
        source_json = str(getattr(cfg.text_objective, "source_caption_json", "")).strip()
        target_json = str(getattr(cfg.text_objective, "target_caption_json", "")).strip()
        if source_json:
            cfg.text_objective.source_caption_json = str(resolve_path(source_json))
        if target_json:
            cfg.text_objective.target_caption_json = str(resolve_path(target_json))
    return cfg




def _parse_device_spec(device_cfg) -> List[str]:
    if isinstance(device_cfg, (list, tuple)) or OmegaConf.is_list(device_cfg):
        devices = [str(d).strip() for d in device_cfg if str(d).strip()]
    else:
        raw = str(device_cfg).strip()
        devices = [d.strip() for d in raw.split(",") if d.strip()]
    return devices or ["cpu"]


def _cuda_index(device: str) -> int:
    if not device.startswith("cuda:"):
        raise ValueError(f"Expected cuda device like 'cuda:0', got: {device}")
    return int(device.split(":")[1])


def _section_backbone_override(section, backbone_name: str, key: str, default):
    overrides = getattr(section, "backbone_overrides", None)
    if overrides is None:
        return default
    backbone_cfg = getattr(overrides, backbone_name, None)
    if backbone_cfg is None:
        return default
    value = getattr(backbone_cfg, key, None)
    return default if value is None else value


def build_models(cfg):
    backbones = list(cfg.model.backbones)
    if (not bool(cfg.model.ensemble)) and len(backbones) > 1:
        raise ValueError("model.ensemble=false only supports a single backbone")

    requested_devices = _parse_device_spec(cfg.model.device)
    if any(dev.startswith("cuda") for dev in requested_devices) and not torch.cuda.is_available():
        print("CUDA not available, fallback to cpu")
        requested_devices = ["cpu"]

    if all(dev.startswith("cuda") for dev in requested_devices):
        valid_count = torch.cuda.device_count()
        for dev in requested_devices:
            idx = _cuda_index(dev)
            if idx < 0 or idx >= valid_count:
                raise ValueError(f"Invalid CUDA device '{dev}', available count={valid_count}")
    elif len(requested_devices) > 1:
        raise ValueError("Multiple devices are only supported for CUDA")

    primary_device = requested_devices[0]
    strategy = str(getattr(cfg.model, "multi_device_strategy", "model_parallel")).strip().lower().replace("-", "_")
    if strategy not in {"model_parallel", "data_parallel"}:
        raise ValueError("model.multi_device_strategy must be one of: model_parallel | data_parallel")
    use_multi_cuda = len(requested_devices) > 1 and all(dev.startswith("cuda") for dev in requested_devices)
    use_dataparallel = use_multi_cuda and strategy == "data_parallel"
    data_parallel_ids = [_cuda_index(dev) for dev in requested_devices] if use_dataparallel else []

    models = []
    stochastic_proxy = bool(getattr(cfg.model, "stochastic_proxy", True))
    attention_dropout = float(getattr(cfg.model, "attention_dropout", 0.1))
    ffn_dropout = float(getattr(cfg.model, "ffn_dropout", 0.1))
    dropout_sample_per_forward = bool(getattr(cfg.model, "dropout_sample_per_forward", True))
    cls_last_k = max(1, int(getattr(cfg.model, "cls_last_k", 1)))
    text_last_k = max(1, int(getattr(cfg.text_objective, "text_last_k", 1)))

    for backbone_idx, backbone_name in enumerate(backbones):
        model_device = requested_devices[backbone_idx % len(requested_devices)] if use_multi_cuda else primary_device
        model = create_model(backbone_name)
        setattr(model, "_backbone_name", backbone_name)
        model_cls_last_k = max(1, int(_section_backbone_override(cfg.model, backbone_name, "cls_last_k", cls_last_k)))
        model_text_last_k = max(1, int(_section_backbone_override(cfg.text_objective, backbone_name, "text_last_k", text_last_k)))
        model_attention_dropout = float(_section_backbone_override(cfg.model, backbone_name, "attention_dropout", attention_dropout))
        model_ffn_dropout = float(_section_backbone_override(cfg.model, backbone_name, "ffn_dropout", ffn_dropout))
        if hasattr(model, "configure_cls_loss_layers"):
            model.configure_cls_loss_layers(model_cls_last_k)
        if hasattr(model, "configure_text_loss_layers"):
            model.configure_text_loss_layers(model_text_last_k)
        if hasattr(model, "configure_stochastic_proxy"):
            model.configure_stochastic_proxy(
                enabled=stochastic_proxy,
                attention_dropout=model_attention_dropout,
                ffn_dropout=model_ffn_dropout,
                sample_per_forward=dropout_sample_per_forward,
            )
        elif not stochastic_proxy:
            model.eval()
        model = model.to(model_device).requires_grad_(False)
        setattr(model, "_inference_device", model_device)
        setattr(model, "_backbone_name", backbone_name)
        if use_dataparallel:
            model = torch.nn.DataParallel(model, device_ids=data_parallel_ids, output_device=data_parallel_ids[0])
            model = model.requires_grad_(False)
            setattr(model, "_backbone_name", backbone_name)
        models.append(model)

    return primary_device, models


def build_dataset_transform(input_res: int):
    return T.Compose([
        T.Resize(int(input_res), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(int(input_res)),
        T.Lambda(lambda img: img.convert("RGB")),
        T.Lambda(lambda img: pil_to_tensor_255(img)),
    ])


def save_adversarial_batch(cfg, run_hash: str, adv_batch: torch.Tensor, path_org_batch: List[str]) -> List[str]:
    saved_paths: List[str] = []
    for idx, src_path in enumerate(path_org_batch):
        folder = os.path.basename(os.path.dirname(src_path))
        name = os.path.basename(src_path)
        stem, ext = os.path.splitext(name)
        save_dir = os.path.join(cfg.output.root, "img", run_hash, folder)
        ensure_dir(save_dir)
        save_name = f"{stem}.png" if ext.lower() in {".jpg", ".jpeg"} else name
        output_path = os.path.join(save_dir, save_name)
        torchvision.utils.save_image(adv_batch[idx], output_path)
        saved_paths.append(output_path)
    return saved_paths


def save_adversarial_batch_snapshot(cfg, run_hash: str, adv_batch: torch.Tensor, path_org_batch: List[str], step: int) -> List[str]:
    saved_paths: List[str] = []
    for idx, src_path in enumerate(path_org_batch):
        folder = os.path.basename(os.path.dirname(src_path))
        name = os.path.basename(src_path)
        stem, ext = os.path.splitext(name)
        save_dir = os.path.join(cfg.output.root, "img_steps", f"step_{int(step):04d}", run_hash, folder)
        ensure_dir(save_dir)
        save_name = f"{stem}.png" if ext.lower() in {".jpg", ".jpeg"} else name
        output_path = os.path.join(save_dir, save_name)
        torchvision.utils.save_image(adv_batch[idx], output_path)
        saved_paths.append(output_path)
    return saved_paths


def build_step_saver(cfg, run_hash: str, path_org_batch: List[str]):
    snapshot_steps = getattr(cfg.output, "snapshot_steps", None)
    if not snapshot_steps:
        return None
    wanted_steps = {int(step) for step in snapshot_steps}

    def _save(step: int, adv_batch: torch.Tensor) -> None:
        if int(step) not in wanted_steps:
            return
        save_adversarial_batch_snapshot(cfg, run_hash, adv_batch.detach().cpu(), path_org_batch, int(step))

    return _save


def _load_fixed_target_tensor(cfg, transform):
    fixed_path = cfg.data.get("fixed_target_image", None)
    if fixed_path is None:
        return None
    fixed_path = str(fixed_path).strip()
    if fixed_path == "" or fixed_path.lower() == "null":
        return None
    abs_path = os.path.abspath(os.path.expanduser(fixed_path))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"data.fixed_target_image not found: {abs_path}")
    return transform(Image.open(abs_path).convert("RGB")).unsqueeze(0)


def _load_caption_map(caption_path: str) -> Dict[str, List[str]]:
    with open(caption_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"caption file must be list[dict]: {caption_path}")
    mapping: Dict[str, List[str]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        image_name = str(row.get("image", "")).strip()
        captions = row.get("caption", [])
        if not image_name:
            continue
        if not isinstance(captions, list):
            captions = [str(captions)]
        mapping[image_name] = [str(x).strip() for x in captions if str(x).strip()]
    return mapping


def _get_caption_banks_for_paths(paths: List[str], cache: Dict[str, Dict[str, List[str]]]) -> List[List[str]]:
    results: List[List[str]] = []
    for p in paths:
        folder = os.path.dirname(p)
        if folder not in cache:
            cache[folder] = _load_caption_map(os.path.join(folder, "caption.json"))
        filename = os.path.basename(p)
        captions = cache[folder].get(filename, [])
        if not captions:
            raise KeyError(f"Missing captions for image '{filename}' under folder '{folder}'")
        results.append(captions)
    return results


def _get_caption_banks_from_mapping(paths: List[str], mapping: Dict[str, List[str]], source_name: str) -> List[List[str]]:
    results: List[List[str]] = []
    for p in paths:
        filename = os.path.basename(p)
        captions = mapping.get(filename, [])
        if not captions:
            raise KeyError(f"Missing captions for image '{filename}' in {source_name}")
        results.append(captions)
    return results


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(int(cfg.seed))
    run_hash = hash_run_config(cfg)
    logger = create_logger(cfg, run_hash)

    device, models = build_models(cfg)
    ensemble_extractor = EnsembleFeatureExtractor(models)
    ensemble_loss = EnsembleFeatureLoss(models)

    transform = build_dataset_transform(int(cfg.model.input_res))
    fixed_target_tensor = _load_fixed_target_tensor(cfg, transform)
    clean_data = build_clean_dataset(cfg, transform)
    if int(cfg.data.num_samples) <= 0:
        raise ValueError("data.num_samples must be positive")
    if int(cfg.data.num_samples) > len(clean_data):
        raise ValueError(f"Requested {cfg.data.num_samples} samples but only {len(clean_data)} are available")
    if fixed_target_tensor is None:
        _clean_for_pairs, target_data = build_pair_datasets(cfg, transform)
        target_loader = torch.utils.data.DataLoader(target_data, batch_size=int(cfg.data.batch_size), shuffle=False)
    else:
        target_loader = None

    clean_loader = torch.utils.data.DataLoader(clean_data, batch_size=int(cfg.data.batch_size), shuffle=False)
    source_aug = build_pipeline(list(cfg.augmentation.source))
    target_aug = build_pipeline(list(cfg.augmentation.target))

    attack_name = str(cfg.optim.attack)
    if attack_name != "oattack_pgd":
        raise ValueError(f"O-Attack release only supports optim.attack=oattack_pgd, got: {attack_name}")
    attack = get_attack(attack_name)

    use_mixed_text_loss = bool(getattr(cfg.text_objective, "enabled", False))
    caption_cache: Dict[str, Dict[str, List[str]]] = {}
    source_caption_override = target_caption_override = None
    source_caption_json = str(getattr(cfg.text_objective, "source_caption_json", "")).strip()
    target_caption_json = str(getattr(cfg.text_objective, "target_caption_json", "")).strip()
    if source_caption_json:
        source_caption_override = _load_caption_map(os.path.abspath(os.path.expanduser(source_caption_json)))
    if target_caption_json:
        target_caption_override = _load_caption_map(os.path.abspath(os.path.expanduser(target_caption_json)))

    processed = 0
    max_samples = int(cfg.data.num_samples)
    print(f"Running O-Attack v2.1 with hash={run_hash}")
    print(f"Device={device}, Backbones={list(cfg.model.backbones)}, Steps={cfg.optim.steps}, Epsilon={cfg.optim.epsilon}")

    iterator = enumerate(clean_loader) if fixed_target_tensor is not None else enumerate(zip(clean_loader, target_loader))
    for img_index, batch in iterator:
        if processed >= max_samples:
            break

        if fixed_target_tensor is not None:
            image_org, _, path_org = batch
            image_org = image_org.to(device)
            image_tgt = fixed_target_tensor.repeat(image_org.size(0), 1, 1, 1).to(device)
            path_tgt_batch = []
        else:
            (image_org, _, path_org), (image_tgt, _, _path_tgt) = batch
            image_org = image_org.to(device)
            image_tgt = image_tgt.to(device)
            path_tgt_batch = list(_path_tgt)

        path_org_batch = list(path_org)
        source_caption_banks = target_caption_banks = None
        if use_mixed_text_loss:
            if source_caption_override is not None:
                source_caption_banks = _get_caption_banks_from_mapping(path_org_batch, source_caption_override, source_caption_json)
            else:
                source_caption_banks = _get_caption_banks_for_paths(path_org_batch, caption_cache)
            if fixed_target_tensor is not None:
                fixed_target_path = os.path.abspath(os.path.expanduser(str(cfg.data.get("fixed_target_image", "")).strip()))
                fixed_target_paths = [fixed_target_path for _ in path_org_batch]
                if target_caption_override is not None:
                    target_caption_banks = _get_caption_banks_from_mapping(fixed_target_paths, target_caption_override, target_caption_json)
                else:
                    target_caption_banks = _get_caption_banks_for_paths(fixed_target_paths, caption_cache)
            else:
                if target_caption_override is not None:
                    target_caption_banks = _get_caption_banks_from_mapping(path_tgt_batch, target_caption_override, target_caption_json)
                else:
                    target_caption_banks = _get_caption_banks_for_paths(path_tgt_batch, caption_cache)

        sample_started_at = time.perf_counter()
        adv_image = attack(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_aug=source_aug,
            target_aug=target_aug,
            image_org=image_org,
            image_tgt=image_tgt,
            target_texts=None,
            source_caption_banks=source_caption_banks,
            target_caption_banks=target_caption_banks,
            logger=logger,
            img_index=img_index,
            step_saver=build_step_saver(cfg, run_hash, path_org_batch),
        )

        remaining = max_samples - processed
        use_count = min(remaining, len(path_org_batch))
        saved_paths = save_adversarial_batch(cfg, run_hash, adv_image[:use_count], path_org_batch[:use_count])
        batch_elapsed = time.perf_counter() - sample_started_at
        if use_count > 0:
            per_sample_elapsed = batch_elapsed / use_count
            target_paths_for_log = path_tgt_batch[:use_count] if path_tgt_batch else []
            for rel_idx, (src_path, out_path) in enumerate(zip(path_org_batch[:use_count], saved_paths)):
                record = {
                    "img_index": img_index + rel_idx,
                    "source_path": src_path,
                    "output_path": out_path,
                    "elapsed_seconds": per_sample_elapsed,
                    "batch_elapsed_seconds": batch_elapsed,
                    "target_type": "image",
                }
                if target_paths_for_log:
                    record["target_path"] = target_paths_for_log[rel_idx]
                logger.log_sample_time(record)
        processed += use_count
        print(f"Processed {processed}/{max_samples}")

    logger.finish()
    print("Done.")
    print(f"Images: {os.path.join(cfg.output.root, 'img', run_hash)}")
    print(f"Logs:   {os.path.join(cfg.output.root, 'logs', run_hash)}")


if __name__ == "__main__":
    main()
