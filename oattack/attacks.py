from typing import Any, Callable, Dict, List, Optional, Tuple

import random
import torch
from tqdm import tqdm

from oattack.models.base import EnsembleFeatureLoss

ATTACK_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {}


def register_attack(name: str):
    def decorator(fn: Callable[..., torch.Tensor]):
        ATTACK_REGISTRY[name] = fn
        return fn

    return decorator


def _log_step_metrics(pbar, logger, img_index: int, epoch: int, metrics: Dict[str, float]) -> None:
    display = {}
    for k, v in metrics.items():
        if isinstance(v, str):
            display[k] = v
        else:
            display[k] = f"{v:.5f}" if "sim" in k else f"{v:.3f}"
    pbar.set_postfix(display)

    payload = {f"img{img_index}_{k}": v for k, v in metrics.items()}
    payload["epoch"] = epoch
    logger.log(payload)


def _summarize_gradient(grad: torch.Tensor) -> Dict[str, float]:
    flat = grad.view(grad.size(0), -1)
    grad_l2 = torch.norm(flat, p=2, dim=1).mean().item()
    grad_abs_mean = torch.mean(torch.abs(grad)).item()
    grad_abs_max = torch.max(torch.abs(grad)).item()
    return {
        "grad_l2": grad_l2,
        "grad_abs_mean": grad_abs_mean,
        "grad_abs_max": grad_abs_max,
    }


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _backbone_name(model, default: Optional[str] = None) -> str:
    backbone = getattr(model, "_backbone_name", None)
    if backbone:
        return str(backbone)
    base = _unwrap_model(model)
    backbone = getattr(base, "_backbone_name", None)
    if backbone:
        return str(backbone)
    if default is not None:
        return default
    raise ValueError("Missing _backbone_name on proxy model")


def _module_device(module, fallback: torch.device) -> torch.device:
    tagged = getattr(module, "_inference_device", None)
    if tagged is not None:
        return torch.device(str(tagged))
    param = next(module.parameters(), None)
    if param is not None:
        return param.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return fallback


def _resolve_cap_for_model(value, backbone_name: str) -> float:
    if isinstance(value, dict):
        if backbone_name in value:
            return float(value[backbone_name])
        if "_default_" in value:
            return float(value["_default_"])
        raise KeyError(f"Missing dropout cap for backbone '{backbone_name}'")
    return float(value)


def _set_proxy_dropout_caps(ensemble_extractor, attn_cap, ffn_cap) -> None:
    if not hasattr(ensemble_extractor, "extractors"):
        return
    for model in ensemble_extractor.extractors:
        proxy_model = _unwrap_model(model)
        backbone = _backbone_name(proxy_model)
        current_attn_cap = max(0.0, min(1.0, _resolve_cap_for_model(attn_cap, backbone)))
        current_ffn_cap = max(0.0, min(1.0, _resolve_cap_for_model(ffn_cap, backbone)))
        if hasattr(proxy_model, "set_dropout_cap"):
            proxy_model.set_dropout_cap(current_attn_cap, current_ffn_cap)


def _section_backbone_override(section, backbone_name: str, key: str, default):
    overrides = getattr(section, "backbone_overrides", None)
    if overrides is None:
        return default
    backbone_cfg = getattr(overrides, backbone_name, None)
    if backbone_cfg is None:
        return default
    value = getattr(backbone_cfg, key, None)
    return default if value is None else value


def _compute_dropout_caps_for_section(section, epoch: int, steps: int, schedule_enabled: bool) -> Tuple[float, float]:
    progress = 1.0 if steps <= 1 else float(epoch) / float(steps - 1)
    attn_min = float(getattr(section, "attention_dropout_min", 0.0))
    ffn_min = float(getattr(section, "ffn_dropout_min", 0.0))
    attn_max = float(getattr(section, "attention_dropout_max", getattr(section, "attention_dropout", 0.0)))
    ffn_max = float(getattr(section, "ffn_dropout_max", getattr(section, "ffn_dropout", 0.0)))
    if schedule_enabled:
        attn_cap = attn_min + (attn_max - attn_min) * progress
        ffn_cap = ffn_min + (ffn_max - ffn_min) * progress
    else:
        attn_cap = float(getattr(section, "attention_dropout", attn_max))
        ffn_cap = float(getattr(section, "ffn_dropout", ffn_max))
    attn_cap = max(0.0, min(1.0, attn_cap))
    ffn_cap = max(0.0, min(1.0, ffn_cap))
    return attn_cap, ffn_cap


def _get_image_dropout_caps(cfg: Any, epoch: int) -> Tuple[float, float]:
    steps = max(1, int(getattr(cfg.optim, "steps", 1)))
    schedule_enabled = bool(getattr(cfg.model, "dropout_schedule", False))
    backbone_overrides = getattr(cfg.model, "backbone_overrides", None)
    if backbone_overrides is None:
        return _compute_dropout_caps_for_section(cfg.model, epoch, steps, schedule_enabled)
    attn_caps: Dict[str, float] = {}
    ffn_caps: Dict[str, float] = {}
    for backbone_name in list(getattr(cfg.model, "backbones", [])):
        section = getattr(backbone_overrides, backbone_name, None)
        if section is None:
            section = cfg.model
        attn_cap, ffn_cap = _compute_dropout_caps_for_section(section, epoch, steps, schedule_enabled)
        attn_caps[str(backbone_name)] = attn_cap
        ffn_caps[str(backbone_name)] = ffn_cap
    attn_caps["_default_"], ffn_caps["_default_"] = _compute_dropout_caps_for_section(cfg.model, epoch, steps, schedule_enabled)
    return attn_caps, ffn_caps


def _get_text_dropout_caps(cfg: Any, epoch: int) -> Tuple[float, float]:
    steps = max(1, int(getattr(cfg.optim, "steps", 1)))
    text_cfg = getattr(cfg, "text_objective", None)
    if text_cfg is None:
        return 0.0, 0.0
    backbone_overrides = getattr(text_cfg, "backbone_overrides", None)
    if backbone_overrides is None:
        return _compute_dropout_caps_for_section(text_cfg, epoch, steps, True)
    attn_caps: Dict[str, float] = {}
    ffn_caps: Dict[str, float] = {}
    for backbone_name in list(getattr(cfg.model, "backbones", [])):
        section = getattr(backbone_overrides, backbone_name, None)
        if section is None:
            section = text_cfg
        attn_cap, ffn_cap = _compute_dropout_caps_for_section(section, epoch, steps, True)
        attn_caps[str(backbone_name)] = attn_cap
        ffn_caps[str(backbone_name)] = ffn_cap
    attn_caps["_default_"], ffn_caps["_default_"] = _compute_dropout_caps_for_section(text_cfg, epoch, steps, True)
    return attn_caps, ffn_caps


def _text_loss_enabled(
    cfg: Any,
    source_caption_banks: Optional[List[List[str]]],
    target_caption_banks: Optional[List[List[str]]],
) -> bool:
    return (
        bool(getattr(cfg.text_objective, "enabled", False))
        and source_caption_banks is not None
        and target_caption_banks is not None
        and len(source_caption_banks) > 0
        and len(target_caption_banks) == len(source_caption_banks)
    )


def _get_text_alpha(cfg: Any) -> float:
    return float(getattr(cfg.text_objective, "alpha", 1.0))
def _get_objective_variance_weights(cfg: Any) -> Tuple[float, float]:
    objective_cfg = getattr(cfg, "oattack_objective", None)
    if objective_cfg is None:
        return 1.0, 1.0
    return (
        float(getattr(objective_cfg, "visual_variance_weight", 1.0)),
        float(getattr(objective_cfg, "text_variance_weight", 1.0)),
    )

def _get_aggregation_mode(cfg: Any) -> str:
    objective_cfg = getattr(cfg, "oattack_objective", None)
    if objective_cfg is None:
        return "per_encoder_mean"
    return str(getattr(objective_cfg, "aggregation_mode", "per_encoder_mean")).lower()


def _get_variance_mode(cfg: Any) -> str:
    objective_cfg = getattr(cfg, "oattack_objective", None)
    if objective_cfg is None:
        return "add"
    return str(getattr(objective_cfg, "variance_mode", "add")).lower()

def _per_component_similarity(feature: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 2:
        return torch.sum(feature * gt, dim=1, keepdim=True)
    if feature.ndim == 3:
        return torch.sum(feature * gt, dim=2)
    raise ValueError(f"Unsupported feature ndim={feature.ndim}, expected 2 or 3")


def _compute_component_variance(components: torch.Tensor, device: torch.device) -> torch.Tensor:
    if components.numel() <= 1:
        return torch.tensor(0.0, device=device)
    return torch.var(components.reshape(-1), unbiased=False)

def _mix_two_captions(candidates: List[str], beta_min: float, beta_max: float) -> Tuple[str, float]:
    pool = [str(x).strip() for x in candidates if str(x).strip()]
    if not pool:
        return "", 0.5
    if len(pool) == 1:
        return pool[0], 0.5

    c1, c2 = random.sample(pool, 2)
    words1 = c1.split()
    words2 = c2.split()

    beta_low = max(0.0, min(1.0, float(beta_min)))
    beta_high = max(0.0, min(1.0, float(beta_max)))
    if beta_low > beta_high:
        beta_low, beta_high = beta_high, beta_low
    beta = random.uniform(beta_low, beta_high)

    m = len(words1)
    n = len(words2)
    p1 = int(beta * m)
    p2 = int((1.0 - beta) * n)
    mixed = words1[:p1] + (words2[-p2:] if p2 > 0 else [])
    if not mixed:
        mixed = words1 or words2
    return " ".join(mixed), beta


def _get_text_mix_mode(cfg: Any) -> str:
    return str(getattr(cfg.text_objective, "mix_mode", "pair")).lower()


def _mix_captions(cfg: Any, candidates: List[str], beta_min: float, beta_max: float) -> Tuple[str, float]:
    mix_mode = _get_text_mix_mode(cfg)
    pool = [str(x).strip() for x in candidates if str(x).strip()]
    if mix_mode in {"pair", "two"}:
        return _mix_two_captions(pool, beta_min, beta_max)
    if mix_mode in {"triplet_concat", "3mix_concat", "three_concat"}:
        if len(pool) < 3:
            return _mix_two_captions(pool, beta_min, beta_max)
        anchor, c1, c2 = random.sample(pool, 3)
        fused, beta = _mix_two_captions([c1, c2], beta_min, beta_max)
        if fused:
            return f"{anchor} {fused}".strip(), beta
        return anchor, beta
    raise ValueError(f"Unsupported text mix mode: {mix_mode}")

def _mean_pool_feature_layers(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 2:
        return feature
    if feature.ndim == 3:
        pooled = torch.mean(feature, dim=1)
        return pooled / pooled.norm(dim=1, keepdim=True).clamp_min(1e-12)
    raise ValueError(f"Unsupported feature ndim={feature.ndim}, expected 2 or 3")


def _collect_encoder_target_scores(
    adv_features: Dict[int, torch.Tensor],
    target_features: Dict[int, torch.Tensor],
    aggregation_mode: str = "per_encoder_mean",
) -> torch.Tensor:
    scores: List[torch.Tensor] = []
    output_device = None
    for index in adv_features.keys():
        adv_feat = adv_features[index]
        output_device = adv_feat.device
        tgt_feat = EnsembleFeatureLoss._align_ground_truth(adv_feat, target_features[index])
        if aggregation_mode == "per_model_layeravg_textpooled":
            per_layer = _per_component_similarity(adv_feat, tgt_feat)
            scores.append(torch.mean(per_layer, dim=1).reshape(-1))
        else:
            adv_mean = _mean_pool_feature_layers(adv_feat)
            tgt_mean = _mean_pool_feature_layers(tgt_feat)
            scores.append(torch.sum(adv_mean * tgt_mean, dim=1).reshape(-1))
    if not scores:
        device = output_device if output_device is not None else torch.device("cpu")
        return torch.empty(0, device=device)
    return torch.cat(scores, dim=0)

def _compute_text_objective(
    cfg: Any,
    ensemble_extractor,
    image_features: Dict[int, torch.Tensor],
    source_caption_banks: Optional[List[List[str]]],
    target_caption_banks: Optional[List[List[str]]],
    output_device: torch.device,
    repeats: int,
    epoch: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if not _text_loss_enabled(cfg, source_caption_banks, target_caption_banks):
        zero = torch.tensor(0.0, device=output_device)
        return zero, zero, {}

    expanded_source_banks = _expand_caption_banks(source_caption_banks, repeats)
    expanded_target_banks = _expand_caption_banks(target_caption_banks, repeats)
    assert expanded_source_banks is not None and expanded_target_banks is not None

    beta_min = float(getattr(cfg.text_objective, "mix_beta_min", 0.0))
    beta_max = float(getattr(cfg.text_objective, "mix_beta_max", 1.0))
    source_texts: List[str] = []
    target_texts: List[str] = []
    betas: List[float] = []
    for src_bank, tgt_bank in zip(expanded_source_banks, expanded_target_banks):
        src_text, beta = _mix_captions(cfg, src_bank, beta_min, beta_max)
        tgt_text, _ = _mix_captions(cfg, tgt_bank, beta_min, beta_max)
        source_texts.append(src_text)
        target_texts.append(tgt_text)
        betas.append(beta)

    img_attn_cap, img_ffn_cap = _get_image_dropout_caps(cfg, epoch)
    text_attn_cap, text_ffn_cap = _get_text_dropout_caps(cfg, epoch)
    _set_proxy_dropout_caps(ensemble_extractor, text_attn_cap, text_ffn_cap)
    text_states = _sample_and_lock_proxy_dropout(ensemble_extractor)
    try:
        diff_scores: List[torch.Tensor] = []
        target_scores: List[torch.Tensor] = []
        source_scores: List[torch.Tensor] = []
        aggregation_mode = _get_aggregation_mode(cfg)
        for model_index, model in enumerate(ensemble_extractor.extractors):
            text_model = _unwrap_model(model)
            if not hasattr(text_model, "encode_text") or model_index not in image_features:
                continue
            model_device = str(_module_device(text_model, fallback=output_device))
            with torch.no_grad():
                src_feat = text_model.encode_text(source_texts, device=model_device)
                tgt_feat = text_model.encode_text(target_texts, device=model_device)
                if src_feat.device != output_device:
                    src_feat = src_feat.to(output_device, non_blocking=True)
                if tgt_feat.device != output_device:
                    tgt_feat = tgt_feat.to(output_device, non_blocking=True)
                src_feat = src_feat.detach()
                tgt_feat = tgt_feat.detach()

            image_feat = image_features[model_index]
            src_mean = _mean_pool_feature_layers(src_feat)
            tgt_mean = _mean_pool_feature_layers(tgt_feat)
            if aggregation_mode == "per_model_layeravg_textpooled" and image_feat.ndim == 3:
                tgt_expand = tgt_mean.unsqueeze(1).expand(-1, image_feat.size(1), -1)
                src_expand = src_mean.unsqueeze(1).expand(-1, image_feat.size(1), -1)
                target_score = torch.mean(torch.sum(image_feat * tgt_expand, dim=2), dim=1)
                source_score = torch.mean(torch.sum(image_feat * src_expand, dim=2), dim=1)
            else:
                image_mean = _mean_pool_feature_layers(image_feat)
                target_score = torch.sum(image_mean * tgt_mean, dim=1)
                source_score = torch.sum(image_mean * src_mean, dim=1)
            diff_scores.append((target_score - source_score).reshape(-1))
            target_scores.append(target_score.reshape(-1))
            source_scores.append(source_score.reshape(-1))
    finally:
        _restore_proxy_dropout(text_states)
        _set_proxy_dropout_caps(ensemble_extractor, img_attn_cap, img_ffn_cap)

    if not diff_scores:
        zero = torch.tensor(0.0, device=output_device)
        return zero, zero, {}

    flat_diff = torch.cat(diff_scores, dim=0)
    flat_target = torch.cat(target_scores, dim=0)
    flat_source = torch.cat(source_scores, dim=0)
    metrics = {
        "text_similarity": float(torch.mean(flat_diff).item()),
        "text_similarity_to_target": float(torch.mean(flat_target).item()),
        "text_similarity_to_source": float(torch.mean(flat_source).item()),
        "text_mix_beta_mean": float(sum(betas) / len(betas)) if betas else 0.0,
    }
    return (
        torch.mean(flat_diff),
        _compute_component_variance(flat_diff, output_device),
        metrics,
    )


def _set_dropout_schedule(cfg: Any, ensemble_extractor, epoch: int) -> Dict[str, float]:
    if not bool(getattr(cfg.model, "stochastic_proxy", False)):
        return {}
    attn_cap, ffn_cap = _get_image_dropout_caps(cfg, epoch)
    _set_proxy_dropout_caps(ensemble_extractor, attn_cap, ffn_cap)

    def _metric_value(val):
        if isinstance(val, dict):
            numeric = [float(x) for k, x in val.items() if k != "_default_"]
            if numeric:
                return float(sum(numeric) / len(numeric))
            return float(val.get("_default_", 0.0))
        return float(val)

    metrics = {
        "proxy_attn_dropout_cap": _metric_value(attn_cap),
        "proxy_ffn_dropout_cap": _metric_value(ffn_cap),
    }
    if isinstance(attn_cap, dict):
        for backbone_name, value in attn_cap.items():
            if backbone_name != "_default_":
                metrics[f"proxy_attn_dropout_cap_{backbone_name}"] = float(value)
    if isinstance(ffn_cap, dict):
        for backbone_name, value in ffn_cap.items():
            if backbone_name != "_default_":
                metrics[f"proxy_ffn_dropout_cap_{backbone_name}"] = float(value)
    if bool(getattr(cfg.text_objective, "enabled", False)):
        text_attn_cap, text_ffn_cap = _get_text_dropout_caps(cfg, epoch)
        metrics["text_proxy_attn_dropout_cap"] = _metric_value(text_attn_cap)
        metrics["text_proxy_ffn_dropout_cap"] = _metric_value(text_ffn_cap)
        if isinstance(text_attn_cap, dict):
            for backbone_name, value in text_attn_cap.items():
                if backbone_name != "_default_":
                    metrics[f"text_proxy_attn_dropout_cap_{backbone_name}"] = float(value)
        if isinstance(text_ffn_cap, dict):
            for backbone_name, value in text_ffn_cap.items():
                if backbone_name != "_default_":
                    metrics[f"text_proxy_ffn_dropout_cap_{backbone_name}"] = float(value)
    return metrics
def _normalize_source_aug_execution(cfg: Any) -> str:
    mode = str(getattr(cfg.optim, "source_aug_execution", "strict")).strip().lower().replace("-", "_")
    if mode in {"strict", "legacy"}:
        return "strict"
    if mode in {"batched", "vectorized", "fast"}:
        return "batched"
    raise ValueError("optim.source_aug_execution must be one of: strict | batched")

def _sample_and_lock_proxy_dropout(ensemble_extractor):
    states = []
    if not hasattr(ensemble_extractor, "extractors"):
        return states
    for model in ensemble_extractor.extractors:
        proxy_model = _unwrap_model(model)
        get_fn = getattr(proxy_model, "get_sample_per_forward", None)
        set_fn = getattr(proxy_model, "set_sample_per_forward", None)
        sample_fn = getattr(proxy_model, "sample_dropout_once", None)
        if get_fn is None or set_fn is None:
            continue
        prev = bool(get_fn())
        states.append((set_fn, prev))
        # Keep target/source paired in this crop under the same sampled proxy state.
        if prev and sample_fn is not None:
            sample_fn()
            set_fn(False)
    return states


def _restore_proxy_dropout(states) -> None:
    for set_fn, prev in states:
        set_fn(prev)

def _expand_caption_banks(caption_banks: Optional[List[List[str]]], repeats: int) -> Optional[List[List[str]]]:
    if caption_banks is None:
        return None
    expanded: List[List[str]] = []
    for _ in range(repeats):
        expanded.extend([list(bank) for bank in caption_banks])
    return expanded


def _build_paired_source_batches(
    source_aug,
    adv_image: torch.Tensor,
    image_org: torch.Tensor,
    repeats: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = adv_image.size(0)
    paired = torch.cat([adv_image, image_org.detach()], dim=0)
    adv_batches: List[torch.Tensor] = []
    source_batches: List[torch.Tensor] = []
    for _ in range(repeats):
        augmented = source_aug(paired)
        adv_batches.append(augmented[:batch_size])
        source_batches.append(augmented[batch_size:])
    return torch.cat(adv_batches, dim=0), torch.cat(source_batches, dim=0)
@register_attack("oattack_pgd")
def oattack_pgd_attack(
    cfg: Any,
    ensemble_extractor,
    ensemble_loss,
    source_aug,
    target_aug,
    image_org: torch.Tensor,
    image_tgt: Optional[torch.Tensor],
    target_texts: Optional[List[str]],
    source_caption_banks: Optional[List[List[str]]],
    target_caption_banks: Optional[List[List[str]]],
    logger,
    img_index: int,
    step_saver=None,
) -> torch.Tensor:
    target_type = str(getattr(cfg.optim, "target_type", "image")).lower()
    if target_type != "image":
        raise ValueError("oattack_pgd only supports optim.target_type=image")
    if image_tgt is None:
        raise ValueError("oattack_pgd requires image_tgt")

    eps_bound = float(cfg.optim.epsilon)
    epsilon = torch.zeros_like(image_org, requires_grad=True)
    optimizer = torch.optim.Adam([epsilon], lr=float(cfg.optim.alpha))
    repeats = max(1, int(getattr(cfg.optim, "source_aug_crops", 10)))
    source_aug_execution = _normalize_source_aug_execution(cfg)
    text_alpha = _get_text_alpha(cfg)
    visual_var_weight, text_var_weight = _get_objective_variance_weights(cfg)
    aggregation_mode = _get_aggregation_mode(cfg)
    variance_mode = _get_variance_mode(cfg)
    pbar = tqdm(range(int(cfg.optim.steps)), desc="Attack progress")

    for epoch in pbar:
        schedule_metrics = _set_dropout_schedule(cfg, ensemble_extractor, epoch)
        pair_states = _sample_and_lock_proxy_dropout(ensemble_extractor)
        try:
            adv_image = image_org + epsilon
            if source_aug_execution == "batched":
                adv_batch, _source_batch = _build_paired_source_batches(
                    source_aug=source_aug,
                    adv_image=adv_image,
                    image_org=image_org,
                    repeats=repeats,
                )

                with torch.no_grad():
                    target_batch = torch.cat([target_aug(image_tgt) for _ in range(repeats)], dim=0)
                    target_features = ensemble_extractor(target_batch)

                adv_features = ensemble_extractor(adv_batch)
                visual_components = _collect_encoder_target_scores(
                    adv_features=adv_features,
                    target_features=target_features,
                    aggregation_mode=aggregation_mode,
                )
                text_mean, text_variance, text_metrics = _compute_text_objective(
                    cfg=cfg,
                    ensemble_extractor=ensemble_extractor,
                    image_features=adv_features,
                    source_caption_banks=source_caption_banks,
                    target_caption_banks=target_caption_banks,
                    output_device=adv_image.device,
                    repeats=repeats,
                    epoch=epoch,
                )
            else:
                visual_components_list: List[torch.Tensor] = []
                text_similarity_vals: List[float] = []
                text_target_vals: List[float] = []
                text_source_vals: List[float] = []
                text_mix_beta_vals: List[float] = []
                text_score_list: List[torch.Tensor] = []

                for _ in range(repeats):
                    adv_single, _source_single = _build_paired_source_batches(
                        source_aug=source_aug,
                        adv_image=adv_image,
                        image_org=image_org,
                        repeats=1,
                    )
                    with torch.no_grad():
                        target_single = target_aug(image_tgt)
                        target_features = ensemble_extractor(target_single)

                    crop_adv_features = ensemble_extractor(adv_single)
                    crop_visual_components = _collect_encoder_target_scores(
                        adv_features=crop_adv_features,
                        target_features=target_features,
                        aggregation_mode=aggregation_mode,
                    )
                    if crop_visual_components.numel() > 0:
                        visual_components_list.append(crop_visual_components.reshape(-1))

                    crop_text_mean, crop_text_variance, crop_text_metrics = _compute_text_objective(
                        cfg=cfg,
                        ensemble_extractor=ensemble_extractor,
                        image_features=crop_adv_features,
                        source_caption_banks=source_caption_banks,
                        target_caption_banks=target_caption_banks,
                        output_device=adv_image.device,
                        repeats=1,
                        epoch=epoch,
                    )
                    if crop_text_metrics:
                        text_score_list.append(crop_text_mean.reshape(1))
                    if "text_similarity" in crop_text_metrics:
                        text_similarity_vals.append(float(crop_text_metrics["text_similarity"]))
                    if "text_similarity_to_target" in crop_text_metrics:
                        text_target_vals.append(float(crop_text_metrics["text_similarity_to_target"]))
                    if "text_similarity_to_source" in crop_text_metrics:
                        text_source_vals.append(float(crop_text_metrics["text_similarity_to_source"]))
                    if "text_mix_beta_mean" in crop_text_metrics:
                        text_mix_beta_vals.append(float(crop_text_metrics["text_mix_beta_mean"]))

                if visual_components_list:
                    visual_components = torch.cat(visual_components_list, dim=0)
                else:
                    visual_components = torch.empty(0, device=adv_image.device)

                if text_score_list:
                    all_text_scores = torch.cat(text_score_list, dim=0)
                    text_mean = torch.mean(all_text_scores)
                    text_variance = _compute_component_variance(all_text_scores, adv_image.device)
                    text_metrics = {
                        "text_similarity": sum(text_similarity_vals) / len(text_similarity_vals) if text_similarity_vals else float(text_mean.item()),
                        "text_similarity_to_target": sum(text_target_vals) / len(text_target_vals) if text_target_vals else 0.0,
                        "text_similarity_to_source": sum(text_source_vals) / len(text_source_vals) if text_source_vals else 0.0,
                        "text_mix_beta_mean": sum(text_mix_beta_vals) / len(text_mix_beta_vals) if text_mix_beta_vals else 0.0,
                    }
                else:
                    text_mean = torch.tensor(0.0, device=adv_image.device)
                    text_variance = torch.tensor(0.0, device=adv_image.device)
                    text_metrics = {}

            visual_mean = torch.mean(visual_components) if visual_components.numel() > 0 else torch.tensor(0.0, device=adv_image.device)
            visual_variance = _compute_component_variance(visual_components, adv_image.device)
            if variance_mode == "subtract":
                total_objective = (
                    visual_mean
                    - visual_var_weight * visual_variance
                    + text_alpha * (text_mean - text_var_weight * text_variance)
                )
            elif variance_mode == "add":
                total_objective = (
                    visual_mean
                    + visual_var_weight * visual_variance
                    + text_alpha * (text_mean + text_var_weight * text_variance)
                )
            else:
                raise ValueError(f"Unsupported oattack variance_mode: {variance_mode}")
            grad = torch.autograd.grad(total_objective, epsilon, create_graph=False)[0]
        finally:
            _restore_proxy_dropout(pair_states)

        metrics = {
            "max_delta": torch.max(torch.abs(epsilon)).item(),
            "mean_delta": torch.mean(torch.abs(epsilon)).item(),
            "visual_mean": float(visual_mean.item()),
            "visual_variance": float(visual_variance.item()),
            "text_mean": float(text_mean.item()),
            "text_variance": float(text_variance.item()),
            "text_alpha": text_alpha,
            "visual_variance_weight": visual_var_weight,
            "text_variance_weight": text_var_weight,
            "variance_mode": variance_mode,
            "visual_mode": "target_only",
            "source_aug_crops": float(repeats),
            "objective": float(total_objective.item()),
            "optimizer": str(getattr(getattr(cfg, "oattack_objective", None), "optimizer", "adam")),
            "aggregation_mode": aggregation_mode,
        }
        metrics.update(text_metrics)
        metrics.update(schedule_metrics)
        metrics.update(_summarize_gradient(grad))
        _log_step_metrics(pbar, logger, img_index, epoch, metrics)

        optimizer.zero_grad()
        epsilon.grad = -grad
        optimizer.step()
        epsilon.data = torch.clamp(epsilon, min=-eps_bound, max=eps_bound)
        if step_saver is not None:
            current_step = epoch + 1
            adv_snapshot = torch.clamp((image_org + epsilon) / 255.0, 0.0, 1.0)
            step_saver(current_step, adv_snapshot)

    adv_image = torch.clamp((image_org + epsilon) / 255.0, 0.0, 1.0)
    logger.log(
        {
            f"img{img_index}_final_max_delta": torch.max(torch.abs(epsilon)).item(),
            f"img{img_index}_final_mean_delta": torch.mean(torch.abs(epsilon)).item(),
        }
    )
    return adv_image



def get_attack(name: str):
    if name not in ATTACK_REGISTRY:
        valid = ", ".join(sorted(ATTACK_REGISTRY.keys()))
        raise ValueError(f"Unknown attack '{name}'. Valid attacks: {valid}")
    return ATTACK_REGISTRY[name]
