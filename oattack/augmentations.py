from typing import Any, Callable, Dict, List

import torch
import torchvision.transforms as T

AUGMENT_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Callable]] = {}


def register_augment(name: str):
    def decorator(builder: Callable[[Dict[str, Any]], Callable]):
        AUGMENT_REGISTRY[name] = builder
        return builder

    return decorator


@register_augment("identity")
def build_identity(_: Dict[str, Any]) -> Callable:
    return torch.nn.Identity()


@register_augment("random_resized_crop")
def build_random_resized_crop(cfg: Dict[str, Any]) -> Callable:
    return T.RandomResizedCrop(
        size=int(cfg["size"]),
        scale=(float(cfg.get("scale", [0.5, 0.9])[0]), float(cfg.get("scale", [0.5, 0.9])[1])),
    )


@register_augment("center_crop")
def build_center_crop(cfg: Dict[str, Any]) -> Callable:
    return T.CenterCrop(size=int(cfg["size"]))


@register_augment("resize")
def build_resize(cfg: Dict[str, Any]) -> Callable:
    return T.Resize(size=int(cfg["size"]), interpolation=T.InterpolationMode.BICUBIC)


@register_augment("random_horizontal_flip")
def build_random_horizontal_flip(cfg: Dict[str, Any]) -> Callable:
    return T.RandomHorizontalFlip(p=float(cfg.get("p", 0.5)))


def build_pipeline(specs: List[Dict[str, Any]]) -> Callable:
    if not specs:
        return torch.nn.Identity()

    ops = []
    for spec in specs:
        name = spec["type"]
        if name not in AUGMENT_REGISTRY:
            valid = ", ".join(sorted(AUGMENT_REGISTRY.keys()))
            raise ValueError(f"Unknown augmentation '{name}'. Valid: {valid}")
        ops.append(AUGMENT_REGISTRY[name](spec))

    if len(ops) == 1:
        return ops[0]
    return T.Compose(ops)
