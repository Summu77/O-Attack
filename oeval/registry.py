from typing import Any, Callable, Dict, List

from .base import BaseVLMEvaluator

VLM_EVAL_REGISTRY: Dict[str, Callable[..., BaseVLMEvaluator]] = {}


def register_vlm(name: str):
    def decorator(builder: Callable[..., BaseVLMEvaluator]):
        VLM_EVAL_REGISTRY[name] = builder
        return builder

    return decorator


def create_vlm(name: str, **kwargs: Any) -> BaseVLMEvaluator:
    if name not in VLM_EVAL_REGISTRY:
        valid = ", ".join(sorted(VLM_EVAL_REGISTRY.keys()))
        raise ValueError(f"Unknown VLM '{name}'. Valid: {valid}")
    return VLM_EVAL_REGISTRY[name](**kwargs)


def list_vlms() -> List[str]:
    return sorted(VLM_EVAL_REGISTRY.keys())
