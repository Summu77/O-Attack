from typing import Any, Dict, Optional

import torch

from oeval.base import BaseVLMEvaluator
from oeval.registry import register_vlm

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except Exception as exc:  # pragma: no cover - handled at runtime
    AutoModelForImageTextToText = None
    AutoProcessor = None
    _MOLMO_IMPORT_ERROR = exc
else:
    _MOLMO_IMPORT_ERROR = None


def _require_molmo2():
    if AutoModelForImageTextToText is None or AutoProcessor is None:
        raise ImportError(
            "Molmo2 requires a transformers build with AutoModelForImageTextToText support."
        ) from _MOLMO_IMPORT_ERROR


def _parse_dtype(dtype: str):
    normalized = dtype.lower().strip()
    if normalized == "auto":
        return "auto"
    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use one of: auto, fp16, bf16, fp32")
    return dtype_map[normalized]


def _resolve_input_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@register_vlm("molmo2_8b")
@register_vlm("molmo_8b")
class Molmo2_8BEvaluator(BaseVLMEvaluator):
    def __init__(
        self,
        model_name: str = "allenai/Molmo2-8B",
        model_dtype: str = "auto",
        model_device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        local_files_only: bool = False,
    ):
        _require_molmo2()
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": _parse_dtype(model_dtype),
            "local_files_only": local_files_only,
        }
        if model_device_map == "auto":
            model_kwargs["device_map"] = "balanced_low_0"
        elif model_device_map in {"balanced", "balanced_low_0", "sequential"}:
            model_kwargs["device_map"] = model_device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        if model_device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
            self.model = self.model.to(model_device_map)
        self.model.eval()
        self.input_device = _resolve_input_device(self.model)

    def describe_image(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image_path},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
