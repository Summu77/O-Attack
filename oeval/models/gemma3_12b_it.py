from typing import Any, Dict, Optional

import re

import torch

from oeval.base import BaseVLMEvaluator
from oeval.registry import register_vlm

try:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
except Exception as exc:  # pragma: no cover - handled at runtime
    AutoProcessor = None
    Gemma3ForConditionalGeneration = None
    _GEMMA3_IMPORT_ERROR = exc
else:
    _GEMMA3_IMPORT_ERROR = None


def _require_gemma3():
    if Gemma3ForConditionalGeneration is None or AutoProcessor is None:
        raise ImportError(
            "Gemma3ForConditionalGeneration is not available. "
            "Install a newer transformers version (>=4.56) to use gemma-3 models."
        ) from _GEMMA3_IMPORT_ERROR


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


def _clean_gemma_output(text: str) -> str:
    if not text:
        return text
    cleaned = text.strip()
    if "\n\n" in cleaned:
        head, tail = cleaned.split("\n\n", 1)
        if re.search(r"description of the image|short sentence", head, flags=re.IGNORECASE):
            cleaned = tail.strip()
    cleaned = re.sub(
        r"^\s*here[’']?s [^\n]*?(description of the image|short sentence)[^\n]*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


@register_vlm("gemma3_12b_it")
@register_vlm("gemma_12b")
class Gemma3_12BITEvaluator(BaseVLMEvaluator):
    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-it",
        model_dtype: str = "auto",
        model_device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        local_files_only: bool = False,
    ):
        _require_gemma3()
        model_kwargs: Dict[str, Any] = {
            "dtype": _parse_dtype(model_dtype),
            "local_files_only": local_files_only,
        }
        if model_device_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
            model_kwargs["device_map"] = model_device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        if model_device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
            self.model = self.model.to(model_device_map)

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)

    def describe_image(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not output_text:
            return ""
        return _clean_gemma_output(output_text[0])
