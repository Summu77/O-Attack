from typing import Any, Dict, Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from oeval.base import BaseVLMEvaluator
from oeval.registry import register_vlm


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


@register_vlm("qwen3_vl_8b_instruct")
class Qwen3VL8BInstructEvaluator(BaseVLMEvaluator):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        model_dtype: str = "auto",
        model_device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        local_files_only: bool = False,
    ):
        model_kwargs: Dict[str, Any] = {
            "dtype": _parse_dtype(model_dtype),
            "device_map": model_device_map,
            "local_files_only": local_files_only,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)

    def describe_image(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip() if output_text else ""
