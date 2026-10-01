from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

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


@register_vlm("llava_16_mistral_7b_hf")
@register_vlm("llava_16_7b_hf")
class Llava16Mistral7BEvaluator(BaseVLMEvaluator):
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        model_dtype: str = "fp16",
        model_device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        local_files_only: bool = False,
    ):
        dtype = _parse_dtype(model_dtype)
        model_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        if model_device_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
            model_kwargs["device_map"] = model_device_map
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        else:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
            self.model = self.model.to(model_device_map)

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        if getattr(self.processor, "patch_size", None) is None:
            self.processor.patch_size = getattr(self.model.config.vision_config, "patch_size", None)
        if getattr(self.processor, "vision_feature_select_strategy", None) is None:
            self.processor.vision_feature_select_strategy = getattr(
                self.model.config, "vision_feature_select_strategy", "default"
            )
        if getattr(self.processor, "num_additional_image_tokens", None) in (None, 0):
            self.processor.num_additional_image_tokens = 1

    def describe_image(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        try:
            inputs = inputs.to(self.model.device, self.model.dtype)
        except TypeError:
            inputs = inputs.to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        begin_token = inputs["input_ids"].shape[1]
        return self.processor.decode(output[0][begin_token:], skip_special_tokens=True).strip()
