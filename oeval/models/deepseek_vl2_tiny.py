from typing import Any, Dict, Optional

import torch

from oeval.base import BaseVLMEvaluator
from oeval.registry import register_vlm

try:
    from transformers import AutoModelForCausalLM
except Exception as exc:  # pragma: no cover - handled at runtime
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

try:
    from deepseek_vl2.models import DeepseekVLV2Processor
    from deepseek_vl2.utils.io import load_pil_images
except Exception as exc_v2:  # pragma: no cover - handled at runtime
    try:
        from deepseek_vl.models import DeepseekVLV2Processor
        from deepseek_vl.utils.io import load_pil_images
    except Exception as exc_v1:  # pragma: no cover - handled at runtime
        DeepseekVLV2Processor = None
        load_pil_images = None
        _DEEPSEEK_VL_IMPORT_ERROR = exc_v1
    else:
        _DEEPSEEK_VL_IMPORT_ERROR = None
else:
    _DEEPSEEK_VL_IMPORT_ERROR = None


def _require_deepseek_vl2():
    if AutoModelForCausalLM is None:
        raise ImportError(
            "Transformers is unavailable for deepseek-vl2."
        ) from _TRANSFORMERS_IMPORT_ERROR
    if DeepseekVLV2Processor is None or load_pil_images is None:
        raise ImportError(
            "deepseek_vl is not installed. Install the DeepSeek-VL2 code package "
            "before using deepseek-vl2-tiny in this evaluator."
        ) from _DEEPSEEK_VL_IMPORT_ERROR


def _parse_dtype(dtype: str):
    normalized = dtype.lower().strip()
    if normalized == "auto":
        return torch.bfloat16
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


def _resolve_model_device_and_dtype(model) -> tuple[torch.device, torch.dtype]:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device, param.dtype
    raise RuntimeError("Could not resolve DeepSeek-VL2 model device/dtype from parameters.")


@register_vlm("deepseek_vl2_tiny")
@register_vlm("deepseek_vl2")
class DeepSeekVL2TinyEvaluator(BaseVLMEvaluator):
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-vl2-tiny",
        model_dtype: str = "bf16",
        model_device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        local_files_only: bool = False,
    ):
        _require_deepseek_vl2()
        dtype = _parse_dtype(model_dtype)
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if model_device_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
            model_kwargs["device_map"] = model_device_map

        self.processor = DeepseekVLV2Processor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if model_device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
            self.model = self.model.to(model_device_map)
        self.model = self.model.eval()
        self.input_device, self.input_dtype = _resolve_model_device_and_dtype(self.model)

    def describe_image(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        )
        for key in list(prepare_inputs.keys()):
            value = prepare_inputs[key]
            if torch.is_tensor(value):
                if value.is_floating_point():
                    prepare_inputs[key] = value.to(device=self.input_device, dtype=self.input_dtype)
                else:
                    prepare_inputs[key] = value.to(device=self.input_device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        generated_tokens = outputs[0][prepare_inputs.input_ids.shape[1] :]
        decoded = self.tokenizer.decode(generated_tokens.cpu().tolist(), skip_special_tokens=True).strip()
        prefix = ""
        sft_format = getattr(prepare_inputs, "sft_format", None)
        if sft_format:
            prefix = str(sft_format[0]).strip()
        if prefix and decoded.startswith(prefix):
            decoded = decoded[len(prefix) :].strip()
        return decoded
