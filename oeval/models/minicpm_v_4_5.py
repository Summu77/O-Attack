from typing import Any, Dict, Optional

import torch
from PIL import Image

from oeval.base import BaseVLMEvaluator
from oeval.registry import register_vlm

try:
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel
except Exception as exc:  # pragma: no cover - handled at runtime
    AutoModel = None
    AutoTokenizer = None
    PreTrainedModel = None
    _MINICPM_IMPORT_ERROR = exc
else:
    _MINICPM_IMPORT_ERROR = None


def _require_minicpm():
    if AutoModel is None or AutoTokenizer is None or PreTrainedModel is None:
        raise ImportError(
            "MiniCPM-V-4_5 requires transformers AutoModel/AutoTokenizer with trust_remote_code support."
        ) from _MINICPM_IMPORT_ERROR


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


@register_vlm("minicpm_v_4_5")
@register_vlm("minicpm_v45")
class MiniCPMV45Evaluator(BaseVLMEvaluator):
    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-V-4_5",
        model_dtype: str = "bf16",
        model_device_map: str = "cuda:0",
        attn_implementation: Optional[str] = None,
        local_files_only: bool = False,
    ):
        _require_minicpm()
        dtype = _parse_dtype(model_dtype)
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
            "attn_implementation": attn_implementation or "sdpa",
        }

        if model_device_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
            model_kwargs["device_map"] = model_device_map
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        else:
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
            self.model = self.model.to(model_device_map)

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

    def describe_image(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [image, prompt]}]
        with torch.inference_mode():
            answer = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                enable_thinking=False,
                stream=False,
                max_new_tokens=max_new_tokens,
            )
        return answer.strip() if isinstance(answer, str) else str(answer).strip()
