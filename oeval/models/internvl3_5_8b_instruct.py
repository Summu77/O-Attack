from typing import Any, Dict, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from oeval.base import BaseVLMEvaluator
from oeval.registry import register_vlm

try:
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:  # pragma: no cover - handled at runtime
    AutoModel = None
    AutoTokenizer = None
    _INTERNVL_IMPORT_ERROR = exc
else:
    _INTERNVL_IMPORT_ERROR = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_internvl():
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError(
            "InternVL dependencies are unavailable. "
            "Install transformers with trust_remote_code support for InternVL."
        ) from _INTERNVL_IMPORT_ERROR


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


def _build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_image(image_path: str, input_size: int = 448, max_num: int = 12):
    image = Image.open(image_path).convert("RGB")
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


@register_vlm("internvl3_5_8b_instruct")
@register_vlm("internvl_8b")
class InternVL35_8BInstructEvaluator(BaseVLMEvaluator):
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3_5-8B-Instruct",
        model_dtype: str = "bf16",
        model_device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        local_files_only: bool = False,
        max_num_tiles: int = 12,
    ):
        _require_internvl()
        dtype = _parse_dtype(model_dtype)
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "use_flash_attn": True,
        }
        if model_device_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
            model_kwargs["device_map"] = model_device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()
        if model_device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
            self.model = self.model.to(model_device_map)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=local_files_only,
        )
        self.max_num_tiles = max_num_tiles

    def describe_image(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        pixel_values = _load_image(image_path, max_num=self.max_num_tiles)
        device = next(self.model.parameters()).device
        target_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(device=device, dtype=target_dtype)
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        question = f"<image>\n{prompt}"
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        return response.strip() if isinstance(response, str) else str(response).strip()
