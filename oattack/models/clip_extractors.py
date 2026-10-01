import os
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer, CLIPModel

from .base import BaseFeatureExtractor

MODEL_REGISTRY: Dict[str, Callable[[], BaseFeatureExtractor]] = {}


def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


class _CLIPExtractor(BaseFeatureExtractor):
    model_id: str = ""
    revision: str = ""
    image_size: int = 224

    def __init__(self):
        super().__init__()
        local_only = os.getenv("OATTACK_LOCAL_FILES_ONLY", "0").lower() not in {"0", "false", "no"}
        cache_dir = _resolve_cache_dir()
        model_ref = _resolve_model_ref(self.model_id)
        kwargs = {"local_files_only": local_only}
        if self.revision and not os.path.isdir(model_ref):
            kwargs["revision"] = self.revision
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.model = CLIPModel.from_pretrained(model_ref, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref, **kwargs)
        self._cls_last_k = 1
        self._text_last_k = 1
        self._attention_modules = []
        self._ffn_dropout_hooks = []
        self._stochastic_enabled = False
        self._dropout_cap_attention = 0.0
        self._dropout_cap_ffn = 0.0
        self._current_attention_dropout = 0.0
        self._current_ffn_dropout = 0.0
        self._sample_per_forward = True
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(
                    self.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                transforms.CenterCrop(self.image_size),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    @staticmethod
    def _clip_prob(p: float) -> float:
        return float(max(0.0, min(1.0, p)))

    def _apply_attention_dropout(self, p: float) -> None:
        p = self._clip_prob(p)
        for module in self._attention_modules:
            module.dropout = p

    def _set_current_dropout(self, attention_dropout: float, ffn_dropout: float) -> None:
        self._current_attention_dropout = self._clip_prob(attention_dropout)
        self._current_ffn_dropout = self._clip_prob(ffn_dropout)
        self._apply_attention_dropout(self._current_attention_dropout)

    def _device(self):
        param = next(self.model.parameters(), None)
        if param is not None:
            return param.device
        buffer = next(self.model.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    def set_dropout_cap(self, attention_dropout: float, ffn_dropout: float) -> None:
        self._dropout_cap_attention = self._clip_prob(attention_dropout)
        self._dropout_cap_ffn = self._clip_prob(ffn_dropout)
        if not self._sample_per_forward:
            self._set_current_dropout(self._dropout_cap_attention, self._dropout_cap_ffn)

    def configure_cls_loss_layers(self, last_k: int) -> None:
        self._cls_last_k = max(1, int(last_k))

    def configure_text_loss_layers(self, last_k: int) -> None:
        self._text_last_k = max(1, int(last_k))

    def get_sample_per_forward(self) -> bool:
        return bool(self._sample_per_forward)

    def set_sample_per_forward(self, enabled: bool) -> None:
        self._sample_per_forward = bool(enabled)

    def sample_dropout_once(self) -> None:
        if not self._stochastic_enabled:
            return
        device = self._device()
        attn = (
            torch.rand((), device=device).item() * self._dropout_cap_attention
            if self._dropout_cap_attention > 0.0
            else 0.0
        )
        ffn = (
            torch.rand((), device=device).item() * self._dropout_cap_ffn
            if self._dropout_cap_ffn > 0.0
            else 0.0
        )
        self._set_current_dropout(attn, ffn)

    def configure_stochastic_proxy(
        self,
        enabled: bool = True,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        sample_per_forward: bool = True,
    ) -> None:
        for hook in self._ffn_dropout_hooks:
            hook.remove()
        self._ffn_dropout_hooks = []
        self._attention_modules = []
        self._stochastic_enabled = bool(enabled)
        self._sample_per_forward = bool(sample_per_forward)

        if not enabled:
            self._set_current_dropout(0.0, 0.0)
            self.model.eval()
            return

        for module in self.model.modules():
            class_name = module.__class__.__name__
            if class_name == "CLIPAttention" and hasattr(module, "dropout"):
                self._attention_modules.append(module)
            if class_name == "CLIPMLP":
                def _mlp_dropout_hook(_module, _inputs, outputs):
                    p = self._current_ffn_dropout
                    if p <= 0.0:
                        return outputs
                    return F.dropout(outputs, p=p, training=True)

                self._ffn_dropout_hooks.append(module.register_forward_hook(_mlp_dropout_hook))

        self.set_dropout_cap(attention_dropout, ffn_dropout)
        if self._sample_per_forward:
            self._set_current_dropout(0.0, 0.0)

        # Keep train mode so CLIP attention dropout is active on every forward.
        self.model.train()

    def forward(self, x):
        if self._stochastic_enabled and self._sample_per_forward:
            self.sample_dropout_once()
        pixel_values = self.normalizer(x)

        if self._cls_last_k <= 1:
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            return image_features

        # Use the last K vision hidden states and return [B, K, D] projected CLS features.
        # Loss will average cosine similarities across K layers.
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        hidden_states = vision_outputs.hidden_states
        if hidden_states is None or len(hidden_states) == 0:
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            return image_features

        k = min(self._cls_last_k, len(hidden_states))
        selected = hidden_states[-k:]
        per_layer_features = []
        for hs in selected:
            cls_token = hs[:, 0, :]
            cls_token = self.model.vision_model.post_layernorm(cls_token)
            proj = self.model.visual_projection(cls_token)
            proj = proj / proj.norm(dim=1, keepdim=True)
            per_layer_features.append(proj)
        return torch.stack(per_layer_features, dim=1)

    def encode_text(self, texts: List[str], device: str = None):
        if self._stochastic_enabled and self._sample_per_forward:
            self.sample_dropout_once()
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if device is None:
            device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        if self._text_last_k <= 1:
            text_features = self.model.get_text_features(**tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            return text_features

        text_outputs = self.model.text_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens.get("attention_mask"),
            output_hidden_states=True,
        )
        hidden_states = text_outputs.hidden_states
        if hidden_states is None or len(hidden_states) == 0:
            text_features = self.model.get_text_features(**tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            return text_features

        eos_positions = tokens["input_ids"].argmax(dim=-1)
        batch_idx = torch.arange(tokens["input_ids"].size(0), device=tokens["input_ids"].device)
        k = min(self._text_last_k, len(hidden_states))
        selected = hidden_states[-k:]
        per_layer_features = []
        for hs in selected:
            eos_hidden = hs[batch_idx, eos_positions]
            eos_hidden = self.model.text_model.final_layer_norm(eos_hidden)
            proj = self.model.text_projection(eos_hidden)
            proj = proj / proj.norm(dim=1, keepdim=True)
            per_layer_features.append(proj)
        return torch.stack(per_layer_features, dim=1)


@register_model("B16")
class ClipB16FeatureExtractor(_CLIPExtractor):
    model_id = "openai/clip-vit-base-patch16"
    revision = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
    image_size = 224


@register_model("B32")
class ClipB32FeatureExtractor(_CLIPExtractor):
    model_id = "openai/clip-vit-base-patch32"
    revision = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    image_size = 224


@register_model("Laion")
class ClipLaionFeatureExtractor(_CLIPExtractor):
    model_id = "laion/CLIP-ViT-G-14-laion2B-s12B-b42K"
    revision = "4b0305adc6802b2632e11cbe6606a9bdd43d35c9"
    image_size = 224


def create_model(name: str) -> BaseFeatureExtractor:
    if name not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown backbone '{name}'. Valid backbones: {valid}")
    return MODEL_REGISTRY[name]()


def _resolve_cache_dir() -> Optional[str]:
    for key in ("OATTACK_HF_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.getenv(key)
        if value:
            return value
    return None


def _resolve_model_ref(model_id: str) -> str:
    local_root = os.getenv("OATTACK_HF_MIRROR_ROOT")
    if local_root:
        local_path = os.path.join(os.path.expanduser(local_root), model_id)
        if os.path.isdir(local_path):
            return local_path
    return model_id
