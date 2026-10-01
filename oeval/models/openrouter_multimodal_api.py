import base64
import os
import re
from pathlib import Path
from typing import Dict, Optional

import requests

from oeval.base import BaseVLMEvaluator
from oeval.registry import register_vlm


def _encode_image(path: Path) -> str:
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }[path.suffix.lower()]
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _strip_reasoning(text: str) -> str:
    cleaned = text.strip()
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1].strip()
    return cleaned


def _clean_caption_text(text: str) -> str:
    cleaned = _strip_reasoning(text)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[*_`#>]+", "", cleaned)
    quoted = re.findall(r'"([^"\n]{8,240})"', cleaned)
    if quoted:
        cleaned = quoted[-1]
    lines = [line.strip(" -\t") for line in cleaned.splitlines() if line.strip()]
    cleaned_lines = []
    intro_pattern = re.compile(
        r"^(?:based on (?:the |this )?image(?: provided)?(?:,|:)?\s*)?"
        r"(?:here(?:['’])?s|here is)\s+(?:a\s+)?short description\s*:\s*",
        flags=re.IGNORECASE,
    )
    for line in lines:
        line = intro_pattern.sub("", line).strip()
        if line:
            cleaned_lines.append(line)
    if cleaned_lines:
        cleaned = cleaned_lines[0]
    elif lines:
        cleaned = lines[0]
    if ":" in cleaned and len(cleaned.split(":", 1)[0]) <= 24:
        cleaned = cleaned.split(":", 1)[1].strip()
    cleaned = re.sub(r"^\d+[\).\s-]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_csv_env(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_provider_payload() -> Dict[str, object] | None:
    provider: Dict[str, object] = {}
    provider_only = _parse_csv_env(os.environ.get("OPENROUTER_PROVIDER_ONLY", ""))
    provider_order = _parse_csv_env(os.environ.get("OPENROUTER_PROVIDER_ORDER", ""))
    if provider_only:
        provider["only"] = provider_only
    if provider_order:
        provider["order"] = provider_order
    if os.environ.get("OPENROUTER_NO_PROVIDER_FALLBACKS", "").strip().lower() in {"1", "true", "yes"}:
        provider["allow_fallbacks"] = False
    return provider or None


class _OpenRouterMultimodalEvaluator(BaseVLMEvaluator):
    def __init__(
        self,
        model_name: str,
        model_dtype: str = "auto",
        model_device_map: str = "api",
        attn_implementation: Optional[str] = None,
        local_files_only: bool = False,
        api_url: str = "https://openrouter.ai/api/v1/chat/completions",
        api_key: Optional[str] = None,
        http_referer: Optional[str] = None,
        app_title: Optional[str] = None,
    ):
        del model_dtype, model_device_map, attn_implementation, local_files_only
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.http_referer = http_referer or os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost")
        self.app_title = app_title or os.environ.get("OPENROUTER_APP_TITLE", "O-Attack VLM Eval")
        self.proxy_url = os.environ.get("OPENROUTER_PROXY_URL", "").strip()
        self.provider = _build_provider_payload()
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY for OpenRouter multimodal inference.")

    def describe_image(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        image_url = _encode_image(Path(image_path))
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.http_referer,
            "X-Title": self.app_title,
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        if not self.model_name.startswith("stepfun/"):
            payload["reasoning"] = {"enabled": False}
        if self.provider:
            payload["provider"] = self.provider
        with requests.Session() as session:
            if self.proxy_url:
                session.proxies.update({"http": self.proxy_url, "https": self.proxy_url})
            resp = session.post(self.api_url, headers=headers, json=payload, timeout=300)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        content = str(content or "").strip()
        if not content:
            raise RuntimeError(f"OpenRouter returned empty content: {str(data)[:500]}")
        return _clean_caption_text(content)


@register_vlm("llama32_11b_vision_openrouter")
@register_vlm("llama32_11b_vision")
class Llama3211BVisionOpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "meta-llama/llama-3.2-11b-vision-instruct", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("kimi_k2_5")
class KimiK25OpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "moonshotai/kimi-k2.5", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("glm46v_vlm")
@register_vlm("glm_4_6v")
class GLM46VVLMOpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "z-ai/glm-4.6v", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("qwen35_397b_a17b")
class Qwen35397BA17BOpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "qwen/qwen3.5-397b-a17b", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("gpt54_vision")
class GPT54VisionOpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "openai/gpt-5.4", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("claude_sonnet_4_6")
class ClaudeSonnet46OpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "anthropic/claude-sonnet-4.6", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("gemini31_flash_lite")
class Gemini31FlashLiteOpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "google/gemini-3.1-flash-lite-preview", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("minimax_m3")
class MiniMaxM3OpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "minimax/minimax-m3", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("mistral_medium_3_5")
class MistralMedium35OpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "mistralai/mistral-medium-3-5", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("llama4_maverick")
class Llama4MaverickOpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "meta-llama/llama-4-maverick", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("step37_flash")
class Step37FlashOpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "stepfun/step-3.7-flash", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("nex_n2_pro_free")
class NexN2ProFreeOpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "nex-agi/nex-n2-pro", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("grok43")
class Grok43OpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "x-ai/grok-4.3", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

@register_vlm("mimo_v25")
class MiMoV25OpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "xiaomi/mimo-v2.5", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("seed20_lite")
class Seed20LiteOpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "bytedance-seed/seed-2.0-lite", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("nemotron_v2")
class NemotronV2OpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "nvidia/nemotron-nano-12b-v2-vl:free", **kwargs):
        super().__init__(model_name=model_name, **kwargs)


@register_vlm("nova2_lite")
class Nova2LiteOpenRouterEvaluator(_OpenRouterMultimodalEvaluator):
    def __init__(self, model_name: str = "amazon/nova-2-lite-v1", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
