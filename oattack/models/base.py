from abc import abstractmethod
from typing import Any, Dict, List

import torch
from torch import Tensor, nn


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    tagged = getattr(module, "_inference_device", None)
    if tagged is not None:
        return torch.device(str(tagged))
    param = next(module.parameters(), None)
    if param is not None:
        return param.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return fallback


class BaseFeatureExtractor(nn.Module):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def encode_text(self, _texts: List[str], _device: str = None) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement encode_text")


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Dict[int, Tensor]:
        features: Dict[int, Tensor] = {}
        output_device = x.device
        for i, model in enumerate(self.extractors):
            model_device = _module_device(model, fallback=output_device)
            model_input = x if model_device == output_device else x.to(model_device, non_blocking=True)
            feat = model(model_input)
            if feat.device != output_device:
                feat = feat.to(output_device, non_blocking=True)
            features[i] = feat
        return features


class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth: List[Tensor] = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor) -> None:
        self.set_ground_truth_image(x)

    @torch.no_grad()
    def set_ground_truth_image(self, x: Tensor) -> None:
        self.ground_truth.clear()
        output_device = x.device
        for model in self.extractors:
            model_device = _module_device(model, fallback=output_device)
            model_input = x if model_device == output_device else x.to(model_device, non_blocking=True)
            gt = model(model_input)
            if gt.device != output_device:
                gt = gt.to(output_device, non_blocking=True)
            self.ground_truth.append(gt)

    @torch.no_grad()
    def set_ground_truth_text(self, texts: List[str], device: str) -> None:
        self.ground_truth.clear()
        output_device = torch.device(device)
        for model in self.extractors:
            text_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            if not hasattr(text_model, "encode_text"):
                raise NotImplementedError(f"{text_model.__class__.__name__} does not implement encode_text")
            model_device = str(_module_device(text_model, fallback=output_device))
            text_feat = text_model.encode_text(texts, device=model_device)
            if text_feat.device != output_device:
                text_feat = text_feat.to(output_device, non_blocking=True)
            self.ground_truth.append(text_feat)

    @staticmethod
    def _align_ground_truth(feature: Tensor, gt: Tensor) -> Tensor:
        if feature.ndim == 1:
            feature = feature.unsqueeze(0)
        if gt.ndim == 1:
            gt = gt.unsqueeze(0)

        # Align layer dimension for multi-layer CLS features [B, K, D].
        if feature.ndim == 3 and gt.ndim == 2:
            gt = gt.unsqueeze(1).expand(-1, feature.size(1), -1)
        elif feature.ndim == 2 and gt.ndim == 3:
            gt = torch.mean(gt, dim=1)
        elif feature.ndim == 3 and gt.ndim == 3 and feature.size(1) != gt.size(1):
            if gt.size(1) == 1:
                gt = gt.expand(-1, feature.size(1), -1)
            else:
                raise ValueError(
                    f"Cannot align layer count: feature K={feature.size(1)} vs ground-truth K={gt.size(1)}"
                )

        if feature.size(0) == gt.size(0):
            return gt
        if feature.size(0) % gt.size(0) == 0:
            rep = feature.size(0) // gt.size(0)
            return gt.repeat_interleave(rep, dim=0)
        raise ValueError(
            f"Cannot align feature batch ({feature.size(0)}) with ground-truth batch ({gt.size(0)})"
        )

    @staticmethod
    def _per_sample_similarity(feature: Tensor, gt: Tensor) -> Tensor:
        if feature.ndim == 2:
            return torch.sum(feature * gt, dim=1)
        if feature.ndim == 3:
            # Mean over K layer-level cosine similarities.
            return torch.sum(feature * gt, dim=2).mean(dim=1)
        raise ValueError(f"Unsupported feature ndim={feature.ndim}, expected 2 or 3")

    def __call__(self, feature_dict: Dict[int, Tensor], _: Any = None) -> Tensor:
        loss = 0.0
        for index, _model in enumerate(self.extractors):
            feature = feature_dict[index]
            gt = self._align_ground_truth(feature, self.ground_truth[index])
            loss += torch.mean(self._per_sample_similarity(feature, gt))
        return loss / len(self.extractors)
