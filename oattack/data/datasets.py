from typing import Any, Callable, Dict, Tuple
from pathlib import Path

import torch
import torchvision

DATASET_REGISTRY: Dict[str, Callable[[Any, Callable], Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]]] = {}


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, label, path


def _apply_sample_start(cfg: Any, dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    sample_start = int(getattr(cfg.data, "sample_start", 0))
    if sample_start < 0:
        raise ValueError("data.sample_start must not be negative")
    if sample_start <= 0:
        return dataset
    if sample_start >= len(dataset):
        raise ValueError(f"data.sample_start={sample_start} is out of range for dataset length={len(dataset)}")
    indices = list(range(sample_start, len(dataset)))
    return torch.utils.data.Subset(dataset, indices)


def register_dataset(name: str):
    def decorator(builder: Callable[[Any, Callable], Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]]):
        DATASET_REGISTRY[name] = builder
        return builder

    return decorator


@register_dataset("folder_pairs")
def build_folder_pairs(cfg: Any, transform: Callable):
    clean_data = ImageFolderWithPaths(cfg.data.clean_root, transform=transform)
    target_data = ImageFolderWithPaths(cfg.data.target_root, transform=transform)
    source_ids = [Path(path).stem for path, _ in clean_data.samples]
    target_ids = [Path(path).stem for path, _ in target_data.samples]
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("Source and target image IDs must each be unique")
    if source_ids != target_ids:
        raise ValueError("Source and target images must have matching IDs in ImageFolder order")
    clean_data = _apply_sample_start(cfg, clean_data)
    target_data = _apply_sample_start(cfg, target_data)
    return clean_data, target_data


def build_pair_datasets(cfg: Any, transform: Callable):
    dataset_type = cfg.data.dataset_type
    if dataset_type not in DATASET_REGISTRY:
        valid = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Valid: {valid}")
    return DATASET_REGISTRY[dataset_type](cfg, transform)


def build_clean_dataset(cfg: Any, transform: Callable):
    dataset_type = cfg.data.dataset_type
    if dataset_type == "folder_pairs":
        clean_data = ImageFolderWithPaths(cfg.data.clean_root, transform=transform)
        return _apply_sample_start(cfg, clean_data)
    raise NotImplementedError(
        f"build_clean_dataset only supports dataset_type='folder_pairs', got '{dataset_type}'"
    )
