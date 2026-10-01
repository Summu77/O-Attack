"""Validate the bundled default configuration, image pairs and file integrity."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _validate_dataset(manifest_name: str, num_pairs: int):
    manifest = json.loads((ROOT / manifest_name).read_text())
    assert manifest["num_pairs"] == len(manifest["pairs"]) == num_pairs
    assert {row["id"] for row in manifest["pairs"]} == set(range(num_pairs))
    assert len({row["id"] for row in manifest["pairs"]}) == len(manifest["pairs"])
    for path, expected in manifest["caption_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    caption_maps = {}
    for side in ("source", "target"):
        path = manifest[f"{side}_captions"]
        rows = json.loads((ROOT / path).read_text())
        assert len(rows) == num_pairs and len({row["image"] for row in rows}) == num_pairs
        assert all(isinstance(row["caption"], list) and len(row["caption"]) == 5
                   and all(isinstance(c, str) and c.strip() for c in row["caption"]) for row in rows)
        caption_maps[side] = {row["image"]: row["caption"] for row in rows}
    dimensions = Counter()
    seen = {"source": set(), "target": set()}
    for index, pair in enumerate(manifest["pairs"]):
        assert pair["index"] == index
        for side in ("source", "target"):
            path = ROOT / pair[f"{side}_image"]
            assert int(path.stem) == pair["id"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == pair[f"{side}_sha256"], path
            assert path.name in caption_maps[side]
            seen[side].add(path)
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.convert("RGB").load()
                dimensions[f"{im.width}x{im.height}"] += 1
    for side, key in [("source", "clean_root"), ("target", "target_root")]:
        actual = sorted(p for p in (ROOT / manifest[key]).rglob("*")
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
        assert set(actual) == seen[side]
        assert [str(pair["id"]) for pair in manifest["pairs"]] == [p.stem for p in actual]
    return manifest, dimensions


def _validate_examples(cfg: dict, default: dict) -> dict:
    import numpy as np
    from oattack.run import build_dataset_transform

    manifest = json.loads((ROOT / "examples/oattack_best_g3/manifest.json").read_text())
    assert manifest["num_images"] == len(manifest["images"]) == 100
    settings = {key: copy.deepcopy(cfg[key]) for key in manifest["settings"]}
    settings["model"].pop("device")
    assert settings == manifest["settings"], "Example generation settings differ from default G3"
    for side in ("source", "target"):
        assert manifest[f"{side}_caption_sha256"] == default["caption_sha256"][default[f"{side}_captions"]]
    assert [(r["sample_start"], r["num_samples"], r["seed"]) for r in manifest["runs"]] == [
        (0, 25, 2023), (25, 25, 2023), (50, 25, 2023), (75, 25, 2023)]
    run_ids = {r["run_hash"] for r in manifest["runs"]}
    assert manifest["linf_pixel_limit"] == cfg["optim"]["epsilon"] == 16
    transform = build_dataset_transform(cfg["model"]["input_res"])
    max_linf = 0.0
    seen = set()
    for record, pair in zip(manifest["images"], default["pairs"]):
        assert record["index"] == pair["index"] and record["id"] == pair["id"]
        assert record["generation_run_hash"] in run_ids
        assert record["source_image"] == pair["source_image"] and record["target_image"] == pair["target_image"]
        path = ROOT / record["adversarial_image"]
        assert path.stem == str(pair["id"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["adversarial_sha256"]
        with Image.open(path) as im:
            assert im.mode == "RGB" and im.size == (224, 224) and im.format == "PNG"
            adv = np.asarray(im).astype(np.int16)
        with Image.open(ROOT / pair["source_image"]) as im:
            source = transform(im).permute(1, 2, 0).numpy().astype(np.int16)
        linf = float(np.abs(adv - source).max())
        assert 0 < linf <= 16, f"Invalid pixel perturbation for {path.name}: {linf}"
        max_linf = max(max_linf, linf)
        seen.add(path)
    assert seen == set((ROOT / "examples/oattack_best_g3/nips17").glob("*.png"))
    return {"images": 100, "generation_runs": 4, "max_linf_pixels": max_linf, "settings_match_default": True}


def validate() -> dict:
    attack_configs = sorted((ROOT / "configs/attack").glob("*.yaml"))
    if attack_configs != [ROOT / "configs/attack/default.yaml"]:
        raise ValueError("The release must contain exactly one default attack configuration")
    cfg = yaml.safe_load(attack_configs[0].read_text())
    assert cfg["seed"] == 2023
    assert cfg["data"]["num_samples"] == 100
    assert cfg["optim"]["epsilon"] == 16 and cfg["optim"]["steps"] == 300
    assert cfg["optim"]["alpha"] == 1.0 and cfg["optim"]["source_aug_crops"] == 10
    assert cfg["model"]["backbones"] == ["Laion", "B16", "B32"]
    assert cfg["oattack_objective"]["variance_mode"] == "subtract"
    assert cfg["oattack_objective"]["aggregation_mode"] == "per_model_layeravg_textpooled"
    for name, vision_k, text_k, low, high in [
        ("Laion", 15, 5, 0.1, 0.15), ("B16", 4, 2, 0.05, 0.1), ("B32", 4, 2, 0.05, 0.1),
    ]:
        vision = cfg["model"]["backbone_overrides"][name]
        text = cfg["text_objective"]["backbone_overrides"][name]
        assert vision["cls_last_k"] == vision_k and text["text_last_k"] == text_k
        for section in (vision, text):
            for kind in ("attention", "ffn"):
                assert section[f"{kind}_dropout_min"] == low
                assert section[f"{kind}_dropout_max"] == high
    default, dimensions = _validate_dataset("data/manifest.json", 100)
    full, full_dimensions = _validate_dataset("data/full_manifest.json", 1000)
    dimensions.update(full_dimensions)
    for side, key in [("source", "clean_root"), ("target", "target_root")]:
        assert default[key] == cfg["data"][key]
        assert default[f"{side}_captions"] == cfg["text_objective"][f"{side}_caption_json"]
    full_by_id = {pair["id"]: pair for pair in full["pairs"]}
    for pair in default["pairs"]:
        for side in ("source", "target"):
            assert pair[f"{side}_sha256"] == full_by_id[pair["id"]][f"{side}_sha256"]
    examples = _validate_examples(cfg, default)
    registry = json.loads((ROOT / "configs/eval/victim_models.json").read_text())
    for item in registry.values():
        assert item["source"] in {"local", "api"}
        assert not item["model_name"].startswith("/")
    judges = json.loads((ROOT / "configs/score/judge_models.json").read_text())
    assert judges["default"] == "glm5" and list(judges["judges"]) == ["glm5"]
    syntax_count = 0
    for folder in ("oattack", "oeval", "oscore", "tools", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
            syntax_count += 1
    report = {"status": "passed", "image_pairs": 1000, "default_pairs": 100,
              "dataset_image_files": 2200, "captions": 11000, "examples": examples,
              "image_dimensions": dict(dimensions), "attack_configs": 1, "judges": 1,
              "victim_models": len(registry), "python_files": syntax_count}
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
