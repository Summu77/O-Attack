"""Regression checks for pairing, gradients, portable entry points and scoring."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from oattack.attacks import oattack_pgd_attack
from oattack.data.datasets import build_pair_datasets
from oattack.models.base import EnsembleFeatureExtractor, EnsembleFeatureLoss
from oattack.run import _parse_device_spec, load_config
from oattack.runtime import python_command
from oeval.build_target_subset import link_target_subset
from oscore.run_similarity import _compute_ratios, _load_predictions, _parse_score, _summarize_scores
from tools import evaluate, score
from tools.validate_release import ROOT, validate


class ReleaseTests(unittest.TestCase):
    def test_default_data_integrity(self):
        report = validate()
        self.assertEqual(report["image_pairs"], 1000)
        self.assertEqual(report["default_pairs"], 100)
        self.assertEqual(report["captions"], 11000)
        self.assertEqual(report["examples"]["max_linf_pixels"], 16)

    def test_full_dataset_and_caption_banks_load(self):
        from oattack.run import _load_caption_map, _get_caption_banks_from_mapping
        cfg = load_config(str(ROOT / "configs/attack/default.yaml"), [
            "data.num_samples=1000", "data.clean_root=data/images/bigscale_1000",
            "data.target_root=data/images/target_images_1000",
            "text_objective.source_caption_json=data/text/bigscale_1000/nips17/caption.json",
            "text_objective.target_caption_json=data/text/target_images_1000/1/caption.json"])
        source, target = build_pair_datasets(cfg, None)
        self.assertEqual(len(source), 1000)
        self.assertEqual(len(target), 1000)
        for dataset, path in [(source, cfg.text_objective.source_caption_json),
                              (target, cfg.text_objective.target_caption_json)]:
            banks = _get_caption_banks_from_mapping([p for p, _ in dataset.samples], _load_caption_map(path), path)
            self.assertEqual(len(banks), 1000)
            self.assertTrue(all(len(bank) == 5 for bank in banks))

    def test_omegaconf_device_list(self):
        self.assertEqual(_parse_device_spec(OmegaConf.create(["cuda:0", "cuda:1"])), ["cuda:0", "cuda:1"])

    def test_source_target_id_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for side, image_id in [("source", "0"), ("target", "1")]:
                folder = root / side / "images"
                folder.mkdir(parents=True)
                Image.new("RGB", (16, 16)).save(folder / f"{image_id}.png")
            cfg = OmegaConf.create({"data": {"dataset_type": "folder_pairs", "clean_root": str(root / "source"),
                                            "target_root": str(root / "target"), "sample_start": 0}})
            with self.assertRaisesRegex(ValueError, "matching IDs"):
                build_pair_datasets(cfg, None)

    def test_target_subset_rebuilt_for_jpeg_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adv, target, out = root / "adv", root / "target", root / "subset"
            for folder in (adv, target, out):
                folder.mkdir()
            Image.new("RGB", (8, 8)).save(adv / "0.png")
            Image.new("RGB", (8, 8)).save(target / "0.jpg")
            Image.new("RGB", (8, 8)).save(out / "99.jpg")
            link_target_subset(adv, target, out, overwrite=True)
            self.assertEqual([p.name for p in out.iterdir()], ["0.jpg"])
            self.assertEqual((out / "0.jpg").resolve(), (target / "0.jpg").resolve())

    def test_missing_target_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adv").mkdir()
            (root / "target").mkdir()
            Image.new("RGB", (8, 8)).save(root / "adv/0.png")
            with self.assertRaisesRegex(ValueError, "Missing target"):
                link_target_subset(root / "adv", root / "target", root / "out", True)

    def test_api_only_evaluation_does_not_query_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adv, target = root / "adv", root / "target"
            adv.mkdir(); target.mkdir()
            Image.new("RGB", (8, 8)).save(adv / "0.png")
            args = argparse.Namespace(models="api", input_root=adv, target_root=target, attack_tag="test",
                                      output_root=root / "eval", target_tag=None, gpus=None,
                                      free_gpu_max_memory_mb=1500, free_gpu_max_util=15, max_local_workers=None,
                                      prompt="Describe.", max_new_tokens=64, openrouter_proxy_url=None)
            with patch.object(evaluate, "parse_args", return_value=args), \
                 patch.object(evaluate, "load_registry", return_value={"api": {"source": "api"}}), \
                 patch.object(evaluate, "ensure_target_subset"), \
                 patch.object(evaluate, "run_inference_job") as jobs, \
                 patch.object(evaluate, "find_free_gpus", side_effect=AssertionError("GPU query")):
                evaluate.main()
                self.assertEqual(jobs.call_count, 2)

    def test_zero_image_summary_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps({"num_images": 0, "num_success": 0, "num_failed": 0}))
            self.assertFalse(evaluate.summary_complete(path))

    def test_changed_inputs_do_not_reuse_old_caption_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps({"num_images": 1, "num_success": 1, "num_failed": 0,
                                        "input_fingerprint": "old", "prompt": "Describe."}))
            self.assertTrue(evaluate.summary_complete(path))
            self.assertFalse(evaluate.summary_complete(path, {"input_fingerprint": "new"}))
            self.assertFalse(evaluate.summary_complete(path, {"prompt": "Changed prompt."}))

    def test_inference_errors_are_kept_in_prediction_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(json.dumps({"relative_path": "0.png", "error": "failed"}) + "\n")
            self.assertEqual(_load_predictions(path)[0]["error"], "failed")
            path.write_text(path.read_text() * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
                _load_predictions(path)

    def test_asr_threshold_and_failure_denominator(self):
        rows = [{"similarity_score": 0.5}, {"similarity_score": 0.8}, {"similarity_score": None, "error": "failed"}]
        summary = _summarize_scores(rows)
        self.assertEqual(summary["num_pairs"], 3)
        self.assertAlmostEqual(summary["asr_gt_0_5"], 1 / 3)
        self.assertAlmostEqual(_compute_ratios([0.5, 0.8], [0.5], 3)[0], 2 / 3)

    def test_invalid_judge_numbers_are_rejected(self):
        self.assertEqual(_parse_score("0.75"), 0.75)
        for text in ["2.0", "10.0", "-0.5", "0.75.4", "NaN"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                _parse_score(text)

    def test_portable_python_default(self):
        import sys
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(python_command(), [sys.executable])

    def test_inference_reports_empty_caption_failure(self):
        from oeval import run_inference
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (8, 8)).save(root / "0.png")
            args = argparse.Namespace(list_models=False, model_key="dummy", model_name="dummy", model_dtype="auto",
                                      model_device_map="auto", attn_implementation=None, local_files_only=False,
                                      proxy_host=None, proxy_port=None, input_root=str(root), limit=None,
                                      run_name="failed", output_root=str(root / "out"), prompt="Describe.", max_new_tokens=64)
            fake = type("EmptyModel", (), {"describe_image": lambda self, **kwargs: ""})()
            with patch.object(run_inference, "parse_args", return_value=args), \
                 patch.object(run_inference, "list_vlms", return_value=["dummy"]), \
                 patch.object(run_inference, "create_vlm", return_value=fake):
                with self.assertRaisesRegex(RuntimeError, "images failed"):
                    run_inference.main()
            summary = json.loads((root / "out/failed/summary.json").read_text())
            self.assertEqual(summary["num_failed"], 1)

    def test_gradient_attack_is_finite_and_bounded(self):
        class Proxy(torch.nn.Module):
            def __init__(self, name, weights):
                super().__init__()
                self._backbone_name = name
                self.register_buffer("weights", torch.tensor(weights))

            def forward(self, x):
                features = F.normalize(x.mean(dim=(-1, -2)) * self.weights, dim=-1)
                return torch.stack([features, F.normalize(features + 0.1, dim=-1)], dim=1)

            def encode_text(self, texts, device=None):
                values = torch.tensor([[len(text) % 7 + 1, sum(map(ord, text)) % 11 + 1, 3] for text in texts],
                                      dtype=torch.float32, device=device)
                values = F.normalize(values, dim=-1)
                return values.unsqueeze(1).expand(-1, 2, -1)

        cfg = load_config(str(ROOT / "configs/attack/default.yaml"), ["optim.steps=4", "optim.alpha=10", "optim.source_aug_crops=2"])
        proxies = [Proxy(name, weights) for name, weights in [("Laion", [1., 2., 3.]), ("B16", [2., 1., 3.]), ("B32", [3., 2., 1.])]]
        source = torch.rand(1, 3, 16, 16) * 200 + 25
        target = torch.rand_like(source) * 255
        logger = type("Logger", (), {"log": lambda self, data: None})()
        adv = oattack_pgd_attack(cfg, EnsembleFeatureExtractor(proxies), EnsembleFeatureLoss(proxies),
                                torch.nn.Identity(), torch.nn.Identity(), source, target, None,
                                [["A panda on a branch.", "A resting panda."]], [["A playful dog.", "A dog chewing."]], logger, 0)
        self.assertTrue(torch.isfinite(adv).all())
        self.assertGreater(float((adv * 255 - source).abs().max()), 0)
        self.assertLessEqual(float((adv * 255 - source).abs().max()), 16.0001)
        self.assertGreaterEqual(float(adv.min()), 0)
        self.assertLessEqual(float(adv.max()), 1)


if __name__ == "__main__":
    unittest.main()
