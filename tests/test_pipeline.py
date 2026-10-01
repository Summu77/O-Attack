"""Exercise caption inference and scoring through a loopback HTTP provider."""
from __future__ import annotations

import base64
import csv
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from oeval import run_inference
from oeval.build_target_subset import link_target_subset
from oeval.models.openrouter_multimodal_api import GPT54VisionOpenRouterEvaluator
from tools import score


class PipelineTests(unittest.TestCase):
    def test_http_caption_score_and_resume(self):
        requests_received = []
        scores = iter(["0.5", "0.8", "0.9"])

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests_received.append(payload)
                if payload["model"] == "z-ai/glm-5":
                    content = next(scores)
                else:
                    image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
                    image_bytes = base64.b64decode(image_url.split(",", 1)[1])
                    self.server.test.assertTrue(image_bytes.startswith(b"\x89PNG") or image_bytes.startswith(b"\xff\xd8"))
                    content = "A dog resting on grass."
                body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.test = self
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}/chat/completions"
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "OPENROUTER_API_KEY": "loopback-test-key", "OPENROUTER_PROXY_URL": "",
                "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1",
            }):
                root = Path(tmp)
                for side in ("adv", "target"):
                    (root / side).mkdir()
                for image_id in (0, 1):
                    Image.new("RGB", (12, 12), "blue").save(root / f"adv/{image_id}.png")
                    Image.new("RGB", (12, 12), "green").save(root / f"target/{image_id}.jpg")
                link_target_subset(root / "adv", root / "target", root / "subset", True)
                evaluator = GPT54VisionOpenRouterEvaluator(api_url=endpoint)
                names = ["gpt54_vision_check_short_sentence_v1", "gpt54_vision_target_target_for_check_short_sentence_v1"]
                for input_root, name in zip([root / "adv", root / "subset"], names):
                    argv = ["run_inference", "--model-key", "gpt54_vision", "--model-name", "openai/gpt-5.4",
                            "--input-root", str(input_root), "--output-root", str(root / "eval"), "--run-name", name]
                    with patch.object(sys, "argv", argv), patch.object(run_inference, "create_vlm", return_value=evaluator):
                        run_inference.main()
                    result = json.loads((root / "eval" / name / "summary.json").read_text())
                    self.assertEqual((result["num_images"], result["num_success"], result["num_failed"]), (2, 2, 0))
                self.assertEqual(len(requests_received), 4)

                argv = ["tools.score", "--runs-root", str(root / "eval"), "--attack-tag", "check", "--models", "gpt54_vision",
                        "--output-dir", str(root / "score"), "--api-url", endpoint, "--sleep", "0", "--strict"]

                def run_score():
                    with patch.object(sys, "argv", argv), patch.object(score, "python_command", return_value=[sys.executable]):
                        score.main()

                run_score()
                self.assertEqual(len(requests_received), 6)
                compact = root / "score/compact_check_glm5.csv"
                row = list(csv.DictReader(compact.open()))[0]
                self.assertAlmostEqual(float(row["avg_similarity"]), 0.65)
                self.assertEqual(float(row["asr_gt_0_5"]), 0.5)
                self.assertEqual(row["num_failed"], "0")
                run_score()
                self.assertEqual(len(requests_received), 6, "Unchanged caption pairs should resume without calls")

                predictions = root / "eval" / names[0] / "predictions.jsonl"
                rows = [json.loads(line) for line in predictions.read_text().splitlines()]
                rows[0]["output"] = "A dog running through grass."
                predictions.write_text("".join(json.dumps(r) + "\n" for r in rows))
                run_score()
                self.assertEqual(len(requests_received), 7, "Changed captions must be rescored")
                row = list(csv.DictReader(compact.open()))[0]
                self.assertAlmostEqual(float(row["avg_similarity"]), 0.85)
                self.assertEqual(float(row["asr_gt_0_5"]), 1.0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
