import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from oattack.utils import ensure_dir
from oeval import create_vlm, list_vlms
from oeval.artifacts import input_fingerprint

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run standalone VLM inference on image folders (independent from attack)."
    )
    parser.add_argument("--input-root", type=str, default=None, help="Folder containing images to evaluate.")
    parser.add_argument("--output-root", type=str, default="./outputs_vlm_eval")
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--model-key", type=str, default="qwen3_vl_8b_instruct")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model-dtype", type=str, default="auto")
    parser.add_argument("--model-device-map", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument(
        "--prompt",
        type=str,
        default="Please describe this image in a short sentence.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--proxy-host", type=str, default=None)
    parser.add_argument("--proxy-port", type=str, default=None)
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def set_proxy_env(proxy_host: str, proxy_port: str):
    proxy = f"http://{proxy_host}:{proxy_port}"
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy


def collect_images(input_root: Path, limit: int = None) -> List[Path]:
    images = [p for p in sorted(input_root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if limit is not None:
        return images[:limit]
    return images


def make_run_dir(args, image_count: int) -> Path:
    if args.run_name:
        run_name = args.run_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = Path(args.input_root).name
        run_name = f"{ts}_{args.model_key}_{src}_{image_count}img"

    run_dir = Path(args.output_root) / run_name
    ensure_dir(str(run_dir))
    ensure_dir(str(run_dir / "meta"))
    ensure_dir(str(run_dir / "responses"))
    return run_dir


def write_json(path: Path, payload: Dict[str, Any]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    if args.list_models:
        print("\n".join(list_vlms()))
        return

    if args.model_key not in list_vlms():
        valid = ", ".join(list_vlms())
        raise ValueError(f"--model-key must be one of: {valid}")

    if not args.input_root:
        raise ValueError("--input-root is required unless --list-models is used.")

    if args.proxy_host and args.proxy_port:
        set_proxy_env(args.proxy_host, args.proxy_port)

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    image_paths = collect_images(input_root, limit=args.limit)
    if not image_paths:
        raise RuntimeError(f"No image files found under {input_root}")
    fingerprint = input_fingerprint(input_root)

    run_dir = make_run_dir(args, image_count=len(image_paths))
    predictions_path = run_dir / "predictions.jsonl"

    write_json(
        run_dir / "meta" / "run_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "input_root": str(input_root.resolve()),
            "num_images": len(image_paths),
        },
    )

    model = create_vlm(
        args.model_key,
        model_name=args.model_name,
        model_dtype=args.model_dtype,
        model_device_map=args.model_device_map,
        attn_implementation=args.attn_implementation,
        local_files_only=bool(args.local_files_only),
    )

    success = 0
    failed = 0
    with predictions_path.open("w", encoding="utf-8") as f:
        for idx, img_path in enumerate(image_paths, start=1):
            rel_path = img_path.relative_to(input_root)
            record: Dict[str, Any] = {
                "index": idx - 1,
                "image_path": str(img_path),
                "relative_path": str(rel_path),
                "prompt": args.prompt,
            }

            try:
                output = model.describe_image(
                    image_path=str(img_path),
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                if not isinstance(output, str) or not output.strip():
                    raise ValueError("The model returned an empty caption.")
                record["output"] = output
                success += 1

                txt_path = run_dir / "responses" / rel_path.with_suffix(".txt")
                ensure_dir(str(txt_path.parent))
                with txt_path.open("w", encoding="utf-8") as tf:
                    tf.write(output + "\n")
            except Exception as exc:
                record["error"] = repr(exc)
                failed += 1

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(image_paths)}] {rel_path}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_key": args.model_key,
        "model_name": args.model_name,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "input_fingerprint": fingerprint,
        "input_root": str(input_root.resolve()),
        "num_images": len(image_paths),
        "num_success": success,
        "num_failed": failed,
        "predictions_jsonl": str(predictions_path),
        "responses_dir": str(run_dir / "responses"),
    }
    write_json(run_dir / "summary.json", summary)

    print("Inference finished.")
    print(f"Run dir: {run_dir}")
    print(json.dumps(summary, ensure_ascii=False))
    if failed:
        raise RuntimeError(f"{failed}/{len(image_paths)} images failed; inspect {predictions_path}")


if __name__ == "__main__":
    main()
