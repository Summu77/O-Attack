# O-Attack

**One Attack to Fool Them All: Highly Transferable Black-Box Adversarial Attacks on Frontier MLLMs**

Sen Nie · Jie Zhang · Zhongqi Wang · Shiguang Shan · Xilin Chen

Institute of Computing Technology, Chinese Academy of Sciences & University of Chinese Academy of Sciences

[Project page](https://summu77.github.io/O-Attack/) · [Paper](https://arxiv.org/abs/2609.33833)

O-Attack exploits a broad, high-level, cross-modally aligned semantic space within surrogate models through semantic space anchoring, progressive sampling, and semantic consensus optimization. Experiments span 10 frontier commercial MLLMs and 14 widely used MLLMs, with six state-of-the-art baselines.

## Code release

Code and the complete **1,000 source–target image pairs** for targeted caption
transfer attacks against vision-language models. The sole default configuration
is **G3** and runs the original **100-pair subset**. The release also includes
**100 previously generated G3 adversarial images**, ready for evaluation.
The complete attack → evaluation → scoring pipeline is retained.
See [the data inventory](data/README.md) for the two caption banks and pairing order.

## Installation

Use Python 3.10 on Linux with an NVIDIA GPU for the tested attack runtime:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m tools.validate_release
```

The dependency versions were recorded from the original working environment.
The main requirements were installed in a fresh Python 3.10 virtual environment,
which passed `pip check`, the regression suite and a three-proxy attack smoke run.
CLIP weights are downloaded from Hugging Face on first use; the three upstream
revisions are pinned in `oattack/models/clip_extractors.py`. Weights are not
bundled. To use an existing local mirror, set `OATTACK_HF_MIRROR_ROOT` to a
directory containing `<organization>/<model>` subdirectories. For cached-only
operation, set `OATTACK_LOCAL_FILES_ONLY=1` after downloading the weights.

The release can be stored and inspected on macOS. GPU validation was performed
on Linux/CUDA; Apple MPS execution has not been validated.

## Generate adversarial images

```bash
bash scripts/attack.sh
```

This reads the sole attack configuration, `configs/attack/default.yaml`, and
processes all 100 pairs. The final console output identifies the image folder:
`outputs/attack/default/img/<hash>/nips17`. Images are saved as lossless PNGs.

For a quick runtime check with the same optimization settings:

```bash
bash scripts/attack.sh --num-samples 1 --output-root outputs/attack/check
```

Use `--device cuda:0,cuda:1` to distribute the three proxies across two GPUs.
`--sample-start` is an index in the original **lexical** image order
(`0, 1, 10, 11, …, 2, …`), not a numeric image ID. The exact order and file
hashes are in `data/manifest.json`.

To process the complete 1,000-pair dataset with the same G3 optimization settings:

```bash
bash scripts/attack.sh --num-samples 1000 \
  --clean-root data/images/bigscale_1000 \
  --target-root data/images/target_images_1000 \
  --set text_objective.source_caption_json=data/text/bigscale_1000/nips17/caption.json \
  --set text_objective.target_caption_json=data/text/target_images_1000/1/caption.json \
  --output-root outputs/attack/full1000
```

Both image paths and both caption paths must change together. The full dataset
uses its original caption bank, which differs from the 100-pair bank, even for
the same image IDs. Its order is recorded in `data/full_manifest.json`.

## Use the provided adversarial examples

The lossless PNGs in `examples/oattack_best_g3/nips17/` are the original 100
G3 outputs, copied without regeneration or recompression. They correspond to
the default 100-pair image/caption bank. Their [manifest](examples/oattack_best_g3/manifest.json)
records image hashes, matching source/target paths and generation settings.
Use this folder as `ADV_ROOT` in the commands below to skip attack generation.

These examples were generated in four 25-image runs, each starting with seed
2023. Running all 100 in one process uses the same settings but follows a
different random-number sequence after the first chunk. See the
[example instructions](examples/oattack_best_g3/README.md) to repeat the original
chunk boundaries.

## Evaluate and score

Set `ADV_ROOT` to the image folder printed by the attack. The example evaluates
the six API models used in the recorded G3 selection:

```bash
export OPENROUTER_API_KEY="your-key"
ADV_ROOT="examples/oattack_best_g3/nips17"
MODELS="gpt54_vision,claude_sonnet_4_6,gemini31_flash_lite,grok43,qwen35_397b_a17b,kimi_k2_5"

bash scripts/evaluate.sh --input-root "$ADV_ROOT" --attack-tag default \
  --models "$MODELS" --max-new-tokens 128

bash scripts/score.sh --runs-root outputs/eval/default --attack-tag default \
  --models "$MODELS" --strict
```

Replace `ADV_ROOT` with your own generated folder when needed. For a full
1,000-image attack, also pass
`--target-root data/images/target_images_1000` to `evaluate.sh` and use a separate
`--attack-tag full1000` consistently for evaluation and scoring.

Evaluation describes both the adversarial image and its matching target image
with the same victim model and prompt. Scoring uses the single retained judge,
GLM-5, to compare these two model-generated captions. It does not compare the
victim's output with the five captions used in the attack objective.
API evaluation and scoring send requests to the configured provider and may
incur charges. Credentials are read from environment variables and are not
bundled. Model IDs preserve the recorded experiments; current provider
availability is not asserted.

The victim registry retains the 24 models listed in the paper; use `--models` to select
those needed. Seven are local models. Several require separate dependency
environments; see [evaluation setup](docs/EVALUATION.md). A fresh `--attack-tag`
should be used for each changed input, prompt or inference setting.

Scoring writes pairwise JSON/CSV, a summary, and a compact CSV with:

- `avg_similarity`: mean over successfully scored caption pairs.
- `asr_gt_0_5`: fraction with a score **strictly greater than 0.5**, divided by
  the total number of paired records. Failed pairs remain in the denominator.
- `num_pairs`, `num_success`, and `num_failed`: completion counts.

The additional `ratio_ge_*` columns use `>=` and are explicitly separate from
the ASR definition. Failed inference or scoring returns a nonzero exit code
after saving diagnostic records. Scoring can resume successful pairs when
their captions, judge model, endpoint and reasoning setting are unchanged.

## Default parameters

| Parameter | Value |
|---|---|
| Seed / samples / batch size | 2023 / 100 / 1 |
| Input resolution | 224 × 224 |
| Proxies | LAION CLIP ViT-G/14, OpenAI CLIP ViT-B/16 and ViT-B/32 |
| Vision / text layers, G/14 | 15 / 5 |
| Vision / text layers, B/16 and B/32 | 4 / 2 |
| Dropout cap ramp, G/14 | 0.10 → 0.15 |
| Dropout cap ramp, B/16 and B/32 | 0.05 → 0.10 |
| Steps / Adam learning rate | 300 / 1.0 |
| Perturbation bound | L∞ ≤ 16 in [0, 255] pixel units (16/255) |
| Source crops / scale | 10 / [0.55, 0.95] |
| Target crop scale | [0.90, 1.00] |
| Text / visual variance weights | 1.0 / 1.0 |

The existing `oattack_pgd` entry implements Adam updates followed by projection
to the perturbation bound. Dropout probabilities are sampled uniformly from
zero to a cap; the cap increases linearly across steps. A proxy sample is held
fixed for paired forwards. The default objective maximizes the mean visual
target similarity and the target-minus-source text similarity, each with its
component variance subtracted. These implementation details are preserved.

## Files and validation

```text
oattack/      Core attack, crops, datasets and three CLIP proxies
oeval/        Victim adapters, inference and matching target subsets
oscore/       Caption scoring and result summaries
configs/      One attack config, one victim registry, one judge registry
data/         Full 1,000-pair dataset plus the original 100-pair default bank
examples/     100 historical G3 adversarial PNGs and their provenance manifest
tools/        Python command-line entry points and release validation
scripts/      Portable shell entry points
requirements/ Separate dependencies for three specialized local evaluators
tests/        Regression checks
docs/         Evaluation setup and release audit
```

```bash
python -m tools.validate_release
python -m unittest discover -s tests -v
```

The supplied audit states the exact validation performed and remaining
publication metadata: [release audit](docs/RELEASE_AUDIT.zh-CN.md).
Historical runs, ablation configurations, repair scripts, backup files,
credentials and model weights are excluded. The provided code repository has
no software license; no license has been selected on behalf of the authors.

## Citation

```bibtex
@article{nie2026one,
  title={One Attack to Fool Them All: Highly Transferable Black-Box Adversarial Attacks on Frontier MLLMs},
  author={Nie, Sen and Zhang, Jie and Wang, Zhongqi and Shan, Shiguang and Chen, Xilin},
  journal={arXiv preprint arXiv:2609.33833},
  year={2026}
}
```
