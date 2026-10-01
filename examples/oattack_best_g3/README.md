# G3 adversarial examples

`nips17/` contains the original 100 best-G3 adversarial PNGs, IDs `0`–`99`,
at 224 × 224 resolution. They use the default 100-pair caption bank and the
G3 settings in `configs/attack/default.yaml`: seed 2023, 300 Adam steps,
learning rate 1 and an L∞ limit of 16 in [0,255] pixel units.

`manifest.json` records each PNG hash, source/target paths, shared optimization
settings and four historical run identifiers. Pixel differences are checked
against the source image **after the attack's bicubic resize and center crop**,
rather than the unprocessed source dimensions.

To evaluate without generating attacks:

```bash
export OPENROUTER_API_KEY="your-key"
MODELS="gpt54_vision,claude_sonnet_4_6,gemini31_flash_lite,grok43,qwen35_397b_a17b,kimi_k2_5"
bash scripts/evaluate.sh --input-root examples/oattack_best_g3/nips17 \
  --attack-tag example_g3 --models "$MODELS" --max-new-tokens 128
bash scripts/score.sh --runs-root outputs/eval/example_g3 \
  --attack-tag example_g3 --models "$MODELS" --strict
```

The examples came from four independent 25-image runs in lexical image order.
Each reset the seed to 2023. To repeat those boundaries with the same settings:

```bash
for START in 0 25 50 75; do
  bash scripts/attack.sh --sample-start "$START" --num-samples 25 \
    --device cuda:0,cuda:1 --output-root "outputs/attack/g3_chunk_${START}"
done
```

Historical generation used two CUDA GPUs with model parallelism. Hardware and
library differences may affect exact output bytes; the default 100-image run
also follows a different random-number trajectory after its first 25 images.
The included outputs are preserved as examples, rather than substituted with
new runs. No historical victim captions or aggregate evaluation results are
bundled; reproduce those using the evaluation/scoring pipeline.
