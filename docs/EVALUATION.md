# Evaluation runtimes

The three pipeline launchers use the active Python environment by default.
They accept `PYTHON_BIN=/absolute/path/to/python` if the shell's `python` points
elsewhere. Python entry points can also be run with `python -m tools.attack`,
`python -m tools.evaluate`, and `python -m tools.score` from the repository root.

Most local models use the main runtime in `requirements.txt`. The original
registry's environment names are retained as routing labels; they are not
required to exist unless you explicitly enable Conda routing.

| Registry labels | Dependencies | Optional Python override |
|---|---|---|
| `llava16`, `qwen8b`, `gemma3_12b`, `internvl35_8b` | `requirements.txt` | `OATTACK_PYTHON_MATTACK` |
| `minicpm_v45` | `requirements/minicpm.txt` | `OATTACK_PYTHON_VLM_NEW` |
| `deepseek_vl2_tiny` | `requirements/deepseek.txt` | `OATTACK_PYTHON_VLM_DEEPSEEK2` |
| `molmo2_8b` | `requirements/molmo.txt` | `OATTACK_PYTHON_VLM_MOLMO` |
| All API models | Main runtime; `OPENROUTER_API_KEY` | `OATTACK_PYTHON_MATTACK` |

Create separate Python 3.10 virtual environments for the three specialized
models and install their corresponding requirements. For example:

```bash
python3.10 -m venv .venv-minicpm
.venv-minicpm/bin/python -m pip install -r requirements/minicpm.txt
export OATTACK_PYTHON_VLM_NEW="$PWD/.venv-minicpm/bin/python"
```

The DeepSeek requirements include the exact upstream code revision installed
in the original environment. Dependencies are version records of the working
environments, not a fully locked transitive environment. A clean installation
of every optional model has not been validated during this release audit.

If reusing Conda, set `CONDA_BIN` explicitly. Child processes then use the
registry's named Conda environments (`mattack`, `vlm_new`, `vlm_deepseek2`,
`vlm_molmo`). An explicit Python override takes precedence over Conda routing.
`CONDA_ENV` selects the attack/scoring environment when Conda routing is used.

Local model names are portable Hugging Face IDs. To use already-downloaded
weights, change the corresponding `model_name` in the one victim registry to
your local directory and set `local_files_only` to `true`. Some upstream
models require authentication or their own code packages; their loader's
errors identify missing dependencies.

For all configured victims:

```bash
bash scripts/evaluate.sh --input-root "$ADV_ROOT" --attack-tag default \
  --models all --gpus 0,1 --max-local-workers 2 --max-new-tokens 128
bash scripts/score.sh --runs-root outputs/eval/default --attack-tag default \
  --models all --strict
```

Only local evaluation queries CUDA GPUs. API-only evaluation does not require
`nvidia-smi`. Evaluation automatically creates a subset of target images with
the same IDs as the adversarial images and rejects missing or duplicate IDs.
Use the same `--models`, `--attack-tag`, and any custom `--target-tag` when
scoring. The original filename suffix `short_sentence_v1` and `qwen_api`
output suffixes are retained for compatibility; the recorded judge model ID
identifies the actual scoring model.
