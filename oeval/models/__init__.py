from importlib import import_module


def _safe_import(module_name: str) -> None:
    try:
        import_module(f"{__name__}.{module_name}")
    except Exception:
        # Each runtime environment only needs the wrapper it actually instantiates.
        # Swallow import failures here so older/newer transformers stacks can coexist.
        return


for _module in [
    "deepseek_vl2_tiny",
    "gemma3_12b_it",
    "internvl3_5_8b_instruct",
    "llava_16_mistral_7b_hf",
    "minicpm_v_4_5",
    "molmo2_8b",
    "openrouter_multimodal_api",
    "qwen3_vl_8b_instruct",
]:
    _safe_import(_module)
