"""openpi's JAX-to-PyTorch conversion for a LoRA fine-tune. openpi's converter (examples/convert_jax_model_to_pytorch.py)
ignores the lora_a / lora_b weights, so a LoRA fine-tune would lose its LoRA part without a word. Here they are merged into
the base weights first: W + scale * (A @ B) along the weight's last two axes (openpi's lora.Einsum, scale alpha / rank; the
feed-forward's LoRA has no scale), then openpi's converter runs. The checkpoint's assets (the normalisation statistics) are
copied next to the converted weights, where openpi looks for them.

    python convert_to_pytorch.py --checkpoint <JAX checkpoint dir> --config <openpi config> --out <PyTorch checkpoint dir>

Runs in openpi's environment; the PyTorch policy then needs openpi's transformers patch (openpi's README, PyTorch support).
"""
import argparse
import importlib.util
import pathlib
import shutil

import numpy as np
import openpi
from openpi.models import gemma as _gemma
from openpi.training import config as _config


def lora_scales(model_config) -> set:
    """The scales of the model's LoRA einsums (alpha / rank, or alpha / sqrt(rank) for rsLoRA)."""
    return {lora.scaling_value for variant in (model_config.paligemma_variant, model_config.action_expert_variant)
            for name, lora in _gemma.get_config(variant).lora_configs.items() if name == "attn"}


def merge_lora(flat: dict, scale: float) -> dict:
    """The flattened PaliGemma parameters with every lora_a / lora_b pair merged into its weight (and removed)."""
    for key_a in [key for key in flat if key.endswith("lora_a")]:
        key_b = key_a[:-1] + "b"
        einsum = key_a.endswith("/lora_a")                  # lora.Einsum: <name>/w, <name>/lora_a; FeedForward: <name>_lora_a
        key_w = key_a[: -len("/lora_a")] + "/w" if einsum else key_a[: -len("_lora_a")]
        flat[key_w] = flat[key_w] + (scale if einsum else 1.0) * np.matmul(flat.pop(key_a), flat.pop(key_b))
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="the JAX checkpoint directory (with params/ and assets/)")
    parser.add_argument("--config", required=True, help="the policy's openpi training config")
    parser.add_argument("--out", required=True, help="the PyTorch checkpoint directory to write")
    parser.add_argument("--precision", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--converter", default=None, help="openpi's examples/convert_jax_model_to_pytorch.py (default: found "
                                                          "next to the openpi package)")
    args = parser.parse_args()
    path = args.converter or pathlib.Path(openpi.__file__).parents[2] / "examples" / "convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("openpi_convert", path)
    convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert)
    model_config = _config.get_config(args.config).model
    scales = lora_scales(model_config)
    if len(scales) > 1:
        raise SystemExit(f"the two experts' LoRA scales differ ({sorted(scales)}): merge them one by one")
    scale = scales.pop() if scales else 1.0
    load = convert.slice_initial_orbax_checkpoint

    def load_merged(checkpoint_dir, restore_precision=None):
        params = load(checkpoint_dir, restore_precision)
        params["paligemma_params"] = merge_lora(params["paligemma_params"], scale)
        return params

    convert.slice_initial_orbax_checkpoint = load_merged
    convert.convert_pi0_checkpoint(args.checkpoint, args.precision, args.out, model_config)
    shutil.copytree(pathlib.Path(args.checkpoint) / "assets", pathlib.Path(args.out) / "assets", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
