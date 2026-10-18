"""The event head's input: the policy's own representation of one call. Runs in the openpi environment, next to the policy.

The call's prefix (the camera images and the prompt, after the policy's input transforms) goes through the policy's first
layer + 1 Gemma blocks, one block at a time with the checkpoint's parameters; the residual stream after the last of them is
averaged over the prompt's valid text tokens and rounded to float16, as the training rows are stored. The fit's rows
(training/fit_event_head.py) and the server's calls (server.py) both use this code, so that the head sees at run time
what it was trained on. Written for openpi's pi0.5 models: three image slots of 256 tokens each, then the prompt's tokens.

Numerics: XLA chooses its bf16 kernels per batch size and per process. The rows of a fit are computed in batches of `batch`,
and a call is scored as `batch` copies of itself, so both see the same batch shape. Equal numbers also need the same kernel
choices, which a shared JAX compilation cache gives (server.py --jax-cache); without it a server's scores differ from
the training numerics by about 0.08 logits on average.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.shared import nnx_utils

POOLS = ("ext", "wrist", "text", "last")    # what pool() returns per row, in this order
IMAGE_TOKENS = 256                          # tokens per image slot


def find_key(tree: dict, key: str):
    """The first value stored under `key` anywhere in a nested dict, or None."""
    if key in tree:
        return tree[key]
    for value in tree.values():
        if isinstance(value, dict):
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def pool(x, prefix_mask, text_start: int):
    """(B, 4, width) float32: the mean over the exterior image's tokens, over the wrist image's tokens and over the valid
    text tokens, and the last valid text token."""
    x = x.astype(jnp.float32)
    text_mask = prefix_mask[:, text_start:].astype(jnp.float32)
    exterior = x[:, 0:IMAGE_TOKENS].mean(1)
    wrist = x[:, IMAGE_TOKENS:2 * IMAGE_TOKENS].mean(1)
    text = (x[:, text_start:] * text_mask[..., None]).sum(1) / text_mask.sum(1, keepdims=True)
    last_index = text_start + text_mask.sum(1).astype(jnp.int32) - 1
    last = jnp.take_along_axis(x, last_index[:, None, None], axis=1)[:, 0]
    return jnp.stack([exterior, wrist, text, last], axis=1)


class LayerFeature:
    """The feature of transformed observations (the dicts the policy's input transform returns)."""

    def __init__(self, policy, train_config, layer: int = 12, pool_name: str = "text", batch: int = 8):
        model = policy._model
        paligemma = _gemma.get_config(train_config.model.paligemma_variant)
        expert = _gemma.get_config(train_config.model.action_expert_variant)
        if not 0 <= layer < paligemma.depth:
            raise ValueError(f"layer {layer}: the model has blocks 0..{paligemma.depth - 1}")
        self.layer = int(layer)
        self.pool_index = POOLS.index(pool_name)
        self.batch = int(batch)
        self.layers_params = find_key(nnx.state(model.PaliGemma.llm).to_pure_dict(), "layers")
        block = _gemma.Block(configs=(paligemma, expert))
        text_start = 3 * IMAGE_TOKENS

        def one_block(params, index, x, positions, mask):
            sliced = jax.tree.map(lambda p: p[index], params)
            (out, _), _ = block.apply({"params": sliced}, [x, None], None, positions, mask, [None, None], deterministic=True)
            return out

        self._block = jax.jit(one_block)
        self._pool = jax.jit(lambda x, prefix_mask: pool(x, prefix_mask, text_start))
        self._embed_prefix = nnx_utils.module_jit(model.embed_prefix)

    def residual(self, stacked: dict):
        """The residual stream after the chosen block and the prefix mask, for a batch of stacked observations."""
        observation = _model.preprocess_observation(None, _model.Observation.from_dict(stacked), train=False)
        tokens, prefix_mask, autoregressive = self._embed_prefix(observation)
        mask = _pi0.make_attn_mask(prefix_mask, autoregressive)[:, None]
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        x = tokens.astype(jnp.bfloat16)
        for index in range(self.layer + 1):
            x = self._block(self.layers_params, index, x, positions, mask)
        return x, prefix_mask

    def features(self, inputs: list) -> np.ndarray:
        """(n, width) float32 features of n transformed observations, rounded to float16; computed in batches of `batch`,
        the last one filled up with copies of its last observation."""
        out = []
        for start in range(0, len(inputs), self.batch):
            chunk = list(inputs[start:start + self.batch])
            count = len(chunk)
            chunk += [chunk[-1]] * (self.batch - count)
            stacked = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *chunk)
            x, prefix_mask = self.residual(stacked)
            pooled = np.asarray(self._pool(x, prefix_mask))
            out.append(pooled[:count, self.pool_index])
        return np.concatenate(out).astype(np.float16).astype(np.float32)

    def feature(self, inputs: dict) -> np.ndarray:
        """(width,) float32: the feature of one transformed observation, computed as `batch` copies of it."""
        return self.features([inputs])[0]
