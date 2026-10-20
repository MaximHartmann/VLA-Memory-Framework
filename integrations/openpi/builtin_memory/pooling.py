"""The memory's input: the frozen backbone's residual stream, pooled over the prefix, from the same prefix pass that fills
the KV cache for the actions (no second pass).

openpi's Gemma module runs its blocks in a scan. PoolingModule has the same parameters; its blocks (PoolingBlock) also return
the residual stream they receive, pooled: the mean over the valid text tokens, over the exterior image's 256 tokens and over
the wrist image's 256 tokens. pooled[l] is the input of block l, the output of block l - 1, so the features after block 12
are pooled[13]. Without pool_mask the module computes exactly what openpi's computes. Written for pi0.5's prefix: three image
slots of 256 tokens, then the prompt's tokens.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import gemma as _gemma

IMAGE_TOKENS = 256      # tokens per image slot
IMAGE_SLOTS = 3         # exterior, wrist, and a third slot (empty in the reference task)


def pool_prefix(residual, prefix_mask):
    """(B, 3, width) float32: the residual stream (B, S, width) averaged over the valid text tokens, over the exterior
    image's tokens and over the wrist image's tokens, in this order."""
    text_start = IMAGE_TOKENS * IMAGE_SLOTS
    residual = residual.astype(jnp.float32)
    text_mask = prefix_mask[:, text_start:].astype(jnp.float32)
    exterior = residual[:, 0:IMAGE_TOKENS].mean(1)
    wrist = residual[:, IMAGE_TOKENS:2 * IMAGE_TOKENS].mean(1)
    text = (residual[:, text_start:] * text_mask[..., None]).sum(1) / text_mask.sum(1, keepdims=True)
    return jnp.stack([text, exterior, wrist], axis=1)


class PoolingBlock(_gemma.Block):
    """openpi's Gemma block; with pool_mask (the prefix's input mask) it also returns the pooled residual stream it
    receives, else None."""

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True, pool_mask=None):  # noqa: FBT002
        pooled = None if pool_mask is None else pool_prefix(xs[0], pool_mask)
        xs, kv_cache = super().__call__(xs, kv_cache, positions, attn_mask, adarms_cond, deterministic)
        return xs, (kv_cache, pooled)


class PoolingModule(_gemma.Module):
    """openpi's Gemma module with the same parameters, whose prefix pass can also return the pooled input of every block."""

    def setup(self):
        # A copy of gemma.Module.setup with PoolingBlock and pool_mask as a sixth broadcast input of the scanned blocks:
        # setup builds the scan, so the block class cannot be exchanged otherwise. The parameter names are openpi's.
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        self.embedder = _gemma.Embedder(vocab_size=_gemma.PALIGEMMA_VOCAB_SIZE, embed_dim=self.configs[0].width,
                                        name="embedder")
        block_cls = nn.remat(PoolingBlock, prevent_cse=False, static_argnums=(5,),   # 5 = deterministic
                             policy=jax.checkpoint_policies.nothing_saveable)
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(configs=self.configs, dropout=self.dropout, dropout_bdims=self.dropout_bdims)
        names = [_gemma._name("final_norm", i) for i in range(len(self.configs))]  # noqa: SLF001
        self.final_norms = [_gemma.RMSNorm(name=name) for name in names]

    def __call__(self, embedded, positions, mask, adarms_cond=None, *, kv_cache=None, deterministic=True,  # noqa: FBT002
                 pool_mask=None):
        """openpi's Module.__call__ (copied, with pool_mask passed on to the blocks): (outputs, kv_cache), and with pool_mask
        (outputs, kv_cache, pooled), pooled float32 [depth, B, 3, width]."""
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        embedded, (kv_cache, pooled) = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic,
                                                   pool_mask)
        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)
        outputs = [norm(e, a)[0] if e is not None else e
                   for norm, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)]
        if pool_mask is None:
            return outputs, kv_cache
        return outputs, kv_cache, pooled


def pooling_module(module: _gemma.Module) -> PoolingModule:
    """A PoolingModule with the settings of openpi's Gemma module `module` (it uses the same parameters)."""
    return PoolingModule(configs=module.configs, embed_dtype=module.embed_dtype, dropout=module.dropout,
                         dropout_bdims=module.dropout_bdims, adarms=module.adarms)
