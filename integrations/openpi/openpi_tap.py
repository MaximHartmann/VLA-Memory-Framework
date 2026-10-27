"""A state tap on an openpi pi0.5 policy, JAX or PyTorch: one backbone block's state from the policy's own prefix pass,
averaged over the prompt's valid tokens (or over a camera's picture tokens), for a head on it. No second pass.

    tap = attach(policy, layer=12)          # after policy_config.create_trained_policy, for either kind of checkpoint
    outputs = policy.infer(observation)     # the actions, as without the tap
    feature = tap.read()[0]                 # (2048,) after block 12

JAX: the policy's sample_actions becomes openpi's (copied) with the prefix pass through builtin_memory/pooling.py, which also
returns the pooled input of every block. PyTorch: a forward hook on the backbone's decoder layer; openpi's PyTorch prefix pass
calls the layers one by one, and its action steps run only the action expert, so the hook fires once per call. Runs in
openpi's environment with this folder (integrations/openpi) and the framework on PYTHONPATH.
"""
from __future__ import annotations

import numpy as np

IMAGE_TOKENS = 256                      # per picture slot; three slots come before the prompt's tokens
POOLS = ("text", "ext", "wrist")        # pooling.py's order


def attach(policy, layer: int = 12, pool: str = "text"):
    """A tap on block `layer` of the policy's backbone: the mean over the prompt's valid tokens ("text") or over a camera's
    picture tokens ("ext", "wrist"). The policy's sample_actions is wrapped; its actions are unchanged."""
    if pool not in POOLS:
        raise ValueError(f"pool {pool!r}: one of {POOLS}")
    return (_attach_torch if getattr(policy, "_is_pytorch_model", False) else _attach_jax)(policy, layer, pool)


def _attach_torch(policy, layer: int, pool: str):
    from vla_memory.taps.torch_tap import TorchStateTap
    start, stop = {"text": (3 * IMAGE_TOKENS, None), "ext": (0, IMAGE_TOKENS), "wrist": (IMAGE_TOKENS, 2 * IMAGE_TOKENS)}[pool]
    tap = TorchStateTap(policy._model, f"paligemma_with_expert.paligemma.language_model.layers.{layer}", start, stop)
    sample = policy._sample_actions

    def sample_actions(device, observation, **kwargs):
        prompt = observation.tokenized_prompt_mask.detach().cpu().numpy().astype(bool)
        tap.valid = np.concatenate([np.ones((len(prompt), 3 * IMAGE_TOKENS), bool), prompt], axis=1)
        return sample(device, observation, **kwargs)

    policy._sample_actions = sample_actions
    return tap


def _attach_jax(policy, layer: int, pool: str):
    import jax
    from flax import nnx
    from builtin_memory.pooling import pooling_module
    from vla_memory.taps.jax_tap import JaxStateTap

    model = policy._model
    model.PaliGemma.llm.module = pooling_module(model.PaliGemma.llm.module)    # the same parameters
    graphdef, state = nnx.split(model)
    run = jax.jit(lambda state, rng, observation, noise, num_steps: sample_actions_pooled(
        nnx.merge(graphdef, state), rng, observation, layer, POOLS.index(pool), noise, num_steps))
    tap = JaxStateTap()

    def sample_actions(rng, observation, noise=None, num_steps=10):
        actions, tap.value = run(state, rng, observation, noise, num_steps)
        return actions

    policy._sample_actions = sample_actions
    return tap


def sample_actions_pooled(model, rng, observation, layer: int, pool_index: int, noise=None, num_steps=10):
    """openpi's Pi0.sample_actions (commit 215abfb, copied) with the prefix pass through the pooling module: (actions, the
    pooled state after block `layer`, (B, width))."""
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models import pi0 as _pi0

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache, pooled = model.PaliGemma.llm([prefix_tokens, None], mask=_pi0.make_attn_mask(prefix_mask, prefix_ar_mask),
                                              positions=positions, pool_mask=prefix_mask)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t,
                                                                                     jnp.broadcast_to(time, batch_size))
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=positions,
                                                 kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        return x_t + dt * model.action_out_proj(suffix_out[:, -model.action_horizon:]), time + dt

    x_0, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0))
    return x_0, pooled[layer + 1][:, pool_index]
