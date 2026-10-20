"""pi0.5 with the built-in memory: a recurrent state inside the policy, as subclasses of openpi's Pi0Config and Pi0.

    features    the frozen backbone's residual stream after block `memory_layer` (12), pooled over the prefix: the means
                over the text, the exterior image and the wrist image, 3 x 2048 numbers, rounded to float16 (pooling.py)
    cell        z-scored with fixed training statistics, Linear(6144 -> 256) + LayerNorm, then a GRU with a state of 256
                numbers (torch's GRUCell: gates r, z, n; h' = (1 - z) n + z h); one update per policy call, all in float32
    read-out    Linear(256 -> 1024), zero-initialised, added to the action expert's adaRMS conditioning (the flow-time
                embedding), which modulates every norm of the action expert; the untrained memory adds exactly 0
    phase head  optional: Linear(256 -> phases) on the state of every call, cross-entropy against the task's phase

The state is zero at the episode start, and the actions of call t see h_t = GRU(h_t-1, x_t). Training reads every call's
features from a cache (MemoryObservation), inference computes them in the call's own prefix pass (sample_actions_memory).
The parameters are named "memory_*", as in the checkpoints of our experiments, so that those load unchanged.
"""
from __future__ import annotations

import dataclasses

import einops
import flax.nnx as nnx
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at

from .pooling import pooling_module

ArrayT = _model.ArrayT


@dataclasses.dataclass(frozen=True)
class MemoryPi0Config(pi0_config.Pi0Config):
    """openpi's model config of pi0.5 plus the memory (module docstring)."""

    pi05: bool = True
    memory_dim: int = 256                   # the GRU's state
    memory_layer: int = 12                  # the features: the residual stream after this block (0-based)
    memory_num_phases: int = 0              # > 0: the phase head, with this many phases
    memory_aux_weight: float = 0.1          # the weight of the phase head's cross-entropy in the loss
    memory_feat_stats: str | None = None    # .npz with "mean" and "std" (6144,) of the training features; for a new memory
    # Training with the backbone frozen: the prefix once, without gradient, and the action expert against its KV cache, as
    # at inference, instead of one joint pass whose rematerialisation recomputes the prefix in the backward pass.
    frozen_prefix_loss: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("the memory enters through the action expert's adaRMS conditioning: pi0.5 only (pi05=True)")

    @override
    def create(self, rng: at.KeyArrayLike) -> MemoryPi0:
        return MemoryPi0(self, rngs=nnx.Rngs(rng))


@at.typecheck
@struct.dataclass
class MemoryObservation(_model.Observation):
    """openpi's Observation plus the call history of a training sample, right-aligned (the current call last)."""

    memory_feats: at.Float[ArrayT, "*b c f"] | None = None    # the cached features of the calls
    memory_mask: at.Bool[ArrayT, "*b c"] | None = None        # True for the rows that are calls
    memory_phase: at.Int[ArrayT, "*b c"] | None = None        # their phase labels, -1 where unknown

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> MemoryObservation:
        """openpi's conversion of a batch dict, plus its memory_feats, memory_mask and memory_phase."""
        observation = super().from_dict(data)
        return observation.replace(memory_feats=data.get("memory_feats"), memory_mask=data.get("memory_mask"),
                                   memory_phase=data.get("memory_phase"))


class MemoryPi0(_pi0.Pi0):
    """openpi's Pi0 (pi0.5) with the memory (module docstring)."""

    def __init__(self, config: MemoryPi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        # The same Gemma parameters, run by a module whose prefix pass can also return the pooled block inputs.
        self.PaliGemma.llm.module = pooling_module(self.PaliGemma.llm.module)
        self.memory_layer = config.memory_layer
        self.memory_num_phases = config.memory_num_phases
        self.memory_aux_weight = config.memory_aux_weight
        self.frozen_prefix_loss = config.frozen_prefix_loss
        self._init_memory(config, rngs)

    def _init_memory(self, config: MemoryPi0Config, rngs: nnx.Rngs):
        """The memory's parameters, all float32, created in a fixed order: a new memory starts from the same initial values
        as in the trainings of the measured checkpoints."""
        feature_dim = 3 * _gemma.get_config(config.paligemma_variant).width
        mean, std = feature_stats(config.memory_feat_stats, feature_dim)
        # the fixed z-scoring of the features: not trained (BatchStat), stored with the parameters
        self.memory_feat_mean = nnx.BatchStat(jnp.asarray(mean))
        self.memory_feat_istd = nnx.BatchStat(jnp.asarray(1.0 / np.maximum(std, 1e-6)))
        f32 = {"dtype": jnp.float32, "param_dtype": jnp.float32}
        dim = config.memory_dim
        self.memory_in = nnx.Linear(feature_dim, dim, rngs=rngs, **f32)
        self.memory_in_norm = nnx.LayerNorm(dim, rngs=rngs, **f32)
        self.memory_gru_x = nnx.Linear(dim, 3 * dim, rngs=rngs, **f32)
        self.memory_gru_h = nnx.Linear(dim, 3 * dim, rngs=rngs, **f32)
        expert_width = _gemma.get_config(config.action_expert_variant).width
        self.memory_out = nnx.Linear(dim, expert_width, rngs=rngs, kernel_init=nnx.initializers.zeros,
                                     bias_init=nnx.initializers.zeros, **f32)
        if config.memory_num_phases > 0:
            self.memory_head = nnx.Linear(dim, config.memory_num_phases, rngs=rngs, **f32)

    # ------------------------------------------------------------------ the memory
    def memory_zero_state(self, batch_size: int):
        """The state at the start of an episode."""
        return jnp.zeros((batch_size, self.memory_out.in_features), jnp.float32)

    def memory_scan(self, features, mask, state):
        """The GRU over a sequence of calls: features (B, C, 6144) (float16 values), mask (B, C) bool, state (B, D) float32.
        A masked call keeps the state. Returns (the last state (B, D), every call's state (B, C, D)), float32."""
        x = (features.astype(jnp.float32) - self.memory_feat_mean.value) * self.memory_feat_istd.value
        gates_x = self.memory_gru_x(self.memory_in_norm(self.memory_in(x)))   # the input part of the gates, every call
        kernel_h = self.memory_gru_h.kernel.value.astype(jnp.float32)
        bias_h = self.memory_gru_h.bias.value.astype(jnp.float32)

        def step(h, inputs):
            gx, valid = inputs
            gh = h @ kernel_h + bias_h
            xr, xz, xn = jnp.split(gx, 3, axis=-1)
            hr, hz, hn = jnp.split(gh, 3, axis=-1)
            r = jax.nn.sigmoid(xr + hr)
            z = jax.nn.sigmoid(xz + hz)
            n = jnp.tanh(xn + r * hn)
            h = jnp.where(valid[:, None], (1.0 - z) * n + z * h, h)
            return h, h

        calls = (jnp.swapaxes(gates_x, 0, 1), jnp.swapaxes(mask.astype(jnp.bool_), 0, 1))
        last, states = jax.lax.scan(step, state, calls)
        return last, jnp.swapaxes(states, 0, 1)

    def phase_loss(self, states, mask, phases):
        """The phase head's mean cross-entropy over the calls with a label, and its accuracy."""
        logits = self.memory_head(states)
        valid = mask & (phases >= 0)
        labels = jnp.clip(phases, 0, self.memory_num_phases - 1)
        cross_entropy = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        count = jnp.maximum(jnp.sum(valid), 1)
        loss = jnp.sum(jnp.where(valid, cross_entropy, 0.0)) / count
        accuracy = jnp.sum(jnp.where(valid, jnp.argmax(logits, -1) == phases, False)) / count
        return loss, accuracy

    @override
    def embed_suffix(self, obs, noisy_actions, timestep, memory_cond=None):
        """openpi's suffix embedding, with the memory's read-out added to the adaRMS conditioning."""
        tokens, input_mask, ar_mask, adarms_cond = super().embed_suffix(obs, noisy_actions, timestep)
        if memory_cond is not None:
            adarms_cond = adarms_cond + memory_cond
        return tokens, input_mask, ar_mask, adarms_cond

    # ------------------------------------------------------------------ training
    @override
    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        """The loss per (sample, action step); see compute_loss_and_metrics."""
        return self.compute_loss_and_metrics(rng, observation, actions, train=train)[0]

    def compute_loss_and_metrics(self, rng, observation, actions, *, train: bool = False):
        """The flow-matching loss per (sample, action step) plus the phase head's weighted cross-entropy (a batch scalar
        added to every entry, so that the mean is action loss + weight x cross-entropy), and scalar metrics. The memory runs
        over the sample's call history (MemoryObservation); its last state conditions the actions."""
        if getattr(observation, "memory_feats", None) is None:
            raise ValueError("the batch has no memory_feats: train with builtin_memory.data's data loader")
        batch_size = observation.memory_feats.shape[0]
        start = self.memory_zero_state(batch_size)
        last, states = self.memory_scan(observation.memory_feats, observation.memory_mask, start)
        memory_cond = self.memory_out(last)
        metrics = {"memory_state_norm": jnp.mean(jnp.linalg.norm(last, axis=-1)),
                   "memory_cond_norm": jnp.mean(jnp.linalg.norm(memory_cond, axis=-1)),
                   "memory_calls": jnp.mean(jnp.sum(observation.memory_mask.astype(jnp.float32), axis=-1))}
        auxiliary = None
        if self.memory_num_phases > 0:
            auxiliary, metrics["aux_acc"] = self.phase_loss(states, observation.memory_mask, observation.memory_phase)
            metrics["aux_ce"] = auxiliary
        action_loss = self.flow_loss(rng, observation, actions, train=train, memory_cond=memory_cond)
        metrics["action_loss"] = jnp.mean(action_loss)
        loss = action_loss if auxiliary is None else action_loss + self.memory_aux_weight * auxiliary
        return loss, metrics

    def flow_loss(self, rng, observation, actions, *, train: bool, memory_cond):
        """openpi's Pi0.compute_loss with the memory's conditioning (a copy: its suffix embedding cannot take it otherwise),
        and with frozen_prefix_loss the prefix computed once, without gradient, as at inference."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        if self.frozen_prefix_loss:
            suffix_out = self._suffix_on_frozen_prefix(observation, prefix_tokens, prefix_mask, prefix_ar_mask, x_t, time,
                                                       memory_cond)
        else:
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time, memory_cond)
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
                                                    adarms_cond=[None, adarms_cond])
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _suffix_on_frozen_prefix(self, observation, prefix_tokens, prefix_mask, prefix_ar_mask, x_t, time, memory_cond):
        """The action expert's output for the noisy actions, attending to the prefix's KV cache, which gets no gradient."""
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)
        kv_cache = jax.lax.stop_gradient(kv_cache)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time, memory_cond)
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=positions,
                                                kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        return suffix_out

    # ------------------------------------------------------------------ inference
    @override
    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        """openpi's interface: the actions of one call from a zero state, as if every call started an episode."""
        return self.sample_actions_memory(rng, observation, None, num_steps=num_steps, noise=noise)[0]

    def sample_actions_memory(self, rng, observation, memory_state, *, num_steps=10, noise=None):
        """One policy call: the prefix pass also yields the call's features, the GRU updates the state with them, and the
        new state conditions the denoising. memory_state: (B, D), None for a zero state. Returns (actions, {"state": (B, D)
        float32, "feats": (B, 6144) float16, and with the phase head "phase_logits": (B, phases)}). openpi's
        Pi0.sample_actions, copied, with the memory added."""
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, pooled = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
                                                 pool_mask=prefix_mask)
        features = pooled[self.memory_layer + 1].reshape(batch_size, -1).astype(jnp.float16)   # the cache's precision
        state = self.memory_zero_state(batch_size) if memory_state is None else memory_state.astype(jnp.float32)
        state, _ = self.memory_scan(features[:, None], jnp.ones((batch_size, 1), jnp.bool_), state)
        memory = {"state": state, "feats": features}
        if self.memory_num_phases > 0:
            memory["phase_logits"] = self.memory_head(state)
        actions = self._denoise(observation, prefix_tokens, prefix_mask, kv_cache, noise, num_steps, self.memory_out(state))
        return actions, memory

    def _denoise(self, observation, prefix_tokens, prefix_mask, kv_cache, noise, num_steps, memory_cond):
        """openpi's flow-matching loop of Pi0.sample_actions (copied), with the memory's conditioning: from noise at t = 1 to
        the actions at t = 0, the action expert attending to the prefix's KV cache."""
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), memory_cond)
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            suffix_length = suffix_tokens.shape[1]
            assert full_attn_mask.shape == (batch_size, suffix_length, prefix_tokens.shape[1] + suffix_length)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=positions,
                                                             kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2      # robust to floating-point error

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0


def feature_stats(path: str | None, feature_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """The features' mean and standard deviation from an .npz with "mean" and "std"; without one zeros and ones (a checkpoint
    that holds the memory brings its own)."""
    if not path:
        return np.zeros(feature_dim, np.float32), np.ones(feature_dim, np.float32)
    stats = np.load(path)
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    if mean.shape != (feature_dim,) or std.shape != (feature_dim,):
        raise ValueError(f"{path}: mean and std must have shape ({feature_dim},), not {mean.shape} and {std.shape}")
    return mean, std
