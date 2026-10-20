"""The stateful policy: openpi's Policy with one memory state per episode, and how it is built from a checkpoint.

Episodes are marked by the framework's episode markers, which the proxy forwards: memory/new_episode (True at the first call
of an episode) and memory/episode (an id; a new id starts a new episode). At an episode start the state is set to zero;
with reset_every_call it is cleared before every call (the reset test, which the proxy's --memory-reset every-call also
gives through the markers). Every memory/ key is removed before the input transforms. Every reply carries

    reply["memory"] = {"episode", "call" (0-based in the episode), "reset" (the call started the episode),
                       "reset_every_call", "state_norm", and with a phase head "phase" and "phase_probs"}

create_policy builds it from a checkpoint like openpi's create_trained_policy, with the memory kept in float32 (its training
precision) and everything else in bfloat16, and with the optional centre crop of the exterior picture.
"""
from __future__ import annotations

import pathlib
import time

import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import download, nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from vla_memory.env_keys import CONTROL_PREFIX
from vla_memory.writers.head.crop import CenterCrop

NEW_EPISODE = CONTROL_PREFIX + "new_episode"     # the framework's episode markers
EPISODE = CONTROL_PREFIX + "episode"


class MemoryPolicy(_policy.Policy):
    """openpi's Policy (JAX) of a MemoryPi0, with the memory state of the current episode (module docstring)."""

    def __init__(self, model, *, reset_every_call: bool = False, **policy_options):
        super().__init__(model, **policy_options)
        self._sample_memory = nnx_utils.module_jit(model.sample_actions_memory)
        self._dim = int(model.memory_out.in_features)
        self.reset_every_call = reset_every_call
        self._state: np.ndarray | None = None
        self._episode = None
        self._call = 0

    @property
    def state(self) -> np.ndarray | None:
        """The memory state after the last call, (1, D) float32 (None before the first call)."""
        return self._state

    def start_episode(self, episode=None) -> None:
        """An episode starts: the state is zero."""
        self._state = np.zeros((1, self._dim), np.float32)
        self._episode = episode
        self._call = 0

    def is_new_episode(self, obs: dict) -> bool:
        """Whether this call starts an episode: the first call, memory/new_episode, or a new memory/episode."""
        episode = obs.get(EPISODE)
        return self._state is None or bool(obs.get(NEW_EPISODE, False)) or (episode is not None and episode != self._episode)

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """One policy call: openpi's Policy.infer (its transforms, a batch of one, the same random number sequence), with
        the memory state in and out."""
        if self.is_new_episode(obs):
            self.start_episode(obs.get(EPISODE))
        state_in = np.zeros_like(self._state) if self.reset_every_call else self._state
        inputs = {key: value for key, value in obs.items() if not key.startswith(CONTROL_PREFIX)}
        inputs = jax.tree.map(lambda x: x, inputs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            sample_kwargs["noise"] = noise[None, ...] if noise.ndim == 2 else noise
        started = time.monotonic()
        actions, memory = self._sample_memory(sample_rng, _model.Observation.from_dict(inputs), jnp.asarray(state_in),
                                              **sample_kwargs)
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), {"state": inputs["state"], "actions": actions})
        model_ms = 1000 * (time.monotonic() - started)
        self._state = np.asarray(memory["state"], np.float32)
        outputs = self._output_transform(outputs)
        outputs["memory"] = self._memory_info(obs.get(EPISODE), memory)
        outputs["policy_timing"] = {"infer_ms": model_ms}
        self._call += 1
        return outputs

    def _memory_info(self, episode, memory: dict) -> dict:
        """reply["memory"] of this call (module docstring)."""
        info = {"episode": episode, "call": self._call, "reset": self._call == 0, "reset_every_call": self.reset_every_call,
                "state_norm": float(np.linalg.norm(self._state))}
        if "phase_logits" in memory:
            logits = np.asarray(memory["phase_logits"], np.float64)[0]
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            info["phase"] = int(np.argmax(probabilities))
            info["phase_probs"] = [float(p) for p in probabilities]
        return info

    @property
    def metadata(self) -> dict:
        """openpi's metadata plus the memory's: its state size and whether it is cleared before every call."""
        return {**super().metadata, "memory": {"dim": self._dim, "reset_every_call": self.reset_every_call}}


def is_memory_parameter(path: tuple) -> bool:
    """Whether a parameter belongs to the memory (its path has a part "memory_*")."""
    return any(str(part).startswith("memory_") for part in path)


def load_params(params_dir) -> dict:
    """A checkpoint's parameters: the memory's in float32, every other one in bfloat16 (as openpi serves them)."""
    flat = traverse_util.flatten_dict(_model.restore_params(params_dir, dtype=None))
    return traverse_util.unflatten_dict(
        {path: value if is_memory_parameter(path) else value.astype(jnp.bfloat16) for path, value in flat.items()})


def create_policy(train_config: _config.TrainConfig, checkpoint_dir, *, default_prompt: str | None = None,
                  center_crop: float = 0.0, reset_every_call: bool = False) -> MemoryPolicy:
    """The stateful policy of a checkpoint. A copy of the JAX part of openpi's create_trained_policy, which restores every
    parameter in bfloat16; center_crop > 0 adds the centre crop of the exterior picture after the model's transforms."""
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))
    model = train_config.model.load(load_params(checkpoint_dir / "params"))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    quantiles = data_config.use_quantile_norm
    crop = [CenterCrop(center_crop)] if center_crop else []
    inputs = [_transforms.InjectDefaultPrompt(default_prompt), *data_config.data_transforms.inputs,
              _transforms.Normalize(norm_stats, use_quantiles=quantiles), *data_config.model_transforms.inputs, *crop]
    outputs = [*data_config.model_transforms.outputs, _transforms.Unnormalize(norm_stats, use_quantiles=quantiles),
               *data_config.data_transforms.outputs]
    return MemoryPolicy(model, reset_every_call=reset_every_call, transforms=inputs, output_transforms=outputs,
                        metadata=train_config.policy_metadata)
