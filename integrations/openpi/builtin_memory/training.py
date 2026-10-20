"""What the memory changes in openpi's training: the frozen backbone, the memory's own learning rate, and a weight loader for
checkpoints without the memory.
"""
from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import numpy as np
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import download, nnx_utils
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders


def frozen_backbone() -> nnx.filterlib.Filter:
    """The frozen parameters: SigLIP and the PaliGemma expert with its LoRA, so that the cached features stay exact, and the
    action expert's base weights. Trained: the action expert's LoRA, the action and time projections, the memory and its
    phase head."""
    llm = nnx_utils.PathRegex(".*llm.*")
    action_expert = nnx_utils.PathRegex(".*llm.*_1.*")
    lora = nnx_utils.PathRegex(".*lora.*")
    return nnx.Any(nnx_utils.PathRegex(".*PaliGemma/img.*"), nnx.All(llm, nnx.Not(action_expert)),
                   nnx.All(action_expert, nnx.Not(lora)))


@dataclasses.dataclass(frozen=True)
class MemoryAdamW(_optimizer.OptimizerConfig):
    """openpi's AdamW (the same settings, one global gradient clip) with memory_lr_scale times the learning rate for every
    parameter whose path contains "memory_": the memory starts from scratch while the action expert continues from a
    converged checkpoint."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0
    memory_lr_scale: float = 10.0

    def create(self, lr: optax.ScalarOrSchedule, weight_decay_mask: at.PyTree | None = None) -> optax.GradientTransformation:
        scale = self.memory_lr_scale
        memory_lr = (lambda step: lr(step) * scale) if callable(lr) else lr * scale
        options = {"b1": self.b1, "b2": self.b2, "eps": self.eps, "weight_decay": self.weight_decay,
                   "mask": weight_decay_mask}

        def labels(params):
            return jax.tree_util.tree_map_with_path(
                lambda path, _: "memory" if "memory_" in jax.tree_util.keystr(path) else "base", params)

        transforms = {"base": optax.adamw(lr, **options), "memory": optax.adamw(memory_lr, **options)}
        groups = optax.multi_transform(transforms, labels)
        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), groups)


@dataclasses.dataclass(frozen=True)
class MemoryWeightLoader(_weight_loaders.CheckpointWeightLoader):
    """openpi's CheckpointWeightLoader for a checkpoint that may lack the memory (and the LoRA weights): what is missing
    keeps its initial value, so a new memory starts with its read-out at zero."""

    missing_regex: str = ".*lora.*|memory_.*"

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _weight_loaders._merge_params(loaded, params, missing_regex=self.missing_regex)  # noqa: SLF001
