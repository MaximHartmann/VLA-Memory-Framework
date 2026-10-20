"""The reference recipe: pi0.5 on the red/blue task (pick up and put back a red, then a blue cube) with the built-in memory.

    model_config(phases)        pi0.5 with LoRA on both experts, action horizon 10, no state tokens, and the memory: GRU(256)
                                over the layer-12 features; a phase head with `phases` phases (4 here; 0: none)
    RedBlueDataConfig           the task's inputs in openpi's LIBERO layout (exterior and wrist picture, 8-number state) and
                                8 action numbers back (7 joint targets and the gripper)
    train_config(...)           the training of the memory: continue a checkpoint for 10,000 steps at batch 16 with the
                                backbone frozen; the action expert's LoRA, the projections and the memory train, cosine
                                1e-5 -> 1e-6 after 200 warm-up steps, the memory at 10x; histories of at most 96 calls
    serving_config(checkpoint)  the TrainConfig a checkpoint is served with (its phase head and norm stats read from it)

Another robot or task brings its own DataConfigFactory in place of RedBlueDataConfig; the model and the training stay.
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import orbax.checkpoint as ocp

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer

from .data import HistoryDataConfig
from .model import MemoryPi0Config
from .training import MemoryAdamW, MemoryWeightLoader, frozen_backbone


def model_config(phases: int = 4, **memory) -> MemoryPi0Config:
    """The reference policy's model with the memory; `memory`: more MemoryPi0Config fields (memory_feat_stats, ...)."""
    return MemoryPi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b_lora",
                           action_expert_variant="gemma_300m_lora", memory_num_phases=phases, **memory)


@dataclasses.dataclass(frozen=True)
class ActionDims(_transforms.DataTransformFn):
    """The model's actions cut to the robot's first `dims` numbers (the rest is padding)."""

    dims: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :self.dims])}


@dataclasses.dataclass(frozen=True)
class RedBlueDataConfig(_config.LeRobotLiberoDataConfig):
    """openpi's LIBERO data config (the same keys, pictures and 8-number state), with 8 action numbers back instead of 7."""

    action_dims: int = 8

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        data_config = super().create(assets_dirs, model_config)
        transforms = _transforms.Group(inputs=data_config.data_transforms.inputs, outputs=[ActionDims(self.action_dims)])
        return dataclasses.replace(data_config, data_transforms=transforms)


def train_config(exp_name: str, *, init_checkpoint: str, repo_ids: tuple[str, ...], feature_indexes: tuple[str, ...],
                 phases: int = 4, feature_stats: str | None = None, steps: int = 10_000, batch_size: int = 16,
                 max_calls: int = 96, name: str = "builtin_memory", checkpoint_base_dir: str = "./checkpoints",
                 **train_options) -> _config.TrainConfig:
    """The training of the memory from init_checkpoint (with or without a memory; its norm stats are kept) on the datasets
    repo_ids with their feature caches; the learning rate decays over `steps`. feature_stats: the features' mean and std
    for a new memory; train_options: other TrainConfig fields (num_train_steps to stop earlier, save_interval, overwrite,
    resume, wandb_enabled, ...)."""
    assets = _config.AssetsConfig(assets_dir=f"{init_checkpoint}/assets", asset_id=checkpoint_asset_id(init_checkpoint))
    task = RedBlueDataConfig(repo_id=repo_ids[0], assets=assets, base_config=_config.DataConfig(prompt_from_task=True))
    options = {"num_train_steps": steps, "num_workers": 0, "ema_decay": None, "save_interval": 1000, "keep_period": 1000,
               **train_options}
    return _config.TrainConfig(
        name=name,
        exp_name=exp_name,
        model=model_config(phases, memory_feat_stats=feature_stats),
        data=HistoryDataConfig(task=task, extra_repo_ids=tuple(repo_ids[1:]), feature_indexes=tuple(feature_indexes),
                               max_calls=max_calls),
        weight_loader=MemoryWeightLoader(f"{init_checkpoint}/params"),
        freeze_filter=frozen_backbone(),
        optimizer=MemoryAdamW(clip_gradient_norm=1.0, memory_lr_scale=10.0),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=1e-5, decay_steps=steps, decay_lr=1e-6),
        batch_size=batch_size,
        checkpoint_base_dir=checkpoint_base_dir,
        **options,
    )


def serving_config(checkpoint_dir: str) -> _config.TrainConfig:
    """The TrainConfig a checkpoint of the reference recipe is served with: its phase head and the asset id of its norm
    stats come from the checkpoint."""
    assets = _config.AssetsConfig(assets_dir=f"{checkpoint_dir}/assets", asset_id=checkpoint_asset_id(checkpoint_dir))
    return _config.TrainConfig(name="builtin_memory", data=RedBlueDataConfig(assets=assets),
                               model=model_config(checkpoint_phases(checkpoint_dir)))


def checkpoint_asset_id(checkpoint_dir: str) -> str:
    """The asset id of a checkpoint's norm stats: the folder of its only assets/**/norm_stats.json."""
    assets = pathlib.Path(checkpoint_dir) / "assets"
    found = sorted(assets.glob("**/norm_stats.json"))
    if len(found) != 1:
        raise ValueError(f"{assets}: expected one norm_stats.json, found {len(found)}")
    return found[0].parent.relative_to(assets).as_posix()


def checkpoint_phases(checkpoint_dir: str) -> int:
    """The number of phases of a checkpoint's phase head, read from its metadata (0: no head)."""
    with ocp.PyTreeCheckpointer() as checkpointer:
        tree = checkpointer.metadata(pathlib.Path(checkpoint_dir).resolve() / "params")["params"]
    head = tree.get("memory_head")
    if head is None:
        return 0
    kernel = head["kernel"]
    kernel = kernel.get("value", kernel) if isinstance(kernel, dict) else kernel
    return int(kernel.shape[-1])
