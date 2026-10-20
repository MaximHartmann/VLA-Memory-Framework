"""The training data of the memory: several LeRobot datasets, the frame filter and every sample's call history.

openpi's data loader reads one LeRobot dataset and turns every batch into openpi's Observation, which has no place for a
call history. create_data_loader builds the same pipeline from openpi's own parts: create_torch_dataset for every dataset
(concatenated in order: the demonstrations first, then e.g. the intervention episodes), transform_dataset, the call
histories after the transforms (sequences.HistoryDataset), TorchDataLoader, and batches as MemoryObservation.
"""
from __future__ import annotations

import dataclasses
import pathlib

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

from .model import MemoryObservation
from .sequences import HistoryDataset, HistorySpec


@dataclasses.dataclass(frozen=True)
class HistoryDataConfig(_config.DataConfigFactory):
    """The data of a memory training: the task's data config (`task`: the first dataset, the norm stats and the transforms)
    and, for the call histories, the datasets after it and one feature cache per dataset (sequences.py)."""

    task: _config.DataConfigFactory = dataclasses.field(default_factory=_config.FakeDataConfig)
    extra_repo_ids: tuple[str, ...] = ()
    feature_indexes: tuple[str, ...] = ()       # the task's dataset first, then one per extra dataset
    max_calls: int = 96                         # the history cap

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        """The task's data config: its transforms are the same for every dataset."""
        return self.task.create(assets_dirs, model_config)

    @property
    def repo_ids(self) -> tuple[str, ...]:
        """Every dataset, in order."""
        return (self.task.repo_id, *self.extra_repo_ids)


class HistoryDataLoader(_data_loader.DataLoaderImpl):
    """openpi's data loader whose batches are MemoryObservation, with the call histories."""

    def __iter__(self):
        for batch in self._data_loader:
            yield MemoryObservation.from_dict(batch), batch["actions"]


def create_data_loader(config: _config.TrainConfig, *, sharding: jax.sharding.Sharding | None = None, shuffle: bool = False,
                       num_batches: int | None = None) -> HistoryDataLoader:
    """openpi's torch data loader over the concatenated datasets of config.data (a HistoryDataConfig), with the frame filter
    and the call histories (module docstring)."""
    data = config.data
    data_config = data.create(config.assets_dirs, config.model)
    datasets = [_data_loader.create_torch_dataset(dataclasses.replace(data_config, repo_id=repo_id),
                                                  config.model.action_horizon, config.model) for repo_id in data.repo_ids]
    raw = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    spec = HistorySpec(sources=tuple(data.feature_indexes), max_calls=data.max_calls)
    dataset = HistoryDataset(_data_loader.transform_dataset(raw, data_config), frame_keys(raw), spec)
    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=config.batch_size // jax.process_count(),
                                          sharding=sharding, shuffle=shuffle, num_batches=num_batches,
                                          num_workers=config.num_workers, seed=config.seed)
    return HistoryDataLoader(data_config, loader)


def frame_keys(dataset) -> list[tuple[np.ndarray, np.ndarray]]:
    """(episode_index, frame_index) per LeRobot dataset in `dataset` (one, or a ConcatDataset of them, each possibly inside
    openpi's TransformedDataset), in the order of the global index."""
    parts = dataset.datasets if isinstance(dataset, torch.utils.data.ConcatDataset) else [dataset]
    keys = []
    for part in parts:
        while hasattr(part, "_dataset"):    # openpi's TransformedDataset (the prompt from the LeRobot task)
            part = part._dataset            # noqa: SLF001
        table = part.hf_dataset.with_format(None)
        keys.append((np.asarray(table["episode_index"], np.int64), np.asarray(table["frame_index"], np.int64)))
    return keys
