"""The call histories of the training samples (numpy only).

A training sample is a frame t of an episode with openpi's usual contents (pictures, prompt, action chunk) plus the cached
features of the episode's policy calls up to and including t, as the memory would have seen them at run time: going back
from t, the gaps between two calls are drawn uniformly from 6..10 frames, and after a call the call is repeated 1-3 times on
the same frame with probability 0.1 (a pause). The walk stops at the episode's first frame or at `max_calls` calls; the most
recent ones are kept, right-aligned, padded and masked:

    memory_feats (max_calls, 6144) float16    memory_mask (max_calls,) bool    memory_phase (max_calls,) int32 (-1: none)

Feature cache of a dataset: an index .npz
    features            str     the path of an (N, 6144) float16 .npy: one row per recorded step of every episode, episode
                                after episode; the means over the text, the exterior image and the wrist image of the frozen
                                policy's residual stream after block 12, as its prefix pass computes them at run time
    start, n, ep        (E,)    per episode: its first row, its number of recorded steps, its LeRobot episode_index
    phase               (N,)    int8 phase label per recorded step (-1: unknown)
    frame_step, frame_n         optional, for a dataset whose episodes leave recorded steps out (split episodes): per
                                episode, in `ep` order, frame_n[e] entries: the recorded step of each of its frames. The
                                history walks over recorded steps, so it holds the left-out steps too (the memory saw them
                                happen). Without them frame f is step f, and an episode has exactly n frames
    train_mask          (N,)    optional bool: the recorded steps usable as training targets (a frame filter)
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np

FEATURES = 6144     # 3 x 2048: the [text, exterior, wrist] means of the residual stream

log = logging.getLogger("builtin_memory.sequences")


@dataclasses.dataclass(frozen=True)
class HistorySpec:
    """The feature caches (one index per dataset, in the datasets' order) and the time augmentation of the call history."""

    sources: tuple[str, ...]
    max_calls: int = 96
    gap_min: int = 6
    gap_max: int = 10
    pause_prob: float = 0.1
    pause_max: int = 3
    seed: int = 0


class FeatureIndex:
    """The feature cache of one dataset (module docstring); the features are memory-mapped when first needed."""

    def __init__(self, path: str):
        index = np.load(path, allow_pickle=False)
        self.path = path
        self.features_path = str(index["features"])
        self.start = np.asarray(index["start"], np.int64)
        self.n = np.asarray(index["n"], np.int64)
        self.ep = np.asarray(index["ep"], np.int64)
        self.phase = np.asarray(index["phase"], np.int32)
        self.train_mask = np.asarray(index["train_mask"], bool) if "train_mask" in index else np.ones(len(self.phase), bool)
        self.position = {int(episode): position for position, episode in enumerate(self.ep)}
        self.frame_step = self._frame_steps(index) if "frame_step" in index else None
        self._features = None
        if not len(self.phase) == len(self.train_mask) == int(self.n.sum()):
            raise ValueError(f"{path}: phase, train_mask and the episodes' steps differ in length")

    def _frame_steps(self, index) -> dict[int, np.ndarray]:
        """Per episode id: the recorded step of each of its frames (strictly increasing, inside the episode)."""
        steps, counts = np.asarray(index["frame_step"], np.int64), np.asarray(index["frame_n"], np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        mapping = {int(episode): steps[offset:offset + count]
                   for episode, offset, count in zip(self.ep, offsets, counts, strict=True)}
        for episode, position in self.position.items():
            frame_steps = mapping[episode]
            if len(frame_steps) and not (frame_steps.min() >= 0 and frame_steps.max() < self.n[position]
                                         and (np.diff(frame_steps) > 0).all()):
                raise ValueError(f"{self.path}: episode {episode}: frame_step is not increasing inside the episode")
        return mapping

    @property
    def features(self) -> np.ndarray:
        """(N, 6144) float16, memory-mapped (opened when first needed, in every data-loader process)."""
        if self._features is None:
            self._features = np.load(self.features_path, mmap_mode="r")
            if self._features.shape != (len(self.phase), FEATURES):
                raise ValueError(f"{self.features_path}: shape {self._features.shape}, "
                                 f"expected ({len(self.phase)}, {FEATURES})")
        return self._features

    def rows(self, episodes: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The cache rows of dataset frames (episode_index, frame_index arrays) and the first rows of their episodes."""
        positions = np.array([self.position[int(episode)] for episode in episodes], np.int64)
        steps = self._steps(episodes, frames)
        if not (steps < self.n[positions]).all():
            raise ValueError(f"{self.path}: a frame lies beyond its episode's cached steps")
        return self.start[positions] + steps, self.start[positions]

    def _steps(self, episodes: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """The recorded step of every dataset frame."""
        if self.frame_step is None:
            for episode in np.unique(episodes):    # the dataset's episodes must be as long as the cache's
                if (episodes == episode).sum() != self.n[self.position[int(episode)]]:
                    raise ValueError(f"{self.path}: episode {episode} has another length in the dataset")
            return frames
        steps = np.empty_like(frames)
        for episode in np.unique(episodes):
            selected = episodes == episode
            mapping = self.frame_step[int(episode)]
            if selected.sum() != len(mapping):
                raise ValueError(f"{self.path}: episode {episode}: {selected.sum()} frames, {len(mapping)} mapped")
            steps[selected] = mapping[frames[selected]]
        return steps


def call_frames(t: int, rng: np.random.Generator, spec: HistorySpec) -> list[int]:
    """The frames (within the episode, oldest first) of the calls up to and including frame t (module docstring)."""
    calls, frame = [], t
    while True:
        calls.append(frame)
        if rng.random() < spec.pause_prob:
            calls.extend([frame] * int(rng.integers(1, spec.pause_max + 1)))
        gap = int(rng.integers(spec.gap_min, spec.gap_max + 1))
        if frame - gap < 0 or len(calls) >= spec.max_calls:
            break
        frame -= gap
    return calls[::-1][-spec.max_calls:]


class HistoryDataset:
    """A dataset (any sequence of sample dicts: the concatenated, transformed training datasets) restricted to the frames
    usable as targets, each sample with its call history. keys: per dataset, in order, its (episode_index, frame_index)
    arrays in index order (data.frame_keys)."""

    def __init__(self, dataset, keys: list[tuple[np.ndarray, np.ndarray]], spec: HistorySpec):
        if len(keys) != len(spec.sources):
            raise ValueError(f"{len(keys)} datasets but {len(spec.sources)} feature caches")
        self._dataset = dataset
        self._spec = spec
        self._sources = [FeatureIndex(path) for path in spec.sources]
        self._rng = np.random.default_rng(spec.seed)
        self._source, self._row, self._episode_start, self._valid = self._index(keys)
        if len(self._source) != len(dataset):
            raise ValueError(f"the datasets hold {len(dataset)} frames, their keys {len(self._source)}")
        log.info("%d of %d frames usable as targets; histories of at most %d calls from %s", len(self._valid), len(dataset),
                 spec.max_calls, list(spec.sources))

    def _index(self, keys) -> tuple[np.ndarray, ...]:
        """Per global frame: its dataset, its cache row and its episode's first row; and the frames usable as targets."""
        sources, rows, starts, keep = [], [], [], []
        for number, ((episodes, frames), source) in enumerate(zip(keys, self._sources, strict=True)):
            frame_rows, episode_starts = source.rows(episodes, frames)
            sources.append(np.full(len(episodes), number, np.int64))
            rows.append(frame_rows)
            starts.append(episode_starts)
            keep.append(source.train_mask[frame_rows])
        return (np.concatenate(sources), np.concatenate(rows), np.concatenate(starts),
                np.flatnonzero(np.concatenate(keep)))

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, index):
        frame = int(self._valid[int(index)])
        sample = dict(self._dataset[frame])
        sample.update(self.history(frame, self._rng))
        return sample

    def history(self, frame: int, rng: np.random.Generator) -> dict:
        """The call history of a global frame (module docstring), drawn with rng."""
        source = self._sources[self._source[frame]]
        start = int(self._episode_start[frame])
        rows = start + np.asarray(call_frames(int(self._row[frame]) - start, rng, self._spec), np.int64)
        size, count = self._spec.max_calls, len(rows)
        features = np.zeros((size, FEATURES), np.float16)
        mask = np.zeros(size, bool)
        phase = np.full(size, -1, np.int32)
        features[size - count:] = source.features[rows]
        mask[size - count:] = True
        phase[size - count:] = source.phase[rows]
        return {"memory_feats": features, "memory_mask": mask, "memory_phase": phase}
