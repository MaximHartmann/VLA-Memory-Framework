"""The retrieval key of the memory: the colour fingerprint of ONE raw wrist frame, scaled to length 1.

The store's keys were computed on the raw wrist frame of every stored moment, the frame the environment sends with every
policy call as memory/wrist_raw (FRAME_SHAPE on the reference bench), with the task's fingerprint
(vla_memory.fingerprint.fingerprint_from_task): a 12 x 16 colour thumbnail plus the area, centroid, extent and edge contact
of the pixels of each task colour; 590 values for the red/blue task. A query key must be computed the same way: with the
same task file, on the same kind of frame.

    wrist_key = WristKey(task)
    wrist_key.key_of(frame)        one frame -> (d,) float32 of length 1 (the proxy, at every policy call)
    wrist_key.keys_of(frames)      several frames -> (n, d) float32, rows of length 1 (the store builder)
"""
from __future__ import annotations

import numpy as np

from ...fingerprint import fingerprint_from_task

KEY_NAME = "fingerprint"        # the key's name in a store's meta.json; the keys are in keys_fingerprint.npy
FRAME_SHAPE = (240, 320, 3)     # the raw wrist frame of the reference bench, on which the reference store's keys were computed


def unit_rows(matrix) -> np.ndarray:
    """Every row scaled to length 1, as float32 (an all-zero row stays as it is)."""
    matrix = np.asarray(matrix, np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 1e-8, norms, 1.0)


class WristKey:
    """The key of raw wrist frames for one task (the colour rules of its fingerprint section)."""

    def __init__(self, task):
        self.fingerprint = fingerprint_from_task(task)

    def keys_of(self, frames) -> np.ndarray:
        """(n, d) float32: the key of every frame, rows of length 1."""
        rows = [self.fingerprint([np.asarray(frame)]) for frame in frames]
        return unit_rows(np.stack(rows))

    def key_of(self, frame) -> np.ndarray:
        """(d,) float32: the key of one frame, of length 1."""
        return self.keys_of([frame])[0]
