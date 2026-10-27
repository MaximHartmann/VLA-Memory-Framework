"""The pooling every state tap shares, and the interface.

A transformer layer returns one vector per token, hidden (B, S, W). A tap reduces it to one vector per row, (B, W): the mean
over the valid tokens of a span [start, stop), or the last valid one. pool_weights gives that as weights (B, S), so every
backend pools with the same einsum("bs,bsw->bw").
"""
from __future__ import annotations

import numpy as np


def pool_weights(valid, batch: int, length: int, start: int = 0, stop: int | None = None, kind: str = "mean") -> np.ndarray:
    """(B, S) float32 pooling weights: the mean over the valid tokens of [start, stop), or ("last") the last valid one.
    valid: (B, S) bool, False for padding; None: every token is valid. A row without a valid token gets zeros."""
    if kind not in ("mean", "last"):
        raise ValueError(f"pool kind {kind!r}: 'mean' or 'last'")
    mask = np.ones((batch, length), bool) if valid is None else np.asarray(valid, bool).reshape(batch, length)
    span = np.zeros(length, bool)
    span[start:stop] = True
    weights = (mask & span).astype(np.float32)
    if kind == "last":
        weights *= np.cumsum(weights, axis=1) == weights.sum(axis=1, keepdims=True)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1.0)


class StateTap:
    """Reads one layer's state during the model's own forward pass, pooled to one vector per row (pool_weights).

    Set `valid` before a call whose tokens include padding; the backend fills `value` while the model runs; read() gives it
    as a (B, W) float32 numpy array."""

    def __init__(self, start: int = 0, stop: int | None = None, kind: str = "mean"):
        self.start, self.stop, self.kind = start, stop, kind
        self.valid = None
        self.value = None

    def weights(self, batch: int, length: int) -> np.ndarray:
        return pool_weights(self.valid, batch, length, self.start, self.stop, self.kind)

    def read(self) -> np.ndarray:
        if self.value is None:
            raise RuntimeError("the tap has captured nothing: did the forward pass reach the tapped layer?")
        return np.asarray(self.value, np.float32)

    def close(self) -> None:
        """Detach the tap from the model (a hook-based backend removes its hook)."""
