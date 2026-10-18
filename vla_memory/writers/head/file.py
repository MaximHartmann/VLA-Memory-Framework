"""The event head's file format: one directory per trained head, written by the fit and read by the policy server.

    head.npz      w (d,) float32: the weights; b (1,) float32: the bias; mean, std (d,) float32: the feature statistics of
                  the training rows; threshold (float64); N (int64): the consecutive "done" calls that write a line; layer
                  (int64): the block after which the policy's residual stream is read, counted from 0; pool (str): the
                  tokens it is averaged over ("text": the prompt's valid text tokens); l2 (float64): the penalty of the fit
    head.json     a free description: the feature, the training episodes, the cross-validation table, when it was saved
    head.sha256   the SHA-256 of head.npz and head.json, one "<digest>  <name>" line each, checked when the head is loaded

The score of a feature vector x is s = ((x - mean) / std) @ w + b in float32; the call says "done" when s >= threshold.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import numpy as np

WEIGHTS = "head.npz"
DESCRIPTION = "head.json"
DIGESTS = "head.sha256"


@dataclasses.dataclass
class EventHead:
    """A trained event head: a logistic regression on z-scored features of one layer of the policy."""

    w: np.ndarray                 # (d,) float32
    b: np.ndarray                 # (1,) float32
    mean: np.ndarray              # (d,) float32
    std: np.ndarray               # (d,) float32
    threshold: float
    consecutive: int = 2          # saved as N
    layer: int = 12
    pool: str = "text"
    l2: float = 0.0
    meta: dict = dataclasses.field(default_factory=dict)

    def scores(self, features) -> np.ndarray:
        """(n,) float32: the scores of n feature vectors (n, d)."""
        z = ((np.asarray(features, np.float32) - self.mean) / self.std).astype(np.float32)
        return z @ self.w + self.b

    def score(self, feature) -> float:
        """The score of one feature vector (d,)."""
        return float(self.scores(np.asarray(feature)[None, :])[0])

    def verdict(self, feature) -> dict:
        """The verdict of one call: its score, whether it says done, and the threshold."""
        score = self.score(feature)
        return {"score": score, "done": bool(score >= self.threshold), "threshold": float(self.threshold)}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_digests(directory: pathlib.Path) -> dict:
    """name -> digest from head.sha256."""
    digests = {}
    for line in (directory / DIGESTS).read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            digests[name] = digest
    return digests


def check_digests(directory: pathlib.Path) -> dict:
    """The files listed in head.sha256 must still have their digests; head.npz must be listed. Returns the digests."""
    digests = read_digests(directory)
    if WEIGHTS not in digests:
        raise ValueError(f"{directory / DIGESTS} does not list {WEIGHTS}")
    for name, digest in digests.items():
        actual = sha256(directory / name)
        if actual != digest:
            raise ValueError(f"{directory / name}: SHA-256 {actual}, but {DIGESTS} says {digest}: the file changed")
    return digests


def load_head(directory, verify: bool = True) -> EventHead:
    """The head saved in `directory`; with verify, its files must match head.sha256."""
    directory = pathlib.Path(directory)
    digests = check_digests(directory) if verify else {}
    arrays = np.load(directory / WEIGHTS)
    meta = {}
    if (directory / DESCRIPTION).exists():
        meta = json.loads((directory / DESCRIPTION).read_text())
    meta["sha256"] = digests
    return EventHead(w=arrays["w"].astype(np.float32), b=arrays["b"].astype(np.float32).reshape(1),
                     mean=arrays["mean"].astype(np.float32), std=arrays["std"].astype(np.float32),
                     threshold=float(arrays["threshold"]), consecutive=int(arrays["N"]), layer=int(arrays["layer"]),
                     pool=str(arrays["pool"]), l2=float(arrays["l2"]) if "l2" in arrays.files else 0.0, meta=meta)


def save_head(directory, head: EventHead, description: dict) -> dict:
    """Write head.npz, head.json (the description) and head.sha256 into a new or empty directory; returns the digests."""
    directory = pathlib.Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"{directory} is not empty; refusing to overwrite a head")
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / WEIGHTS, w=head.w.astype(np.float32), b=head.b.astype(np.float32).reshape(1),
             mean=head.mean.astype(np.float32), std=head.std.astype(np.float32), threshold=np.float64(head.threshold),
             N=np.int64(head.consecutive), l2=np.float64(head.l2), layer=np.int64(head.layer), pool=np.array(head.pool))
    (directory / DESCRIPTION).write_text(json.dumps(description, indent=1) + "\n")
    digests = {name: sha256(directory / name) for name in (WEIGHTS, DESCRIPTION)}
    (directory / DIGESTS).write_text("".join(f"{digest}  {name}\n" for name, digest in digests.items()))
    return digests
