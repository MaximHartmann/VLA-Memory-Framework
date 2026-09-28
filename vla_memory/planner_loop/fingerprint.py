"""Retrieval keys ("fingerprints") of observations: a fixed-length unit vector per observation such that similar
situations have a high cosine similarity.

ColourBlobFingerprint is what the reference task uses: a coarse colour thumbnail of every camera image (scene
layout) plus, for each object colour, the area, centroid, extent and edge contact of that colour's pixels (where
the objects are, how large they appear, whether an object touches the wrist view's edge). It is cheap (about 2 ms,
no GPU) and was enough for the two-cube task. For scenes without distinctive object colours, replace it with a
learned image embedding (DINOv2 / SigLIP) by implementing Fingerprint.__call__.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_CH = {"r": 0, "g": 1, "b": 2}


class Fingerprint:
    def __call__(self, images: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


def thumbnail(img, th=12, tw=16) -> np.ndarray:
    im = np.asarray(img)[..., :3]; h, w = im.shape[:2]
    ys = (np.arange(th) * h // th).clip(0, h - 1); xs = (np.arange(tw) * w // tw).clip(0, w - 1)
    return (im[np.ix_(ys, xs)].astype(np.float32) / 255.0).ravel()


def colour_blob(img, rules) -> np.ndarray:
    """Area (log-scaled), centroid, extent and edge contact of the pixels satisfying all channel-difference rules."""
    im = np.asarray(img)[..., :3].astype(np.int16)
    m = np.ones(im.shape[:2], dtype=bool)
    for a, b, d in rules:
        m &= (im[..., _CH[a]] - im[..., _CH[b]]) > d
    h, w = m.shape; n = int(m.sum())
    if n < 4:
        return np.zeros(7, np.float32)
    ys, xs = np.nonzero(m)
    return np.array([np.log1p(n) / np.log1p(h * w), xs.mean() / w, ys.mean() / h, (xs.max() - xs.min() + 1) / w,
                     (ys.max() - ys.min() + 1) / h, float(xs.min() == 0), float(ys.min() == 0)], np.float32)


class ColourBlobFingerprint(Fingerprint):
    def __init__(self, colours: dict[str, list], thumb=(12, 16), w_thumb=1.0, w_blob=6.0):
        self.colours = {k: [tuple(r) for r in v] for k, v in colours.items()}
        self.thumb = tuple(thumb); self.w_thumb = w_thumb; self.w_blob = w_blob

    def __call__(self, images):
        parts = []
        for img in images:
            parts.append(self.w_thumb * thumbnail(img, *self.thumb))
            for rules in self.colours.values():
                parts.append(self.w_blob * colour_blob(img, rules))
        v = np.concatenate(parts).astype(np.float32); n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v


class ThumbnailFingerprint(Fingerprint):
    """Layout-only key (no object colours): usable for any scene, less discriminative."""

    def __init__(self, thumb=(12, 16)):
        self.thumb = tuple(thumb)

    def __call__(self, images):
        v = np.concatenate([thumbnail(img, *self.thumb) for img in images]).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v


def fingerprint_from_task(task) -> Fingerprint:
    """Build the fingerprint described in the task file (section `fingerprint`)."""
    f = dict(task.fingerprint or {})
    if "colours" in f:
        return ColourBlobFingerprint(f["colours"], thumb=f.get("thumbnail", (12, 16)),
                                     w_thumb=f.get("w_thumb", 1.0), w_blob=f.get("w_blob", 6.0))
    return ThumbnailFingerprint(thumb=f.get("thumbnail", (12, 16)))
