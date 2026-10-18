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
    """A th x tw colour thumbnail of the image (pixels sampled on a grid), as one flat vector of values in 0..1."""
    image = np.asarray(img)[..., :3]
    height, width = image.shape[:2]
    rows = (np.arange(th) * height // th).clip(0, height - 1)
    columns = (np.arange(tw) * width // tw).clip(0, width - 1)
    return (image[np.ix_(rows, columns)].astype(np.float32) / 255.0).ravel()


def colour_mask(img, rules) -> np.ndarray:
    """Pixels satisfying all channel-difference rules [channel_a, channel_b, min_difference] of one colour."""
    im = np.asarray(img)[..., :3].astype(np.int16)
    mask = np.ones(im.shape[:2], dtype=bool)
    for a, b, d in rules:
        mask &= (im[..., _CH[a]] - im[..., _CH[b]]) > d
    return mask


def colour_blob(img, rules) -> np.ndarray:
    """Area (log-scaled), centroid, extent and edge contact of the pixels satisfying all channel-difference rules."""
    mask = colour_mask(img, rules)
    height, width = mask.shape
    count = int(mask.sum())
    if count < 4:
        return np.zeros(7, np.float32)
    ys, xs = np.nonzero(mask)
    return np.array([np.log1p(count) / np.log1p(height * width), xs.mean() / width, ys.mean() / height,
                     (xs.max() - xs.min() + 1) / width, (ys.max() - ys.min() + 1) / height,
                     float(xs.min() == 0), float(ys.min() == 0)], np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    """The vector scaled to length 1 (an all-zero vector stays as it is)."""
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        return vector / norm
    return vector


class ColourBlobFingerprint(Fingerprint):
    def __init__(self, colours: dict[str, list], thumb=(12, 16), w_thumb=1.0, w_blob=6.0):
        self.colours = {k: [tuple(r) for r in v] for k, v in colours.items()}
        self.thumb = tuple(thumb)
        self.w_thumb = w_thumb
        self.w_blob = w_blob

    def __call__(self, images):
        parts = []
        for img in images:
            parts.append(self.w_thumb * thumbnail(img, *self.thumb))
            for rules in self.colours.values():
                parts.append(self.w_blob * colour_blob(img, rules))
        return _unit(np.concatenate(parts).astype(np.float32))


class ThumbnailFingerprint(Fingerprint):
    """Layout-only key (no object colours): usable for any scene, less discriminative."""

    def __init__(self, thumb=(12, 16)):
        self.thumb = tuple(thumb)

    def __call__(self, images):
        return _unit(np.concatenate([thumbnail(img, *self.thumb) for img in images]).astype(np.float32))


def fingerprint_from_task(task) -> Fingerprint:
    """Build the fingerprint described in the task file (section `fingerprint`)."""
    spec = dict(task.fingerprint or {})
    if "colours" in spec:
        return ColourBlobFingerprint(spec["colours"], thumb=spec.get("thumbnail", (12, 16)),
                                     w_thumb=spec.get("w_thumb", 1.0), w_blob=spec.get("w_blob", 6.0))
    return ThumbnailFingerprint(thumb=spec.get("thumbnail", (12, 16)))
