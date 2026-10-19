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

_CHANNEL_INDEX = {"r": 0, "g": 1, "b": 2}


class Fingerprint:
    """The interface: the images of one observation (exterior, then wrist) -> one unit vector."""

    def __call__(self, images: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


def thumbnail(image, rows=12, columns=16) -> np.ndarray:
    """A rows x columns colour thumbnail of the image (pixels sampled on a grid), as one flat vector of values in 0..1."""
    pixels = np.asarray(image)[..., :3]
    height, width = pixels.shape[:2]
    row_indices = (np.arange(rows) * height // rows).clip(0, height - 1)
    column_indices = (np.arange(columns) * width // columns).clip(0, width - 1)
    return (pixels[np.ix_(row_indices, column_indices)].astype(np.float32) / 255.0).ravel()


def colour_mask(image, rules) -> np.ndarray:
    """Pixels satisfying all channel-difference rules [channel_a, channel_b, min_difference] of one colour."""
    pixels = np.asarray(image)[..., :3].astype(np.int16)
    mask = np.ones(pixels.shape[:2], dtype=bool)
    for channel_a, channel_b, min_difference in rules:
        mask &= (pixels[..., _CHANNEL_INDEX[channel_a]] - pixels[..., _CHANNEL_INDEX[channel_b]]) > min_difference
    return mask


def colour_blob(image, rules) -> np.ndarray:
    """Area (log-scaled), centroid, extent and edge contact of the pixels satisfying all channel-difference rules; all
    zeros for fewer than 4 pixels (no blob)."""
    mask = colour_mask(image, rules)
    height, width = mask.shape
    count = int(mask.sum())
    if count < 4:
        return np.zeros(7, np.float32)
    row_indices, column_indices = np.nonzero(mask)
    return np.array([np.log1p(count) / np.log1p(height * width), column_indices.mean() / width, row_indices.mean() / height,
                     (column_indices.max() - column_indices.min() + 1) / width, (row_indices.max() - row_indices.min() + 1) / height,
                     float(column_indices.min() == 0), float(row_indices.min() == 0)], np.float32)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """The vector scaled to length 1 (an all-zero vector stays as it is)."""
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        return vector / norm
    return vector


class ColourBlobFingerprint(Fingerprint):
    """The thumbnail of every image and the blob of every object colour in it, weighted and joined into one unit vector.
    colours: colour name -> channel-difference rules (the task file's fingerprint.colours)."""

    def __init__(self, colours: dict[str, list], thumbnail_shape=(12, 16), thumbnail_weight=1.0, blob_weight=6.0):
        self.colours = {name: [tuple(rule) for rule in rules] for name, rules in colours.items()}
        self.thumbnail_shape = tuple(thumbnail_shape)
        self.thumbnail_weight = thumbnail_weight
        self.blob_weight = blob_weight

    def __call__(self, images):
        parts = []
        for image in images:
            parts.append(self.thumbnail_weight * thumbnail(image, *self.thumbnail_shape))
            for rules in self.colours.values():
                parts.append(self.blob_weight * colour_blob(image, rules))
        return _unit_vector(np.concatenate(parts).astype(np.float32))


class ThumbnailFingerprint(Fingerprint):
    """Layout-only key (no object colours): usable for any scene, less discriminative."""

    def __init__(self, thumbnail_shape=(12, 16)):
        self.thumbnail_shape = tuple(thumbnail_shape)

    def __call__(self, images):
        return _unit_vector(np.concatenate([thumbnail(image, *self.thumbnail_shape) for image in images]).astype(np.float32))


def fingerprint_from_task(task) -> Fingerprint:
    """Build the fingerprint described in the task file (section `fingerprint`: colours, thumbnail, w_thumb, w_blob); the
    layout-only key when the task declares no colours."""
    spec = dict(task.fingerprint or {})
    if "colours" in spec:
        return ColourBlobFingerprint(spec["colours"], thumbnail_shape=spec.get("thumbnail", (12, 16)),
                                     thumbnail_weight=spec.get("w_thumb", 1.0), blob_weight=spec.get("w_blob", 6.0))
    return ThumbnailFingerprint(thumbnail_shape=spec.get("thumbnail", (12, 16)))
