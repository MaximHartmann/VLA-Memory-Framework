"""The store's wrist pictures, decoded from their PNG files when first needed and kept in a bounded cache.

An entry's picture is its raw wrist frame (240 x 320 x 3 uint8 in the reference store). The reference store holds about
11,000 of them (334 MB as PNG, 2.5 GB decoded), so they are not decoded at the start: Pictures decodes a file at its first
use (about 1 ms) and keeps the `capacity` most recently used pictures (230 KB each: 1,024 pictures take 236 MB).
Capacity 0 decodes at every use.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from PIL import Image


def decode_png(path) -> np.ndarray:
    """One PNG file as a uint8 RGB array."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


class Pictures:
    """The pictures of a store by entry index: decoded at first use, the `capacity` most recently used ones kept."""

    def __init__(self, paths, capacity: int):
        self.paths = list(paths)
        self.capacity = int(capacity)
        self.decoded = 0                        # files decoded so far (cache misses)
        self._cache: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        index = int(index)
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        picture = decode_png(self.paths[index])
        self.decoded += 1
        if self.capacity > 0:
            self._remember(index, picture)
        return picture

    def _remember(self, index: int, picture: np.ndarray) -> None:
        """Keep the picture; drop the least recently used ones beyond the capacity."""
        self._cache[index] = picture
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    @property
    def cached(self) -> int:
        """How many pictures the cache holds now."""
        return len(self._cache)

    def cache_bytes(self) -> int:
        """The bytes the cached pictures take."""
        return sum(picture.nbytes for picture in self._cache.values())
