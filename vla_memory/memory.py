"""Long-term memory interface. A memory store answers one question at every planner decision: 'what do we know
from the past that resembles the current situation?' and renders the answer as chat content for the planner model.

Implementations: planner_loop.ExemplarStore (labelled frames of past episodes, retrieved by visual similarity).
Any other store (text rules keyed by skill, preference profiles, outcome-weighted entries) implements the same
four methods.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class MemoryStore:
    """The four methods every long-term memory store implements (module docstring)."""

    def __len__(self) -> int:
        """How many entries the store holds (0: nothing stored yet)."""
        raise NotImplementedError

    def retrieve(self, images: Sequence[np.ndarray], k: int = 4, exclude_episode: Any = None) -> list[tuple[int, float]]:
        """Return [(entry index, similarity)] of the k entries most similar to the current images."""
        raise NotImplementedError

    def fewshot_content(self, hits: list[tuple[int, float]], skill_index: int, skill: str, encode_png) -> list[dict]:
        """Render retrieved entries as OpenAI chat content items (text + image_url), labelled for the current skill."""
        raise NotImplementedError

    def describe_hit(self, index: int) -> dict:
        """Compact metadata of one entry for the decision log."""
        raise NotImplementedError
