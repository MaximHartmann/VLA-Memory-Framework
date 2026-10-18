"""The retrieval rule of the memory: which stored moment one policy call is shown.

neighbour(store, key, skill, width, exclude_episode) applies hard filters and then takes the most similar admissible entry
(k = 1, cosine similarity of the keys, key.py):

    skill              the entry's skill is the call's skill
    gripper state      open and closed are kept apart: the call's state from its MEASURED finger width, the entry's from its
                       own measured width (gripper.py)
    closed on nothing  a gripper closed on nothing is shown only the skill's release entries (moments that open the
                       gripper); if the skill has none, all its closed entries
    own episode        never an entry of exclude_episode (the store builder labels an episode without its own moments)

With the neighbour comes the rule that admitted it (RULES): open, closed, release or closed_fallback. The training labels
were retrieved with the same rule, so the policy was trained on exactly such neighbours.
"""
from __future__ import annotations

import numpy as np

from .gripper import gripper_state
from .store import MomentStore

RULES = ("open", "closed", "release", "closed_fallback")


def candidates(store: MomentStore, skill: str, width: float, exclude_episode: str | None = None) -> tuple[np.ndarray, str]:
    """The entries a call may be shown (a bool mask over the store) and the rule that admitted them (one of RULES)."""
    state = gripper_state(width)
    mask = store.skill == skill
    if exclude_episode is not None:
        mask &= store.episode != exclude_episode
    if state == "open":
        return mask & ~store.closed, "open"
    mask &= store.closed
    if state == "closed_nothing" and (mask & store.release).any():
        return mask & store.release, "release"
    if state == "closed_nothing":
        return mask, "closed_fallback"
    return mask, "closed"


def neighbour(store: MomentStore, key, skill: str, width: float,
              exclude_episode: str | None = None) -> tuple[int, float, str]:
    """(entry index or -1, cosine similarity or NaN, rule) of the most similar admissible entry for one call."""
    mask, rule = candidates(store, skill, width, exclude_episode)
    if not mask.any():
        return -1, float("nan"), rule
    key = np.asarray(key, np.float32)
    norm = float(np.linalg.norm(key))
    query = key / norm if norm > 1e-8 else key
    similarity = store.keys @ query
    similarity[~mask] = -np.inf
    best = int(similarity.argmax())
    return best, float(similarity[best]), rule
