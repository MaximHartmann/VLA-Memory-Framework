"""The training rows of the event head: the (frame, history) pairs of a recorded episode that the head learns from.

A row is one frame of an episode shown with a prompt that holds k history lines; the head answers "is the current skill,
skill k + 1, complete?". The reference h says how many skills are complete at every frame, as the history at a policy call
there would show it (the rules writer's lines, a simulator's tracker, a person's annotation):

    label 0  h == k   the frame shows skill k + 1 in progress, with the correct history (a negative)
    label 1  h > k    the skill is complete but its line is not written yet: what the head must detect (a positive)

Rows are taken at every `grid`-th frame: a negative with the correct history wherever h <= k_max, and a positive with the
stale history at the frames of the first `window` frames after each of the first k_max + 1 boundaries. Prompts with more
than k_max lines are not used: with max_lines lines the current skill is the last one, whose completion writes no line.
"""
from __future__ import annotations

import numpy as np

GRID = 4          # a row at every 4th frame
WINDOW = 24       # positives: the first 24 frames after a boundary


def boundaries(reference, count: int) -> list:
    """b_1 .. b_count: the first frame at which the reference has reached k (None if it never does)."""
    h = np.asarray(reference)
    out = []
    for k in range(1, count + 1):
        hits = np.flatnonzero(h >= k)
        out.append(int(hits[0]) if len(hits) else None)
    return out


def completed(edges: list, frames: int) -> np.ndarray:
    """(frames,) int: how many of the boundaries lie at or before each frame (a history only grows)."""
    h = np.zeros(frames, int)
    for edge in edges:
        if edge is not None:
            h[edge:] += 1
    return h


def probe_rows(reference, k_max: int = 2, grid: int = GRID, window: int = WINDOW) -> list[tuple[int, int, int]]:
    """The rows of one episode as (frame, k, label), sorted by frame and k."""
    edges = boundaries(reference, k_max + 1)
    h = completed(edges, len(reference))
    rows = {}
    for frame in range(0, len(h), grid):
        if h[frame] <= k_max:
            rows[(frame, int(h[frame]))] = 0
        for k, edge in enumerate(edges):
            if edge is not None and edge <= frame < edge + window:
                rows[(frame, k)] = int(h[frame] > k)
    return [(frame, k, label) for (frame, k), label in sorted(rows.items())]


def reference_from_lines(lines) -> np.ndarray:
    """The reference of an episode from a writer's lines after each control step (lines[t]: the lines written after steps
    0..t, as RuleMonitor.label_rows returns them): a line written after step t is shown from frame t + 1 on."""
    lines = np.asarray(lines, int)
    return np.concatenate([[0], lines[:-1]]).astype(int)
