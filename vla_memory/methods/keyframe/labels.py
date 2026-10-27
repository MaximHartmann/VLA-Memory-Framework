"""The training labels of the keyframe policy: which wrist picture every frame of a demonstration gets as its keyframe.

The events of an episode are the moments at which its history grows (picked up red, placed red, picked up blue in the
reference task): boundary b_k is the first frame whose history holds k lines. A frame t after b_e (e = the number of
boundaries at or before t) gets the wrist picture of frame

    s = b_e + crc32("<split>:<episode>:<t>") % (min(16, t - b_e) + 1)

a fixed draw from the frames b_e .. min(b_e + 16, t): never later than the frame itself, as at run time, where the keyframe
is taken at the call at which the event is signalled, up to a call after it. A frame before b_1 gets no keyframe (an all-zero
picture, which the policy's input transform leaves masked out).
"""
from __future__ import annotations

import zlib
from typing import Sequence

WINDOW = 16          # the keyframe is drawn from the boundary and the 16 frames after it (17 frames)


def history_counts(prompts: Sequence[str], instruction: str, history, plan: Sequence[str] | None = None) -> list[int]:
    """The number of history lines of every frame's prompt; history: a HistoryPrompt of the task. Every prompt must be the
    instruction followed by a history of the plan's skills in order (default: the task's skills)."""
    skills = tuple(plan) if plan else history.task.skills
    rendered = {history.render(instruction, skills[:k]): k for k in range(len(skills) + 1)}
    counts = []
    for prompt in prompts:
        if prompt not in rendered:
            raise ValueError(f"{prompt!r} is not {instruction!r} with a history of the task's skills")
        counts.append(rendered[prompt])
    return counts


def boundaries(counts: Sequence[int], events: int) -> list[int]:
    """b_1 .. b_events: the first frame with each history count. The count must start at 0, rise by one at a time and reach
    `events`."""
    steps = {later - earlier for earlier, later in zip(counts, counts[1:])}
    if not counts or counts[0] != 0 or counts[-1] != events or not steps <= {0, 1}:
        raise ValueError(f"history counts from {counts[:1]} to {counts[-1:]} with steps {sorted(steps)}: expected 0 to "
                         f"{events}")
    return [list(counts).index(k) for k in range(1, events + 1)]


def source_frame(split: str, episode: int, frame: int, boundary: int) -> int:
    """The frame whose wrist picture is the keyframe of `frame`, for the event at `boundary` (module docstring)."""
    span = min(WINDOW, frame - boundary) + 1
    return boundary + zlib.crc32(f"{split}:{episode}:{frame}".encode()) % span


def label_episode(split: str, episode: int, frames: int, edges: Sequence[int]) -> list[tuple[int, int, int]]:
    """(event, boundary, source frame) of every frame of one episode; (0, -1, -1) before the first event."""
    rows = []
    for frame in range(frames):
        event = sum(edge <= frame for edge in edges)
        if event == 0:
            rows.append((0, -1, -1))
        else:
            boundary = edges[event - 1]
            rows.append((event, boundary, source_frame(split, episode, frame, boundary)))
    return rows
