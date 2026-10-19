"""Statistics of labelled demonstrations for the rules author's variant hard: what the writer's own signals showed at the
moments the training labels switched to the next history line, relative to the grasp cycle that the gripper signals show
(no rules involved), and the prompt text made of them.
"""
from __future__ import annotations

import re
import statistics
from typing import Iterable, Sequence

import numpy as np

from ...task import TaskSpec
from .author_rules_spec import gripper_thresholds
from ...methods.history.prompt import HistoryPrompt

_STATE_TEXT = {"holding": "still holding the object", "released": "open again after a set-down",
               "closed on nothing": "closed on nothing", "no grasp yet": "not yet grasping"}


def _cycles(rows: np.ndarray, open_width: float, empty_width: float, squeeze_force: float) -> list[dict]:
    """Grasp cycles from the gripper signals alone (no rules): start of the squeeze, end (fingers opened or closed fully)."""
    cycles = []
    current = None
    for step, (width, effort) in enumerate(rows[:, :2]):
        if current is None:
            if empty_width < width < open_width and effort > squeeze_force:
                current = {"grasp": step}
        elif width >= open_width or width <= empty_width:
            current["end"] = step
            current["opened"] = bool(width >= open_width)
            cycles.append(current)
            current = None
    if current is not None:
        cycles.append(current)
    return cycles


def _cycle_at(cycles: list[dict], step: int) -> dict | None:
    """The last grasp cycle that began at or before the step."""
    for cycle in reversed(cycles):
        if cycle["grasp"] <= step:
            return cycle
    return None


# ---------------------------------------------------------------------------------------------------------------- statistics
def demo_event_summary(episodes: Iterable[dict], task: TaskSpec, window: Sequence[int] = (-16, -8, -4, 0, 4, 8),
                       **gripper_overrides) -> list[dict]:
    """Statistics of the writer's signals at the label switches of labelled episodes ({"rows", "reference"}, as for
    validate.py): per history line k, at the first step where the label count reaches k, relative to the grasp cycle that
    the gripper signals show (no rules involved). If the fingers still hold: the fingertip rise above the grasp point, the
    highest rise of that hold, the upward speed, the steps since the grasp. If they opened after a set-down: the release
    height above the grasp point, the fingertip rise above and the distance from the release point, the steps since the
    release. Plus the median course of the rise around the switch (`window`, control steps). Lengths in cm."""
    gripper = gripper_thresholds(task, **gripper_overrides)
    cap = (task.history or {}).get("max_lines")
    if cap is None:
        cap = len(task.skills)
    per_line: dict[int, dict[str, list]] = {}
    for episode in episodes:
        _collect_switches(episode, cap, gripper, window, per_line)
    return _summaries(per_line, task, window)


def _collect_switches(episode: dict, cap: int, gripper: dict, window: Sequence[int], per_line: dict) -> None:
    """Add one episode's values at each label switch to the per-line lists."""
    rows = np.asarray(episode["rows"], float)
    reference = np.minimum(np.asarray(episode["reference"], int), cap)
    heights = 100 * rows[:, 4]
    positions = 100 * rows[:, 2:5]
    cycles = _cycles(rows, gripper["open_width"], gripper["empty_width"], gripper["squeeze_force"])
    for line in range(1, int(reference.max(initial=0)) + 1):
        step = int(np.flatnonzero(reference >= line)[0])
        collected = per_line.setdefault(line, {"n": [], "state": []})
        collected["n"].append(1)
        around = [step + offset for offset in window]
        state, values = _switch_statistics(step, cycles, heights, positions, around)
        collected["state"].append(state)
        for key, value in values:
            collected.setdefault(key, []).append(value)


def _switch_statistics(step: int, cycles: list[dict], heights, positions, around: list[int]) -> tuple[str, list[tuple]]:
    """The gripper's state at a label switch and the statistics of that state (none for no grasp / closed on nothing)."""
    cycle = _cycle_at(cycles, step)
    if cycle is None:
        return "no grasp yet", []
    grasp = cycle["grasp"]
    if cycle.get("end") is None or cycle["end"] > step:
        end = cycle.get("end", len(heights) - 1)
        return "holding", _holding_statistics(heights, grasp, end, step, around)
    if cycle["opened"]:
        return "released", _released_statistics(heights, positions, grasp, cycle["end"], step, around)
    return "closed on nothing", []


def _holding_statistics(heights, grasp: int, end: int, step: int, around: list[int]) -> list[tuple]:
    """Still holding at the switch: the rise above the grasp point, the highest rise of the hold, the upward speed, the
    steps since the grasp, and the course of the rise around the switch."""
    speed = heights[step] - heights[step - 1] if step > 0 else 0.0
    return [("rise", float(heights[step] - heights[grasp])),
            ("max_rise", float(heights[grasp:end + 1].max() - heights[grasp])),
            ("speed", float(speed)),
            ("steps_since_grasp", float(step - grasp)),
            ("rise_around", _course(heights, grasp, around))]


def _released_statistics(heights, positions, grasp: int, release: int, step: int, around: list[int]) -> list[tuple]:
    """Opened after the grasp: the release height above the grasp point, the rise above and the distance from the release
    point, the steps since the release, and the course of the rise around the switch."""
    return [("release_height", float(heights[release] - heights[grasp])),
            ("up", float(heights[step] - heights[release])),
            ("away", float(np.linalg.norm(positions[step] - positions[release]))),
            ("steps_since_release", float(step - release)),
            ("up_around", _course(heights, release, around))]


def _course(heights, origin: int, around: list[int]) -> list:
    """The height above the origin step at each step of `around` (clamped to the episode)."""
    last = len(heights) - 1
    return [heights[min(max(step, 0), last)] - heights[origin] for step in around]


def _summaries(per_line: dict, task: TaskSpec, window: Sequence[int]) -> list[dict]:
    """The collected lists -> one summary per history line: the state counts, min / median / max of each statistic
    (2 decimals) and the median course around the switch (1 decimal)."""
    plan = list(task.skills)
    history = HistoryPrompt(task)
    summaries = []
    for line in sorted(per_line):
        collected = per_line[line]
        phrase = history.event(plan[line - 1]) if line <= len(plan) else None
        summary = {"line": line, "phrase": phrase, "n": len(collected["n"]), "states": _state_counts(collected["state"]),
                   "window": list(window)}
        for key, values in collected.items():
            if key in ("n", "state"):
                continue
            if key.endswith("_around"):
                summary[key] = [round(float(statistics.median(column)), 1) for column in zip(*values)]
            else:
                summary[key] = [round(float(statistic(values)), 2) for statistic in (min, statistics.median, max)]
        summaries.append(summary)
    return summaries


def _state_counts(states: list[str]) -> dict[str, int]:
    """How often each state occurred, in the order of first occurrence."""
    counts: dict[str, int] = {}
    for state in states:
        counts[state] = counts.get(state, 0) + 1
    return counts


# ---------------------------------------------------------------------------------------------------------------- prompt text
def _number_text(value, decimals: int = 1) -> str:
    """A number with the given decimals; a negative zero loses its sign (no "-0.0")."""
    text = f"{value:.{decimals}f}"
    if re.fullmatch(r"-0\.?0*", text):
        return text[1:]
    return text


def _min_median_max(values, unit: str, decimals: int = 1) -> str:
    """Three numbers as "min / median / max" with the unit appended."""
    return " / ".join(_number_text(value, decimals) for value in values) + unit


def demo_summary_text(summary: list[dict]) -> str:
    """The prompt text of demo_event_summary (variant hard)."""
    lines = []
    for line in summary:
        lines.append(_switch_text(line))
        if "rise" in line:
            lines += _holding_text(line)
        if "up" in line:
            lines += _released_text(line)
    return "\n".join(lines)


def _switch_text(line: dict) -> str:
    """The headline of one history line: how many demonstrations, and the gripper's state at the switch."""
    states = ", ".join(f"{_STATE_TEXT.get(state, state)} in {count} of {line['n']}" for state, count in line["states"].items())
    return (f'History line {line["line"]}, "{line["phrase"]}" ({line["n"]} demonstrations): at the switch the gripper was '
            f'{states}.')


def _holding_text(line: dict) -> list[str]:
    """The statistics of the switches at which the fingers still held the object."""
    course = ", ".join(f"{offset:+d} steps {_number_text(value)} cm" for offset, value in zip(line["window"], line["rise_around"]))
    return [f"  fingertip rise above the grasp point at the switch: {_min_median_max(line['rise'], ' cm')}",
            f"  highest rise during that hold: {_min_median_max(line['max_rise'], ' cm')}",
            f"  upward speed at the switch: {_min_median_max(line['speed'], ' cm per step', 2)}",
            f"  control steps since the grasp began: {_min_median_max(line['steps_since_grasp'], '', 0)}",
            "  rise above the grasp point around the switch (median): " + course]


def _released_text(line: dict) -> list[str]:
    """The statistics of the switches at which the fingers had opened after a set-down."""
    course = ", ".join(f"{offset:+d} steps {_number_text(value)} cm" for offset, value in zip(line["window"], line["up_around"]))
    return [f"  release height above the grasp point: {_min_median_max(line['release_height'], ' cm')}",
            f"  fingertip rise above the release point at the switch: {_min_median_max(line['up'], ' cm')}",
            f"  straight-line distance from the release point at the switch: {_min_median_max(line['away'], ' cm')}",
            f"  control steps since the release: {_min_median_max(line['steps_since_release'], '', 0)}",
            "  rise above the release point around the switch (median): " + course]


def pick_evenly(files: Sequence[str], count: int) -> list[str]:
    """`count` files spread evenly over a sorted list (first and last included)."""
    files = sorted(files)
    if count >= len(files):
        return list(files)
    return [files[int(round(position))] for position in np.linspace(0, len(files) - 1, count)]
