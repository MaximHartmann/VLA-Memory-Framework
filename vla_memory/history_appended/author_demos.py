"""Statistics of labelled demonstrations for the rules author's variant hard: what the writer's own signals showed at the
moments the training labels switched to the next history line, relative to the grasp cycle that the gripper signals show
(no rules involved), and the prompt text made of them.
"""
from __future__ import annotations

import re
import statistics
from typing import Iterable, Sequence

import numpy as np

from ..task import TaskSpec
from .author_rules_spec import gripper_thresholds
from .prompt import HistoryPrompt

_STATE_TEXT = {"holding": "still holding the object", "released": "open again after a set-down",
               "closed on nothing": "closed on nothing", "no grasp yet": "not yet grasping"}


def _cycles(rows: np.ndarray, open_width: float, empty_width: float, squeeze_force: float) -> list[dict]:
    """Grasp cycles from the gripper signals alone (no rules): start of the squeeze, end (fingers opened or closed fully)."""
    out = []
    current = None
    for t, (width, effort) in enumerate(rows[:, :2]):
        if current is None:
            if empty_width < width < open_width and effort > squeeze_force:
                current = {"grasp": t}
        elif width >= open_width or width <= empty_width:
            current["end"] = t
            current["opened"] = bool(width >= open_width)
            out.append(current)
            current = None
    if current is not None:
        out.append(current)
    return out


def _cycle_at(cycles: list[dict], t: int) -> dict | None:
    """The last grasp cycle that began at or before step t."""
    for cycle in reversed(cycles):
        if cycle["grasp"] <= t:
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
    per: dict[int, dict[str, list]] = {}
    for episode in episodes:
        _collect_switches(episode, cap, gripper, window, per)
    return _summaries(per, task, window)


def _collect_switches(episode: dict, cap: int, gripper: dict, window: Sequence[int], per: dict) -> None:
    """Add one episode's values at each label switch to the per-line lists (`per`)."""
    rows = np.asarray(episode["rows"], float)
    reference = np.minimum(np.asarray(episode["reference"], int), cap)
    z = 100 * rows[:, 4]
    p = 100 * rows[:, 2:5]
    cycles = _cycles(rows, gripper["open_width"], gripper["empty_width"], gripper["squeeze_force"])
    for k in range(1, int(reference.max(initial=0)) + 1):
        t = int(np.flatnonzero(reference >= k)[0])
        stats = per.setdefault(k, {"n": [], "state": []})
        stats["n"].append(1)
        around = [t + offset for offset in window]
        state, values = _switch_statistics(t, cycles, z, p, around)
        stats["state"].append(state)
        for key, value in values:
            stats.setdefault(key, []).append(value)


def _switch_statistics(t: int, cycles: list[dict], z, p, around: list[int]) -> tuple[str, list[tuple]]:
    """The gripper's state at a label switch and the statistics of that state (none for no grasp / closed on nothing)."""
    cycle = _cycle_at(cycles, t)
    if cycle is None:
        return "no grasp yet", []
    grasp = cycle["grasp"]
    if cycle.get("end") is None or cycle["end"] > t:
        end = cycle.get("end", len(z) - 1)
        return "holding", _holding_statistics(z, grasp, end, t, around)
    if cycle["opened"]:
        return "released", _released_statistics(z, p, grasp, cycle["end"], t, around)
    return "closed on nothing", []


def _holding_statistics(z, grasp: int, end: int, t: int, around: list[int]) -> list[tuple]:
    """Still holding at the switch: the rise above the grasp point, the highest rise of the hold, the upward speed, the
    steps since the grasp, and the course of the rise around the switch."""
    speed = z[t] - z[t - 1] if t > 0 else 0.0
    return [("rise", float(z[t] - z[grasp])),
            ("max_rise", float(z[grasp:end + 1].max() - z[grasp])),
            ("speed", float(speed)),
            ("steps_since_grasp", float(t - grasp)),
            ("rise_around", _course(z, grasp, around))]


def _released_statistics(z, p, grasp: int, release: int, t: int, around: list[int]) -> list[tuple]:
    """Opened after the grasp: the release height above the grasp point, the rise above and the distance from the release
    point, the steps since the release, and the course of the rise around the switch."""
    return [("release_height", float(z[release] - z[grasp])),
            ("up", float(z[t] - z[release])),
            ("away", float(np.linalg.norm(p[t] - p[release]))),
            ("steps_since_release", float(t - release)),
            ("up_around", _course(z, release, around))]


def _course(z, origin: int, around: list[int]) -> list:
    """The height above the origin step at each step of `around` (clamped to the episode)."""
    last = len(z) - 1
    return [z[min(max(i, 0), last)] - z[origin] for i in around]


def _summaries(per: dict, task: TaskSpec, window: Sequence[int]) -> list[dict]:
    """The collected lists -> one summary per history line: the state counts, min / median / max of each statistic
    (2 decimals) and the median course around the switch (1 decimal)."""
    plan = list(task.skills)
    history = HistoryPrompt(task)
    out = []
    for k in sorted(per):
        stats = per[k]
        phrase = history.event(plan[k - 1]) if k <= len(plan) else None
        summary = {"line": k, "phrase": phrase, "n": len(stats["n"]), "states": _state_counts(stats["state"]),
                   "window": list(window)}
        for key, values in stats.items():
            if key in ("n", "state"):
                continue
            if key.endswith("_around"):
                summary[key] = [round(float(statistics.median(column)), 1) for column in zip(*values)]
            else:
                summary[key] = [round(float(f(values)), 2) for f in (min, statistics.median, max)]
        out.append(summary)
    return out


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
    course = ", ".join(f"{offset:+d} steps {_number_text(value)} cm" for offset, value in zip(line["window"], line["rise_around"]))
    return [f"  fingertip rise above the grasp point at the switch: {_min_median_max(line['rise'], ' cm')}",
            f"  highest rise during that hold: {_min_median_max(line['max_rise'], ' cm')}",
            f"  upward speed at the switch: {_min_median_max(line['speed'], ' cm per step', 2)}",
            f"  control steps since the grasp began: {_min_median_max(line['steps_since_grasp'], '', 0)}",
            "  rise above the grasp point around the switch (median): " + course]


def _released_text(line: dict) -> list[str]:
    course = ", ".join(f"{offset:+d} steps {_number_text(value)} cm" for offset, value in zip(line["window"], line["up_around"]))
    return [f"  release height above the grasp point: {_min_median_max(line['release_height'], ' cm')}",
            f"  fingertip rise above the release point at the switch: {_min_median_max(line['up'], ' cm')}",
            f"  straight-line distance from the release point at the switch: {_min_median_max(line['away'], ' cm')}",
            f"  control steps since the release: {_min_median_max(line['steps_since_release'], '', 0)}",
            "  rise above the release point around the switch (median): " + course]


def pick_evenly(files: Sequence[str], n: int) -> list[str]:
    """n files spread evenly over a sorted list (first and last included)."""
    files = sorted(files)
    if n >= len(files):
        return list(files)
    return [files[int(round(i))] for i in np.linspace(0, len(files) - 1, n)]
