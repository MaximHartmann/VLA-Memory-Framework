"""Offline validation of a rules writer: replay recorded episodes through the task file's history.rules and compare, at
every policy call, the history the writer would show with a reference. Run it before a rule set drives a robot.

Episode files (.npz), one per episode:
    rows       (T, 5) float   the signals observed at each control step t, before its action: finger width, squeeze,
                              fingertip x, y, z (the same columns the environment sends as memory/proprio_steps)
    reference  (T,)   int     the number of completed skills the reference shows at step t: a ground-truth tracker in
                              simulation, the per-frame labels of the training demonstrations, or a person's annotation
    signal_<name> (T,) float  optional: a signal the task declares (task file `signals`) for its `trigger: signals` rules,
                              e.g. signal_contact_force
The writer at step t has seen rows[0..t]. Its history is compared with the reference at the policy calls t = 0, k, 2k, ...
(k = --call-every; 1 compares every frame, e.g. against demonstration labels).

Both counts are capped at the task's history.max_lines, the longest history the policy sees. Per history line: in how
many episodes the writer and the reference switch in the same call, earlier, later; missed (reference only) and false
(writer only) lines; and the share of calls whose history equals the reference.

    python -m vla_memory.history_appended.validate --episodes 'exported/*.npz' [--task task.yaml] [--call-every 8]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from typing import Iterable

import numpy as np

from ..task import TaskSpec
from .rules import RuleMonitor

SIGNAL_PREFIX = "signal_"       # an episode's array of a declared signal: signal_<name>


def episode_signals(ep: dict) -> dict:
    """The declared signals of an episode dict or file, {name: (T,)}, from its signal_<name> arrays."""
    signals = {}
    for key in ep:
        if key.startswith(SIGNAL_PREFIX):
            signals[key[len(SIGNAL_PREFIX):]] = np.asarray(ep[key])
    return signals


def _first_call(counts: np.ndarray, calls: np.ndarray, line: int) -> int | None:
    """The first policy call at which the count has reached `line`, or None if it never does."""
    hits = np.flatnonzero(counts[calls] >= line)
    if len(hits) == 0:
        return None
    return int(calls[hits[0]])


def _new_line_statistics() -> dict:
    return {"both": 0, "same_call": 0, "earlier": 0, "later": 0, "missed": 0, "false": 0, "offsets": []}


def validate(task: TaskSpec, episodes: Iterable[dict], call_every: int = 8, **gripper_overrides) -> dict:
    """Replay the episodes through the task's rules and compare with their reference at the policy calls (module
    docstring). Returns the counts over all calls and the per-line result under "lines"."""
    monitor = RuleMonitor(task, **gripper_overrides)
    cap = (task.history or {}).get("max_lines")
    if cap is None:
        cap = len(monitor.plan)
    n_skills = min(cap, len(monitor.plan))
    per = {line: _new_line_statistics() for line in range(1, n_skills + 1)}
    calls_equal = 0
    calls_total = 0
    n_episodes = 0
    for episode in episodes:
        equal, calls = _compare_episode(monitor, episode, cap, call_every, per)
        calls_equal += equal
        calls_total += calls
        n_episodes += 1
    lines = {}
    for line, stats in per.items():
        lines[monitor.plan[line - 1]] = _line_summary(stats)
    return {"episodes": n_episodes, "calls": calls_total, "calls_equal": calls_equal,
            "agreement": calls_equal / max(calls_total, 1), "call_every": call_every, "lines": lines}


def _compare_episode(monitor: RuleMonitor, episode: dict, cap: int, call_every: int, per: dict) -> tuple[int, int]:
    """One episode: the writer's counts against the reference at the policy calls; updates the per-line statistics.
    Returns (calls at which the histories are equal, calls)."""
    rows = np.asarray(episode["rows"], float)
    reference = np.minimum(np.asarray(episode["reference"], int), cap)
    writer = np.minimum(monitor.label_rows(rows, signals=episode_signals(episode) or None), cap)
    calls = np.arange(0, len(writer), max(1, int(call_every)))
    for line, stats in per.items():
        _count_switch(stats, _first_call(writer, calls, line), _first_call(reference, calls, line))
    return int((writer[calls] == reference[calls]).sum()), len(calls)


def _count_switch(stats: dict, writer_call, reference_call) -> None:
    """One episode's switch to a line: both (same call, earlier or later), missed (reference only) or false (writer only)."""
    if writer_call is not None and reference_call is not None:
        stats["both"] += 1
        stats["offsets"].append(writer_call - reference_call)
        if writer_call == reference_call:
            stats["same_call"] += 1
        elif writer_call < reference_call:
            stats["earlier"] += 1
        else:
            stats["later"] += 1
    elif reference_call is not None:
        stats["missed"] += 1
    elif writer_call is not None:
        stats["false"] += 1


def _line_summary(stats: dict) -> dict:
    """The per-line result: the counts plus the median and the largest absolute offset (control steps)."""
    offsets = stats.pop("offsets")
    median = statistics.median(offsets) if offsets else None
    largest = max((abs(offset) for offset in offsets), default=None)
    return {**stats, "median_offset_steps": median, "max_abs_offset_steps": largest}


def load_episodes(pattern: str):
    """The episode files matching the glob, sorted, as dicts: rows, reference, file and the signal_<name> arrays."""
    for path in sorted(glob.glob(pattern)):
        data = np.load(path)
        episode = {"rows": data["rows"], "reference": data["reference"], "file": path}
        for key in data.files:
            if key.startswith(SIGNAL_PREFIX):
                episode[key] = data[key]
        yield episode


def report(res: dict) -> str:
    """The result of validate() as a text table."""
    out = [f"{res['episodes']} episodes, {res['calls']} calls (every {res['call_every']} steps): history equal to the reference "
           f"on {res['calls_equal']} ({100 * res['agreement']:.2f} %)",
           f"{'completed skill':28s} {'both':>5s} {'same call':>9s} {'earlier':>7s} {'later':>5s} {'missed':>6s} {'false':>5s}  median offset"]
    for skill, s in res["lines"].items():
        med = "-" if s["median_offset_steps"] is None else f"{s['median_offset_steps']:+.0f} steps"
        out.append(f"{skill:28s} {s['both']:5d} {s['same_call']:9d} {s['earlier']:7d} {s['later']:5d} {s['missed']:6d} {s['false']:5d}  {med}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default=None, help="glob of episode .npz files (rows, reference)")
    parser.add_argument("--task", default=None, help="task YAML/JSON with history.rules (default: the packaged reference task)")
    parser.add_argument("--call-every", type=int, default=8, help="policy call period in control steps (1 = compare every frame)")
    parser.add_argument("--squeeze-force", type=float, default=None, help="override proprio.squeeze_force (0.5 for a grasped flag)")
    parser.add_argument("--json", default=None, help="also write the result as JSON")
    args = parser.parse_args(argv)
    if not args.episodes:
        parser.error("give --episodes")
    task = TaskSpec.load(args.task) if args.task else TaskSpec.default()
    overrides = {} if args.squeeze_force is None else {"squeeze_force": args.squeeze_force}
    result = validate(task, load_episodes(args.episodes), args.call_every, **overrides)
    print(report(result))
    if args.json:
        with open(args.json, "w") as file:
            json.dump(result, file, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
