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
from .identify import make_identifier
from .prompt import HistoryPrompt
from .rules import RuleMonitor, parse_rules

SIGNAL_PREFIX = "signal_"       # an episode's array of a declared signal: signal_<name>


def episode_signals(ep: dict) -> dict:
    """The declared signals of an episode dict or file, {name: (T,)}, from its signal_<name> arrays."""
    return {k[len(SIGNAL_PREFIX):]: np.asarray(ep[k]) for k in ep if k.startswith(SIGNAL_PREFIX)}


def _first(counts: np.ndarray, calls: np.ndarray, i: int):
    hit = np.flatnonzero(counts[calls] >= i)
    return int(calls[hit[0]]) if len(hit) else None


def validate(task: TaskSpec, episodes: Iterable[dict], call_every: int = 8, **gripper_overrides) -> dict:
    mon = RuleMonitor(task, **gripper_overrides)
    cap = (task.history or {}).get("max_lines"); cap = len(mon.plan) if cap is None else cap; n_skills = min(cap, len(mon.plan))
    per = {i: {"both": 0, "same_call": 0, "earlier": 0, "later": 0, "missed": 0, "false": 0, "offsets": []} for i in range(1, n_skills + 1)}
    agree = calls_total = n_ep = 0
    for ep in episodes:
        rows = np.asarray(ep["rows"], float); ref = np.minimum(np.asarray(ep["reference"], int), cap)
        writer = np.minimum(mon.label_rows(rows, signals=episode_signals(ep) or None), cap)
        calls = np.arange(0, len(writer), max(1, int(call_every)))
        agree += int((writer[calls] == ref[calls]).sum()); calls_total += len(calls); n_ep += 1
        for i in range(1, n_skills + 1):
            w, r = _first(writer, calls, i), _first(ref, calls, i)
            s = per[i]
            if w is not None and r is not None:
                s["both"] += 1; s["offsets"].append(w - r)
                s["same_call" if w == r else ("earlier" if w < r else "later")] += 1
            elif r is not None:
                s["missed"] += 1
            elif w is not None:
                s["false"] += 1
    lines = {}
    for i, s in per.items():
        off = s.pop("offsets")
        lines[mon.plan[i - 1]] = {**s, "median_offset_steps": statistics.median(off) if off else None,
                                  "max_abs_offset_steps": max((abs(x) for x in off), default=None)}
    return {"episodes": n_ep, "calls": calls_total, "calls_equal": agree, "agreement": agree / max(calls_total, 1),
            "call_every": call_every, "lines": lines}


def load_episodes(pattern: str):
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        yield {"rows": z["rows"], "reference": z["reference"], "file": f,
               **{k: z[k] for k in z.files if k.startswith(SIGNAL_PREFIX)}}


def report(res: dict) -> str:
    out = [f"{res['episodes']} episodes, {res['calls']} calls (every {res['call_every']} steps): history equal to the reference "
           f"on {res['calls_equal']} ({100 * res['agreement']:.2f} %)",
           f"{'completed skill':28s} {'both':>5s} {'same call':>9s} {'earlier':>7s} {'later':>5s} {'missed':>6s} {'false':>5s}  median offset"]
    for skill, s in res["lines"].items():
        med = "-" if s["median_offset_steps"] is None else f"{s['median_offset_steps']:+.0f} steps"
        out.append(f"{skill:28s} {s['both']:5d} {s['same_call']:9d} {s['earlier']:7d} {s['later']:5d} {s['missed']:6d} {s['false']:5d}  {med}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default=None, help="glob of episode .npz files (rows, reference)")
    ap.add_argument("--task", default=None, help="task YAML/JSON with history.rules (default: the packaged reference task)")
    ap.add_argument("--call-every", type=int, default=8, help="policy call period in control steps (1 = compare every frame)")
    ap.add_argument("--squeeze-force", type=float, default=None, help="override proprio.squeeze_force (0.5 for a grasped flag)")
    ap.add_argument("--json", default=None, help="also write the result as JSON")
    args = ap.parse_args(argv)
    if not args.episodes:
        ap.error("give --episodes")
    task = TaskSpec.load(args.task) if args.task else TaskSpec.default()
    over = {} if args.squeeze_force is None else {"squeeze_force": args.squeeze_force}
    res = validate(task, load_episodes(args.episodes), args.call_every, **over)
    print(report(res))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
