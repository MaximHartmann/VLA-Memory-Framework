#!/usr/bin/env python3
"""Memory-only baseline: no planner model. The current skill counts as complete when the majority of the k retrieved
exemplars (labelled for that skill) say so; same hysteresis and monotone index as the planner. Scored like
offline_shadow_test.py on recorded episodes. Useful to see how much of the monitor's accuracy the memory alone gives.

  python scripts/offline_knn_monitor.py --demos /path/to/demos --memory stores/ep0-49_s8.npz --episodes 200-209 --k 4
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vla_memory import TaskSpec  # noqa: E402
from vla_memory.planner_loop import ExemplarStore, fingerprint_from_task, parse_ranges  # noqa: E402
from build_exemplar_store import parse_range  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", required=True); ap.add_argument("--memory", required=True)
    ap.add_argument("--episodes", default="200-209"); ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--task", default=None); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--votes", type=int, default=1); ap.add_argument("--min-calls", type=int, default=2)
    ap.add_argument("--plan", default=None, help="skills of the plan, comma-separated (default: the task's skills in their listed order)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    task = TaskSpec.load(a.task) if a.task else TaskSpec.default()
    phase_to_skill = parse_ranges(task.demo_phases["phase_to_skill"]); skills = task.skills
    plan = [s.strip() for s in a.plan.split(",")] if a.plan else list(skills)
    if any(s not in skills for s in plan):
        ap.error(f"--plan contains skills outside the task vocabulary: {[s for s in plan if s not in skills]}")
    st = ExemplarStore(fingerprint_from_task(task), task, a.memory); rows = []
    for ep in parse_range(a.episodes):
        z = np.load(pathlib.Path(a.demos) / f"episode_{ep:04d}.npz"); EXT, WRI = z["exterior"], z["wrist"]
        gt = np.array([phase_to_skill[int(p)] for p in z["phase"]]); gt_switch = [int(np.argmax(gt >= k)) for k in range(1, len(skills))]
        idx = 0; consec = 0; calls_in = 0; pred = []; votes = []
        frames = list(range(0, len(gt), a.stride))
        for f in frames:
            cur = skills.index(plan[idx]); pred.append(cur); calls_in += 1
            hits = st.retrieve((EXT[f], WRI[f]), k=a.k, exclude_episode=ep)
            done_votes = [st.meta[j]["subgoal"] > cur for j, _ in hits]
            done = sum(done_votes) > len(done_votes) / 2
            votes.append((done, int(gt[f]) > cur)); consec = consec + 1 if done else 0
            if done and consec >= a.votes and calls_in >= a.min_calls and idx < len(plan) - 1:
                idx += 1; consec = 0; calls_in = 0
        pred = np.array(pred); gtc = gt[frames]
        pl_switch = [next((frames[i] for i in range(len(pred)) if pred[i] >= k), None) for k in range(1, len(skills))]
        delays = [None if p is None else p - g for p, g in zip(pl_switch, gt_switch)]
        rows.append(dict(episode=ep, prompt_accuracy=float(np.mean(pred == gtc)), vote_accuracy=float(np.mean([x == y for x, y in votes])),
                         switch_delay_frames=delays, early=sum(1 for d in delays if d is not None and d < 0)))
        print(json.dumps(rows[-1]))
    d = [x for r in rows for x in r["switch_delay_frames"] if x is not None]
    agg = dict(n_episodes=len(rows), prompt_accuracy=float(np.mean([r["prompt_accuracy"] for r in rows])),
               vote_accuracy=float(np.mean([r["vote_accuracy"] for r in rows])), switches_found=f"{len(d)}/{(len(skills) - 1) * len(rows)}",
               switch_delay_median=float(np.median(d)) if d else None, early_switches=int(sum(1 for x in d if x < 0)))
    print("AGG", json.dumps(agg))
    if a.out:
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True); (pathlib.Path(a.out) / "summary.json").write_text(json.dumps(rows, indent=1))
        (pathlib.Path(a.out) / "aggregate.json").write_text(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
