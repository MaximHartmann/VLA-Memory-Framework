#!/usr/bin/env python3
"""Shadow-mode test of the planner on recorded episodes (no robot, no policy): the planner plans from the episode's
instruction, then sees one frame every --stride frames (one policy call) and advances on its own decisions; the
frame's ground-truth skill (from the phase labels) scores every decision.

Metrics per episode: plan == reference; prompt accuracy (planner's skill vs ground truth at every call); switch
delay in frames per transition (negative = early); raw "done" vote accuracy before hysteresis; latency and tokens.

  python scripts/offline_shadow_test.py --demos /path/to/demos --episodes 200-209 --memory stores/ep0-49_s8.npz --out runs/shadow
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vla_memory import OpenAICompatibleVLM, TaskSpec  # noqa: E402
from vla_memory.planner_loop import ExemplarStore, Planner, fingerprint_from_task, parse_ranges  # noqa: E402


def parse_range(s):
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-"); out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", required=True); ap.add_argument("--episodes", default="200-209")
    ap.add_argument("--stride", type=int, default=8); ap.add_argument("--task", default=None)
    ap.add_argument("--url", default="http://127.0.0.1:8100/v1"); ap.add_argument("--model", default="Qwen3.8-27B-INT4")
    ap.add_argument("--thinking", action="store_true"); ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--min-calls", type=int, default=2); ap.add_argument("--memory", default=None)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--upscale", type=int, default=1); ap.add_argument("--proprio", action="store_true",
                    help="feed the recorded gripper closedness (state column 7) as proprioception (no hand height in the demos)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    task = TaskSpec.load(args.task) if args.task else TaskSpec.default()
    phase_to_skill = parse_ranges(task.demo_phases["phase_to_skill"]); skills = task.skills
    store = None
    if args.memory:
        store = ExemplarStore(fingerprint_from_task(task), task, args.memory); print(f"memory: {len(store)} entries from {args.memory}")
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vlm = OpenAICompatibleVLM(args.url, args.model, chat_template_kwargs={"enable_thinking": bool(args.thinking)})
    rows = []
    for ep in parse_range(args.episodes):
        z = np.load(pathlib.Path(args.demos) / f"episode_{ep:04d}.npz")
        gt = np.array([phase_to_skill[int(p)] for p in z["phase"]])
        gt_switch = [int(np.argmax(gt >= k)) for k in range(1, len(skills))]
        gt_plan = [skills[k] for k in dict.fromkeys(int(x) for x in gt)]   # the skills in the order the demo performed them
        instr = str(z["instruction"])
        pl = Planner(vlm, task, memory=store, k=args.k, votes_to_advance=args.votes, min_calls_per_skill=args.min_calls,
                     log=str(out / f"planner_ep{ep:04d}.jsonl"), use_wrist=not args.no_wrist, upscale=args.upscale, exclude_episode=ep)
        t0 = time.time(); plan_rec = pl.start_episode(instr, f"demo{ep}")
        frames = list(range(0, len(gt), args.stride)); pred, votes, lat = [], [], []
        for f in frames:
            pro = {"gripper_closedness": float(z["state"][f, 7])} if args.proprio else None
            rec = pl.observe((z["exterior"][f], z["wrist"][f]), f, proprio=pro)
            pred.append(skills.index(rec["skill"]) if rec["skill"] in skills else -1)
            if rec.get("vlm"):
                cur = skills.index(rec["skill"]); votes.append((bool(rec["vlm"]["current_skill_done"]), int(gt[f]) > cur)); lat.append(rec["ms"])
        pred = np.array(pred); gtc = gt[frames]
        pl_switch = [next((frames[i] for i in range(len(pred)) if pred[i] >= k), None) for k in range(1, len(skills))]
        delays = [None if p is None else p - g for p, g in zip(pl_switch, gt_switch)]
        row = dict(episode=ep, instruction=instr, plan=plan_rec["plan"], plan_source=plan_rec["source"],
                   plan_matches_demo=(plan_rec["plan"] == gt_plan), n_calls=len(frames), n_frames=len(gt),
                   prompt_accuracy=round(float(np.mean(pred == gtc)), 3), gt_switch_frames=gt_switch, planner_switch_frames=pl_switch,
                   switch_delay_frames=delays, vote_accuracy=round(float(np.mean([a == b for a, b in votes])), 3) if votes else None,
                   false_done_votes=int(sum(1 for a, b in votes if a and not b)), missed_done_votes=int(sum(1 for a, b in votes if b and not a)),
                   n_votes=len(votes), mean_ms=round(float(np.mean(lat)), 0) if lat else None, wall_s=round(time.time() - t0, 1))
        rows.append(row); pl.close(); print(json.dumps(row), flush=True)
        (out / "summary.json").write_text(json.dumps(rows, indent=1))
    d = [x for r in rows for x in r["switch_delay_frames"] if x is not None]
    agg = dict(n_episodes=len(rows), plan_accuracy=float(np.mean([r["plan_matches_demo"] for r in rows])),
               prompt_accuracy=float(np.mean([r["prompt_accuracy"] for r in rows])),
               vote_accuracy=float(np.mean([r["vote_accuracy"] for r in rows if r["vote_accuracy"] is not None])),
               switches_found=f"{len(d)}/{(len(skills) - 1) * len(rows)}", switch_delay_median=float(np.median(d)) if d else None,
               early_switches=int(sum(1 for x in d if x < 0)), mean_ms=float(np.mean([r["mean_ms"] for r in rows if r["mean_ms"]])))
    (out / "aggregate.json").write_text(json.dumps(agg, indent=1)); print("AGG", json.dumps(agg))


if __name__ == "__main__":
    main()
