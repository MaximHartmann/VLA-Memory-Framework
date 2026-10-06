#!/usr/bin/env python3
"""Relabel a red/blue LeRobot dataset with PER-FRAME subgoal prompts.

Every frame's `task_index` becomes the subgoal of the waypoint phase it was
recorded in (generate_demos.py stores `phase` per frame in the staged npz):

    phases 0-3    pick up the red block     approach, descend, close, lift
    phases 4-6    place the red block       lower, open, up
    phases 7-11   pick up the blue block    the neutral retreat (7) is folded in,
                                            so the goal direction stays continuous
    phases 12-14  place the blue block

Mapped by PHASE, not by frame number. 318 frames is the median episode, but the
converge logic stretches phases 2/9/10 up to 378 frames, and fixed frame cuts
(0-84 / 84-150 / 150-252 / 252-318) would mislabel exactly those episodes.

Only `task_index` changes. Images, state and actions are copied unchanged (the
fast path of rewrite_actions_absolute.py: minutes of I/O, no PNG re-encoding),
so the data's norm stats are unaffected. meta/ is rewritten to match:
tasks.jsonl (the four subgoals), episodes.jsonl (each episode's task list),
info.json (total_tasks) and the task_index entries of episodes_stats.jsonl.

INTEGRITY: before touching an episode, its parquet `state` column must equal the
backing npz exactly -- the same check as rewrite_actions_absolute.py -- so a
subgoal can never be attached to another episode's frames. Every episode must
also visit the four subgoals in order, and the finished meta/ is re-read and
checked against the parquet files before the script reports success.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SUBGOALS = ("pick up the red block", "place the red block", "pick up the blue block", "place the blue block")
PHASE_TO_SUBGOAL = {**dict.fromkeys(range(0, 4), 0), **dict.fromkeys(range(4, 7), 1),
                    **dict.fromkeys(range(7, 12), 2), **dict.fromkeys(range(12, 15), 3)}


def fail(msg: str) -> int:
    print(f"[relabel] ABORT: {msg}", file=sys.stderr)
    return 3


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="existing LeRobot dataset root")
    ap.add_argument("--dst", required=True, help="new dataset root to create")
    ap.add_argument("--raw-dir", default="data/demos_optb_full")
    ap.add_argument("--start", type=int, default=0,
                    help="index of the first npz backing this split (val starts at 200)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = pathlib.Path(args.src), pathlib.Path(args.dst)
    raw = sorted(pathlib.Path(args.raw_dir).glob("episode_*.npz"))
    parquets = sorted((src / "data").rglob("episode_*.parquet"))
    if not parquets:
        return fail(f"no parquet files under {src}/data")
    if dst.exists() and not args.dry_run:
        return fail(f"{dst} exists; refusing to overwrite")
    print(f"[relabel] {len(parquets)} episodes in {src}; backing npz from index {args.start}", flush=True)

    stats, tasks_of, rows_of = {}, {}, {}
    t0 = time.time()
    for i, pqf in enumerate(parquets):
        ep = int(pqf.stem.split("_")[-1])
        z = np.load(raw[args.start + ep])
        table = pq.read_table(pqf)

        state = np.asarray(table.column("state").to_pylist(), dtype=np.float32)
        if state.shape != z["state"].shape or not np.allclose(state, z["state"], atol=1e-6):
            return fail(f"state mismatch: {pqf.name} vs {raw[args.start + ep].name}")
        phase = z["phase"].astype(int)
        if len(phase) != table.num_rows:
            return fail(f"{pqf.name}: {table.num_rows} rows but {len(phase)} phase labels")
        unknown = set(np.unique(phase).tolist()) - set(PHASE_TO_SUBGOAL)
        if unknown:
            return fail(f"{pqf.name}: phases {sorted(unknown)} have no subgoal")
        idx = np.array([PHASE_TO_SUBGOAL[p] for p in phase], dtype=np.int64)
        if np.any(np.diff(idx) < 0) or set(idx.tolist()) != {0, 1, 2, 3}:
            return fail(f"{pqf.name}: subgoals not visited in order 0-1-2-3: {np.unique(idx).tolist()}")

        col = pa.array(idx, type=table.schema.field("task_index").type)
        table = table.set_column(table.schema.get_field_index("task_index"), "task_index", col)
        stats[ep] = {"min": [int(idx.min())], "max": [int(idx.max())], "mean": [float(idx.mean())],
                     "std": [float(idx.std())], "count": [int(idx.size)]}
        tasks_of[ep] = [SUBGOALS[j] for j in sorted(set(idx.tolist()))]
        rows_of[ep] = table.num_rows
        if not args.dry_run:
            out = dst / pqf.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out)
        if i == 0 or (i + 1) % 25 == 0:
            counts = np.bincount(idx, minlength=4).tolist()
            print(f"[relabel] {i + 1}/{len(parquets)} ({time.time() - t0:.0f}s)  {pqf.name}: "
                  f"{table.num_rows} frames, per subgoal {counts}", flush=True)

    if args.dry_run:
        print(f"[relabel] DRY RUN ok: {len(parquets)} episodes verified, nothing written")
        return 0

    meta = dst / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for f in (src / "meta").glob("*"):
        shutil.copy2(f, meta / f.name)
    write_jsonl(meta / "tasks.jsonl", [{"task_index": j, "task": t} for j, t in enumerate(SUBGOALS)])
    episodes = read_jsonl(meta / "episodes.jsonl")
    for rec in episodes:
        rec["tasks"] = tasks_of[rec["episode_index"]]
    write_jsonl(meta / "episodes.jsonl", episodes)
    ep_stats = read_jsonl(meta / "episodes_stats.jsonl")
    for rec in ep_stats:
        rec["stats"]["task_index"] = stats[rec["episode_index"]]
    write_jsonl(meta / "episodes_stats.jsonl", ep_stats)
    info = json.loads((meta / "info.json").read_text())
    info["total_tasks"] = len(SUBGOALS)
    (meta / "info.json").write_text(json.dumps(info, indent=4))

    # Re-read what was written and check it against the parquet files.
    tasks = read_jsonl(meta / "tasks.jsonl")
    episodes = read_jsonl(meta / "episodes.jsonl")
    info = json.loads((meta / "info.json").read_text())
    written = sorted((dst / "data").rglob("episode_*.parquet"))
    problems = []
    if [t["task"] for t in tasks] != list(SUBGOALS):
        problems.append("tasks.jsonl does not list the four subgoals in order")
    if info["total_tasks"] != len(SUBGOALS) or info["total_episodes"] != len(written):
        problems.append(f"info.json: total_tasks {info['total_tasks']}, total_episodes "
                        f"{info['total_episodes']} vs {len(written)} parquet files")
    if info["total_frames"] != sum(rows_of.values()):
        problems.append(f"info.json total_frames {info['total_frames']} != {sum(rows_of.values())} rows")
    for rec in episodes:
        ep = rec["episode_index"]
        if rec["tasks"] != list(SUBGOALS) or rec["length"] != rows_of[ep]:
            problems.append(f"episodes.jsonl episode {ep}: tasks {rec['tasks']}, length {rec['length']}")
    for pqf in written[:: max(1, len(written) // 10)]:  # spot-check written columns
        ti = pq.read_table(pqf, columns=["task_index"]).column("task_index").to_numpy()
        if ti.min() != 0 or ti.max() != 3:
            problems.append(f"{pqf.name}: task_index range {ti.min()}..{ti.max()}")
    if problems:
        return fail("; ".join(problems[:5]))

    dur = time.time() - t0
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1e9
    print(f"[relabel] done: {len(written)} episodes / {sum(rows_of.values())} frames in {dur:.0f}s -> {dst} "
          f"({size:.2f} GB); meta verified: 4 tasks, every episode lists all four, lengths match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
