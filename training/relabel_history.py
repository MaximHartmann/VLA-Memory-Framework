#!/usr/bin/env python3
"""Relabel a monolith red/blue LeRobot dataset with PER-FRAME "history prompts" (UniMem-style textual event memory).

Every frame's task becomes   "<episode instruction> History: <events so far>."   where the events are derived from
the waypoint phase stored in the staged npz (generate_demos.py), using the SAME phase boundaries as
relabel_subgoals.py, so the history text changes exactly where the sub-goal prompt would change:

    phases 0-3    History: none yet.
    phases 4-6    History: picked up the red block.                      (after the lift, incl. lowering/release/retreat)
    phases 7-11   History: picked up the red block, placed the red block.
    phases 12-14  History: picked up the red block, placed the red block, picked up the blue block.

This is the fourth ablation arm between "subgoal" (the prompt IS the current sub-goal) and "monolith" (the prompt
never changes): the instruction stays monolithic and the policy is told what it has already done. At test time the
bench produces the same text from its event tracker (bench flag --history-oracle, same retreat rule as the oracle).

Only `task_index`, tasks.jsonl, episodes.jsonl (tasks), episodes_stats.jsonl (task_index stats) and info.json
(total_tasks) change; images/state/actions are copied unchanged (norm stats unaffected). Integrity check as in
relabel_subgoals.py: the parquet `state` column must equal the backing npz exactly.
"""
from __future__ import annotations
import argparse, json, pathlib, shutil, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

HIST = ("none yet.",
        "picked up the red block.",
        "picked up the red block, placed the red block.",
        "picked up the red block, placed the red block, picked up the blue block.")
PHASE_TO_HIST = {**dict.fromkeys(range(0, 4), 0), **dict.fromkeys(range(4, 7), 1),
                 **dict.fromkeys(range(7, 12), 2), **dict.fromkeys(range(12, 15), 3)}


def history_prompt(instruction: str, h: int) -> str:
    return f"{instruction.strip()} History: {HIST[h]}"


def fail(msg):
    print(f"[relabel-hist] ABORT: {msg}", file=sys.stderr); return 3


def read_jsonl(p): return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
def write_jsonl(p, rows): pathlib.Path(p).write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--raw-dir", default="data/demos_v2_full"); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)
    raw = sorted(pathlib.Path(a.raw_dir).glob("episode_*.npz"))
    parquets = sorted((src / "data").rglob("episode_*.parquet"))
    if not parquets: return fail(f"no parquet files under {src}/data")
    if dst.exists() and not a.dry_run: return fail(f"{dst} exists; refusing to overwrite")
    src_tasks = {r["task_index"]: r["task"] for r in read_jsonl(src / "meta" / "tasks.jsonl")}
    src_eps = {r["episode_index"]: r for r in read_jsonl(src / "meta" / "episodes.jsonl")}
    task_ids: dict[str, int] = {}   # new task string -> new index (assigned in order of first appearance)
    stats, tasks_of, rows_of = {}, {}, {}
    t0 = time.time()
    for i, pqf in enumerate(parquets):
        ep = int(pqf.stem.split("_")[-1]); z = np.load(raw[a.start + ep]); table = pq.read_table(pqf)
        state = np.asarray(table.column("state").to_pylist(), dtype=np.float32)
        if state.shape != z["state"].shape or not np.allclose(state, z["state"], atol=1e-6):
            return fail(f"state mismatch: {pqf.name} vs {raw[a.start + ep].name}")
        phase = z["phase"].astype(int)
        if len(phase) != table.num_rows: return fail(f"{pqf.name}: {table.num_rows} rows but {len(phase)} phases")
        unknown = set(np.unique(phase).tolist()) - set(PHASE_TO_HIST)
        if unknown: return fail(f"{pqf.name}: phases {sorted(unknown)} unmapped")
        old_idx = np.asarray(table.column("task_index").to_pylist())
        instr = {src_tasks[int(k)] for k in np.unique(old_idx)}
        if len(instr) != 1: return fail(f"{pqf.name}: expected one instruction per episode, got {instr}")
        instr = instr.pop()
        h = np.array([PHASE_TO_HIST[p] for p in phase], dtype=np.int64)
        if np.any(np.diff(h) < 0) or set(h.tolist()) != {0, 1, 2, 3}:
            return fail(f"{pqf.name}: history states not visited in order 0-1-2-3: {np.unique(h).tolist()}")
        strings = [history_prompt(instr, k) for k in range(4)]
        for s_ in strings: task_ids.setdefault(s_, len(task_ids))
        idx = np.array([task_ids[strings[k]] for k in h], dtype=np.int64)
        col = pa.array(idx, type=table.schema.field("task_index").type)
        table = table.set_column(table.schema.get_field_index("task_index"), "task_index", col)
        stats[ep] = {"min": [int(idx.min())], "max": [int(idx.max())], "mean": [float(idx.mean())],
                     "std": [float(idx.std())], "count": [int(idx.size)]}
        tasks_of[ep] = strings; rows_of[ep] = table.num_rows
        if not a.dry_run:
            out = dst / pqf.relative_to(src); out.parent.mkdir(parents=True, exist_ok=True); pq.write_table(table, out)
        if i == 0 or (i + 1) % 25 == 0:
            print(f"[relabel-hist] {i + 1}/{len(parquets)} ({time.time() - t0:.0f}s) {pqf.name}: {table.num_rows} frames, "
                  f"per history state {np.bincount(h, minlength=4).tolist()}; tasks so far {len(task_ids)}", flush=True)
    if a.dry_run:
        print(f"[relabel-hist] DRY RUN ok: {len(parquets)} episodes verified, {len(task_ids)} distinct prompts"); return 0
    meta = dst / "meta"; meta.mkdir(parents=True, exist_ok=True)
    for f in (src / "meta").glob("*"): shutil.copy2(f, meta / f.name)
    write_jsonl(meta / "tasks.jsonl", [{"task_index": j, "task": t} for t, j in sorted(task_ids.items(), key=lambda kv: kv[1])])
    episodes = read_jsonl(meta / "episodes.jsonl")
    for rec in episodes: rec["tasks"] = tasks_of[rec["episode_index"]]
    write_jsonl(meta / "episodes.jsonl", episodes)
    ep_stats = read_jsonl(meta / "episodes_stats.jsonl")
    for rec in ep_stats: rec["stats"]["task_index"] = stats[rec["episode_index"]]
    write_jsonl(meta / "episodes_stats.jsonl", ep_stats)
    info = json.loads((meta / "info.json").read_text()); info["total_tasks"] = len(task_ids)
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    # verify
    tasks = {r["task_index"]: r["task"] for r in read_jsonl(meta / "tasks.jsonl")}
    written = sorted((dst / "data").rglob("episode_*.parquet"))
    if len(written) != len(parquets): return fail("written episode count differs")
    for pqf in written:
        ti = np.asarray(pq.read_table(pqf, columns=["task_index"]).column("task_index").to_pylist())
        if not all(int(k) in tasks for k in np.unique(ti)): return fail(f"{pqf.name}: task_index outside tasks.jsonl")
    print(f"[relabel-hist] OK: {len(written)} episodes, {len(tasks)} prompts, e.g. {tasks[0]!r} ... {tasks[3]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
