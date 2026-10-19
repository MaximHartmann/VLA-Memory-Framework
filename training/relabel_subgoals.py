#!/usr/bin/env python3
"""Relabel a red/blue LeRobot dataset with per-frame sub-goal prompts: every frame's task becomes the skill of the waypoint
phase it was recorded in (the recordings store a `phase` per frame).

    phases 0-3    pick up the red block     approach, descend, close, lift
    phases 4-6    place the red block       lower, open, up
    phases 7-11   pick up the blue block    the neutral retreat (7) is folded in, so the goal direction stays continuous
    phases 12-14  place the blue block

The mapping goes by phase, not by frame number: the phases' lengths vary from episode to episode, so fixed frame cuts
would mislabel some episodes.

Only `task_index` changes. Images, state and actions are copied unchanged (minutes of I/O, no re-encoding), so the data's
norm stats are unaffected. meta/ is rewritten to match: tasks.jsonl (the four sub-goals), episodes.jsonl (each episode's
task list), info.json (total_tasks) and the task_index entries of episodes_stats.jsonl.

Integrity: before an episode is touched, its parquet `state` column must equal the backing recording exactly, so a
sub-goal can never be attached to another episode's frames. Every episode must also visit the four sub-goals in order,
and the finished meta/ is re-read and checked against the parquet files before the script reports success.

    python training/relabel_subgoals.py --src <converted dataset> --dst <new dataset> --raw-dir <recordings> [--start N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # the shared helpers, also when the script is loaded by its path
from lerobot_tools import episode_files, read_jsonl, states_match, task_index_column, task_index_stats, update_task_meta  # noqa: E402

SUBGOALS = ("pick up the red block", "place the red block", "pick up the blue block", "place the blue block")
PHASE_TO_SUBGOAL = {**dict.fromkeys(range(0, 4), 0), **dict.fromkeys(range(4, 7), 1),
                    **dict.fromkeys(range(7, 12), 2), **dict.fromkeys(range(12, 15), 3)}


class RelabelError(Exception):
    """A dataset that cannot be relabelled; its message is printed and the script exits with code 3."""


def fail(message: str) -> int:
    """Report why the script stops; the exit code 3."""
    print(f"[relabel] ABORT: {message}", file=sys.stderr)
    return 3


def parse_args() -> argparse.Namespace:
    """The source and destination datasets, the recordings behind them and the dry run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="existing LeRobot dataset root")
    parser.add_argument("--dst", required=True, help="new dataset root to create")
    parser.add_argument("--raw-dir", default="data/demos_optb_full")
    parser.add_argument("--start", type=int, default=0,
                        help="index of the first npz backing this split (val starts at 200)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def episode_subgoals(parquet: pathlib.Path, archive, table) -> np.ndarray:
    """The sub-goal index of every frame of one episode, from the recording's phases; the episode must visit the four
    sub-goals in order."""
    phase = archive["phase"].astype(int)
    if len(phase) != table.num_rows:
        raise RelabelError(f"{parquet.name}: {table.num_rows} rows but {len(phase)} phase labels")
    unknown = set(np.unique(phase).tolist()) - set(PHASE_TO_SUBGOAL)
    if unknown:
        raise RelabelError(f"{parquet.name}: phases {sorted(unknown)} have no subgoal")
    task_indices = np.array([PHASE_TO_SUBGOAL[value] for value in phase], dtype=np.int64)
    if np.any(np.diff(task_indices) < 0) or set(task_indices.tolist()) != {0, 1, 2, 3}:
        raise RelabelError(f"{parquet.name}: subgoals not visited in order 0-1-2-3: {np.unique(task_indices).tolist()}")
    return task_indices


def relabel_episode(parquet: pathlib.Path, recording: pathlib.Path):
    """One episode: its table with the new task_index column, and the sub-goal indices, after the state check."""
    archive = np.load(recording)
    table = pq.read_table(parquet)
    if not states_match(table, archive):
        raise RelabelError(f"state mismatch: {parquet.name} vs {recording.name}")
    task_indices = episode_subgoals(parquet, archive, table)
    return task_index_column(table, task_indices), task_indices


def relabel_dataset(args, source: pathlib.Path, destination: pathlib.Path, recordings: list, parquets: list) -> tuple:
    """Every episode: the task_index statistics, the task list and the row count by episode index; the parquet files are
    written unless --dry-run."""
    stats, tasks_of, rows_of = {}, {}, {}
    started = time.time()
    for position, parquet in enumerate(parquets):
        episode = int(parquet.stem.split("_")[-1])
        table, task_indices = relabel_episode(parquet, recordings[args.start + episode])
        stats[episode] = task_index_stats(task_indices)
        tasks_of[episode] = [SUBGOALS[index] for index in sorted(set(task_indices.tolist()))]
        rows_of[episode] = table.num_rows
        if not args.dry_run:
            target = destination / parquet.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
        if position == 0 or (position + 1) % 25 == 0:
            counts = np.bincount(task_indices, minlength=4).tolist()
            print(f"[relabel] {position + 1}/{len(parquets)} ({time.time() - started:.0f}s)  {parquet.name}: "
                  f"{table.num_rows} frames, per subgoal {counts}", flush=True)
    return stats, tasks_of, rows_of


def verify_output(destination: pathlib.Path, rows_of: dict) -> list[str]:
    """Re-read the written meta/ and check it against the parquet files; returns the problems found."""
    meta = destination / "meta"
    tasks = read_jsonl(meta / "tasks.jsonl")
    episodes = read_jsonl(meta / "episodes.jsonl")
    info = json.loads((meta / "info.json").read_text())
    written = episode_files(destination)
    problems = []
    if [task["task"] for task in tasks] != list(SUBGOALS):
        problems.append("tasks.jsonl does not list the four subgoals in order")
    if info["total_tasks"] != len(SUBGOALS) or info["total_episodes"] != len(written):
        problems.append(f"info.json: total_tasks {info['total_tasks']}, total_episodes "
                        f"{info['total_episodes']} vs {len(written)} parquet files")
    if info["total_frames"] != sum(rows_of.values()):
        problems.append(f"info.json total_frames {info['total_frames']} != {sum(rows_of.values())} rows")
    for record in episodes:
        episode = record["episode_index"]
        if record["tasks"] != list(SUBGOALS) or record["length"] != rows_of[episode]:
            problems.append(f"episodes.jsonl episode {episode}: tasks {record['tasks']}, length {record['length']}")
    for parquet in written[:: max(1, len(written) // 10)]:  # spot-check written columns
        task_indices = pq.read_table(parquet, columns=["task_index"]).column("task_index").to_numpy()
        if task_indices.min() != 0 or task_indices.max() != 3:
            problems.append(f"{parquet.name}: task_index range {task_indices.min()}..{task_indices.max()}")
    return problems


def main() -> int:
    """Relabel the dataset, rewrite its meta/ and verify the result; exit code 3 on any problem."""
    args = parse_args()
    source, destination = pathlib.Path(args.src), pathlib.Path(args.dst)
    recordings = sorted(pathlib.Path(args.raw_dir).glob("episode_*.npz"))
    parquets = episode_files(source)
    if not parquets:
        return fail(f"no parquet files under {source}/data")
    if destination.exists() and not args.dry_run:
        return fail(f"{destination} exists; refusing to overwrite")
    print(f"[relabel] {len(parquets)} episodes in {source}; backing npz from index {args.start}", flush=True)
    started = time.time()
    try:
        stats, tasks_of, rows_of = relabel_dataset(args, source, destination, recordings, parquets)
    except RelabelError as error:
        return fail(str(error))
    if args.dry_run:
        print(f"[relabel] DRY RUN ok: {len(parquets)} episodes verified, nothing written")
        return 0
    update_task_meta(source, destination, list(SUBGOALS), tasks_of, stats)
    problems = verify_output(destination, rows_of)
    if problems:
        return fail("; ".join(problems[:5]))
    size = sum(file.stat().st_size for file in destination.rglob("*") if file.is_file()) / 1e9
    print(f"[relabel] done: {len(episode_files(destination))} episodes / {sum(rows_of.values())} frames in "
          f"{time.time() - started:.0f}s -> {destination} ({size:.2f} GB); meta verified: 4 tasks, every episode lists all "
          f"four, lengths match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
