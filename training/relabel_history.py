#!/usr/bin/env python3
"""Relabel a red/blue LeRobot dataset with per-frame history prompts: every frame's task becomes

    "<episode instruction> History: <events so far>."

where the events come from the waypoint phase stored in the recordings, with the same phase boundaries as
relabel_subgoals.py, so the history text changes exactly where the sub-goal prompt would change:

    phases 0-3    History: none yet.
    phases 4-6    History: picked up the red block.                      (after the lift, incl. lowering/release/retreat)
    phases 7-11   History: picked up the red block, placed the red block.
    phases 12-14  History: picked up the red block, placed the red block, picked up the blue block.

The instruction stays the full sentence and the policy is told what it has already done; at run time the history method
produces the same text from a writer's completed skills (vla_memory.methods.history). Only `task_index`, tasks.jsonl,
episodes.jsonl (tasks), episodes_stats.jsonl (task_index stats) and info.json (total_tasks) change; images, state and actions
are copied unchanged (norm stats unaffected). Integrity check as in relabel_subgoals.py: the parquet `state` column must
equal the backing recording exactly.

    python training/relabel_history.py --src <converted dataset> --dst <new dataset> --raw-dir <recordings> [--start N]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # the shared helpers, also when the script is loaded by its path
from lerobot_tools import episode_files, read_tasks, states_match, task_index_column, task_index_stats, update_task_meta  # noqa: E402

HISTORIES = ("none yet.",
             "picked up the red block.",
             "picked up the red block, placed the red block.",
             "picked up the red block, placed the red block, picked up the blue block.")
PHASE_TO_HISTORY = {**dict.fromkeys(range(0, 4), 0), **dict.fromkeys(range(4, 7), 1),
                    **dict.fromkeys(range(7, 12), 2), **dict.fromkeys(range(12, 15), 3)}


class RelabelError(Exception):
    """A dataset that cannot be relabelled; its message is printed and the script exits with code 3."""


def history_prompt(instruction: str, history: int) -> str:
    """The prompt of a frame with `history` events behind it."""
    return f"{instruction.strip()} History: {HISTORIES[history]}"


def fail(message: str) -> int:
    """Report why the script stops; the exit code 3."""
    print(f"[relabel-hist] ABORT: {message}", file=sys.stderr)
    return 3


def parse_args() -> argparse.Namespace:
    """The source and destination datasets, the recordings behind them and the dry run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--raw-dir", default="data/demos_v2_full")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def episode_histories(parquet: pathlib.Path, archive, table) -> np.ndarray:
    """The history state (0 to 3) of every frame of one episode, from the recording's phases, visited in order."""
    phase = archive["phase"].astype(int)
    if len(phase) != table.num_rows:
        raise RelabelError(f"{parquet.name}: {table.num_rows} rows but {len(phase)} phases")
    unknown = set(np.unique(phase).tolist()) - set(PHASE_TO_HISTORY)
    if unknown:
        raise RelabelError(f"{parquet.name}: phases {sorted(unknown)} unmapped")
    histories = np.array([PHASE_TO_HISTORY[value] for value in phase], dtype=np.int64)
    if np.any(np.diff(histories) < 0) or set(histories.tolist()) != {0, 1, 2, 3}:
        raise RelabelError(f"{parquet.name}: history states not visited in order 0-1-2-3: {np.unique(histories).tolist()}")
    return histories


def episode_instruction(parquet: pathlib.Path, table, source_tasks: dict) -> str:
    """The one instruction of an episode (its frames' task before the relabelling)."""
    old_indices = np.asarray(table.column("task_index").to_pylist())
    instructions = {source_tasks[int(index)] for index in np.unique(old_indices)}
    if len(instructions) != 1:
        raise RelabelError(f"{parquet.name}: expected one instruction per episode, got {instructions}")
    return instructions.pop()


def relabel_episode(parquet: pathlib.Path, recording: pathlib.Path, source_tasks: dict, task_ids: dict):
    """One episode: its table with the new task_index column, its four prompts and the history state of every frame.
    task_ids (prompt -> new task index) grows with the prompts first seen here."""
    archive = np.load(recording)
    table = pq.read_table(parquet)
    if not states_match(table, archive):
        raise RelabelError(f"state mismatch: {parquet.name} vs {recording.name}")
    histories = episode_histories(parquet, archive, table)
    instruction = episode_instruction(parquet, table, source_tasks)
    prompts = [history_prompt(instruction, history) for history in range(4)]
    for prompt in prompts:
        task_ids.setdefault(prompt, len(task_ids))
    task_indices = np.array([task_ids[prompts[history]] for history in histories], dtype=np.int64)
    return task_index_column(table, task_indices), prompts, histories


def relabel_dataset(args, source: pathlib.Path, destination: pathlib.Path, recordings: list, parquets: list,
                    task_ids: dict) -> tuple:
    """Every episode: the task_index statistics and the prompts by episode index; the parquet files are written unless
    --dry-run."""
    source_tasks = read_tasks(source)
    stats, tasks_of = {}, {}
    started = time.time()
    for position, parquet in enumerate(parquets):
        episode = int(parquet.stem.split("_")[-1])
        table, prompts, histories = relabel_episode(parquet, recordings[args.start + episode], source_tasks, task_ids)
        task_indices = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)
        stats[episode] = task_index_stats(task_indices)
        tasks_of[episode] = prompts
        if not args.dry_run:
            target = destination / parquet.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
        if position == 0 or (position + 1) % 25 == 0:
            print(f"[relabel-hist] {position + 1}/{len(parquets)} ({time.time() - started:.0f}s) {parquet.name}: "
                  f"{table.num_rows} frames, per history state {np.bincount(histories, minlength=4).tolist()}; tasks so far "
                  f"{len(task_ids)}", flush=True)
    return stats, tasks_of


def verify_output(destination: pathlib.Path, episode_count: int) -> dict:
    """Re-read the written meta/ and check every parquet file against it; returns task_index -> prompt."""
    tasks = read_tasks(destination)
    written = episode_files(destination)
    if len(written) != episode_count:
        raise RelabelError("written episode count differs")
    for parquet in written:
        task_indices = np.asarray(pq.read_table(parquet, columns=["task_index"]).column("task_index").to_pylist())
        if not all(int(index) in tasks for index in np.unique(task_indices)):
            raise RelabelError(f"{parquet.name}: task_index outside tasks.jsonl")
    return tasks


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
    task_ids: dict[str, int] = {}   # new task string -> new index (assigned in order of first appearance)
    try:
        stats, tasks_of = relabel_dataset(args, source, destination, recordings, parquets, task_ids)
        if args.dry_run:
            print(f"[relabel-hist] DRY RUN ok: {len(parquets)} episodes verified, {len(task_ids)} distinct prompts")
            return 0
        prompts = [prompt for prompt, _ in sorted(task_ids.items(), key=lambda item: item[1])]
        update_task_meta(source, destination, prompts, tasks_of, stats)
        tasks = verify_output(destination, len(parquets))
    except RelabelError as error:
        return fail(str(error))
    print(f"[relabel-hist] OK: {len(episode_files(destination))} episodes, {len(tasks)} prompts, e.g. {tasks[0]!r} ... {tasks[3]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
