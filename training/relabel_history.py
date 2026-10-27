#!/usr/bin/env python3
"""Relabel a LeRobot dataset of the task's demonstrations with per-frame history prompts: every frame's task becomes

    "<episode instruction> History: <events so far>."

where the events come from the waypoint phase stored in the recordings, with the same phase boundaries as
relabel_subgoals.py, so the history text changes exactly where the sub-goal prompt would change. The phrases, the prefix and
the longest history (`history.max_lines`) come from the task file (--task, default the packaged reference task), the same
renderer the history method uses at run time. In the reference task, a red/blue pick-and-place:

    phases 0-3    History: none yet.
    phases 4-6    History: picked up the red block.                      (after the lift, incl. lowering/release/retreat)
    phases 7-11   History: picked up the red block, placed the red block.
    phases 12-14  History: picked up the red block, placed the red block, picked up the blue block.

The instruction stays the full sentence and the policy is told what it has already done; at run time the history method
produces the same text from a writer's completed skills (vla_memory.methods.history). An episode of a task whose plan varies
records its own plan and the plan position of every frame (see relabel_subgoals.py). Only `task_index`, tasks.jsonl,
episodes.jsonl (tasks), episodes_stats.jsonl (task_index stats) and info.json (total_tasks) change; images, state and
actions are copied unchanged (norm stats unaffected). Integrity check as in relabel_subgoals.py: the parquet `state` column
must equal the backing recording exactly.

    python training/relabel_history.py --src <converted dataset> --dst <new dataset> --raw-dir <recordings> [--start N]
        [--task task.yaml]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # the shared helpers, also when the script is loaded by its path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the framework, when it is not installed
from lerobot_tools import (RelabelError, episode_files, episode_plan, episode_positions, read_tasks, states_match,  # noqa: E402
                           task_index_column, task_index_stats, update_task_meta)
from vla_memory.methods.history.prompt import HistoryPrompt  # noqa: E402
from vla_memory.task import TaskSpec  # noqa: E402
from vla_memory.writers.vlm.labels import parse_ranges  # noqa: E402


def fail(message: str) -> int:
    """Report why the script stops; the exit code 3."""
    print(f"[relabel-hist] ABORT: {message}", file=sys.stderr)
    return 3


def parse_args() -> argparse.Namespace:
    """The source and destination datasets, the recordings behind them and the dry run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--raw-dir", required=True, help="the recorded demonstrations (episode_*.npz)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--task", default=None, help="task YAML/JSON (default: the packaged red_blue_blocks task)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def max_lines(task: TaskSpec) -> int:
    """The longest history of the task's prompts: history.max_lines, else every skill but the last."""
    return int((task.history or {}).get("max_lines", len(task.skills) - 1))


def episode_histories(parquet: pathlib.Path, archive, table, task: TaskSpec) -> tuple[np.ndarray, list[str]]:
    """The number of history lines of every frame of one episode, from the recording's phases or plan positions (the skills
    before the active one, at most max_lines), and the episode's plan; its skills must be visited in order."""
    plan = episode_plan(archive, task.skills, parquet.name)
    positions = episode_positions(archive, parse_ranges((task.demo_phases or {}).get("phase_to_skill", {})), parquet.name,
                                  table.num_rows)
    if positions.min() < 0 or positions.max() >= len(plan):
        raise RelabelError(f"{parquet.name}: plan positions {sorted(set(positions.tolist()))} outside the plan of "
                           f"{len(plan)} skills")
    histories = np.minimum(positions, max_lines(task))
    expected = {min(position, max_lines(task)) for position in range(len(plan))}
    if np.any(np.diff(histories) < 0) or set(histories.tolist()) != expected:
        raise RelabelError(f"{parquet.name}: history states not visited in order {sorted(expected)}: "
                           f"{np.unique(histories).tolist()}")
    return histories, plan


def episode_instruction(parquet: pathlib.Path, table, source_tasks: dict) -> str:
    """The one instruction of an episode (its frames' task before the relabelling)."""
    old_indices = np.asarray(table.column("task_index").to_pylist())
    instructions = {source_tasks[int(index)] for index in np.unique(old_indices)}
    if len(instructions) != 1:
        raise RelabelError(f"{parquet.name}: expected one instruction per episode, got {instructions}")
    return instructions.pop()


def relabel_episode(parquet: pathlib.Path, recording: pathlib.Path, source_tasks: dict, task_ids: dict, task: TaskSpec,
                    history: HistoryPrompt):
    """One episode: its table with the new task_index column, its prompts (one per history state) and the history state of
    every frame. task_ids (prompt -> new task index) grows with the prompts first seen here."""
    archive = np.load(recording)
    table = pq.read_table(parquet)
    if not states_match(table, archive):
        raise RelabelError(f"state mismatch: {parquet.name} vs {recording.name}")
    histories, plan = episode_histories(parquet, archive, table, task)
    instruction = episode_instruction(parquet, table, source_tasks)
    prompts = [history.render(instruction, plan[:count]) for count in range(int(histories.max()) + 1)]
    for prompt in prompts:
        task_ids.setdefault(prompt, len(task_ids))
    task_indices = np.array([task_ids[prompts[count]] for count in histories], dtype=np.int64)
    return task_index_column(table, task_indices), prompts, histories


def relabel_dataset(args, task: TaskSpec, source: pathlib.Path, destination: pathlib.Path, recordings: list,
                    parquets: list, task_ids: dict) -> tuple:
    """Every episode: the task_index statistics and the prompts by episode index; the parquet files are written unless
    --dry-run."""
    source_tasks = read_tasks(source)
    history = HistoryPrompt(task)
    stats, tasks_of = {}, {}
    started = time.time()
    for position, parquet in enumerate(parquets):
        episode = int(parquet.stem.split("_")[-1])
        table, prompts, histories = relabel_episode(parquet, recordings[args.start + episode], source_tasks, task_ids, task,
                                                    history)
        task_indices = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)
        stats[episode] = task_index_stats(task_indices)
        tasks_of[episode] = prompts
        if not args.dry_run:
            target = destination / parquet.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
        if position == 0 or (position + 1) % 25 == 0:
            per_state = np.bincount(histories, minlength=max_lines(task) + 1).tolist()
            print(f"[relabel-hist] {position + 1}/{len(parquets)} ({time.time() - started:.0f}s) {parquet.name}: "
                  f"{table.num_rows} frames, per history state {per_state}; tasks so far {len(task_ids)}", flush=True)
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
    task = TaskSpec.load_or_default(args.task)
    recordings = sorted(pathlib.Path(args.raw_dir).glob("episode_*.npz"))
    parquets = episode_files(source)
    if not parquets:
        return fail(f"no parquet files under {source}/data")
    if destination.exists() and not args.dry_run:
        return fail(f"{destination} exists; refusing to overwrite")
    task_ids: dict[str, int] = {}   # new task string -> new index (assigned in order of first appearance)
    try:
        stats, tasks_of = relabel_dataset(args, task, source, destination, recordings, parquets, task_ids)
        if args.dry_run:
            print(f"[relabel-hist] DRY RUN ok: {len(parquets)} episodes verified, {len(task_ids)} distinct prompts")
            return 0
        prompts = [prompt for prompt, _ in sorted(task_ids.items(), key=lambda item: item[1])]
        update_task_meta(source, destination, prompts, tasks_of, stats)
        tasks = verify_output(destination, len(parquets))
    except RelabelError as error:
        return fail(str(error))
    print(f"[relabel-hist] OK: {len(episode_files(destination))} episodes, {len(tasks)} prompts, e.g. {tasks[0]!r} ... "
          f"{tasks[min(3, len(tasks) - 1)]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
