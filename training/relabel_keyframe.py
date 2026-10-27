#!/usr/bin/env python3
"""The training data of the keyframe policy: the memoryless policy's frames and the wrist picture of the last key moment.

Input, per split: two LeRobot datasets (v2.1, the cameras stored as images) of the same recorded episodes, frame by frame:

    --train FRAMES:EVENTS   the training split (--val for the validation split)
        FRAMES   the memoryless policy's dataset: the frame columns image, wrist_image, state, actions and the plain
                 instruction as each frame's task (convert_to_lerobot.py --no-videos)
        EVENTS   the history policy's dataset of the same frames (relabel_history.py): its prompts say how many events
                 (history lines) have happened at every frame

The two must hold the same frames (both cameras' PNG bytes, state, actions, timestamps and indices are compared). The
history count of a frame gives its event; every frame after the first event gets the wrist picture of a frame drawn from the
16 frames after that event, never later than the frame itself (vla_memory.methods.keyframe.labels); a frame before the first
event gets an all-zero picture.

Output: <root>/<repo-id>_train and <root>/<repo-id>_val with every frame of FRAMES, its plain instruction as the task, and

    memory_image   the keyframe (stored like wrist_image; all zero: no keyframe yet)

plus keyframe_frames.csv in each dataset folder: per frame its event, the event's first frame and the source frame of the
keyframe (-1: none). Run where lerobot is installed (the openpi environment).

    python training/relabel_keyframe.py --train <frames train>:<history train> --val <frames val>:<history val> \
        --root <dataset root> --repo-id <user>/<name>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # the shared helpers, also when the script is loaded by its path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the framework, when it is not installed
from lerobot_tools import (create_memory_dataset, decode_png, episode_files, output_dataset, read_info, read_tasks,  # noqa: E402
                           write_manifest)
from vla_memory.methods.history.prompt import HistoryPrompt  # noqa: E402
from vla_memory.methods.keyframe import labels  # noqa: E402
from vla_memory.task import TaskSpec  # noqa: E402

COLUMNS = ("image", "wrist_image", "state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index")
MANIFEST = "keyframe_frames.csv"
MANIFEST_COLUMNS = ("episode_index", "frame_index", "event", "boundary", "source_frame")


# ----------------------------------------------------------------------------------------------------------- reading
def dataset_pair(text: str) -> tuple[pathlib.Path, pathlib.Path]:
    """'FRAMES:EVENTS' -> (frames dataset, events dataset)."""
    frames, _, events = text.rpartition(":")
    if not frames or not events:
        raise argparse.ArgumentTypeError(f"{text!r}: expected FRAMES:EVENTS")
    return pathlib.Path(frames).expanduser(), pathlib.Path(events).expanduser()


def read_episode(path: pathlib.Path) -> dict:
    """One episode's columns: the PNG bytes of both cameras, state, actions, timestamps and indices."""
    table = pq.read_table(path, columns=list(COLUMNS))
    episode = {name: [cell["bytes"] for cell in table.column(name).to_pylist()] for name in ("image", "wrist_image")}
    for name in COLUMNS[2:]:
        episode[name] = np.stack(table.column(name).to_numpy(zero_copy_only=False))
    return episode


def first_difference(frames: dict, events: dict) -> str | None:
    """The first column in which the two episodes differ (the task aside), or None."""
    for name in COLUMNS[:-1]:
        left, right = frames[name], events[name]
        if isinstance(left, list):
            same = left == right
        else:
            same = left.shape == right.shape and np.array_equal(left, right)
        if not same:
            return name
    return None


# ----------------------------------------------------------------------------------------------------------- writing
def write_episode(dataset, episode: dict, prompts: list[str], rows: list[tuple]) -> None:
    """Add one episode with its keyframes to the dataset."""
    wrists = [decode_png(data) for data in episode["wrist_image"]]
    empty = np.zeros_like(wrists[0])
    for frame, (_, _, source) in enumerate(rows):
        dataset.add_frame({"image": decode_png(episode["image"][frame]), "wrist_image": wrists[frame],
                           "memory_image": empty if source < 0 else wrists[source], "state": episode["state"][frame],
                           "actions": episode["actions"][frame], "task": prompts[frame]})
    dataset.save_episode()


def label_pair(split: str, index: int, frames_path: pathlib.Path, events_path: pathlib.Path, tasks: tuple, history,
               events: int) -> tuple[dict, list[str], list[tuple]]:
    """One episode: its frames, its plain prompts and its keyframe labels, after checking that both datasets hold it."""
    frames, history_episode = read_episode(frames_path), read_episode(events_path)
    difference = first_difference(frames, history_episode)
    if difference is not None:
        raise SystemExit(f"{split} {frames_path.name}: column {difference} differs between the two datasets")
    episode = int(frames["episode_index"][0])
    if episode != index:
        raise SystemExit(f"{split} {frames_path.name}: episode index {episode}, expected {index}")
    prompts = [tasks[0][int(task_index)] for task_index in frames["task_index"].tolist()]
    with_history = [tasks[1][int(task_index)] for task_index in history_episode["task_index"].tolist()]
    counts = [labels.history_counts([text], prompt, history)[0] for prompt, text in zip(prompts, with_history)]
    edges = labels.boundaries(counts, events)
    return frames, prompts, labels.label_episode(split, episode, len(prompts), edges)


def write_split(split: str, pair: tuple, args: argparse.Namespace, history, events: int) -> dict:
    """One output dataset and its manifest; returns its counts."""
    frames_root, events_root = pair
    files = episode_files(frames_root)[:args.limit]
    if [file.name for file in files] != [file.name for file in episode_files(events_root)[:args.limit]]:
        raise SystemExit(f"{split}: the two datasets have different episode files")
    repo_id, path = output_dataset(args, split)
    dataset = create_memory_dataset(repo_id, path, read_info(frames_root))
    tasks = (read_tasks(frames_root), read_tasks(events_root))
    manifest = []
    for index, frames_path in enumerate(files):
        events_path = events_root / frames_path.relative_to(frames_root)
        frames, prompts, rows = label_pair(split, index, frames_path, events_path, tasks, history, events)
        write_episode(dataset, frames, prompts, rows)
        manifest += [[index, frame, *row] for frame, row in enumerate(rows)]
        print(f"[relabel_keyframe] {repo_id}: episode {index}, {len(manifest)} frames", flush=True)
    dataset.stop_image_writer()
    write_manifest(path / MANIFEST, MANIFEST_COLUMNS, manifest)
    per_event = {str(event): sum(1 for row in manifest if row[2] == event) for event in range(events + 1)}
    return {"episodes": len(files), "frames": len(manifest), "frames_per_event": per_event}


# ------------------------------------------------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    """The two datasets per split, the output folder and name, the task file and the smoke-test limit."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=dataset_pair, required=True, metavar="FRAMES:EVENTS", help="the training split")
    parser.add_argument("--val", type=dataset_pair, default=None, metavar="FRAMES:EVENTS", help="the validation split")
    parser.add_argument("--root", type=pathlib.Path, required=True, help="the folder the new datasets go into")
    parser.add_argument("--repo-id", required=True, help="the new datasets' name: <repo-id>_train and <repo-id>_val")
    parser.add_argument("--task", default=None, help="task YAML/JSON (default: the packaged red_blue_blocks task)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N episodes of each split (a smoke test)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Write the keyframe datasets of the given splits and print their counts."""
    args = parse_args(argv)
    splits = {split: pair for split, pair in (("train", args.train), ("val", args.val)) if pair is not None}
    for split in splits:
        if output_dataset(args, split)[1].exists():
            raise SystemExit(f"{output_dataset(args, split)[1]} exists; refusing to overwrite")
    task = TaskSpec.load_or_default(args.task)
    history = HistoryPrompt(task)
    events = int((task.history or {}).get("max_lines", len(task.skills) - 1))
    summary = {split: write_split(split, pair, args, history, events) for split, pair in splits.items()}
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
