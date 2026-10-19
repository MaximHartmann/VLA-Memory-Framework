#!/usr/bin/env python3
"""Make the training data of the memory policy: the memory store and the datasets with a retrieved memory per frame.

Input: LeRobot datasets (v2.1, the cameras stored as images, not video) with the frame columns image, wrist_image, state
[8] and actions [8] and the skill as each frame's task (convert_to_lerobot.py --no-videos, then relabel_subgoals.py).
Each dataset is named with a source label, which its store entries keep as provenance:

    --train PATH:LABEL   a training dataset (repeatable); the store is built from these only
    --val PATH:LABEL     a validation dataset (repeatable)

Output: the store under --store (entries.jsonl, keys_fingerprint.npy, wrist/NNNNN.png, meta.json, chunks.npz) and the
datasets <root>/<repo-id>_train (all training datasets, in the order given) and <root>/<repo-id>_val, with every frame of
the input and

    memory_image   the wrist picture of the frame's nearest stored moment of ANOTHER episode (all-zero: no memory)
    task           "<skill>. memory: <sentence>", or the bare skill for a frame without memory

plus memory_frames.csv in each dataset folder: per frame the retrieved entry, its similarity and rule, and the prompt.
Building the store and labelling the frames is vla_memory.methods.memory_policy.build; this script reads, calls it and
writes. The finger width comes from the dataset's state (finger_width: the red/blue convention). The task file (--task,
default the packaged red/blue task) gives the colours of the retrieval key and the skill list. Run where lerobot is installed
(the openpi environment) with the framework on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

from vla_memory.methods.memory_policy import build
from vla_memory.methods.memory_policy.gripper import gripper_state
from vla_memory.methods.memory_policy.key import WristKey
from vla_memory.methods.memory_policy.pictures import Pictures
from vla_memory.methods.memory_policy.store import MomentStore, save_store
from vla_memory.task import TaskSpec

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # the shared helpers, also when the script is loaded by its path
from lerobot_tools import create_memory_dataset, decode_png, episode_files, output_dataset, read_info, read_tasks, write_manifest  # noqa: E402

FINGER_SCALE = 0.04          # m: the red/blue state's last value is 1 - (one finger's opening) / FINGER_SCALE
PICTURE_CACHE = 2048         # decoded store pictures kept while writing (230 KB each)
MANIFEST = "memory_frames.csv"
MANIFEST_COLUMNS = ("episode_index", "frame_index", "episode", "source", "skill", "width", "gripper", "has_memory",
                    "neighbour", "neighbour_episode", "neighbour_source", "similarity", "rule", "prompt")
HAS_MEMORY = MANIFEST_COLUMNS.index("has_memory")


def finger_width(state: np.ndarray) -> np.ndarray:
    """The measured finger width (m, both fingers together) from the red/blue state, whose last value is
    1 - (one finger's opening) / 0.04: width = 2 * 0.04 * (1 - state[..., 7]). Another robot needs its own rule here."""
    return (2.0 * FINGER_SCALE * (1.0 - state[..., 7])).astype(np.float32)


@dataclasses.dataclass
class Memory:
    """What the frames are labelled against: the saved store, its pictures and the key it was built with."""

    store: MomentStore
    pictures: Pictures
    wrist_key: WristKey


# ----------------------------------------------------------------------------------------------------------- reading
def dataset_argument(text: str) -> tuple[pathlib.Path, str]:
    """'PATH:LABEL' -> (path, label)."""
    path, _, label = text.rpartition(":")
    if not path or not label:
        raise argparse.ArgumentTypeError(f"{text!r}: expected PATH:LABEL")
    return pathlib.Path(path).expanduser(), label


def column(table, name: str) -> np.ndarray:
    """A vector column of a parquet table as a float32 array (frames, size)."""
    return np.stack(table.column(name).to_numpy(zero_copy_only=False)).astype(np.float32)


def read_episode(path: pathlib.Path, tasks: dict[int, str]) -> dict:
    """One episode file: its index, the PNG bytes of both cameras, state, actions and the skill of every frame."""
    table = pq.read_table(path, columns=["image", "wrist_image", "state", "actions", "task_index"])
    return {"index": int(path.stem.split("_")[-1]),
            "image_png": [cell["bytes"] for cell in table.column("image").to_pylist()],
            "wrist_png": [cell["bytes"] for cell in table.column("wrist_image").to_pylist()],
            "state": column(table, "state"),
            "actions": column(table, "actions"),
            "skills": [tasks[int(index)] for index in table.column("task_index").to_numpy()]}


def episodes(root: pathlib.Path, label: str, split: str):
    """(build.Episode, the raw episode) of every episode of one dataset, in file order."""
    tasks = read_tasks(root)
    for path in episode_files(root):
        raw = read_episode(path, tasks)
        episode = build.Episode(episode_id=f"{label}_{split}_{raw['index']:06d}", source=label,
                                joints=raw["state"][:, :7], actions=raw["actions"],
                                widths=finger_width(raw["state"]), skills=raw["skills"])
        yield episode, raw


# ------------------------------------------------------------------------------------------------------- the store
def make_store(train: list, store_dir: pathlib.Path, wrist_key: WristKey, skills) -> int:
    """Build the store from the training datasets and save it; returns the number of entries."""
    parts = build.StoreParts()
    for root, label in train:
        for episode, raw in episodes(root, label, "train"):
            wrist = {step: decode_png(raw["wrist_png"][step]) for step in build.call_moments(len(episode))}
            parts.extend(build.episode_entries(episode, wrist, wrist_key, skills))
    meta = build.store_meta(parts.entries, parts.keys)
    save_store(store_dir, parts.entries, parts.keys, parts.pngs, meta, q_now=parts.q_now, chunk=parts.chunks)
    return len(parts.entries)


# ---------------------------------------------------------------------------------------------------------- writing
def write_episode(dataset, episode: build.Episode, raw: dict, memory: Memory) -> list:
    """Label every frame of one episode and add the episode to the dataset; returns its manifest rows."""
    wrist = [decode_png(data) for data in raw["wrist_png"]]
    labels = build.label_episode(memory.store, episode, wrist, memory.wrist_key)
    index = dataset.num_episodes
    zeros = np.zeros(dataset.features["memory_image"]["shape"], np.uint8)
    for step, label in enumerate(labels):
        picture = memory.pictures[label.neighbour] if label.neighbour >= 0 else zeros
        dataset.add_frame({"image": decode_png(raw["image_png"][step]), "wrist_image": wrist[step],
                           "memory_image": picture, "state": raw["state"][step], "actions": raw["actions"][step],
                           "task": label.prompt})
    dataset.save_episode()
    return [manifest_row(index, step, episode, label, memory.store) for step, label in enumerate(labels)]


def manifest_row(episode_index: int, step: int, episode: build.Episode, label: build.FrameLabel,
                 store: MomentStore) -> list:
    """One line of memory_frames.csv."""
    width = float(episode.widths[step])
    has_memory = label.neighbour >= 0
    neighbour_episode = str(store.episode[label.neighbour]) if has_memory else ""
    neighbour_source = str(store.source[label.neighbour]) if has_memory else ""
    similarity = f"{label.similarity:.4f}" if has_memory else ""
    return [episode_index, step, episode.episode_id, episode.source, episode.skills[step], f"{width:.4f}",
            gripper_state(width), int(has_memory), label.neighbour, neighbour_episode, neighbour_source, similarity,
            label.rule, label.prompt]


def write_split(split: str, datasets: list, memory: Memory, args: argparse.Namespace) -> dict:
    """One output dataset (and its manifest) from the input datasets of a split; returns its counts."""
    repo_id, path = output_dataset(args, split)
    dataset = create_memory_dataset(repo_id, path, read_info(datasets[0][0]))
    rows = []
    for root, label in datasets:
        for episode, raw in episodes(root, label, split):
            rows.extend(write_episode(dataset, episode, raw, memory))
            print(f"[relabel] {repo_id}: {episode.episode_id}, {len(rows)} frames", flush=True)
    dataset.stop_image_writer()
    write_manifest(path / MANIFEST, MANIFEST_COLUMNS, rows)
    without = sum(1 for row in rows if not row[HAS_MEMORY])
    return {"episodes": dataset.num_episodes, "frames": len(rows), "no_memory": without}


# ------------------------------------------------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    """The input datasets with their labels, the store, the output folder and name, and the task file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=dataset_argument, action="append", required=True, metavar="PATH:LABEL",
                        help="a training dataset and its source label (repeatable)")
    parser.add_argument("--val", type=dataset_argument, action="append", default=[], metavar="PATH:LABEL",
                        help="a validation dataset and its source label (repeatable)")
    parser.add_argument("--store", type=pathlib.Path, required=True, help="the store directory to write")
    parser.add_argument("--root", type=pathlib.Path, required=True, help="the folder the new datasets go into")
    parser.add_argument("--repo-id", required=True, help="the new datasets' name: <repo-id>_train and <repo-id>_val")
    parser.add_argument("--task", default=None, help="task YAML/JSON (default: the packaged red_blue_blocks task)")
    return parser.parse_args(argv)


def splits(args: argparse.Namespace) -> dict:
    """split -> its input datasets, for the splits that have inputs."""
    return {split: datasets for split, datasets in (("train", args.train), ("val", args.val)) if datasets}


def check_arguments(args: argparse.Namespace) -> None:
    """Unique source labels per split (they make the episode ids); no output dataset and no store there already."""
    for split, datasets in splits(args).items():
        labels = [label for _, label in datasets]
        if len(set(labels)) != len(labels):
            raise SystemExit(f"--{split}: every dataset needs its own source label, got {labels}")
        if output_dataset(args, split)[1].exists():
            raise SystemExit(f"{output_dataset(args, split)[1]} exists; refusing to overwrite")
    if args.store.exists() and any(args.store.iterdir()):
        raise SystemExit(f"{args.store} is not empty; refusing to overwrite")


def main(argv=None) -> int:
    """Build the store from the training datasets, then write every split with its memories; print the counts."""
    args = parse_args(argv)
    check_arguments(args)
    task = TaskSpec.load_or_default(args.task)
    wrist_key = WristKey(task)
    summary = {"store_entries": make_store(args.train, args.store, wrist_key, task.skills)}
    print(f"[relabel] store: {summary['store_entries']} entries in {args.store}", flush=True)
    store = MomentStore.load(args.store)
    memory = Memory(store, Pictures(store.pictures, PICTURE_CACHE), wrist_key)
    for split, datasets in splits(args).items():
        summary[split] = write_split(split, datasets, memory, args)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
