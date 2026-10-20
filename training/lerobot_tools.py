"""Shared helpers of the relabelling scripts: the files of a LeRobot dataset (v2.1) as they read and write them.

    read_jsonl, write_jsonl        the JSON-lines files of meta/
    read_info, read_tasks          meta/info.json (the cameras must be stored as images) and task_index -> task
    episode_files                  the parquet file of every episode, in order
    decode_png                     a PNG cell of a parquet table as a uint8 RGB array
    feature                        a column of info.json as LeRobotDataset.create takes it
    create_memory_dataset          an empty dataset with the input's columns plus memory_image
    write_manifest                 a per-frame CSV next to a written dataset
    output_dataset                 the repo id and the folder of a split's output dataset
    states_match                   whether an episode's parquet state column equals the backing recording
    episode_plan, episode_positions   a recorded episode's skills in order, and the position in them of the skill of every frame
    RelabelError                   a dataset the relabelling scripts cannot relabel (message printed, exit code 3)
    task_index_column, task_index_stats, update_task_meta
                                   a relabelled task_index column, its statistics, and the meta/ files that describe it

Run where lerobot and pyarrow are installed (the openpi environment).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import shutil

import numpy as np
import pyarrow as pa
from PIL import Image


class RelabelError(Exception):
    """A dataset that cannot be relabelled; its message is printed and the script exits with code 3."""


def read_jsonl(path) -> list[dict]:
    """The records of a JSON-lines file (blank lines skipped)."""
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows) -> None:
    """One JSON object per line."""
    pathlib.Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_info(root: pathlib.Path) -> dict:
    """meta/info.json of a dataset; the two cameras must be stored as images."""
    info = json.loads((root / "meta/info.json").read_text())
    for name in ("image", "wrist_image"):
        if info["features"][name]["dtype"] != "image":
            raise SystemExit(f"{root}: {name} is stored as {info['features'][name]['dtype']}; convert with --no-videos")
    return info


def read_tasks(root: pathlib.Path) -> dict[int, str]:
    """task_index -> task string of a dataset (meta/tasks.jsonl)."""
    return {int(row["task_index"]): row["task"] for row in read_jsonl(root / "meta/tasks.jsonl")}


def episode_files(root: pathlib.Path) -> list[pathlib.Path]:
    """The parquet file of every episode of a dataset, in order."""
    return sorted((root / "data").rglob("episode_*.parquet"))


def decode_png(data: bytes) -> np.ndarray:
    """A PNG cell of a parquet table as a uint8 RGB array."""
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def feature(spec: dict) -> dict:
    """A column of info.json as LeRobotDataset.create takes it (the shape as a tuple)."""
    return {**spec, "shape": tuple(spec["shape"])}


def create_memory_dataset(repo_id: str, path: pathlib.Path, info: dict):
    """An empty LeRobot dataset with the input's columns plus memory_image (stored like the wrist image)."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    features = {name: feature(info["features"][name]) for name in ("image", "wrist_image", "state", "actions")}
    features["memory_image"] = feature(info["features"]["wrist_image"])
    return LeRobotDataset.create(repo_id=repo_id, root=path, fps=info["fps"], robot_type=info["robot_type"],
                                 features=features, use_videos=False, image_writer_processes=4, image_writer_threads=8)


def write_manifest(path: pathlib.Path, columns, rows: list) -> None:
    """A CSV with a header row."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def output_dataset(args: argparse.Namespace, split: str) -> tuple[str, pathlib.Path]:
    """The repo id and the folder of a split's output dataset: <repo-id>_<split> under --root."""
    repo_id = f"{args.repo_id}_{split}"
    return repo_id, args.root / repo_id


def states_match(table, archive) -> bool:
    """Whether a parquet table's state column equals the backing recording's state array (shape and values)."""
    state = np.asarray(table.column("state").to_pylist(), dtype=np.float32)
    return state.shape == archive["state"].shape and bool(np.allclose(state, archive["state"], atol=1e-6))


def episode_plan(archive, task_skills, name: str) -> list[str]:
    """The skills of a recorded episode in order: its own `plan` (skill strings, all of them skills of the task) when the
    recording has one, as an episode of a task whose plan varies does, else the task's skills in their order."""
    if "plan" not in archive.files:
        return list(task_skills)
    plan = [str(skill) for skill in archive["plan"]]
    unknown = [skill for skill in plan if skill not in task_skills]
    if unknown:
        raise RelabelError(f"{name}: plan skills {unknown} are not skills of the task")
    return plan


def episode_positions(archive, phase_to_skill: dict, name: str, rows: int) -> np.ndarray:
    """The position in the episode's plan of the skill active at every frame (int64, `rows` frames), as the recording gives
    it: `skill_index` for a task whose plan varies, else the waypoint `phase` of each frame mapped by phase_to_skill
    ({phase: position}, the task file's demo_phases.phase_to_skill)."""
    if "skill_index" in archive.files:
        positions = np.asarray(archive["skill_index"]).astype(np.int64)
        if len(positions) != rows:
            raise RelabelError(f"{name}: {rows} rows but {len(positions)} plan positions")
        return positions
    if "phase" not in archive.files or not phase_to_skill:
        raise RelabelError(f"{name}: the recording has no `skill_index`, and either it has no `phase` or the task file has no "
                           f"demo_phases.phase_to_skill")
    phase = archive["phase"].astype(int)
    if len(phase) != rows:
        raise RelabelError(f"{name}: {rows} rows but {len(phase)} phase labels")
    unknown = set(np.unique(phase).tolist()) - set(phase_to_skill)
    if unknown:
        raise RelabelError(f"{name}: phases {sorted(unknown)} have no subgoal")
    return np.array([phase_to_skill[value] for value in phase], dtype=np.int64)


def task_index_column(table, task_indices: np.ndarray):
    """The table with its task_index column replaced."""
    column = pa.array(task_indices, type=table.schema.field("task_index").type)
    return table.set_column(table.schema.get_field_index("task_index"), "task_index", column)


def task_index_stats(task_indices: np.ndarray) -> dict:
    """The task_index statistics of one episode, as episodes_stats.jsonl holds them."""
    return {"min": [int(task_indices.min())], "max": [int(task_indices.max())], "mean": [float(task_indices.mean())],
            "std": [float(task_indices.std())], "count": [int(task_indices.size)]}


def update_task_meta(source: pathlib.Path, destination: pathlib.Path, tasks: list[str], tasks_of: dict, stats: dict) -> pathlib.Path:
    """Copy meta/ from the source dataset and rewrite what a new task_index column changes: tasks.jsonl (the task strings
    in index order), each episode's tasks in episodes.jsonl, its task_index statistics in episodes_stats.jsonl and
    total_tasks in info.json. Returns the meta directory."""
    meta = destination / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for file in (source / "meta").glob("*"):
        shutil.copy2(file, meta / file.name)
    write_jsonl(meta / "tasks.jsonl", [{"task_index": index, "task": task} for index, task in enumerate(tasks)])
    episodes = read_jsonl(meta / "episodes.jsonl")
    for record in episodes:
        record["tasks"] = tasks_of[record["episode_index"]]
    write_jsonl(meta / "episodes.jsonl", episodes)
    episode_stats = read_jsonl(meta / "episodes_stats.jsonl")
    for record in episode_stats:
        record["stats"]["task_index"] = stats[record["episode_index"]]
    write_jsonl(meta / "episodes_stats.jsonl", episode_stats)
    info = json.loads((meta / "info.json").read_text())
    info["total_tasks"] = len(tasks)
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    return meta
