#!/usr/bin/env python3
"""Recorded episodes (.npz) -> a LeRobot dataset openpi can train on.

Run it in the openpi environment (lerobot is installed there, Isaac Sim is not). The schema follows openpi's own LIBERO
converter: the proprioception in `state`, the actions in `actions`, images with the dtype `image`. Images are stored at their
native size (320 x 240 in the reference recordings); the resize to 224 x 224 happens in the training transform, exactly as at
inference, so that it is one shared step rather than two that can drift apart.

--action-space absolute converts the recorded velocity actions of the arm, (q_cmd - q_meas) / dt, back to the absolute joint
targets q_cmd = q_meas + action * dt. The inversion is exact, because a recording stores q_meas (state[:, :7]) and its fps.
Why absolute targets: the velocity targets are spiky (near zero while a waypoint is held, rare large bursts) and their spread
differs 69-fold across the seven joints, while the absolute targets are smooth and close to the state, which a policy learns
more reliably. The gripper command (index 7) is absolute in [0, 1] already and is never changed.

    python training/convert_to_lerobot.py --raw-dir <recordings> --root <dataset root> [--repo-id <user>/<name>]
        [--no-videos] [--action-space absolute] [--start N] [--limit N] [--overwrite]
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

import numpy as np


def parse_args() -> argparse.Namespace:
    """The recordings, the dataset to write, the frame rate, the image encoding and the action space."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="the folder of the staged .npz episodes")
    parser.add_argument("--repo-id", default="vla_memory/red_blue_blocks")
    parser.add_argument("--root", required=True, help="the LeRobot root folder the dataset is written to")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--use-videos", action="store_true", default=True,
                        help="MP4 per episode instead of PNG per frame; far kinder to a network file system")
    parser.add_argument("--no-videos", dest="use_videos", action="store_false")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete an existing --root folder, with everything in it, before writing")
    parser.add_argument("--start", type=int, default=0, help="first staged episode index (for train/val splits)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--action-space", choices=["velocity", "absolute"], default="velocity",
                        help="velocity: actions exactly as recorded, (q_cmd - q_meas)/dt.  "
                             "absolute: arm actions converted to absolute joint targets q_cmd. "
                             "The gripper (index 7) is ALREADY absolute in [0,1] and is never "
                             "touched by either mode.")
    return parser.parse_args()


def to_absolute(episode) -> np.ndarray:
    """The arm actions as absolute joint targets, q_cmd = q_meas + velocity * dt; the gripper command stays."""
    dt = 1.0 / float(episode["fps"])
    actions = np.array(episode["actions"], dtype=np.float32, copy=True)
    actions[:, :7] = episode["state"][:, :7] + actions[:, :7] * dt
    return actions


def episode_actions(episode, action_space: str) -> np.ndarray:
    """The actions of an episode in the chosen action space."""
    if action_space == "absolute":
        return to_absolute(episode)
    return episode["actions"]


def describe_input(probe, args) -> dict:
    """Print the shapes of the recordings (and the spread of the action targets); returns the shapes."""
    height, width = probe["exterior"].shape[1:3]
    shapes = {"height": height, "width": width, "state_dim": probe["state"].shape[1], "action_dim": probe["actions"].shape[1]}
    print(f"[conv] image {height}x{width}  state {shapes['state_dim']}  actions {shapes['action_dim']}", flush=True)
    print(f"[conv] action space: {args.action_space}", flush=True)
    if args.action_space == "absolute":
        absolute = to_absolute(probe)
        print(f"[conv]   arm target std per-dim: {np.round(absolute[:, :7].std(0), 4)}", flush=True)
        print(f"[conv]   (velocity was          : {np.round(probe['actions'][:, :7].std(0), 4)})", flush=True)
    return shapes


def create_dataset(args, root: pathlib.Path, shapes: dict):
    """An empty LeRobot dataset with the two cameras, the state and the actions."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    image = {"dtype": "image", "shape": (shapes["height"], shapes["width"], 3), "names": ["height", "width", "channel"]}
    features = {"image": image, "wrist_image": dict(image),
                "state": {"dtype": "float32", "shape": (shapes["state_dim"],), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (shapes["action_dim"],), "names": ["actions"]}}
    return LeRobotDataset.create(repo_id=args.repo_id, root=str(root), fps=args.fps, robot_type="franka_fer",
                                 use_videos=args.use_videos, features=features, image_writer_threads=8,
                                 image_writer_processes=4)


def add_episode(dataset, file: pathlib.Path, action_space: str) -> int:
    """Add every frame of one recording to the dataset; returns the number of frames."""
    episode = np.load(file, allow_pickle=True)
    actions = episode_actions(episode, action_space)
    task = str(episode["instruction"])
    frames = episode["state"].shape[0]
    for frame in range(frames):
        dataset.add_frame({"image": episode["exterior"][frame], "wrist_image": episode["wrist"][frame],
                           "state": episode["state"][frame].astype(np.float32),
                           "actions": actions[frame].astype(np.float32), "task": task})
    dataset.save_episode()
    return frames


def report_size(root: pathlib.Path, use_videos: bool) -> None:
    """Print the number of files and the size of the written dataset."""
    files = [path for path in root.rglob("*") if path.is_file()]
    size = sum(path.stat().st_size for path in files)
    print(f"[conv] on disk: {len(files)} files, {size / 1e9:.2f} GB ({'video' if use_videos else 'PNG'} encoding)", flush=True)


def main() -> int:
    """Convert the recordings of --raw-dir (from --start, at most --limit) into the dataset at --root."""
    args = parse_args()
    raw = pathlib.Path(args.raw_dir)
    files = sorted(raw.glob("episode_*.npz"))[args.start:]
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no episodes in {raw}")
        return 2
    print(f"[conv] {len(files)} staged episodes from {raw}", flush=True)
    shapes = describe_input(np.load(files[0], allow_pickle=True), args)
    root = pathlib.Path(args.root)
    if root.exists():
        if not args.overwrite:
            print(f"[conv] {root} exists; pass --overwrite")
            return 2
        shutil.rmtree(root)
    dataset = create_dataset(args, root, shapes)
    total = 0
    for file in files:
        frames = add_episode(dataset, file, args.action_space)
        total += frames
        print(f"[conv] {file.name}: {frames} frames  (total {total})", flush=True)
    print(f"\n[conv] wrote {len(files)} episodes / {total} frames to {root}", flush=True)
    report_size(root, args.use_videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
