#!/usr/bin/env python3
"""Fit the event head of a history policy from recorded rollouts of that policy.

Input: one .npz file per recorded episode of the history policy (the policy's own rollouts, not demonstrations), with

    exterior     (T, H, W, 3) uint8   the exterior camera at every control step, as the environment sends it
    wrist        (T, H, W, 3) uint8   the wrist camera
    instruction  str                  the instruction of the episode
    reference    (T,) int             the number of completed skills at every step, as the history at a policy call there
                                      shows it; or, instead,
    rows         (T, 5) float         finger width, squeeze, fingertip x, y, z at every step (memory/proprio_steps), which
                                      the task's rules writer labels: reference[t] = the lines it wrote after steps 0..t-1
    state        (T, D) float         optional: the policy's state input (default zeros of --state-dim)

The steps:
1. rows: the training rows of every episode (vla_memory.writers.head.rows), every 4th frame;
2. features (the openpi environment and one GPU): the policy, loaded as the head server loads it, computes the feature of
   every row's frames and prompt (head_features); they are saved to --features, and a later run with an existing
   --features file goes straight to the fit, without openpi and without a GPU;
3. the fit (CPU): head_fit.fit_head; the head is saved to --out (head_file: head.npz, head.json, head.sha256).

    python training/fit_event_head.py --episodes 'rollouts/*.npz' --config <policy config> \
        --checkpoint <checkpoint dir> --center-crop 0.95 --features rows.npz --out <head dir>

Images are letterboxed to --image-size before the policy's input transform, as the reference environment sends them. With
--jax-cache the compiled kernels are kept; a head server started on a copy of that cache computes the features with the same
kernels, so that its scores equal the training numerics (head_features.py).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import pathlib
import sys

import numpy as np

from vla_memory.methods.history.prompt import HistoryPrompt
from vla_memory.writers.head import fit as head_fit
from vla_memory.writers.head import rows as head_rows
from vla_memory.writers.head.file import save_head
from vla_memory.writers.rules.rules import RuleMonitor
from vla_memory.writers.rules.validate import episode_signals
from vla_memory.task import TaskSpec


# ---------------------------------------------------------------------- episodes and rows
def episode_files(patterns) -> list[pathlib.Path]:
    """The episode files matching the globs, sorted and without duplicates."""
    paths = {path for pattern in patterns for path in glob.glob(str(pathlib.Path(pattern).expanduser()))}
    files = sorted(pathlib.Path(path) for path in paths)
    if not files:
        raise SystemExit(f"no episode files match {patterns}")
    return files


def reference_of(episode, task: TaskSpec) -> np.ndarray:
    """The episode's reference count, given or labelled from its signal rows by the task's rules writer."""
    if "reference" in episode.files:
        return np.asarray(episode["reference"], int)
    if "rows" not in episode.files:
        raise SystemExit("an episode file needs `reference` or `rows`")
    lines = RuleMonitor(task).label_rows(episode["rows"], signals=episode_signals(episode) or None)
    return head_rows.reference_from_lines(lines)


def episode_rows(files, task: TaskSpec, k_max: int) -> list[list]:
    """The rows of every episode, [(frame, k, label), ...] per file."""
    rows = []
    for path in files:
        with np.load(path) as episode:
            rows.append(head_rows.probe_rows(reference_of(episode, task), k_max))
    return rows


# ---------------------------------------------------------------------- features
def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """The image resized with padding to size x size, as the reference environment sends it to the policy."""
    from openpi_client import image_tools
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, size, size))


def row_observations(episode, rows: list, prompts: HistoryPrompt, task: TaskSpec, args) -> list:
    """The policy's observation of every row: the letterboxed frames, the state and the prompt with k history lines."""
    instruction = str(episode["instruction"])
    exterior = episode["exterior"]              # an .npz array is read from the file at every access: read it once
    wrist = episode["wrist"]
    states = episode["state"] if "state" in episode.files else None
    observations = []
    for frame, k, _ in rows:
        state = states[frame] if states is not None else np.zeros(args.state_dim, np.float32)
        observations.append({"observation/image": letterbox(exterior[frame], args.image_size),
                             "observation/wrist_image": letterbox(wrist[frame], args.image_size),
                             "observation/state": np.asarray(state, np.float32),
                             "prompt": prompts.render(instruction, task.skills[:k])})
    return observations


def compute_features(files, rows: list, task: TaskSpec, args) -> np.ndarray:
    """(n, width) float16: the feature of every row of every episode (openpi environment)."""
    from vla_memory.writers.head.server import create_policy, use_compilation_cache
    if args.jax_cache:
        use_compilation_cache(args.jax_cache)
    from vla_memory.writers.head.features import LayerFeature
    policy, train_config = create_policy(args)
    feature = LayerFeature(policy, train_config, layer=args.layer, batch=args.batch)
    prompts = HistoryPrompt(task)
    features = []
    for path, rows_of_episode in zip(files, rows):
        with np.load(path) as episode:
            observations = row_observations(episode, rows_of_episode, prompts, task, args)
        inputs = [policy._input_transform(observation) for observation in observations]
        features.append(feature.features(inputs).astype(np.float16))
        print(f"[fit_event_head] {path.name}: {len(rows_of_episode)} rows", flush=True)
    return np.concatenate(features)


def save_features(path: pathlib.Path, files, rows: list, features: np.ndarray, args) -> None:
    """The features file: the features, the rows as (episode, frame, k, label), the episode files and how they were computed."""
    flat = [(index, frame, k, label) for index, episode in enumerate(rows) for frame, k, label in episode]
    table = np.asarray(flat, np.int64).reshape(-1, 4)
    source = {"config": args.config, "checkpoint": args.checkpoint, "center_crop": args.center_crop, "layer": args.layer,
              "pool": "text", "image_size": args.image_size, "batch": args.batch}
    np.savez(path, features=features, rows=table, episodes=np.array([file.name for file in files]), source=json.dumps(source))


def load_features(path: pathlib.Path) -> dict:
    """The features file as save_features wrote it."""
    with np.load(path) as saved:
        return {"features": saved["features"], "rows": saved["rows"], "episodes": [str(name) for name in saved["episodes"]],
                "source": json.loads(str(saved["source"]))}


# ---------------------------------------------------------------------- the fit
def description(saved: dict, report: dict) -> dict:
    """head.json: what the head reads, how it was fitted, and on which episodes."""
    source = saved["source"]
    saved_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    feature = (f"residual stream after Gemma block {source['layer']} of {source['config']} ({source['checkpoint']}), mean "
               f"over the valid text tokens; images letterboxed to {source['image_size']}, centre crop "
               f"{source['center_crop']}, batch {source['batch']}")
    score = "s = ((x - mean) / std) @ w + b (float32); a call says done if s >= threshold; a line after N consecutive calls"
    return {"what": "event head of the history policy", "saved_at": saved_at, "feature": feature, "score": score, **report,
            "episodes": saved["episodes"]}


def fit(saved: dict, args) -> None:
    """Fit the head on the saved features and save it to --out with its description."""
    table = saved["rows"]
    l2_grid = tuple(args.l2) if args.l2 else head_fit.L2_GRID
    head, report = head_fit.fit_head(saved["features"].astype(np.float32), table[:, 3], table[:, 0].tolist(), l2_grid=l2_grid,
                                     folds=args.folds, fpr=args.fpr, consecutive=args.consecutive,
                                     layer=saved["source"]["layer"])
    digests = save_head(args.out, head, description(saved, report))
    print(json.dumps({key: report[key] for key in ("l2", "threshold", "threshold_oof", "training")}, indent=1))
    print(f"[fit_event_head] head saved to {args.out}: {digests}")


# ---------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    """The episodes, the features file, the head directory, the policy and the fit's parameters."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", action="append", default=[], metavar="GLOB", help="episode files (repeatable)")
    parser.add_argument("--features", type=pathlib.Path, required=True,
                        help="the rows' features: written when absent, else read (and the episodes are not needed)")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="the head's directory (new or empty)")
    parser.add_argument("--task", default=None, help="task YAML/JSON (default: the packaged red_blue_blocks task)")
    parser.add_argument("--config", help="the policy's openpi training config")
    parser.add_argument("--checkpoint", help="the policy's checkpoint directory")
    parser.add_argument("--center-crop", type=float, default=0.0, help="as the policy is served (reference: 0.95)")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--layer", type=int, default=12, help="the block after which the residual stream is read")
    parser.add_argument("--batch", type=int, default=8, help="rows per forward pass (the server's --head-batch)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=8, help="the state's size when the episodes carry none")
    parser.add_argument("--jax-cache", default=None,
                        help="JAX's persistent compilation cache; a server started on a copy uses the same kernels")
    parser.add_argument("--l2", type=float, action="append", default=[], help="L2 values to choose from (repeatable)")
    parser.add_argument("--folds", type=int, default=head_fit.FOLDS)
    parser.add_argument("--fpr", type=float, default=head_fit.FPR, help="the out-of-fold false-positive share allowed")
    parser.add_argument("--consecutive", type=int, default=2, help="consecutive 'done' calls that write a line (N)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Compute the features when the features file does not exist yet, then fit the head."""
    args = parse_args(argv)
    if not args.features.exists():
        if not (args.episodes and args.config and args.checkpoint):
            raise SystemExit("computing the features needs --episodes, --config and --checkpoint")
        task = TaskSpec.load_or_default(args.task)
        files = episode_files(args.episodes)
        k_max = int((task.history or {}).get("max_lines", len(task.skills) - 1)) - 1
        rows = episode_rows(files, task, k_max)
        save_features(args.features, files, rows, compute_features(files, rows, task, args), args)
    fit(load_features(args.features), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
