#!/usr/bin/env python3
"""Fit the event head of a policy from the policy's own runs (not demonstrations), for the method it will be served with.

Two ways in:

1. Recordings, for any policy whose server hands out its state (CPU only): the files the proxy writes with --record-head
   while another writer runs the robot (vla_memory/writers/head/record.py). At every call they hold the policy's state
   summaries with the number of completed skills the input showed (label 0), and after each completed skill the policy's
   state for the old input (label 1). Every summary the server hands out is tried; the one with the best cross-validated
   AUROC is kept.

    python training/fit_event_head.py --recordings 'runs/*.npz' --out <head dir> [--task task.yaml]

2. Pictures of recorded episodes, for openpi's pi0.5 (the openpi environment and one GPU, integrations/openpi on
   PYTHONPATH): one .npz file per episode with

    exterior     (T, H, W, 3) uint8   the exterior camera at every control step, as the environment sends it
    wrist        (T, H, W, 3) uint8   the wrist camera
    instruction  str                  the instruction of the episode
    reference    (T,) int             the number of completed skills at every step, as the history at a policy call there
                                      shows it; or, instead,
    rows         (T, 5) float         finger width, squeeze, fingertip x, y, z at every step (memory/proprio_steps), which
                                      the task's rules writer labels: reference[t] = the lines it wrote after steps 0..t-1
    state        (T, D) float         optional: the policy's state input (default zeros of --state-dim)
    plan, objects  (n,) str           optional, for a task whose plan varies: the skills of the episode in order, or the names
                                      of its objects (the plan is then the task order of those objects)

   Every 4th frame becomes a row (vla_memory.writers.head.rows) that shows the policy what its method (--method) shows it
   with k skills done: history lines, the next skill as the prompt, or the keyframe picture; its label says whether skill
   k + 1 is done. The policy, loaded as openpi_server.py serves it, computes each row's state in its own pass (openpi_tap.py,
   block --layer averaged over the prompt's tokens). The states are saved to --features; a later run with that file goes
   straight to the fit. Images are letterboxed to --image-size, as the reference environment sends them.

    python training/fit_event_head.py --episodes 'rollouts/*.npz' --config <policy config> \
        --checkpoint <checkpoint dir> --center-crop 0.95 --method subgoal --features rows.npz --out <head dir>

The fit (CPU, both ways): head_fit.fit_head; the head is saved to --out (head.npz, head.json, head.sha256).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the framework, when it is not installed
from vla_memory.methods.history.prompt import HistoryPrompt  # noqa: E402
from vla_memory.writers.head import fit as head_fit  # noqa: E402
from vla_memory.writers.head import rows as head_rows  # noqa: E402
from vla_memory.writers.head.file import save_head, state_name  # noqa: E402
from vla_memory.writers.rules.rules import RuleMonitor  # noqa: E402
from vla_memory.writers.rules.validate import episode_signals  # noqa: E402
from vla_memory.task import TaskSpec  # noqa: E402

SCORE = "s = ((x - mean) / std) @ w + b (float32); a call says done if s >= threshold; a skill after N consecutive calls"


# ---------------------------------------------------------------------- recordings (any policy)
def lines_of(task: TaskSpec, plan) -> int:
    """How many skills a head completes in an episode with this plan: history.max_lines, at most all but the last skill."""
    max_lines = (task.history or {}).get("max_lines")
    return len(plan) - 1 if max_lines is None else min(len(plan) - 1, int(max_lines))


def load_recordings(patterns, task: TaskSpec) -> dict:
    """The rows of the proxy's recordings: the state summaries (n, candidates, width), the labels, the episode of every row
    and the summaries' names. Rows whose input already shows every skill the head can complete are left out."""
    files = episode_files(patterns)
    values, labels, groups, names, method = [], [], [], None, None
    for index, path in enumerate(files):
        with np.load(path) as recording:
            file_names = [str(name) for name in recording["names"]]
            if names is not None and file_names != names:
                raise SystemExit(f"{path.name} holds the summaries {file_names}, the others {names}")
            names, meta, rows = file_names, json.loads(str(recording["meta"])), recording["rows"]
            keep = rows[:, 1] < lines_of(task, meta["plan"])
            values.append(recording["values"][keep])
            labels.append(rows[keep, 2])
            groups += [index] * int(keep.sum())
            method = meta.get("method")
    return {"values": np.concatenate(values), "labels": np.concatenate(labels), "groups": groups, "names": names,
            "method": method, "files": [path.name for path in files]}


def fit_recordings(args) -> None:
    """Fit a head on every state summary of the recordings and save the one with the best mean AUROC."""
    data = load_recordings(args.recordings, TaskSpec.load_or_default(args.task))
    candidates = {}
    for index, name in enumerate(map(state_name, data["names"])):
        head, report = head_fit.fit_head(data["values"][:, index].astype(np.float32), data["labels"], data["groups"],
                                         l2_grid=tuple(args.l2) or head_fit.L2_GRID, folds=args.folds, fpr=args.fpr,
                                         consecutive=args.consecutive, state=name)
        auroc = max(row["mean_auroc"] for row in report["inner_cv"])
        candidates[name] = (auroc, head, report)
        print(f"[fit_event_head] {name}: mean AUROC {auroc:.4f}", flush=True)
    name = max(candidates, key=lambda candidate: candidates[candidate][0])
    _, head, report = candidates[name]
    saved_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    description = {"what": f"event head of the {data['method']} policy", "saved_at": saved_at, "state": name, "score": SCORE,
                   **report, "candidates": {key: round(value[0], 5) for key, value in candidates.items()},
                   "recordings": data["files"]}
    finish(head, report, description, args)


# ---------------------------------------------------------------------- pictures (openpi's pi0.5)
def episode_files(patterns) -> list[pathlib.Path]:
    """The files matching the globs, sorted and without duplicates."""
    paths = {path for pattern in patterns for path in glob.glob(str(pathlib.Path(pattern).expanduser()))}
    files = sorted(pathlib.Path(path) for path in paths)
    if not files:
        raise SystemExit(f"no files match {patterns}")
    return files


def episode_objects(episode) -> list[str] | None:
    """The names of the objects the episode file declares (`objects`), or None: all of the task's."""
    if "objects" in episode.files:
        return [str(name) for name in episode["objects"]]
    return None


def plan_of(episode, task: TaskSpec) -> list[str]:
    """The skills of the episode in order: its `plan`, else the task order of its objects."""
    if "plan" in episode.files:
        return [str(skill) for skill in episode["plan"]]
    return task.default_plan(episode_objects(episode))


def reference_of(episode, task: TaskSpec) -> np.ndarray:
    """The episode's reference count, given or labelled from its signal rows by the task's rules writer."""
    if "reference" in episode.files:
        return np.asarray(episode["reference"], int)
    if "rows" not in episode.files:
        raise SystemExit("an episode file needs `reference` or `rows`")
    monitor = RuleMonitor(task, plan=plan_of(episode, task))
    lines = monitor.label_rows(episode["rows"], signals=episode_signals(episode) or None)
    return head_rows.reference_from_lines(lines)


def episode_rows(files, task: TaskSpec, k_max: int) -> list[list]:
    """The rows of every episode, [(frame, k, label), ...] per file."""
    rows = []
    for path in files:
        with np.load(path) as episode:
            rows.append(head_rows.probe_rows(reference_of(episode, task), k_max))
    return rows


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """The image resized with padding to size x size, as the reference environment sends it to the policy."""
    from openpi_client import image_tools
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, size, size))


def keyframe_frames(reference: np.ndarray, count: int, every: int) -> list[int]:
    """The frame whose wrist picture is the keyframe of key moment k = 1..count: the policy call after the one at which the
    moment's line becomes due, as the head writer takes it (it decides after a call), on the grid of calls every `every` steps."""
    frames = []
    for edge in head_rows.boundaries(reference, count):
        call = -(-((len(reference) if edge is None else edge) + every) // every) * every
        frames.append(min(call, len(reference) - 1))
    return frames


def row_observations(episode, rows: list, prompts: HistoryPrompt, task: TaskSpec, args) -> list:
    """The policy's observation of every row (frame, k, label): the letterboxed frames, the state, and what the method shows
    the policy for k completed skills: the instruction with k history lines (history), skill k + 1 as the prompt (subgoal),
    or the instruction with the wrist picture of key moment k in the memory image (keyframe; none for k = 0)."""
    instruction = str(episode["instruction"])
    exterior = episode["exterior"]              # an .npz array is read from the file at every access: read it once
    wrist = episode["wrist"]
    states = episode["state"] if "state" in episode.files else None
    plan = plan_of(episode, task)
    if args.method == "keyframe":
        pictures = keyframe_frames(reference_of(episode, task), max(k for _, k, _ in rows), args.call_every)
    observations = []
    for frame, k, _ in rows:
        state = states[frame] if states is not None else np.zeros(args.state_dim, np.float32)
        observation = {"observation/image": letterbox(exterior[frame], args.image_size),
                       "observation/wrist_image": letterbox(wrist[frame], args.image_size),
                       "observation/state": np.asarray(state, np.float32)}
        if args.method == "history":
            observation["prompt"] = prompts.render(instruction, plan[:k])
        elif args.method == "subgoal":
            observation["prompt"] = plan[min(k, len(plan) - 1)]
        else:
            observation["prompt"] = instruction
            if k:
                observation["observation/memory_image"] = np.asarray(wrist[pictures[k - 1]])
        observations.append(observation)
    return observations


def compute_features(files, rows: list, task: TaskSpec, args) -> np.ndarray:
    """(n, width) float16: the state of every row of every episode, from the policy's own pass (openpi environment)."""
    from openpi_server import create_policy
    from openpi_tap import attach
    policy = create_policy(args.config, args.checkpoint, args.center_crop, args.default_prompt)
    tap = attach(policy, layer=args.layer)
    prompts = HistoryPrompt(task)
    features = []
    for path, rows_of_episode in zip(files, rows):
        with np.load(path) as episode:
            observations = row_observations(episode, rows_of_episode, prompts, task, args)
        states = []
        for observation in observations:
            policy.infer(observation)
            states.append(tap.read()[0])
        features.append(np.stack(states).astype(np.float16))
        print(f"[fit_event_head] {path.name}: {len(rows_of_episode)} rows", flush=True)
    return np.concatenate(features)


def save_features(path: pathlib.Path, files, rows: list, features: np.ndarray, args) -> None:
    """The features file: the features, the rows as (episode, frame, k, label), the episode files and how they were computed."""
    flat = [(index, frame, k, label) for index, episode in enumerate(rows) for frame, k, label in episode]
    table = np.asarray(flat, np.int64).reshape(-1, 4)
    source = {"config": args.config, "checkpoint": args.checkpoint, "center_crop": args.center_crop, "layer": args.layer,
              "pool": "text", "image_size": args.image_size, "method": args.method, "pass": "same"}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, rows=table, episodes=np.array([file.name for file in files]), source=json.dumps(source))


def load_features(path: pathlib.Path) -> dict:
    """The features file as save_features wrote it."""
    with np.load(path) as saved:
        return {"features": saved["features"], "rows": saved["rows"], "episodes": [str(name) for name in saved["episodes"]],
                "source": json.loads(str(saved["source"]))}


def fit_features(saved: dict, args) -> None:
    """Fit the head on the saved features of the picture rows and save it to --out with its description."""
    source, table = saved["source"], saved["rows"]
    state = state_name(f"layer{source['layer']}:{source.get('pool', 'text')}")
    head, report = head_fit.fit_head(saved["features"].astype(np.float32), table[:, 3], table[:, 0].tolist(),
                                     l2_grid=tuple(args.l2) or head_fit.L2_GRID, folds=args.folds, fpr=args.fpr,
                                     consecutive=args.consecutive, state=state)
    computed = "the policy's own pass" if source.get("pass") == "same" else f"a separate pass, batch {source.get('batch')}"
    feature = (f"residual stream after Gemma block {source['layer']} of {source.get('config')} ({source.get('checkpoint')}), "
               f"mean over the valid text tokens; images letterboxed to {source.get('image_size')}, centre crop "
               f"{source.get('center_crop')}, {computed}")
    saved_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    description = {"what": f"event head of the {source.get('method', 'history')} policy", "saved_at": saved_at,
                   "state": state, "feature": feature, "score": SCORE, **report, "episodes": saved["episodes"]}
    finish(head, report, description, args)


# ---------------------------------------------------------------------- main
def finish(head, report: dict, description: dict, args) -> None:
    """Save the head and print what was fitted."""
    digests = save_head(args.out, head, description)
    print(json.dumps({key: report[key] for key in ("l2", "threshold", "threshold_oof", "training")}, indent=1))
    print(f"[fit_event_head] head on {head.state} saved to {args.out}: {digests}")


def parse_args(argv=None) -> argparse.Namespace:
    """The recordings or the picture episodes, the head directory, the policy and the fit's parameters."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", action="append", default=[], metavar="GLOB",
                        help="the proxy's --record-head files (repeatable)")
    parser.add_argument("--episodes", action="append", default=[], metavar="GLOB",
                        help="picture episodes, openpi's pi0.5 only (repeatable)")
    parser.add_argument("--features", type=pathlib.Path, default=None,
                        help="with --episodes: the rows' states, written when absent, else read (the episodes then unused)")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="the head's directory (new or empty)")
    parser.add_argument("--task", default=None, help="task YAML/JSON (default: the packaged red_blue_blocks task)")
    parser.add_argument("--config", help="pictures: the policy's openpi training config")
    parser.add_argument("--checkpoint", help="pictures: the policy's checkpoint directory")
    parser.add_argument("--center-crop", type=float, default=0.0, help="pictures: as the policy is served (ours: 0.95)")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--layer", type=int, default=12, help="pictures: the block whose output is read")
    parser.add_argument("--method", choices=("history", "subgoal", "keyframe"), default="history",
                        help="pictures: what the policy is shown for k completed skills: history lines, the next skill as "
                             "the prompt, or the keyframe picture (the method the head will be served with)")
    parser.add_argument("--call-every", type=int, default=8, help="pictures, keyframe: control steps between policy calls")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=8, help="pictures: the state's size when the episodes carry none")
    parser.add_argument("--l2", type=float, action="append", default=[], help="L2 values to choose from (repeatable)")
    parser.add_argument("--folds", type=int, default=head_fit.FOLDS)
    parser.add_argument("--fpr", type=float, default=head_fit.FPR, help="the out-of-fold false-positive share allowed")
    parser.add_argument("--consecutive", type=int, default=2, help="consecutive 'done' calls that complete a skill (N)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Fit on the recordings; or compute the picture rows' states when the features file does not exist yet, then fit."""
    args = parse_args(argv)
    if args.recordings:
        fit_recordings(args)
        return 0
    if args.features is None:
        raise SystemExit("give --recordings, or --episodes with --features")
    if not args.features.exists():
        if not (args.episodes and args.config and args.checkpoint):
            raise SystemExit("computing the features needs --episodes, --config and --checkpoint")
        task = TaskSpec.load_or_default(args.task)
        files = episode_files(args.episodes)
        k_max = int((task.history or {}).get("max_lines", len(task.skills) - 1)) - 1
        rows = episode_rows(files, task, k_max)
        save_features(args.features, files, rows, compute_features(files, rows, task, args), args)
    fit_features(load_features(args.features), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
