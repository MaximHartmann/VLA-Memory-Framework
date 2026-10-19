#!/usr/bin/env python3
"""Build an exemplar store (the long-term memory) from recorded demonstration episodes.

  python scripts/build_exemplar_store.py --demos /path/to/demos --episodes 0-49 --stride 8 --out stores/ep0-49_s8.npz
  python scripts/build_exemplar_store.py --task my_task.yaml ...

Reads episode_NNNN.npz files with the camera stacks named as in --cameras, a 'state' array whose column
--gripper-column is the gripper closedness (0 open .. 1 closed) and a per-frame 'phase' (see labels.PhaseLabeller).
Every --stride-th frame becomes one entry, labelled from its phase.
"""
import argparse
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vla_memory import TaskSpec  # noqa: E402
from vla_memory.fingerprint import fingerprint_from_task  # noqa: E402
from vla_memory.writers.vlm import ExemplarStore, PhaseLabeller, load_npz_episodes  # noqa: E402
from vla_memory.writers.vlm.labels import parse_episode_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    """The demonstrations, the episodes and stride, the output file, the task and the recordings' layout."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos", required=True)
    parser.add_argument("--episodes", default="0-19")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--cameras", nargs="+", default=["exterior", "wrist"])
    parser.add_argument("--gripper-column", type=int, default=7)
    return parser.parse_args()


def main():
    """Build and save the store, then print how its entries are labelled."""
    args = parse_args()
    task = TaskSpec.load_or_default(args.task)
    episode_ids = parse_episode_ids(args.episodes)
    store = ExemplarStore.build(fingerprint_from_task(task), task, load_npz_episodes(args.demos, episode_ids, args.cameras),
                                PhaseLabeller(task, state_gripper_index=args.gripper_column), args.stride, args.out,
                                camera_names=args.cameras)
    print(f"built {len(store)} entries from {len(episode_ids)} episodes -> {args.out}")
    for object_name in task.objects:
        print(f" {object_name}:", dict(Counter(entry[task.field(object_name)] for entry in store.meta)))
    print(" gripper:", dict(Counter(entry["gripper"] for entry in store.meta)),
          "\n skill:", dict(Counter(entry["subgoal"] for entry in store.meta)))


if __name__ == "__main__":
    main()
