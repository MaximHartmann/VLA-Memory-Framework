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
from vla_memory.planner_loop import ExemplarStore, PhaseLabeller, fingerprint_from_task, load_npz_episodes  # noqa: E402
from vla_memory.planner_loop.labels import parse_episode_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--episodes", default="0-19")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default=None)
    ap.add_argument("--cameras", nargs="+", default=["exterior", "wrist"])
    ap.add_argument("--gripper-column", type=int, default=7)
    a = ap.parse_args()
    task = TaskSpec.load(a.task) if a.task else TaskSpec.default()
    eps = parse_episode_ids(a.episodes)
    st = ExemplarStore.build(fingerprint_from_task(task), task, load_npz_episodes(a.demos, eps, a.cameras),
                             PhaseLabeller(task, state_gripper_index=a.gripper_column), a.stride, a.out, camera_names=a.cameras)
    print(f"built {len(st)} entries from {len(eps)} episodes -> {a.out}")
    for o in task.objects:
        print(f" {o}:", dict(Counter(m[task.field(o)] for m in st.meta)))
    print(" gripper:", dict(Counter(m["gripper"] for m in st.meta)), "\n skill:", dict(Counter(m["subgoal"] for m in st.meta)))


if __name__ == "__main__":
    main()
