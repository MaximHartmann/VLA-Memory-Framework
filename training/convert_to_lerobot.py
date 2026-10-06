#!/usr/bin/env python3
"""Stage 2: staged .npz episodes -> a LeRobot dataset openpi can train on.

Runs inside openpi.sif (lerobot lives there, Isaac Sim does not).

Schema follows openpi's own LIBERO converter, whose comment is the contract:
"OpenPi assumes that proprio is stored in `state` and actions in `actions`.
 LeRobot assumes that dtype of image data is `image`."

Images are stored at the NATIVE 320x240. resize_with_pad to 224x224 happens in
the training transform, exactly as it does at inference -- keeping the raw data
means the resize is one shared step rather than two that can drift apart.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/workspace/data/demos_raw")
    ap.add_argument("--repo-id", default="hartmann/red_blue_blocks")
    ap.add_argument("--root", default="/workspace/data/lerobot")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--use-videos", action="store_true", default=True,
                    help="MP4 per episode instead of PNG per frame; far kinder to BeeGFS")
    ap.add_argument("--no-videos", dest="use_videos", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--start", type=int, default=0,
                    help="first staged episode index (for train/val splits)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--action-space", choices=["velocity", "absolute"], default="velocity",
                    help="velocity: actions exactly as recorded, (q_cmd - q_meas)/dt.  "
                         "absolute: arm actions converted to absolute joint targets q_cmd. "
                         "The gripper (index 7) is ALREADY absolute in [0,1] and is never "
                         "touched by either mode.")
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    raw = pathlib.Path(args.raw_dir)
    files = sorted(raw.glob("episode_*.npz"))[args.start:]
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no episodes in {raw}"); return 2
    print(f"[conv] {len(files)} staged episodes from {raw}", flush=True)

    def to_absolute(d):
        """Recover absolute joint targets from the recorded velocity actions.

        generate_demos.py:301 records  act_arm = (q_cmd - q_meas) / dt  and the
        npz separately stores q_meas as state[:, :7] plus the fps, so the
        inversion  q_cmd = q_meas + act_arm * dt  is exact rather than an
        approximation -- no Isaac re-recording is needed.

        WHY: the velocity target spans a 69x std range across the seven joints
        (0.025 .. 1.736) and is spiky -- mostly near zero while holding a
        waypoint, with rare large bursts. q_cmd spans 5x (0.059 .. 0.291) and is
        smooth: 56.3% of consecutive steps change by less than 0.001 rad. The
        step-20000 checkpoint reproduced that structure in its errors, emitting
        arm velocities with 3.4-4.6x the expert's normalised std while the
        gripper -- the one already-absolute dimension -- came out at 1.07x.

        CAVEAT: q_cmd is very close to q_meas (near-identical per-dim std), so a
        model can score well by echoing its own state input and barely moving.
        The 10-step action horizon has to extrapolate ~0.67 s, which mitigates
        this, but it is the first thing to check on the new checkpoint: compare
        predicted q_cmd against the state that was fed in.
        """
        dt = 1.0 / float(d["fps"])
        act = np.array(d["actions"], dtype=np.float32, copy=True)
        act[:, :7] = d["state"][:, :7] + act[:, :7] * dt
        return act

    probe = np.load(files[0], allow_pickle=True)
    H, W = probe["exterior"].shape[1:3]
    state_dim = probe["state"].shape[1]
    act_dim = probe["actions"].shape[1]
    print(f"[conv] image {H}x{W}  state {state_dim}  actions {act_dim}", flush=True)
    print(f"[conv] action space: {args.action_space}", flush=True)
    if args.action_space == "absolute":
        _a = to_absolute(probe)
        print(f"[conv]   arm target std per-dim: {np.round(_a[:, :7].std(0), 4)}", flush=True)
        print(f"[conv]   (velocity was          : {np.round(probe['actions'][:, :7].std(0), 4)})", flush=True)

    root = pathlib.Path(args.root)
    if root.exists():
        if not args.overwrite:
            print(f"[conv] {root} exists; pass --overwrite"); return 2
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=str(root),
        fps=args.fps,
        robot_type="franka_fer",
        use_videos=args.use_videos,
        features={
            "image":       {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "state":       {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
            "actions":     {"dtype": "float32", "shape": (act_dim,),   "names": ["actions"]},
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    total = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        acts = to_absolute(d) if args.action_space == "absolute" else d["actions"]
        task = str(d["instruction"])
        T = d["state"].shape[0]
        for t in range(T):
            ds.add_frame({
                "image": d["exterior"][t],
                "wrist_image": d["wrist"][t],
                "state": d["state"][t].astype(np.float32),
                "actions": acts[t].astype(np.float32),
                "task": task,
            })
        ds.save_episode()
        total += T
        print(f"[conv] {f.name}: {T} frames  (total {total})", flush=True)

    print(f"\n[conv] wrote {len(files)} episodes / {total} frames to {root}", flush=True)
    n_files = sum(1 for _ in root.rglob("*") if _.is_file())
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"[conv] on disk: {n_files} files, {size/1e9:.2f} GB "
          f"({'video' if args.use_videos else 'PNG'} encoding)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
