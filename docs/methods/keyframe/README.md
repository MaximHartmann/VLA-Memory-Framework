# The keyframe policy: a picture of the last key moment

## The idea

The memory is one picture. The policy keeps the plain instruction as its prompt, and its third image slot (pi0.5 has three,
the reference policy uses two) shows the wrist camera's picture of the last key moment of its own episode: picked up red,
placed red, picked up blue in the reference task; before the first one the slot stays empty. The picture tells the policy
which phase of the task it is in where the live cameras alone look alike, such as the start and the moment after the first
cube was put back; the history method does the same with text.

```
before the first event   instruction + exterior + wrist + (empty slot)
from "picked up red" on  instruction + exterior + wrist + the wrist picture taken when the red block was picked up
from "placed red" on     instruction + exterior + wrist + the wrist picture taken when it was placed
```

The key moments are the skills the writer completes, as for the history method: here the simulator's ground truth, or the
rules writer on the robot's gripper and hand signals. At the call at which the writer completes a skill the method keeps
that call's raw wrist frame and sends it with every observation from then on. The policy is trained on demonstrations relabelled with causally chosen
keyframes: a frame after an event gets the wrist picture of a frame drawn from the 16 frames after that event, never later
than the frame itself, since at run time the picture is taken at the call that signals the event, up to a call later.

## Prerequisites

- The recorded demonstrations as two LeRobot datasets of the same frames: the memoryless policy's, with the plain
  instruction as each frame's task (`training/convert_to_lerobot.py --no-videos`), and the history policy's, whose prompts
  mark the events (`training/relabel_history.py`).
- A policy that takes a third picture, `observation/memory_image`, and ignores it while the picture is absent or all zero
  (pi0.5 has a third image slot with a mask), trained on that dataset.
- At run time: an environment that sends the raw wrist frame (`memory/wrist_raw`) and what the writer reads
  (`memory/oracle_prompt` for the simulator's events, `memory/proprio_steps` for the rules writer), as
  `vla_memory.env_keys.ControlKeys` builds them. The later "placed" keyframe also needs the joint state in
  `observation/state` (the 7 arm joints first); the forward kinematics of the reference Franka arm is built in.

## How to use it in the framework

1. **Training data:**

   ```
   python training/relabel_keyframe.py --train <frames train>:<history train> --val <frames val>:<history val> \
       --root <dataset root> --repo-id <user>/<name>
   ```

   It checks that both datasets of a split hold the same frames and writes `<name>_train` and `<name>_val` with the image
   column `memory_image` and a per-frame manifest, `keyframe_frames.csv`.
2. **Training:** one policy from `pi05_base` with the memoryless policy's recipe (30,000 steps) on
   `vla_memory/red_blue_v2_abs_keyframe_train`, the local dataset step 1 writes with `--repo-id
   vla_memory/red_blue_v2_abs_keyframe` (not distributed; `training/README.md`), the third image slot reading `memory_image`.
3. **Run:** serve a checkpoint and start the proxy with the keyframe method, which keeps the plain instruction as the prompt:

   ```
   python -m vla_memory.proxy --writer oracle --method keyframe --listen-port <port> --upstream-port <policy port> \
       [--keyframe-mode off] [--keyframe-placed-later]
   ```

   `--writer rules` instead of `oracle` for the rules writer. `--keyframe-mode off` takes the keyframes and never sends
   them: the control. `--keyframe-placed-later` takes the
   "placed" keyframe at the first call from its signal on at which the fingertips are at least 10 cm above the placed cube
   (`--keyframe-placed-height`), at the latest at the third call after the signal (`--keyframe-placed-calls`). Every proxy
   log record says which event the current keyframe belongs to, the step it was taken at and whether it was sent.

## Our findings

Reference task, checkpoint 29000 (the best of 30 on the development seeds 1-20, 19 of 20), 100 fresh seeds of 500 control
steps:

| policy and writer | full tasks of 100 |
|---|---|
| keyframe policy, the simulator's events (`--writer oracle`) | 78 |
| keyframe policy, the rules writer (`--writer rules`) | 75 |
| keyframe policy, no keyframe (`--keyframe-mode off`) | 0 |
| history policy: ground truth / rules / VLM / event head | 82 / 79 / 77 / 78 |
| memoryless policy | 11 |

- The keyframe policy does as well as the history policy with every writer (paired sign tests, all p >= 0.27) and far
  better than the memoryless policy (p < 1e-6). Without the keyframe it lifts red and holds it to the end: the picture is
  what starts the place phase.
- It grasps better than the history policy (0.68 to 0.70 lifts per grasp attempt against 0.34 to 0.42) and loses that margin
  by grasping red again after placing it (10 episodes with the simulator's events, 16 with the rules writer). The cause is a
  "placed red" keyframe taken low: below 8 cm above the cube 25 of 60 episodes grasped red again, from 8 cm on 1 of 127. The
  simulator's place event fires 6 cm above the cube, while the training keyframes show the hand about 10 cm up.
- The later "placed" keyframe removes these re-grasps but does not raise the full tasks significantly. On new seeds
  (2001-2100) the re-grasps fell from 17 of 94 to 0 of 90 with the rules writer (p = 6.1e-5) and from 5 to 1 with the
  simulator's events; the full tasks went from 74 to 82 with the rules writer (p = 0.15) and from 83 to 78 with the
  simulator's events (p = 0.42). The failures that remain are grasps. The original moment therefore stays the default.
- The writers are on time: with the simulator's events all 264 keyframes were taken at the simulator's call; the rules
  writer took 258 of 263 at the same call, and the others one call off or, in 3 cases, at a wrong moment.
- A keyframe costs about 3 ms per policy call, the proxy hop (172 against 170 ms); the later "placed" keyframe adds 1 to
  3 ms.
