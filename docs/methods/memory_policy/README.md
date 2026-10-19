# The memory policy: a policy that reads its memory

## The idea

The memory is handed to the policy itself, as a hint next to its cameras. A store holds moments of recorded episodes, one
per policy call: the wrist camera's picture at that moment and the movement made in the next 8 actions, written as one
sentence. At every policy call the proxy looks up the stored moment most similar to the current one and passes it on: its
wrist picture goes into the policy's third image slot (pi0.5 has three, the reference policy uses two), its sentence is
appended to the skill prompt:

```
pick up the red block. memory: release at step 1, up 2 cm
```

The policy is trained to read such hints. Every training frame gets the nearest stored moment of ANOTHER episode
(leave-one-episode-out), so the policy learns how far a hint from a similar but different scene can be trusted, and it stays
in control. One frame in ten gets no memory (an all-zero picture and the bare skill prompt), so that a call without a hint
is in the training distribution. The idea follows retrieval-augmented policies such as RICL (Sridhar et al., 2025, arXiv
2508.02062), which put retrieved experience into the policy's own input.

- **The sentence**: the hand's displacement from the current joints to the last joint target of the next 8 actions (forward
  kinematics, whole centimetres, largest first: up/down, left/right, forward/back), then the gripper: "release at step k"
  when it is closed now (measured finger width) and opens, "close at step k", "gripper stays closed", or "hold still".
- **The retrieval**: the single most similar moment by the colour fingerprint of the raw wrist frame, among the moments of
  the same skill and gripper state (open, closed on an object, closed on nothing, from the measured finger width); a
  gripper closed on nothing is only shown moments that release.

## Prerequisites

- Recorded episodes as LeRobot datasets: both cameras stored as images at their native size, the joint state with the
  gripper's closedness last, absolute joint targets as actions, and the skill as each frame's task
  (`training/convert_to_lerobot.py --no-videos --action-space absolute`, then `training/relabel_subgoals.py`). The forward
  kinematics of the Franka arm of the reference bench is built in; another arm brings its own chain.
- openpi with our transforms (`training/openpi/`): `RedBlueMemoryInputs` puts `observation/memory_image` into the third
  image slot (an absent or all-zero picture stays masked out); `RedBlueDropMemory` removes the memory for the control.
- At run time: an environment that sends the raw wrist frame (`memory/wrist_raw`) and the finger width
  (`memory/proprio_steps`), as `vla_memory.env_keys.ControlKeys` builds them, and a writer that sequences the skills with
  the sub-goal method (the VLM writer, the rules writer, or the ground truth).

## How to use it in the framework

1. **Store and training data.** Every dataset is named with its source label; the store is built from the training
   datasets only, and every frame of every dataset gets its memory:

   ```
   python training/relabel_memory.py --train <demos>:demo --train <corrections>:expert --val <demos val>:demo \
       --store <store dir> --root <dataset root> --repo-id <user>/<name>
   ```

   It writes the store (`entries.jsonl`, the keys, the wrist pictures, `meta.json`) and the datasets `<name>_train` and
   `<name>_val` with the image column `memory_image`, the prompt with the memory and a per-frame manifest.
2. **Training.** Two policies from `pi05_base` on the same frames: `pi05_red_blue_v2_memory` reads the memory, the control
   `pi05_red_blue_v2_memory_control` gets the frames without it, so that their difference is the memory, not the data.
   Both configs read `vla_memory/red_blue_v2_abs_memory_train`, the local dataset step 1 writes with `--repo-id
   vla_memory/red_blue_v2_abs_memory` (not distributed, like the store; `training/README.md`). In the openpi checkout with
   our changes, the norm statistics once per config, then per policy:

   ```
   uv run scripts/compute_norm_stats.py --config-name pi05_red_blue_v2_memory
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_red_blue_v2_memory --exp-name=<name> --overwrite
   ```

3. **Run.** Serve a checkpoint with its config name and add the memory to the proxy:

   ```
   python -m vla_memory.proxy --writer rules --method subgoal --listen-port <port> --upstream-port <policy port> \
       --with memory_policy --memory-store <store dir> [--memory-mode off|wrong]
   ```

   `--memory-mode off` sends no memory (the bare skill prompt, no picture) and `wrong` a random moment of another skill:
   the two checks of whether the policy reads its memory at all.

## Our findings

Reference task, checkpoint 19999. Training data: 200 demonstrations and 75 episodes in which the red cube was nudged and a
scripted expert took over (87,768 frames); the store holds 10,966 moments of them. The nudge benchmark pushes the red cube
1.5-3 cm and turns it 10-30 degrees on every red approach (50 seeds).

| policy | 100 fresh seeds | nudge benchmark: full tasks of 50 | red lifted after the nudge |
|---|---|---|---|
| memory policy, with memory | 98 | 35 | 44 |
| memory policy, without memory (`--memory-mode off`) | 96 | 41 | 46 |
| control, same data trained without memory | 96 | 40 | 41 |
| sub-goal policy (reference) | 94 | 29 | 31 |
| RECAP-lite | | 33 | 39 |
| full RECAP round | | 27 | 34 |

- The policy reads its memory: its first action chunk differs between memory and none by a median 0.0165 rad at step 5000
  and 0.0070 rad at step 19999, against a noise floor of 0.0032 rad between two launches.
- On fresh seeds all variants are within noise. The recovery gain after a nudge comes from training on the expert
  corrections, not from the memory: the control, which never saw a memory, recovers as well, and the retrieved memory does
  not help recovery (35 with against 41 without, within noise).
- Why the memory cannot help here: in the failures the wrist camera lost the cube in 40 % of the calls after the nudge (3 %
  in the recoveries), where the wrist-picture key says nothing about where the cube is; at the closed-gripper moments after
  the nudge the hint said "keep closed" in 80 % of the calls, since the store's closed moments are mostly holds that worked.
