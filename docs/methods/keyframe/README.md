# The keyframe policy: a picture of the last key moment

## The idea

A VLA policy (a network that turns camera pictures and a text prompt into robot movements) decides from what it sees now.
After the robot has put the red block back, the table looks the way it did at the start, so the policy cannot tell what has
already been done. The [history method](../history/README.md) answers with words. The keyframe policy answers with **a
picture**.

The policy gets the plain instruction, as a memoryless policy would, and its two cameras show the scene now. In its **third
image slot** (pi0.5 reads three pictures; the reference policy uses two) it gets the wrist camera's picture of **the last key
moment of its own episode**: the moment the writer last said that a skill was done. In the reference task the key moments are
"picked up red", "placed red" and "picked up blue". Before the first one, the slot is empty.

![The keyframe policy: the third picture is the last key moment](detailed/keyframe_slots.svg)

- A **writer** decides when a skill is done, as for the history method (we measured the simulator's ground truth, the rules
  writer and the policy's event head). At the call at which the writer completes a skill, the method keeps that call's wrist
  picture and sends it with every observation from then on. The next key moment replaces it.
- The policy is **trained on demonstrations relabelled the same way**: a frame after a key moment gets the wrist picture of a
  frame drawn from the 16 frames after that moment, never from the frame's own future. So the start and the moment after the
  first block was put back no longer look alike to it: the slot is empty in the first and shows the put-down in the second.

**What it cannot do:**

- **It depends completely on the picture,** and the picture must come at the right moment. With the picture never sent, the
  policy lifted the red block and held it in the air until the end of the episode. A "placed red" picture taken with the
  hand lower than in the training pictures can make the robot grasp red again.
- **It keeps only the last key moment.** The next picture replaces the old one.
- **We cannot read what it remembers.** We did not test whether the policy reads what the picture shows or only that a
  picture is there.
- **It needs a policy trained for it,** with a third image slot that it ignores while the slot is empty. The environment must
  send the raw wrist picture.
- **Grasping is still its main weakness.** Most failures that remain are grasps: with the simulator's events, 7 of the 22
  failed episodes never lifted the red block and 5 never lifted the blue block.

How it works in the code, the later "placed" picture and how the training data are made:
[detailed/README.md](detailed/README.md).

## Prerequisites

- The recorded demonstrations as two LeRobot datasets of the same frames: the memoryless policy's, with the cameras stored as
  images and the plain instruction as each frame's task (`training/convert_to_lerobot.py --no-videos`), and the history
  policy's, whose prompts mark the key moments (`training/relabel_history.py`).
- A policy that takes a third picture, `observation/memory_image`, and ignores it while the picture is missing or all zero
  (pi0.5 has a third image slot with a mask), trained on the keyframe dataset.
- At run time, an environment that sends the raw wrist picture (`memory/wrist_raw`) and what the writer reads
  (`memory/oracle_prompt` for the simulator's key moments, `memory/proprio_steps` for the rules writer), as
  `vla_memory.env_keys.ControlKeys` builds them.
- For the later "placed" picture only: the joint state in `observation/state` (the 7 arm joints first). The calculation of
  where the hand is from the joint angles is built in for the reference Franka arm.

## How to use it in the framework

1. **Make the training data:**

   ```
   python training/relabel_keyframe.py --train <frames train>:<history train> --val <frames val>:<history val> \
       --root <dataset root> --repo-id <user>/<name> [--task task.yaml]
   ```

   It checks that both datasets of a split hold the same frames and writes `<name>_train` and `<name>_val`, with the new
   image column `memory_image` and a per-frame list, `keyframe_frames.csv`.
2. **Train** the policy on that dataset, with its third image slot reading `memory_image` (the reference policy: the
   memoryless policy's recipe, 30,000 steps; `training/README.md`).
3. **Run:** serve a checkpoint and start the proxy with the keyframe method. The prompt stays the plain instruction:

   ```
   python -m vla_memory.proxy --writer oracle --method keyframe --task task.yaml --listen-port <port> \
       --upstream-port <policy port> [--keyframe-mode off] [--keyframe-placed-later]
   ```

   - `--task` can be left out for the reference task. `--writer rules` or `--writer head` replaces the simulator's events.
   - `--keyframe-mode off` takes the pictures but never sends them: the control.
   - `--keyframe-placed-later` takes the picture of a placing event ("placed red" in the reference task) only once the
     fingertips are at least 10 cm above the placed cube (`--keyframe-placed-height`), at the latest at the third call
     after the signal (`--keyframe-placed-calls`).
   - With `--log FILE`, every line of the proxy's log says which key moment the current picture belongs to, the step it was
     taken at and whether it was sent.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim. The keyframe policy
(checkpoint 29000: the policy after 29,000 training steps, the best of 30 saved checkpoints on the development seeds 1-20, 19
of 20) on 100 fresh seeds of 500 control steps, a policy call every 8 steps. A seed fixes where the cubes start; the fresh
seeds were not used to choose anything. A **full task** means: red picked up and put back, then blue picked up and put back.

| policy and writer | full tasks of 100 |
|---|---|
| keyframe policy, the simulator's key moments (`--writer oracle`) | 78 |
| keyframe policy, the rules writer (`--writer rules`) | 75 |
| keyframe policy, the policy's event head (`--writer head`) | 79 |
| keyframe policy, the picture never sent (`--keyframe-mode off`) | 0 |
| history policy: ground truth / rules / VLM / event head | 82 / 79 / 77 / 75 |
| memoryless policy | 11 |

- **As good as the history policy with every writer.** Seeds that only one of two policies completed split almost
  evenly: for the keyframe policy with the simulator's events against the history policy with the ground truth, 12 seeds
  went only to the keyframe policy and 16 only to the history policy, a split that could easily be chance. The same
  holds across all eight pairs of these two kinds of runs (the keyframe policy with the simulator's or the rules writer's key
  moments against the four history runs): none shows a clear difference.
- **Far better than the memoryless policy:** 78 against 11. The picture is the reason: without it the policy lifts red and
  holds it until the end.
- **The later "placed" picture removes the re-grasps but does not clearly raise the full tasks.** On 100 new seeds
  (2001-2100) the re-grasps fell from 17 of 94 episodes to 0 of 90 with the rules writer (a drop far too large to be
  chance) and from 5 of 92 to 1 of 89 with the simulator's events. The full tasks went from 74 to 82 with the rules
  writer and from 83 to 78 with the simulator's events: differences that chance produces easily. The failures that
  remain are grasps. The original moment stays the default.
- **The policy's event head takes the "placed" picture later still.** It decides after a call and needs two "done"
  verdicts in a row, so its picture came a call or two after the simulator's, with the hand higher (a median 12.8 against
  9.9 cm above the red cube's centre), and no episode grasped red again. It completed 79, against 78 with the simulator's
  events ([the head writer](../../writers/head/README.md)); an earlier setup of the head reached 87, the difference lying
  in failed grasps.
- **The writers are on time.** With the simulator's events all 264 pictures were taken at the simulator's call. The rules
  writer took 258 of 263 at the same call; 2 came one call late, and 3 a little early, with the hand 5.5 to 5.8 cm above the
  cube, just under the simulator's 6 cm rule.
- **A picture costs about 3 ms per policy call,** the proxy's hop (172 against 170 ms); the later "placed" picture adds 1 to
  3 ms.
