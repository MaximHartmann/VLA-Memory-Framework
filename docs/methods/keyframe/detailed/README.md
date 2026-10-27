# The keyframe policy in detail

This page explains how the keyframe policy works: how the proxy keeps the picture of the last key moment and sends it at
every call, how the later "placed" picture works, how the training data are made, and how to use the method with your own
policy and task. What the method is for and what we found are on the short page, [docs/methods/keyframe](../README.md).

![The keyframe policy: the third picture is the last key moment](keyframe_slots.svg)

**In short.** The prompt stays the plain instruction. pi0.5 reads three pictures: the outside camera, the wrist camera and a
third slot that the memoryless policy leaves empty. The method puts into that third slot the wrist picture of the last **key
moment** of the episode: the call at which the writer last completed a skill. For frames that look alike but need different
actions, this picture is the only input that differs, so training teaches the policy to look at it. Unlike the history line,
a picture needs no wording that must match the training data, and it can show where things were. The method does not decide
when a skill is done; the writer does.

## The pieces

| file | what it does |
|---|---|
| `vla_memory/methods/keyframe/method.py` | `KeyframeMethod`: the options; keeps the keyframe and adds it to the observation |
| `vla_memory/methods/keyframe/placed.py` | `PlacedLater`: the option that takes the picture of a placing event once the hand has risen |
| `vla_memory/methods/keyframe/labels.py` | the training labels: which frame's wrist picture every training frame gets |
| `vla_memory/methods/memory_policy/query.py` | `wrist_frame`: the call's raw wrist picture, or the policy's own wrist image as a fallback |
| `vla_memory/methods/memory_policy/kinematics.py` | `FrankaKinematics`: where the fingertips are, from the 7 joint angles (for the later picture) |
| `vla_memory/proxy/relay.py` | `Relay.infer`: what happens at every policy call |
| `training/relabel_keyframe.py` | makes the training dataset (section 6) |

The parts that every writer and method share (the start-up, the server, the connection to the policy) are explained step by
step in the [oracle walkthrough](../../../writers/oracle/detailed/README.md).

## 1. Start-up

This page follows one run of

```
python -m vla_memory.proxy --writer oracle --method keyframe --task vla_memory/tasks/red_blue_blocks.yaml \
    --listen-port 8001 --upstream-port 8000
```

1. **The command line is read twice** (`parse_command_line`). `resolve_method("keyframe")` finds
   `vla_memory.methods.keyframe.method:KeyframeMethod` and imports it. The second pass reads the options the method adds
   (`KeyframeMethod.add_arguments`):

   | option | meaning | default |
   |---|---|---|
   | `--keyframe-mode normal\|off` | `off` takes and logs the pictures but never sends them (the control) | normal |
   | `--keyframe-max-events N` | the most key moments that take a new picture; never the last skill of the episode's plan, so a value above the plan's length minus one changes nothing | the task file's `history.max_lines` (3) |
   | `--keyframe-placed-later` | take a placing event's picture only once the hand has risen (section 4) | off |
   | `--keyframe-placed-height M` | with the option above: the fingertips' height above the placed object, in m | 0.1 |
   | `--keyframe-placed-calls N` | with the option above: the latest call after the signal | 3 |

2. **The task file is loaded** (`TaskSpec.load_or_default`) and `KeyframeMethod.build(args, task)` makes the method:
   - it sets `max_events` from `--keyframe-max-events`, else from `history.max_lines` of the task file (else the number of
     the task's skills minus one);
   - it takes the name of the policy's wrist image from `--image-keys` (the second entry, `observation/wrist_image` by
     default): the fallback when the environment sends no raw picture;
   - with `--keyframe-placed-later` it builds `PlacedLater.for_task(task, height, calls)` (section 4).
3. **`KeyframeMethod.__init__`** calls `start_episode`: no keyframe yet (`keyframe = None`, `event = 0`).
4. **When the environment connects**, `KeyframeMethod.metadata()` adds the settings to what the proxy tells it:
   `{"keyframe": {"mode": ..., "max_events": ...}}`.

## 2. What the environment sends

- the policy's usual entries, and the plain instruction as `prompt`;
- `memory/wrist_raw`: the raw wrist picture of this call (240 x 320 in the reference setup). The training pictures are of this
  kind. Without it the method falls back to the policy's own wrist image and says so once in the log;
- what the writer reads: `memory/oracle_prompt` for the oracle; `memory/proprio_steps` and the wrist picture for the rules
  writer;
- for `--keyframe-placed-later` only: the joint state in `observation/state` (or `observation/joint_position`), the 7 arm
  joints first.

`vla_memory/env_keys.py` (`ControlKeys`) builds the `memory/` entries.

## 3. One policy call

![One policy call with the oracle writer and the keyframe method](keyframe_call.svg)

`Relay.infer` handles the call:

1. **It sorts the entries** (`split_control_keys`) and **starts a new episode if one begins** (`_begin_episode`):
   `KeyframeMethod.start_episode` forgets the keyframe, and `PlacedLater.reset` forgets a waiting one.
2. **The writer decides, before the call** (`writer.before_call`). The oracle points at the skill that `memory/oracle_prompt`
   names; the skills before it count as completed.
3. **The method sets the prompt** (`Relay._decide_prompt`): `KeyframeMethod.prompt` returns the instruction unchanged.
4. **`KeyframeMethod.before_policy`** does the work of this method:
   1. `events = min(len(writer.completed()), most_shown(max_events, writer))`: how many key moments there have been. The
      helper `most_shown` (`methods/base.py`) caps the count at `max_events` and at the number of skills in the episode's
      plan minus one, because the last skill of a plan has no key moment in the training data.
   2. If `events` is larger than the event of the current keyframe (`self.event`), a skill was completed since the keyframe
      was taken, and `_take` stores a new one: `wrist_frame` gives this call's `memory/wrist_raw` (else the policy's wrist
      image, "fallback"); it is copied into `self.keyframe`, and the event number and the step are noted.
   3. If there is a keyframe and the mode is `normal`, it goes into the observation as `observation/memory_image`. In mode
      `off` it is kept and logged but never sent.
   4. It returns the log fields: `keyframe_mode`, and under `keyframe`: `source` (the writer), `events`, `stored` (a picture
      was taken at this call), `event`, `step`, `sent`, and `frame` ("raw" or "fallback") at a call that stores one.
5. **The call** (`_forward`): the observation goes to the policy server. On the policy's side, an input transform must put
   `observation/memory_image` into the third image slot (in openpi, `right_wrist_0_rgb`) and switch that slot's mask on (the
   mask tells the network whether to look at the slot). A missing or all-zero picture gives an empty slot with the mask off.
6. **After the call** the writer looks at the reply (`writer.after_call`; the oracle does nothing), and the reply goes back
   with the log record.

If two skills are completed within one call, the one picture of that call becomes the keyframe of the newer event. Key
moments after the `max_events`-th, and the completion of the plan's last skill, do not replace the picture. A new episode
clears it.

## 4. The later "placed" picture (`--keyframe-placed-later`)

By default the picture of a key moment is the wrist picture of the call at which the writer signals it. For a placing event,
that call can come while the hand is still low: the simulator's rule fires with the fingertips 6 cm above the cube, while the
training pictures of "placed red" show the hand about 10 cm up. A picture lower than any seen in training can make the robot
grasp the placed block again. With the option, `_take_later` replaces `_take`:

1. **Which events wait.** `PlacedLater.for_task` takes the placing verbs from the task file (`TaskSpec.placed_verbs`:
   `keyframe.placed_verbs`, else the verbs of the rules writer's `set_down` rules, here "place") and the resting height of an
   object's centre (`keyframe.rest_z`, 0.76 m). `PlacedLater.is_placing` says whether the newest completed skill has one of
   these verbs.
2. **The hand's height.** `PlacedLater.fingertip_height` computes the fingertips' height from the 7 joint angles with the
   forward kinematics (`FrankaKinematics.tcp_position`) and subtracts `rest_z`.
3. **Waiting.** A placing event does not take its picture at once. `_wait_for_hand` asks `PlacedLater.wait` at every call:
   take it now if the fingertips are at least `--keyframe-placed-height` above the object ("height"); take it now if this is
   the `--keyframe-placed-calls`-th call after the signal ("fallback"); else keep waiting ("waiting"). Until then the
   previous keyframe stays in the slot.
4. **A newer event** that comes while a placing event waits is taken at once, and the waiting one is dropped
   (`dropped_pending`).
5. **The log record** gets `placed_later`: the `height` and, while an event waits or is taken, `pending_calls`,
   `signal_step` and `reason` (waiting, height or fallback).

## 5. A whole episode of the reference task

| calls | the writer | the keyframe sent with the observation |
|---|---|---|
| until the red block is 9 cm up | no skill completed | none: the slot is empty and masked out |
| from the first call after the lift | "pick up the red block" completed | the wrist picture of this call, red lifted |
| from the first call after the hand has left the red block | "place the red block" completed | the wrist picture of this call (or, with the later option, of the first call with the fingertips 10 cm above the cube) |
| from the first call after the blue block is 9 cm up | "pick up the blue block" completed | the wrist picture of this call |
| after the blue block is set down | "place the blue block" completed: the plan's last skill | unchanged: the last skill of the plan takes no picture |

With `--writer off` no skill is ever completed, so no picture is taken and the policy gets an empty slot at every call. With
`--keyframe-mode off` the pictures are taken and logged but never sent: the control that shows whether a policy reads the
picture.

## 6. How the training data are made

![How the training keyframes are drawn from the frames of a demonstration](keyframe_labels.svg)

**Two datasets of the same frames.** `training/relabel_keyframe.py` reads two LeRobot datasets (the data format the training
code reads) per split:

- the memoryless policy's, with the cameras stored as images (`training/convert_to_lerobot.py --no-videos`) and the plain
  instruction as each frame's task;
- the history policy's (`training/relabel_history.py`), whose prompts tell how many skills are completed at every frame.

```
python training/relabel_keyframe.py --train <frames train>:<history train> --val <frames val>:<history val> \
    --root <dataset root> --repo-id <user>/<name> [--task task.yaml] [--limit N]
```

Run it where `lerobot` is installed. `--limit N` handles only the first N episodes of
each split, as a quick test. For every episode the script (`label_pair`):

1. compares the two datasets column by column (both cameras' PNG bytes, state, actions, timestamps and indices) and stops at
   the first difference (`first_difference`);
2. reads the number of history lines of every frame (`labels.history_counts`), and checks that each prompt is the
   instruction followed by a history of the task's skills in order;
3. finds the key moments b1, b2 and b3: the first frame whose history has 1, 2 and 3 lines (`labels.boundaries`);
4. gives every frame its event e (the number of key moments at or before it) and a **source frame**, the frame whose wrist
   picture becomes this frame's keyframe (`labels.label_episode`, `labels.source_frame`). The source frame is drawn from the
   frames b_e to min(b_e + 16, t), where t is the frame itself, by a fixed draw per frame (a checksum of the split, the
   episode and the frame, so every run gives the same result). A frame before b1 gets an all-zero picture;
5. writes the frame with its plain instruction and the new image column `memory_image` (stored like the wrist picture).

The output datasets are `<root>/<repo-id>_train` and `<root>/<repo-id>_val`. Each folder also gets `keyframe_frames.csv`:
for every frame its event, the first frame of the event and the source frame (-1: none).

**Why a window of 16 frames, never later than the frame itself.** At run time the picture is taken at the first call after
the writer signals the key moment: 0 to 7 control steps after the signal (a writer that decides after the call, like the head
writer, is one call later: up to 15). Drawing the training picture from a window, instead of always taking the same frame,
trains the policy on this jitter. The rule "never later than the frame itself" keeps a frame just after a key moment from
getting a picture from its own future. In the reference demonstrations the key moments are at least 66 frames apart, so the
window never reaches the next key moment. The window assumes that the writer signals at the key moment of the labels; for
"placed red" the simulator's rule fires a little earlier than the labels change, which the later "placed" picture corrects
(section 4).

**Training.** The policy is trained like the memoryless policy, with two differences: the dataset above, and an input
transform in which the column `memory_image` fills the third image slot, with its mask on for every frame that has a
picture. The prompt is the plain instruction. The normalisation statistics (the means and spreads that scale the numbers
going into the network) can stay those of the memoryless policy, because the pictures do not change them
([training/README.md](../../../../training/README.md)).

## 7. Using it with your own policy and task

1. **A policy with a free image slot.** pi0.5 has a third slot (`right_wrist_0_rgb`). Another VLA needs a slot of its own for
   the picture, with a mask or an equivalent that makes it ignore an empty slot.
2. **The task file:** the `history` section (its `max_lines` is the most key moments; the last skill of a plan never takes a
   picture) and, for the later picture, the `keyframe` section: `rest_z`, the height of the centre of a resting object, and
   `placed_verbs`, the verbs of the placing skills (default: the verbs of the `set_down` rules).
3. **Two datasets of the same frames:** the plain one and the history one (`training/relabel_history.py --task
   your_task.yaml` gives the key moments). Then `training/relabel_keyframe.py --task your_task.yaml` writes the dataset with
   `memory_image`. Its window (16 frames after each key moment) fits a writer that signals at most a call or two after the
   event; for a slower writer, widen `WINDOW` in `labels.py` and make the dataset again.
4. **Train** with your policy's own recipe, with the new column in the third slot.
5. **Check the key moments of your writer** against the training moments: they must be the moments where your history labels
   change. For a placing event, check at what height the hand is when the writer signals it. If it is lower than in the
   training pictures, use `--keyframe-placed-later` with a height from your data (`--keyframe-placed-height`). Another arm
   needs its own forward kinematics: `PlacedLater` takes any object with a method `tcp_position(joints)` as `kinematics`.
6. **Run** the proxy with `--method keyframe`, and once with `--keyframe-mode off`, the control that shows whether your
   policy reads the picture.

## 8. Checking a run

With `--log FILE` every line is one call:

- `keyframe.event`: which key moment the current picture belongs to (empty before the first);
- `keyframe.step`: the control step (a tick of the robot's controller, 15 per second) at which it was taken;
- `keyframe.sent`: whether it went to the policy, and `keyframe.stored`: whether it was taken at this call;
- `keyframe.frame`: "raw" or "fallback", at a call that took a picture;
- `prompt`: the plain instruction, at every call.

## 9. Limits to keep in mind

- **The policy relies on its picture.** A writer that never signals a key moment, or an environment that never sends the raw
  wrist picture (the method then sends the policy's own wrist image, which is not what the policy was trained on), leaves the
  policy without the input it learned to use.
- **The picture must match the training pictures.** A key moment signalled at another point of the movement than in the
  labels gives a picture the policy has never seen.
- **One picture only.** A task that needs an earlier key moment than the last needs more slots, or the history method.
- **The event head is fitted per policy.** A head writer for the keyframe policy needs a head fitted for that policy.

## Where the idea comes from

- **UniMem** (Osterberg, Wang and Schwager, 2026, arXiv 2608.22869) writes events as text into the prompt and, at the same
  moment, stores the camera views as keyframes (at most four). Text alone was enough for a sequential task and failed on a
  task where positions matter (TableClean: 0 with text only, against 80 with text and keyframes). This is the case for
  pictures.
- **KEMO** (Zeng et al., 2026, arXiv 2606.23589) picks keyframes with a kinematic rule (the robot slowing down near a
  manipulation event) and feeds each one to pi0.5 as a few extra tokens. Keyframes at events beat uniformly sampled and recent
  frames.
- **EventVLA** (Yang et al., 2026, arXiv 2606.20092) predicts with a head on the VLA's last-layer states when a keyframe is
  due, and keeps the first frame, two anchors and up to five keyframes.
- **MemER** (Sridhar et al., 2025, arXiv 2510.20328) lets a vision-language model choose and track up to eight keyframes and
  write the next sub-instruction for a pi0.5 low-level policy.
- **ARM** (James and Davison, 2021, arXiv 2105.14829) defines the classic keyframe rule of imitation learning: a moment where
  the gripper changes state or the arm comes to rest (also used in C2F-ARM, arXiv 2106.12534, and PerAct, arXiv 2209.05451).
  Such a rule fires at the grasp, before the lift: earlier than the moments our policies are trained on.

What is different here: the key moment is a completed skill, signalled by a writer that can be exchanged, not a rule of the
keyframe method itself. There is exactly one picture, the last one, in a slot that already exists, so the policy's network
is unchanged, and the policy is trained on demonstrations alone.
