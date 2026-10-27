# The rules writer: the robot's own signals

## The idea

A pick-and-place skill ends with a movement the robot can measure on itself. "Pick up" is done when the fingers hold the
block and the hand has lifted it. "Place" is done when the hand has set the block down, opened, and moved away from it. The
rules writer watches for exactly these two moments. It needs no model and no tracker, only three readings every robot arm
has at every control step:

- the **finger width**: how far apart the two fingers are;
- the **squeeze**: how hard the fingers press (the finger force; a gripper that only reports "holding" or "not holding"
  works too);
- the **fingertip position**, computed from the measured joint angles.

From these readings the writer follows one grasp at a time through four states, *free*, *holding*, *lifted*, *released*
([the picture](detailed/README.md#5-the-grasp-cycle)), and two moments fire a trigger:

- **lift**: the fingers hold something and have risen a set height (9 cm in the reference task) above where they closed;
- **set down**: the fingers opened low down, near where they grasped (opening higher up counts as a drop), and then moved a
  set distance up (6 cm) or away (8 cm).

The task file says which trigger completes which verb, with the thresholds. The reference task's:

```yaml
history:
  rules:
    pick up: {trigger: lift, rise: 0.09}
    place:   {trigger: set_down, release_below: 0.08, up: 0.06, away: 0.08}
  object_from: colour
proprio:
  open_width: 0.06
  empty_width: 0.015
  squeeze_force: 8.0
```

When a trigger fits the verb of the current skill, the skill is complete and the writer moves on to the next skill of the
plan, the task file's order of the skills. It never moves back. With `object_from: colour` it also looks at the wrist
camera at every pick and names the held block by its colour: if the robot picks up blue while the plan expects red, blue's
skills move to the front of the plan.

**Why these thresholds.** The policy was trained on demonstrations whose labels switch at fixed moments: "pick up" ends when
the block is 9 cm up, "place" ends after the hand has moved back from the set-down block. The writer must switch at the same
moments; one that switches earlier makes the policy grasp the same block again.

**What it cannot do well:**

- It only knows what the gripper feels: a block set down in the wrong place, or one that fell over, still counts as placed.
- Without the colour check it cannot tell which block was picked; the colour check needs clearly different colours.
- It follows the task file's order and cannot turn a new instruction into a plan; the VLM writer can.
- It never takes a skill back.
- Its thresholds must be measured from the policy's training demonstrations; wrong thresholds make the policy fail even
  when the writer is right about the event.
- It needs the readings of every control step, not only of the policy calls.

Skills without a grasp (open a drawer, press a button) use a third trigger, `signals`: an ordered list of conditions on the
robot's signals. How the code works and how to set it up for a new robot or task: [detailed/README.md](detailed/README.md).

## Prerequisites

- A task file with `history.rules` and the gripper thresholds under `proprio`, measured where the policy's training labels
  switch (the detailed page, section 1, says how).
- An environment that sends, with every policy call, `memory/proprio_steps` (the finger width, squeeze and fingertip x, y,
  z of every control step since the previous call) and, for the colour check, the wrist picture (`memory/wrist_raw`).
  `vla_memory.env_keys.ControlKeys` builds these entries.
- Recorded episodes with a reference (a tracker, the training labels or a person's annotation), to check a set of rules
  before it drives a robot.

## How to use it in the framework

```
python -m vla_memory.proxy --writer rules --method subgoal --task task.yaml --listen-port <port> --upstream-port <policy port>
python -m vla_memory.writers.rules.validate --episodes 'recorded/*.npz' --task task.yaml
python -m vla_memory.writers.rules.author --variant easy --out authored.yaml --task task.yaml
```

- **The proxy** runs the writer; `--task` can be left out for the reference task, and any method can replace `subgoal`. Its
  log shows at every call whether a skill was completed (`advanced`, `completed`), the grasp's state (`writer_state`) and
  the colour check's answer at a pick (`identified`).
- **The validator** replays recorded episodes through the rules and reports, per skill, how often the writer switches in
  the same policy call as the reference, earlier, later or not at all. Run it before a set of rules drives a robot.
- **The author** lets the planning model draft `history.rules` from the task's completion criteria; check its draft with
  the validator.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim, π0.5 policies; 100 fresh
seeds of 500 control steps, a policy call every 8 steps:

| method | full tasks of 100 | against the simulator's ground truth |
|---|---|---|
| sub-goal | 94 | the same policy call in 275 of 286 switches, never more than one call apart |
| history | 79 | the history equal to the ground truth's in 6,297 of 6,300 policy calls |
| keyframe | 75 | |

- With every method the rules writer is within noise of the simulator's ground truth (92, 82 and 78) and, where both ran,
  of the VLM writer. For a task whose skill order is known it can replace both, in real time and without a model.
- The colour check named the held block correctly in all 957 recorded picks, including the 31 in which the robot took the
  blue block first.
- It adds 0.16 ms to a policy call.
- Only the reference task has run live; other tasks so far only in the framework's tests.
