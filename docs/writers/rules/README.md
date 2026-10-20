# The rules writer: the robot's own signals

## The idea

A pick-and-place skill ends with a movement the robot can measure on itself. "Pick up" is done when the fingers hold the
block and the hand has lifted it. "Place" is done when the hand has set the block down, opened, and moved away from it. The
rules writer watches for exactly these two moments. It needs no model and no tracker, only three readings every robot arm
has at every control step:

- the **finger width**: how far apart the two fingers are;
- the **squeeze**: how hard the fingers press on something (the finger force; a gripper that only reports "holding" or "not
  holding" works too);
- the **fingertip position**: where the fingertips are, computed from the measured joint angles.

From these readings the writer follows one grasp cycle at a time: *free*, *holding*, *lifted*, *released*, and back to
*free* ([the picture](detailed/README.md#4-the-grasp-cycle)). Two moments of the cycle fire a trigger:

- **lift**: the fingers hold something and have risen a set height (9 cm in the reference task) above the point where they
  closed;
- **set down**: the fingers opened low down, close to where they grasped (opening higher up counts as a drop), and then
  moved a set distance up (6 cm) or away (8 cm).

The task file says which trigger completes which verb, and gives the thresholds. The reference task's:

```yaml
history:
  max_lines: 3
  rules:
    pick up: {trigger: lift, rise: 0.09}
    place:   {trigger: set_down, release_below: 0.08, up: 0.06, away: 0.08}
  object_from: colour
proprio:
  open_width: 0.06
  empty_width: 0.015
  squeeze_force: 8.0
```

When a trigger fires and fits the verb of the current skill, the writer marks the skill as complete and moves on to the next
skill of the plan. It never moves back, so a block that slips out and is grasped again is not "picked up" twice. The plan is
the task file's order of the skills; when the environment names the episode's objects (`memory/objects`), the plan covers
only those.

**Which object?** The grasp cycle tells *when* something was picked up, not *what*. With `object_from: colour` the writer
looks at the wrist image at every pick and names the held object by its colour. If the robot picks up the blue block while
the plan expects the red one, the writer moves blue's skills to the front of the plan, so the right skill is completed.

**Why these thresholds.** The policy was trained on demonstrations whose labels switch at fixed moments: "pick up" ends when
the block is 9 cm up, "place" ends after the hand has retreated from the set-down block. The writer must switch at the same
moments; one that switches earlier than the labels makes the policy grasp the same block again. The same rules can label
recorded episodes (`RuleMonitor.label_rows`), so the training labels and the decisions at run time come from one definition.

**Other cases.** Verbs that share a trigger may have thresholds of their own, and the writer uses those of the current skill:
a cube put into a box is let go just above where it was grasped, a cube put on a tall stack much higher. Skills without a
grasp (open a drawer, press a button) use `trigger: signals`: an ordered list of conditions on the robot's signals.

**Timing.** The writer decides before the policy call, from the readings of every control step since the previous call, so
the policy sees the next skill in the same call. It adds about 0.16 ms to a policy call in the proxy (median of a full
evaluation run); the rules themselves take about 0.03 ms of that, the colour check at a pick about 0.08 ms.

**What it cannot do.**

- It only knows what the gripper feels. A block that was set down but fell over, or landed in the wrong place, still counts
  as placed.
- Without the colour check it cannot tell which object was picked and assumes the plan's. The colour check needs objects of
  clearly different colours.
- It follows the task file's order (apart from picks that the colour check moves forward). It cannot turn a new instruction
  into a plan; the VLM writer can.
- It never takes a skill back. If a person puts a placed block back where it was, the writer still counts it as placed.
- Its thresholds must be measured from the policy's training demonstrations; thresholds that do not match make the policy
  fail even when the writer is right about the event.
- It needs the readings of every control step, not only those at the policy calls; a short event between calls is missed
  otherwise.

A walkthrough of the code, from the command line to the last policy call: [detailed/README.md](detailed/README.md).

## Prerequisites

- A task file with a `history` section (`rules`, `object_from`, `max_lines`) and the gripper thresholds under `proprio`,
  measured at the moments at which the policy's training labels change.
- An environment that sends, with every policy call:
  - `memory/proprio_steps`: the finger width, the squeeze and the fingertip position x, y, z of every control step since the
    previous call;
  - the wrist image (`memory/wrist_raw`) for the colour check;
  - `memory/signal_steps` for `trigger: signals` rules;
  - `memory/objects` for a task whose episodes hold only some of its objects.

  `vla_memory.env_keys.ControlKeys` builds these entries.
- Recorded episodes with a reference (a ground-truth tracker, the training labels or a person's annotation), to check a set
  of rules before it drives a robot.

## How to use it in the framework

```
python -m vla_memory.proxy --writer rules --method subgoal --task task.yaml --listen-port <port> --upstream-port <policy port>
python -m vla_memory.writers.rules.validate --episodes 'recorded/*.npz' --task task.yaml [--call-every 8]
python -m vla_memory.writers.rules.author --variant easy --out authored.yaml --task task.yaml
```

`--task` can be left out for the reference task, and `--method history` or another method can replace `subgoal`.

- **The proxy's log** holds, for every call: `advanced` (a skill was completed), `skill_index`, `completed` (the skills
  completed at this call), `writer_state` (the grasp cycle's state) and `identified` (the colour check's answer at a pick).
- **The validator** replays recorded episodes (`rows`: the readings per step; `reference`: the number of completed skills per
  step) and reports, for every skill, how often the writer switches in the same policy call as the reference, earlier,
  later, not at all, or without a reason. Run it before a set of rules drives a robot.
- **The author** lets the planning model write `history.rules`: from the task's completion criteria (`--variant easy`) or
  from statistics of the signals in labelled demonstrations (`--variant hard`). Its rules are used only after the validator
  has checked them.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim, pi0.5 policies; 100 fresh
seeds of 500 control steps, a policy call every 8 steps:

| method | full tasks of 100 | against the simulator's ground truth |
|---|---|---|
| sub-goal | 94 | the same policy call in 275 of 286 switches, never more than one call apart; 55 s per episode (84 s with the VLM planner) |
| history | 79 | the history equal to the ground truth's in 6297 of 6300 policy calls |
| keyframe | 75 | |

- With every method the rules writer is within noise of the simulator's ground truth and, where both ran, of the VLM
  planner. For a task whose skill order is known it can replace both, in real time and without a model; the VLM planner is
  needed to turn a new instruction into a plan.
- The colour check named the held block correctly in all 957 recorded picks, including the 31 in which the robot took the
  blue block first.
- Limits of these measurements: the validator checks when the skills switch, not which object they name; the colour check
  was measured separately, on recorded picks. Only the reference task has run live.
