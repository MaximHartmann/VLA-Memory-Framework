# The rules writer: the robot's own signals

## The idea

A skill of a pick-and-place task ends with a movement every robot can measure on itself: the block is lifted, or it is set
down and the hand moves away. The rules writer follows one grasp cycle at every control step, from the finger width, the
squeeze force and the fingertip position: *holding* (the fingers stopped between two widths and squeeze), *lift* (the
fingertips rose a set height above the grasp point), *set-down* (the fingers opened less than a set height above the grasp
point; higher up is a drop), *retreat* (the fingertips moved a set distance up or away from the release point). The task
file maps each verb to the trigger that completes it and gives the thresholds:

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

A trigger completes the current skill only if its verb matches, and the pointer only moves forward, so a block that slipped
out and is grasped again does not complete a skill twice. Verbs that share a trigger may carry thresholds of their own, which
the grasp cycle takes from the current skill's verb: a cube put into a container is let go just above where it was
grasped, a cube stacked on a tall stack several centimetres higher. Skills without a grasp (open a drawer, press a button) use
`trigger: signals`: ordered phases of thresholds on signals the task declares. A colour identifier on the wrist image names
the object held at each pick, so a robot that takes the blue block first completes the right skill and the plan is
reordered (the pending skills of that object in the current stage of the task's `plan_stages`; one stage covers every verb).
The plan is the task order; for an episode whose objects the environment names (`memory/objects`) it covers those objects
only. The writer decides before the policy call, from the control steps since the last call, so the prompt of the same
call shows the decision; it needs no model and takes 0.16 ms per call. The same rules label recorded episodes
(`RuleMonitor.label_rows`), so training labels and run-time decisions come from one definition.

## Prerequisites

- A task file with a `history` section (`rules`, `object_from`, `max_lines`) and the gripper thresholds under `proprio`,
  taken from the moments at which the policy's training labels change.
- An environment that sends `memory/proprio_steps` (finger width, squeeze, fingertip x, y, z of every control step since
  the last call), the wrist image (`memory/wrist_raw`) for the colour identifier, and `memory/signal_steps` for
  `trigger: signals` rules, and `memory/objects` for a task whose episodes hold only some of its objects;
  `vla_memory.env_keys.ControlKeys` builds these keys.
- Recorded episodes with a reference (a ground-truth tracker, the training labels or a person's annotation), to check a rule
  set before it drives a robot.

## How to use it in the framework

```
python -m vla_memory.proxy --writer rules --method subgoal --listen-port <port> --upstream-port <policy port> [--task task.yaml]
python -m vla_memory.writers.rules.validate --episodes 'recorded/*.npz' [--task task.yaml] [--call-every 8]
python -m vla_memory.writers.rules.author --variant easy --out authored.yaml [--task task.yaml]
```

`--method history` or another method instead of `subgoal`. The proxy's log record of a call holds `advanced`,
`skill_index`, `completed` (the skills completed at this call), `writer_state` and `identified` (the colour identifier's
answer at a pick). Before a rule set drives a robot, the validator replays recorded episodes (`rows`: the signals per step,
`reference`: the completed skills per step) and reports, per skill, how often the writer switches in the same policy call
as the reference, earlier, later, missed or false. The rules author lets the planning model write `history.rules` from the
task's completion criteria (`--variant easy`) or from signal statistics of labelled demonstrations (`--variant hard`); its
rules are used only after the validator has checked them.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim, pi0.5 policies; 100 fresh
seeds of 500 control steps, a policy call every 8 steps:

| method | full tasks of 100 | against the simulator's ground truth |
|---|---|---|
| sub-goal | 94 | the same policy call in 275 of 286 switches, never more than one call apart; 55 s per episode (84 s with the VLM planner) |
| history | 79 | the history equal to the ground truth's in 6297 of 6300 policy calls |
| keyframe | 75 | |

- With each method the rules writer is within noise of the simulator's ground truth and, where both ran, of the VLM
  planner. For a task whose skill order is known it replaces both, in real time and without a model; the planner is needed
  to turn a new instruction into a plan.
- The colour identifier named every one of 957 recorded picks correctly, including the 31 in which the robot took the blue
  block first.
- Limits: the validator checks when the skills switch, not which object they name; the identifier is checked separately on
  recorded picks. One task has run live.
