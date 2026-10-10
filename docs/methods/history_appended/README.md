# The history-appended prompt

## The idea

The memory sits inside the policy's own input. The instruction stays the full sentence, exactly as a memoryless policy would
receive it, and a short text is appended that says what has already been done:

```
Pick up the red block, place it back, and then grasp the blue block. History: picked up the red block, placed the red block.
```

A policy fine-tuned on frames labelled this way tells apart phases that look alike from the cameras, such as the start and
the moment after the first cube was put back, and sequences the task itself. The external part only writes a line when a
skill is complete. The idea follows UniMem (Osterberg et al., 2026, arXiv 2608.22869).

Who writes the line is exchangeable; the policy only sees the text. Three writers, chosen with `--mode`:

| writer | what it reads | cost per policy call | real robot |
|---|---|---|---|
| `rules` | finger width, squeeze force and fingertip position of every control step; the wrist image at a pick, to name the object | 0.16 ms, no model | yes |
| `vlm` | the camera images, judged by the planner loop's VLM monitor | about 3 s per question | yes |
| `oracle` | the simulator's ground truth | none | no |

**The rules writer** follows one grasp cycle at every control step, from signals every robot has: *holding* (the fingers
stopped between two widths and squeeze), *lift* (the fingertips rose a set height above the grasp point), *set-down* (the
fingers opened less than a set height above the grasp point; higher up is a drop), *retreat* (the fingertips moved a set
distance up or away from the release point). The task file maps each verb to the trigger that completes it and gives the
thresholds:

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
out and is grasped again does not produce a second line. Skills without a grasp (open a drawer, press a button) use
`trigger: signals`: ordered phases of thresholds on the robot's signals. A colour identifier on the wrist image names the
object held at each pick, so a robot that takes the blue block first gets the right line and the plan is reordered. The same
rules label recorded episodes (`RuleMonitor.label_rows`), so training labels and run-time history come from one definition.

## Prerequisites

- A policy fine-tuned on instruction-plus-history prompts. For the reference task the demonstrations were relabelled per
  frame with the four history states (`training/relabel_history.py`).
- A task file with a `history` section (the phrase per verb, `max_lines`, `rules`, `object_from`) and the gripper thresholds
  under `proprio`. The thresholds are taken from the moments at which the training labels change.
- For `rules`: an environment that sends `memory/proprio_steps` (finger width, squeeze, fingertip x, y, z of every control
  step) and the wrist image; `vla_memory.env_keys.ControlKeys` builds these keys. For `vlm`: the planner loop's model. For
  `oracle`: the ground-truth skill as `memory/oracle_prompt`.

## How to use it in the framework

```
python -m vla_memory.history_appended.proxy --mode rules  --listen-port <port> --upstream-port <policy port> [--task task.yaml]
python -m vla_memory.history_appended.proxy --mode vlm    --listen-port <port> --upstream-port <policy port> [--memory store.npz]
python -m vla_memory.history_appended.proxy --mode oracle --listen-port <port> --upstream-port <policy port>
python -m vla_memory.history_appended.validate --episodes 'recorded/*.npz' --call-every 8
```

The environment connects to `<port>` instead of the policy server and otherwise runs unchanged. Before a rule set drives a
robot, the validator replays recorded episodes (`rows`: the signals per step, `reference`: the completed skills per step)
and reports, per line, how often the writer switches in the same policy call as the reference, earlier, later, missed or
false. The same writer sequences the planner loop's sub-goal policy: `python -m vla_memory.planner_loop.proxy --mode rules`.
Two switches serve timing experiments: `--delay-calls N` shows the history N calls late, `--max-lines N` caps the lines.

## Our findings

Reference task, 100 fresh seeds of 500 control steps, the same history policy, only the writer differs:

| writer | full tasks of 100 | history equal to the ground truth's |
|---|---|---|
| `oracle` | 82 | reference |
| `rules` | 79 | 6297 of 6300 policy calls |
| `vlm` | 77 | 94 % of calls |
| memoryless policy, same instruction, no history | 11 | |

- The three writers are within noise of each other; which episode fails is mostly the policy's own grasp, not the memory. The
  rules writer therefore replaces both the simulator's ground truth and the VLM monitor, in real time and without a model.
- The history solves what sinks the memoryless policy: after placing the red cube, 84 of 87 episodes went on to the blue one,
  against 17 of 31 without history. The gap to the planner loop's sub-goal policy (82 against 92) is grasping (87 against 95
  episodes lifted the red cube), a property of the policy trained on the long prompt, not of the writer.
- Timing matters as much as the event. On the development seeds, a line one or two policy calls late (up to a second) cost
  nothing, six calls late slowed every stage by about the delay (11 of 20 instead of 17), a line written early, at the grasp
  instead of after the lift, gave 0 of 20, and one missed line 0 of 20 as well. A writer must never write early and never
  miss a line; being late is tolerable.
- The colour identifier named every one of 957 recorded picks correctly, including the 31 in which the robot took the blue
  block first.
- As sequencer of the sub-goal policy the rules writer completes 94 of 100 (ground-truth switching 92, the VLM planner 94),
  switches in the same call as the ground truth in 275 of 286 switches and never more than one call apart, at 55 s per
  episode against 84 s with the VLM.
- Limits: the validator checks when the lines come, not which object they name; the identifier is checked separately on
  recorded picks. One task has run live.
