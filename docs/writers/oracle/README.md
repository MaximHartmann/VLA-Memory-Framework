# The oracle writer: the simulator's ground truth

## The idea

In simulation the environment knows where every object is, so a tracker can tell when a skill is complete. The oracle writer
takes its word. With every policy call the environment sends the skill its tracker considers current as
`memory/oracle_prompt`, and before the call the writer sets its pointer to that skill's index in the task's skill order. The
method of the run shows the pointer in the same call: the skill itself (sub-goal), the instruction with a line per earlier
skill (history), or the picture taken when the pointer moved (keyframe). An oracle prompt that is not one of the task's
skills is forwarded to the policy unchanged, and the pointer stays where it was.

The oracle is the reference the other writers are measured against: as far as the tracker is right, it is never late, never
early and never misses a skill. A real robot has no such tracker.

## Prerequisites

- A simulator with a ground-truth tracker of the task's skills that sends `memory/oracle_prompt` with every policy call
  (`vla_memory.env_keys.ControlKeys` builds the key).
- A tracker that switches where the policy's training labels switch. In the reference task "pick up" ends once the block is
  9 cm up and "place" once the fingertips are 6 cm above or 8 cm away from the set-down block, the moments at which the
  labels of the demonstrations change (`vla_memory/tasks/red_blue_blocks.yaml`, `history.rules`).

## How to use it in the framework

```
python -m vla_memory.proxy --writer oracle --method subgoal --listen-port <port> --upstream-port <policy port>
```

`--method history` or another method instead of `subgoal`. The writer has no options and adds no fields to the proxy's log;
every log record holds the call's oracle prompt (`oracle_prompt`). Without `memory/oracle_prompt` the writer forwards the
instruction and says so once in the log.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim, pi0.5 policies; 100 fresh
seeds of 500 control steps, a policy call every 8 steps:

| method | full tasks of 100 |
|---|---|
| sub-goal | 92 |
| history | 82 |
| keyframe | 78 |

- The oracle is an upper bound only for when the skills switch, not for the result: with the sub-goal method the rules writer
  and the VLM planner completed 94 each, with the history method 79 and 77. The failures with every writer are mostly the
  policy's own grasps.
