# The oracle writer: the simulator's ground truth

## The idea

A simulator knows exactly where everything is. So at any moment it can tell which skill the robot is working on: still
picking up the red block, or already placing it. The oracle writer takes the simulator's word for it.

The task's skills in order form the **plan**: pick up the red block, place it, pick up the blue block, place it. At every
policy call (one request to the policy for its next actions), the simulator sends the skill it considers current, as
`memory/oracle_prompt`. The oracle finds this skill in the plan and makes it the current one; every skill before it counts
as done. In the same call, the method (what the policy receives, chosen with `--method`) shows this to the policy: the
skill itself (sub-goal method), the instruction with one line per finished skill (history method), or the wrist picture
taken when the last skill was finished (keyframe method).

Two special cases:

- A text that is not one of the task's skills goes to the policy as it is, and the current skill stays.
- If the objects change from episode to episode (the environment names them in `memory/objects`), the plan follows the
  order in which the simulator reports the skills. So the finished skills are always the ones reported earlier.

The oracle is the yardstick for the other writers: as long as the simulator's tracker is right, the oracle is never early,
never late and never misses a skill. A real robot has no such tracker, so the oracle works only in simulation.

## Prerequisites

- A simulator with a **tracker**: code that works out the current skill from the simulator's true state and sends it as
  `memory/oracle_prompt` with every policy call (`vla_memory.env_keys.ControlKeys` builds this entry).
- The tracker must switch at the same moments as the policy's training labels. In the reference task, "pick up" ends once
  the block is 9 cm up, and "place" ends once the fingertips are 6 cm above or 8 cm away from the block that was set down:
  exactly where the labels of the demonstrations change (the rules writer uses the same numbers, `history.rules` in
  `vla_memory/tasks/red_blue_blocks.yaml`).

## How to use it in the framework

```
python -m vla_memory.proxy --writer oracle --method subgoal --task task.yaml --listen-port <port> --upstream-port <policy port>
```

- `--task` can be left out for the reference task, and `--method history` or `--method keyframe` can replace `subgoal`.
- The oracle has no options of its own.
- With `--log FILE`, every line of the proxy's log shows the call's `oracle_prompt` next to the `prompt` the policy got.
- If the simulator sends no `memory/oracle_prompt`, the writer passes the instruction through and says so once in the log.

How the code works, step by step, and how to connect your own simulator: [detailed/README.md](detailed/README.md).

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim, π0.5 policies; 100 episodes
with fresh seeds, each 500 control steps long, with a policy call every 8 steps:

| method | full tasks of 100 |
|---|---|
| sub-goal | 92 |
| history | 82 |
| keyframe | 78 |

- The oracle is the yardstick for *when* the skills switch, not a limit on success: with the sub-goal method the rules writer
  and the VLM writer each completed 94, with the history method 79 and 77.
- With every writer, most of the failures are the policy's own grasps.
