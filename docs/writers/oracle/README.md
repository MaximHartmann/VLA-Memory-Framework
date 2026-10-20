# The oracle writer: the simulator's ground truth

## The idea

A simulator knows exactly where everything is. So it can tell, at any moment, which skill the robot is working on: still
picking up the red block, or already placing it. The oracle writer simply takes the simulator's word for it.

With every policy call, the simulator sends the skill it considers current, as `memory/oracle_prompt`. The oracle looks this
skill up in the plan and makes it the current one; every skill before it in the plan counts as done. The method then shows
this to the policy in the same call: the skill itself (sub-goal method), the instruction with one line per finished skill
(history method), or the picture taken when the last skill was finished (keyframe method).

Two special cases:

- If the simulator sends a text that is not one of the task's skills, the oracle passes it to the policy as it is and keeps
  the current skill.
- If the objects change from episode to episode (the environment names them in `memory/objects`), the plan follows the
  order in which the simulator reports the skills. That way the finished skills are always the ones reported before.

The oracle is the yardstick for the other writers: as long as the simulator's tracker is right, the oracle is never early,
never late and never misses a skill. A real robot has no such tracker, so the oracle only works in simulation.

## Prerequisites

- A simulator with a tracker that sends `memory/oracle_prompt` with every policy call (`vla_memory.env_keys.ControlKeys`
  builds this entry).
- The tracker must switch at the same moments as the policy's training labels. In the reference task, "pick up" ends once
  the block is 9 cm up, and "place" ends once the fingertips are 6 cm above or 8 cm away from the block that was set down.
  These are exactly the moments where the labels of the demonstrations change (`vla_memory/tasks/red_blue_blocks.yaml`,
  `history.rules`).

## How to use it in the framework

```
python -m vla_memory.proxy --writer oracle --method subgoal --task task.yaml --listen-port <port> --upstream-port <policy port>
```

`--task` can be left out for the reference task, and `--method history` or another method can replace `subgoal`. The oracle
has no options of its own. Every line of the proxy's log holds the call's oracle prompt (`oracle_prompt`). If the simulator
sends no `memory/oracle_prompt`, the writer passes the instruction through and says so once in the log.

A walkthrough of the code, from the command line to the last policy call: [detailed/README.md](detailed/README.md).

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim, pi0.5 policies; 100 fresh
seeds of 500 control steps, a policy call every 8 steps:

| method | full tasks of 100 |
|---|---|
| sub-goal | 92 |
| history | 82 |
| keyframe | 78 |

- The oracle is the yardstick for *when* the skills switch, not a limit on success: with the sub-goal method the rules writer
  and the VLM writer each completed 94, with the history method 79 and 77.
- With every writer, most of the failures are the policy's own grasps.
