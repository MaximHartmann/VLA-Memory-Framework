# Measured results

How many of 100 episodes completed the whole task, for each writer (columns) and method (rows). The task is the reference
task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim. Each method has its own π0.5 policy,
trained for it. Each episode starts from a fresh seed and runs 500 control steps, with a policy call every 8 steps. An
empty cell means that the combination works but was not measured.

| method \ writer | `oracle` | `rules` | `vlm` | `head` |
|---|---|---|---|---|
| `subgoal` | 92 | 94 | 94 | 91 |
| `history` | 82 | 79 | 77 | 75 |
| `keyframe` | 78 | 75 | | 79 |

- For each method the writers are within noise of each other: the differences are about as large as chance alone
  produces.
- An earlier setup of the head writer, with heads fitted on saved pictures instead of the proxy's recordings, gave 92, 72
  to 78 and 87 (in the order of the table's rows): the runs differ mostly in which grasps fail.
- The memoryless policy, trained on the full instruction, completes 11.
- The VLM writer ran without an example store with the sub-goal method (87 with a store of 20 recorded episodes), and with
  that store with the history method.

The pages of the writers and methods give the conditions, the timing and what failed:
[oracle](../writers/oracle/README.md), [rules](../writers/rules/README.md), [vlm](../writers/vlm/README.md),
[head](../writers/head/README.md); [subgoal](../methods/subgoal/README.md), [history](../methods/history/README.md),
[keyframe](../methods/keyframe/README.md).
