# Measured results

Full tasks out of 100 fresh seeds on the reference task: pick up and put back a red and then a blue cube with a Franka arm
in Isaac Sim; a pi0.5 policy trained for each method; 500 control steps, a policy call every 8 steps. An empty cell: the
combination is supported but was not measured.

| method \ writer | `oracle` | `rules` | `vlm` | `head` |
|---|---|---|---|---|
| `subgoal` | 92 | 94 | 94 | |
| `history` | 82 | 79 | 77 | 78 |
| `keyframe` | 78 | 75 | | |

For each method the writers are within noise of each other; the memoryless policy, trained on the full instruction, completes
11. The VLM writer ran without a store with the sub-goal method (87 with a store of 20 recorded episodes) and with that store
with the history method. The pages of the writers and methods give the conditions, the timing and what failed:
[oracle](../writers/oracle/README.md), [rules](../writers/rules/README.md), [vlm](../writers/vlm/README.md),
[head](../writers/head/README.md); [subgoal](../methods/subgoal/README.md), [history](../methods/history/README.md),
[keyframe](../methods/keyframe/README.md).
