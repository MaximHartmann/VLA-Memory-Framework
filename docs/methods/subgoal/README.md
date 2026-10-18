# The sub-goal method: the current skill as the prompt

## The idea

The VLA policy is trained on a few short skills ("pick up the red block", "place the red block", ...). The sub-goal method
hands it, at every policy call, the prompt of the skill the writer considers current; the user's long instruction never
reaches the policy. The policy stays unchanged and memoryless: the memory is the writer's plan and its pointer, the working
memory of the episode, held by the proxy. A writer that decides before the policy call (the simulator's ground truth, the
rules writer, the VLM planner) shows a completed skill in the same call. With the VLM planner as the writer this is the
planner loop ([the VLM writer](../../writers/vlm/README.md)).

## Prerequisites

- A VLA policy trained on the skills of the task, served over the openpi websocket protocol (another server: `--transport`).
  For the reference task the demonstrations were relabelled per frame with the active skill
  (`training/relabel_subgoals.py`).
- A task file whose skills are the policy's training prompts (`vla_memory/tasks/red_blue_blocks.yaml`).
- A writer and what it needs: the environment's ground truth (`oracle`), the robot's signals (`rules`), or the planning model
  (`vlm`).

## How to use it in the framework

```
python -m vla_memory.proxy --writer vlm --method subgoal --listen-port <port> --upstream-port <policy port> [--task task.yaml]
```

`--writer rules` or `--writer oracle` instead of `vlm`; `--writer vlm --method subgoal` is the default. The environment
connects to `<port>` instead of the policy server and otherwise runs unchanged. With the oracle, an oracle prompt that is not
one of the task's skills is forwarded as it is. The method has no options and adds no fields to the proxy's log; every log
record holds the prompt the policy received (`prompt`).

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim; the sub-goal policy (pi0.5);
100 fresh seeds of 500 control steps, a policy call every 8 steps:

| writer | full tasks of 100 | median episode |
|---|---|---|
| `oracle` (ground-truth switching) | 92 | 50 s |
| `rules` | 94 | 55 s |
| `vlm`, no store | 94 | 84 s |
| `vlm`, 20 stored episodes | 87 | 133 s |
| memoryless policy trained on the full instruction, no writer | 11 | |

- The writers are within noise of each other; the failures that remain are mostly grasps.
- The sub-goal policy grasps more reliably than the history policy (95 against 87 of 100 episodes lifted the red cube),
  which is why it completes more tasks with the same writer (92 against 82 with the ground truth).
