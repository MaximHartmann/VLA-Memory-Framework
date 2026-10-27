# The sub-goal method: the current skill as the prompt

## The idea

A VLA policy (vision-language-action policy: a network that turns camera pictures and a text prompt into robot movements)
has no memory. After the robot has put the red block back, the table looks as it did at the start. The policy cannot tell
whether it should pick up red again or move on to blue. A policy trained on the full instruction completed only 11 of 100
tasks in our tests.

The sub-goal method keeps the memory outside the policy:

- the policy is trained on a few **short skills** only, such as "pick up the red block" or "place the red block";
- a **writer** keeps the **plan** (the skills in order) and decides when the current skill is done;
- at every policy call the method gives the policy **only the current skill** as its prompt. The user's long instruction
  never reaches the policy.

![The sub-goal method: the policy only ever sees the current skill](detailed/subgoal_method.svg)

The policy stays memoryless; the memory is the writer's plan, kept by the proxy (the relay between the environment and the
policy server). With the VLM writer this setup is called the planner loop.

**What matters most: when the writer switches.** The policy learned from demonstrations whose labels switch to the next skill
at fixed moments, and the writer has to switch at the same moments. In our first runs the simulator switched from "place the
red block" to "pick up the blue block" as soon as the fingers opened. The policy then heard "pick up" while its open hand was
still right above the red cube, and grasped it again. Switching only after the hand had moved back up, where the training
labels switch, fixed it ([the picture](detailed/README.md#3-when-to-switch)).

**What it cannot do:**

- **The policy is blind to the task as a whole.** It does whatever skill it is given. It cannot notice a wrong plan or a
  wrong switch.
- **Every skill must be trained.** A new skill needs demonstrations, and its prompt must be worded exactly as in the training
  data.
- **A missed or wrong switch can cost the episode.** If the writer never moves on, the policy repeats the finished skill. If
  it moves on too early, the policy may redo what it was doing.

How it works in the code, how the policy is trained and when to switch: [detailed/README.md](detailed/README.md).

## Prerequisites

- A VLA policy trained on the task's skills, served over openpi's websocket protocol (another kind of server through
  `--transport`). For the reference task every frame of the demonstrations was relabelled with the skill that was active
  (`training/relabel_subgoals.py`).
- A task file whose skills are worded exactly like the policy's training prompts (the reference task:
  `vla_memory/tasks/red_blue_blocks.yaml`).
- A writer and what it needs: the simulator's ground truth (`oracle`), the robot's gripper readings (`rules`), the planning
  model (`vlm`), or a policy server that hands out the policy's state together with an event head fitted for that policy
  (`head`).

## How to use it in the framework

```
python -m vla_memory.proxy --writer vlm --method subgoal --task task.yaml --listen-port <port> --upstream-port <policy port>
```

- `--task` can be left out for the reference task.
- `--writer` can be `oracle`, `rules`, `vlm` or `head`. Without `--writer` and `--method` the proxy runs `vlm` with
  `subgoal`. Do not use `--writer off`: it hands the policy the full instruction, which a skill-trained policy cannot follow.
- The method has no options of its own.
- With `--log FILE` the proxy writes one line per policy call; its `prompt` field is the skill the policy received.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim, with the sub-goal policy
(π0.5). 100 fresh seeds (a seed fixes where the cubes start) of 500 control steps (a control step is one tick of the robot's
controller, 15 per second), a policy call every 8 steps. A **full task** means: red picked up and put back, then blue picked
up and put back.

| writer | full tasks of 100 | median episode |
|---|---|---|
| `oracle` (the simulator's ground truth) | 92 | 50 s |
| `rules` | 94 | 55 s |
| `vlm`, no example store | 94 | 84 s |
| `vlm`, 20 stored episodes | 87 | 133 s |
| `head` (the policy's event head) | 91 | 52 s |
| no writer: a policy trained on the full instruction | 11 | 60 s |

- **The writers do about equally well.** Their differences are small enough to be chance. The failures that remain are mostly
  failed grasps, not memory.
- **The sub-goal policy grasps more reliably than the history policy.** It lifted the red cube in 95 of 100 episodes, against
  87, and completed more tasks with the same writer (92 against 82 with the ground truth), although that difference could
  still be chance.
- **Switching at the right moment matters.** Before the fix, the policy grasped the red cube again in 14 to 16 of 20
  episodes. After it, the best checkpoints completed 18 to 20 of 20 tasks, and no checkpoint grasped a placed cube again.
