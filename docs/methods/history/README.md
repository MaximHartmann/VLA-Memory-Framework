# The history method: the instruction with what has been done

## The idea

The memory sits inside the policy's own input. The instruction stays the full sentence, exactly as a memoryless policy would
receive it, and a short text is appended that says what has already been done:

```
Pick up the red block, place it back, and then grasp the blue block. History: picked up the red block, placed the red block.
```

A policy fine-tuned on frames labelled this way tells apart phases that look alike from the cameras, such as the start and
the moment after the first cube was put back, and sequences the task itself. The method writes one line per skill the writer
has completed; who decides that a skill is complete is exchangeable, the policy only sees the text. The idea follows UniMem
(Osterberg et al., 2026, arXiv 2608.22869).

## Prerequisites

- A policy fine-tuned on instruction-plus-history prompts. For the reference task the demonstrations were relabelled per
  frame with the four history states (`training/relabel_history.py`).
- A task file with a `history` section: the phrase per verb, the joiner, `max_lines` (the longest history in the training
  data).
- A writer and what it needs:
  - `rules`: the robot's signals
  - `vlm`: the planning model
  - `oracle`: the environment's ground truth

## How to use it in the framework

```
python -m vla_memory.proxy --writer rules --method history --listen-port <port> --upstream-port <policy port> [--task task.yaml]
```

Another writer with `--writer NAME`. The environment connects to `<port>` instead of the policy server and otherwise runs
unchanged. With the oracle, an oracle prompt that is not one of the task's skills is
forwarded as it is. Two options serve timing experiments: `--delay-calls N` shows the history N calls late, `--max-lines N`
caps the lines (1: every skill after the first is missed).

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim; the history policy (pi0.5,
checkpoint 28000); 100 fresh seeds of 500 control steps, a policy call every 8 steps; only the writer differs:

| writer | full tasks of 100 | history equal to the ground truth's |
|---|---|---|
| `oracle` | 82 | reference |
| `rules` | 79 | 6297 of 6300 policy calls |
| `vlm` | 77 | 94 % of calls |
| memoryless policy, same instruction, no history | 11 | |

- The writers are within noise of each other; which episode fails is mostly the policy's own grasp, not the memory.
- The history solves what sinks the memoryless policy: after placing the red cube, 84 of 87 episodes went on to the blue one,
  against 17 of 31 without history. The gap to the sub-goal method (82 against 92 with the ground truth) is grasping (87
  against 95 episodes lifted the red cube), a property of the policy trained on the long prompt, not of the writer.
- Timing matters as much as the event. On the development seeds, a line one or two policy calls late (up to a second) cost
  nothing, six calls late slowed every stage by about the delay (11 of 20 instead of 17), a line written early, at the grasp
  instead of after the lift, gave 0 of 20, and one missed line 0 of 20 as well. A writer must never write early and never
  miss a line; being late is tolerable.
