# The history method: the instruction with what has been done

## The idea

A VLA policy decides from what it sees now. After the robot has put the red block back, the table looks the way it did at
the start, so the policy cannot tell whether it should pick up red again or go on to blue. A policy trained on the plain
instruction completed only 11 of 100 tasks in our tests.

The history method gives the policy a memory it can read: **a short text about what has been done**. The instruction stays
exactly as the user wrote it. The method adds one sentence at its end:

```
Pick up the red block, place it back, and then grasp the blue block. History: picked up the red block, placed the red block.
```

![The history method: the prompt grows with what has been done](detailed/history_prompt.svg)

- A **writer** decides when a skill is done: the simulator's ground truth, rules on the gripper's readings, a
  vision-language model, or the policy itself through its event head (a small add-on that reads the policy's inner numbers).
  For every finished skill the method adds one line to the history, such as "picked up the red block".
- The policy is **trained on prompts like this**. In the demonstrations, every frame gets the history that is true at that
  frame. So the policy learns to read it, and two moments that look alike are no longer alike for it: "History: none yet."
  at the start, "History: picked up the red block, placed the red block." after red is back on the table.
- Nothing is kept inside the policy. The memory is the text in its input. The proxy builds the text from the writer's plan.
  Who writes the lines can be exchanged; the policy only sees the words. The idea follows UniMem (Osterberg, Wang and
  Schwager, 2026, arXiv 2608.22869).

**What matters most: when a line is written.** The policy believes its history. A line that comes too early, or never,
sends it the wrong way; a line that comes a little late does no harm. The test is under "Our findings".

**What it cannot do:**

- **It trusts the text.** A wrong line misleads the policy for the rest of the episode, and a line is never taken back. On
  20 development seeds, a line written too early (at the grasp, before the lift) gave 0 full tasks, and one line that never
  came gave 0 as well. A line one or two policy calls late (up to about a second) cost nothing.
- **It does not make grasping better.** Most failures that remain are grasps that fail: of the 18 failed episodes with the
  ground truth, 13 never lifted the red block. The history policy lifted the red block in 87 of 100 episodes, the sub-goal
  policy in 95.
- **It says what was done, not where things are.** The history lists finished skills. It holds no picture and no position.
  Our task never needs a position, so we could not test this limit; UniMem reports that a text history alone fails on tasks
  where positions matter.
- **It needs a policy trained for it.** The policy must be fine-tuned on prompts with exactly this wording
  (`training/relabel_history.py`). Another policy, or another wording, needs new training.
- **It needs a good writer.** A writer must never be early and never miss a line. The rules writer needs thresholds measured
  from the training labels, the VLM writer needs seconds for every question, and the head writer's head must be fitted for
  the policy.
- **It remembers one episode only.** The history starts again with "none yet" at every new episode.

A walkthrough of how the method works, step by step, with the code and the training data:
[detailed/README.md](detailed/README.md).

## Prerequisites

- A policy fine-tuned on instruction-plus-history prompts. For the reference task, the demonstrations were relabelled per
  frame with the four histories (`training/relabel_history.py`).
- A task file with a `history` section: the words of the text, one past-tense phrase per verb, and `max_lines`, the longest
  history in the training data (`vla_memory/tasks/red_blue_blocks.yaml`).
- A writer and what it needs:
  - `oracle`: a simulator whose tracker tells the current skill
  - `rules`: the robot's gripper readings of every control step, and the wrist picture
  - `vlm`: the planning model
  - `head`: a policy server that returns the policy's event head

## How to use it in the framework

```
python -m vla_memory.proxy --writer rules --method history --task task.yaml --listen-port <port> --upstream-port <policy port> \
    [--delay-calls N] [--max-lines N]
```

`--task` can be left out for the reference task. Another writer with `--writer NAME`. The environment connects to `<port>`
instead of the policy server and otherwise runs unchanged. Two options serve timing experiments: `--delay-calls N` shows
the history N policy calls late, and `--max-lines N` caps the number of lines (with 1, every skill after the first is
missed). With the oracle writer, a prompt from the simulator that is not one of the task's skills is passed on as it is.
Every line of the proxy's log holds the prompt the policy received (`prompt`).

The training data for the policy comes from the demonstrations of the memoryless policy, with only the prompts changed:

```
python training/relabel_history.py --src <converted dataset> --dst <history dataset> --raw-dir <recordings> [--task task.yaml]
```

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim. The history policy (pi0.5,
checkpoint 28000: the policy after 28,000 training steps) on 100 fresh seeds of 500 control steps, a policy call every 8
steps. A seed fixes where the cubes start; the fresh seeds were not used to choose anything. Only the writer differs. A
**full task** means: red picked up and put back, then blue picked up and put back.

| writer | full tasks of 100 | the history compared with the ground truth's |
|---|---|---|
| `oracle` (the simulator's ground truth) | 82 | the reference |
| `rules` | 79 | the same in 6,297 of the 6,300 policy calls |
| `vlm` | 77 | the same in 94.6 % of the calls |
| `head` | 78 | 236 of the 253 lines came exactly one policy call after the ground truth's |
| the memoryless policy: same instruction, no history | 11 | |

- **No writer is clearly better than another.** Seeds that only one of two writers completed split almost evenly. For the
  ground truth against the rules writer, 14 seeds went only to the ground truth and 11 only to the rules writer: a split
  that could easily be chance. Which episode fails is mostly the policy's own grasp, not the memory.
- **The history solves what sinks the memoryless policy.** After the red cube was put back, 84 of 87 episodes went on to
  the blue one, against 17 of 31 without history.
- **The gap to the sub-goal method is grasping.** With the ground truth the sub-goal policy completed 92 and the history
  policy 82. The sub-goal policy lifted the red cube in 95 episodes, the history policy in 87. A likely reason, not tested: a
  prompt with only the current skill is shorter and less ambiguous than the full instruction with history lines.
- **Timing matters as much as the event.** On 20 development seeds, a line one or two policy calls late cost nothing (16 of
  20 full tasks, 17 with the exact writer). Six calls late made the policy wait for its history and slowed the episode (11 of
  20). A line written early, at the grasp instead of after the lift, gave 0 of 20, and one missed line gave 0 of 20 as well.
  A writer must never write early and never miss a line; being a little late is fine.
- **The method itself costs almost nothing.** The history adds at most 19 tokens (word pieces) to the prompt. With the rules
  writer a policy call takes 168 ms (the median, through the proxy), of which the writer and the method add 0.16 ms. The VLM
  writer needs about 3 s for a question and the head writer about 0.5 s for a call; see their pages.
