# The head writer: the policy writes its own history

## The idea

Every method needs a writer that says when a skill is done. The other writers watch the robot from the outside: the
simulator's tracker, rules on the gripper's readings, or a vision-language model looking at the cameras. The head writer
asks a different question: does the policy itself already know when a skill is done? It sees the same pictures and the same
prompt, so the answer may already be somewhere inside its own computation.

![One policy call with the head writer and the history method](detailed/head_call.svg)

How it works, in short:

- At every policy call, the policy server also runs a small add-on, the **event head**. It reads the policy's inner
  representation of this call: the output of one of its internal layers (block 12 of 18), averaged over the words of the
  prompt. That is a list of 2,048 numbers.
- A **logistic regression**, a weighted sum of these 2,048 numbers plus a constant, turns them into one score. A score of at
  least 0.571 means "done".
- The verdict travels back to the proxy together with the actions. When the head says "done" at **two calls in a row**, the
  writer writes the skill's line into the history ("picked up the red block"). The policy sees the new line from the next
  call on.
- The policy itself is not changed. The head is fitted afterwards, on recordings of the policy's own runs, labelled by the
  rules writer.

**Why this is worth trying.** If it works, the robot needs no extra sensors and no extra model to keep track of its task:
the policy writes its own memory. It also answers a research question: does a VLA represent the progress of its task
inside? Our answer is yes, and a simple linear read-out is enough to get it out (see the findings).

**It depends on the policy, completely.** The head reads the policy's own internal numbers:

- another policy, another fine-tune, even another checkpoint of the same model has a different inner representation, so
  the head must be fitted again, on that policy's own runs;
- it was fitted on prompts with history lines, so it belongs to the history method; with another method the policy sees
  other prompts, and the head would have to be fitted for those;
- the server has to reach inside the model: as shipped it reads the backbone of an openpi π0.5 policy, so it works with
  openpi π0.5 policies only.

**What it cannot do well:**

- **It is slow as built.** The head computes the policy's first 13 blocks a second time for every call, which adds about
  545 ms per call. Reading the layer from the policy's
  own computation would make it almost free; that is not built.
- **It copies its teacher.** It learned from the rules writer's labels, on one policy, one instruction and one camera setup,
  in simulation.
- **It misses rare events**, such as a block dropped right after the lift, or a hand that slides along the table after the
  release instead of rising.
- **It decides after the policy call**, so a line reaches the policy one call later than with the other writers.
- **The last skill writes no line** (as in the policy's training data), so it never reports that the whole task is
  finished.

A walkthrough of how it works: where the head sits, why it is built this way, how it is trained and the code:
[detailed/README.md](detailed/README.md).

## Prerequisites

- A policy served with openpi (π0.5), here the history policy ([the history method](../../methods/history/README.md)).
- Recorded runs of that policy, not demonstrations: the camera pictures of every control step, and either the gripper
  readings the rules writer uses (finger width, squeeze, fingertip position) or the number of finished skills at every
  step.
- The openpi environment and one GPU, to compute the features for the fit and to serve the policy; the fit itself runs on
  the CPU.

## How to use it in the framework

1. **Record runs** of the history policy with another writer, for example `--writer rules`, and keep the pictures. One
   `.npz` file per episode with `exterior`, `wrist`, `instruction`, and either `rows` (the gripper readings, which the rules
   writer turns into labels) or `reference` (the number of finished skills per step). `training/fit_event_head.py`
   describes the arrays.
2. **Fit the head:**

   ```
   python training/fit_event_head.py --episodes 'rollouts/*.npz' --config <policy config> \
       --checkpoint <checkpoint dir> --center-crop 0.95 --features rows.npz --out <head dir> [--jax-cache DIR]
   ```

   `--config` is the policy's openpi training config. `--center-crop` is the centre crop of the outside picture that the
   policy uses at inference (0.95 for our policies); use the same value here and at the server. The result is a folder with
   `head.npz` (the weights and everything the head needs), `head.json` (how it was fitted) and `head.sha256` (a checksum,
   checked whenever the head is loaded). No fitted head and no policy checkpoint come with the repository.
3. **Serve the policy together with its head** (in the openpi environment). Every reply then carries
   `event_head: {score, done, threshold, ms}`:

   ```
   python -m vla_memory.writers.head.server --port <server port> --config <policy config> \
       --checkpoint <checkpoint dir> --center-crop 0.95 --head-dir <head dir> [--head-batch 8] [--jax-cache DIR]
   ```

4. **Run the proxy with the head as the writer:**

   ```
   python -m vla_memory.proxy --writer head --method history --task task.yaml --listen-port <port> \
       --upstream-port <server port> [--consecutive 2]
   ```

   `--task` can be left out for the reference task. `--consecutive` is how many "done" verdicts in a row write a line.
   Every line of the proxy's log holds, under `event_head`, the score, the verdict, the count of "done" verdicts in a row,
   and the skill whose line was written, if any (`wrote`).

`--head-batch 8` (the default) and `--jax-cache` make the server's scores match the fit's numbers exactly; the detailed page
explains why.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim. The history policy
(checkpoint 28000), and a head fitted on 129 of its own runs labelled by the rules writer. 100 fresh seeds of 500 control
steps, the same policy every time; only the writer of the history differs:

| writer of the history | full tasks of 100 | time per policy call | 500-step episode |
|---|---|---|---|
| the policy's event head | 78 | 714 ms | 86 s |
| simulator's ground truth | 82 | 169 ms | 51 s |
| rules writer | 79 | 172 ms | 51 s |
| VLM writer | 77 | 1,630 ms | 143 s |

- **No clear difference:** all four writers are close enough that the differences could be chance. Of the 22 failed
  episodes, 20 had no writer error, mostly failed grasps; 2 failed after a missed line.
- **The head's lines against the simulator's tracker:** 250 of 253 written, 236 of them exactly one call after the tracker,
  7 in the same call, 6 later (up to 11 calls), 1 two calls early while the cube was rising (that episode succeeded),
  3 missed, none false. On the 3,744 calls at which the skill was not done, the head said "done" 19 times (0.5 %); asking for
  two in a row turned only one of these into a line.
- **The teacher matters most.** Heads fitted on the demonstrations told the classes apart perfectly on other demonstrations
  (AUROC 0.999 to 1.000) but not on the policy's own runs (0.85 to 0.93): they said "done" at the grasp, before the lift.
  Fitted on the policy's own runs instead, the head reached AUROC 0.991 and wrote no early and no false line on 49 fresh
  runs. AUROC is the chance that a "done" moment gets a higher score than a "not done" moment: 0.5 is guessing, 1.0 perfect.
- **A small multilayer perceptron did no better** than the logistic regression (rollout AUROC 0.856 to 0.921 against 0.849
  to 0.928); the problem was the training data, not the size of the head. The detailed page explains the choice.
- **The price is time:** the exact head pass adds about 545 ms per call, more than the 533 ms that an 8-step action chunk
  lasts at 15 Hz.
