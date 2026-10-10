# RECAP-lite: learning from the policy's own rollouts

## The idea

The memory goes into the policy's weights. The policy runs in the simulator, its successful executions are kept, turned into
training episodes and the policy is fine-tuned on them, mixed with a few original demonstrations. This is the filtered-cloning
core of RECAP from π*0.6 (Physical Intelligence, 2025, arXiv 2511.14759): the autonomous rollouts and the training on them are
kept; the human corrections, the value function, the per-transition advantage indicator and the repeated rounds are dropped.
What remains is filtered behaviour cloning on the agent's own successes, the cheapest member of the family and one that needs
no new training code, because the successful rollouts simply become demonstrations. At run time nothing changes: the result
is a drop-in replacement for the policy inside the planner loop.

## Prerequisites

- A trained sub-goal policy and its training pipeline (openpi for pi0.5; `training/`).
- A simulator or robot that records, per control step, the joints, the gripper, the executed actions and the camera frames,
  with a success criterion per episode.
- The scripts that turn demonstrations into training data (`training/convert_to_lerobot.py`, `training/relabel_subgoals.py`);
  kept rollouts go through the same ones.

## How to use it in the framework

1. **Roll out** the policy closed-loop on fresh seeds with ground-truth sub-prompts and record every step.
2. **Filter**: keep only the episodes that completed the whole task, cut shortly after the final placement (181 of 200 in the
   reference run).
3. **Convert** them like demonstrations and **mix** in original demonstrations as replay against drift (40 of 221 episodes).
4. **Fine-tune** from the parent's weights with the same LoRA settings, a short schedule and a lower learning rate
   (`training/README.md`, "Running a fine-tuning"; the reference run: 5000 steps, 1e-5 to 1e-6, cosine).
5. **Select** the checkpoint on development seeds and evaluate it like every other policy. It replaces the parent in the
   planner loop (`--upstream-port` of the new policy server); the proxy, the planner and the store stay the same.

The framework's part is the data side; the training itself belongs to the policy's own code base.

## Our findings

Reference task, 100 fresh seeds, full tasks of 100:

| policy | ground-truth sub-prompts | the planner loop's monitor with a 20-episode store |
|---|---|---|
| parent (sub-goal policy, step 20000) | 92 | 90 |
| RECAP-lite, one round (step 4999) | 94 | 94 |

- One round keeps everything the parent could do and recovers some of its failures (6 of the parent's 8), but adds 4 new
  ones of the same kind: the gripper closing repeatedly beside a cube it had nudged. The differences are within noise
  (paired by seed: 7 wins, 89 ties, 4 losses).
- The residual failure is exactly what the filtering discards: the states that need a recovery never enter the training
  data. Closing that gap needs the dropped ingredients (corrections by an expert, a value model, the advantage indicator and
  several rounds).
