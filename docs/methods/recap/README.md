# RECAP-lite: learning from the policy's own rollouts

**Status: in evaluation.** The fine-tuned policy exists and matches its parent on the development seeds; its run on the
100 fresh seeds is in progress.

The third method writes memory into the policy's weights. Where the planner loop keeps memory outside the policy and the
history-appended prompt puts it into the policy's input, this method lets the policy learn from what it has itself done:
its own successful executions become training data. The idea follows RECAP from π*0.6 (Physical Intelligence, 2025,
arXiv 2511.14759), "RL with Experience and Corrections via Advantage-conditioned Policies".

## The original method and what we kept

| RECAP ingredient | RECAP-lite | why |
|---|---|---|
| autonomous rollouts of the current policy | kept | the experience itself |
| human corrections by teleoperation when the policy fails | dropped | no teleoperation; a scripted expert in the simulator could take this role later |
| a value function trained on the rollouts | dropped | needs a second model and training path; few episodes give noisy values |
| a per-transition good/bad indicator, training on all data, conditioning on "good" at test time | replaced by episode-level filtering: keep the successes, discard the failures | without a value function the only label is episode success |
| several rounds of rollout, train, rollout | one round | budget |
| offline RL pre-training on diverse data | not applicable | with only successful demonstrations that stage reduces to plain behaviour cloning, which is the parent checkpoint |

What remains is filtered behaviour cloning on the agent's own successful executions: the cheapest member of the family,
and one that needs no new training code, because the successful rollouts simply become demonstrations.

## How our version works

1. **Roll out.** The selected policy (checkpoint 20000 of the sub-prompt arm) runs closed-loop in the simulator on 200
   new seeds with the ground-truth oracle providing the sub-prompts, exactly as in its evaluation. Per step the joints,
   the gripper, the policy's own executed joint targets and both camera frames are recorded.
2. **Filter.** Only episodes that completed the whole task by the milestone ladder are kept, 181 of 200, cut 16 steps
   after the final placement like the demonstrations.
3. **Convert.** The kept rollouts are written in the demonstration format and pass through the same conversion and
   relabelling as the demonstrations, so every frame carries its sub-prompt. The actions are what the policy actually
   did, not a scripted generator's targets.
4. **Mix.** 40 original demonstrations are added as replay against drift: 221 episodes, 65,752 frames, about 18 percent
   demonstrations.
5. **Fine-tune.** Training continues from the parent's weights (parameter norm identical at step 0) with the same LoRA
   settings and normalisation, a short schedule of 5000 steps and a lower peak learning rate (1e-5 to 1e-6, cosine).
6. **Evaluate** like every other policy: a screen of each checkpoint on the 20 development seeds, the pre-registered
   selection rule, then the 100 fresh seeds against the parent.

At run time nothing changes: the result is a sub-prompt policy that speaks the same four commands as its parent and
replaces it inside the planner loop, with the same proxy, planner and exemplar memory. The framework's part of this
method is therefore the data side (turning rollouts into training episodes); the training itself is specific to the
policy's own code base, openpi for pi0.5.

## Results so far (reference task)

Development seeds, full task / mean score per checkpoint: 500: 15/20, 5.90; 1000: 10, 5.20; 1500: 14, 6.00; 2000: 13,
6.10; 2500: 19, 6.90; 3000: 18, 6.60; 3500: 17, 6.50; 4000: 19, 6.85; 4500: 20, 7.00; 4999: 20, 7.00. The parent scores
20/20 and 7.00 on the same seeds, so the fine-tune dips after the switch to the new data and recovers to the parent's
level; the development set cannot show more than that. Selection by the rule ended in a tie between 4500 and 4999,
decided by the offline validation loss (0.0021796 against 0.0021824) in favour of 4999. That loss is higher than the
parent's 0.00185 on the demonstration validation set, which is expected: the policy was trained mostly on its own
rollout actions, whose style differs from the scripted demonstrations, so this loss ranks RECAP checkpoints among
themselves but does not compare them with the parent. The comparison that counts is the 100-seed run, in progress.

## What the full method would add

The states that filtering discards are exactly the ones that need recovery: in the reference task the fingers nudge
the cube on the approach, the policy then grasps the displaced cube by its edge and chatters. Closing that gap means the
dropped ingredients: a scripted expert in the simulator taking over when a stall is detected (the corrections), a small
value model on the exemplar fingerprint and proprioception, the advantage indicator written into the prompt as text
("Advantage: good"), and two or three rounds. Each of these fits the existing tools, and the indicator uses the same
mechanism as the history-appended prompt.
