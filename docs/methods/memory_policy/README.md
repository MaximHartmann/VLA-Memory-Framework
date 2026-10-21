# The memory policy: a policy that reads its memory

## The idea

A VLA policy decides from what it sees at this moment: the camera pictures, the robot's state and the prompt. It cannot look
up how a similar situation went before. All it knows about that, it learned in training and carries in its weights.

The memory policy gives it something to look at: a hint from recorded episodes. A **store** holds moments of those episodes,
one for every policy call. Each moment keeps the picture of the wrist camera and what the robot did during the next 8
actions, written as one short sentence. At every policy call the proxy looks for the stored moment that looks most like the
current one and passes it on:

- its wrist picture goes into the policy's **third picture slot** (π0.5 has three; the reference policy only uses two);
- its sentence is added to the end of the skill prompt:

```
pick up the red block. memory: release at step 1, up 2 cm
```

![One policy call with the memory policy](detailed/memory_policy_call.svg)

**Why this kind of memory?** The sub-goal policy is told which skill to do, but not how to do it in this exact picture. The
store can show how a similar picture was handled before: by a demonstration, by the policy itself, or by a scripted expert
that took over from the policy. The hope was that such a hint helps with the failures that are left, mostly grasps that miss
or slip. The memory lives outside the weights, so it could grow without training again, as long as its moments look like the
ones the policy was trained on. The idea follows retrieval-augmented policies such as RICL (Sridhar et al., 2025, arXiv
2508.02062), which put retrieved experience into the policy's own input.

**The policy is trained to read hints.** Every training frame gets the nearest stored moment of *another* episode. So the
policy learns how far the hint of a similar but different scene can be trusted, and it stays in control. One frame in ten gets
no hint (an all-zero picture and the bare skill prompt), so a call without a hint is something the policy has seen too.

**What the sentence says.** First how the hand moves from now to the end of the next 8 actions, in whole centimetres, largest
movement first (up or down, left or right, forward or back). Then what the gripper does: "release at step k", "close at step
k" or "gripper stays closed". A moment with no words at all reads "hold still".

**How the moment is found.** It is the single most similar moment by the colour fingerprint of the raw wrist picture, among
the moments of the same skill whose gripper was open (or closed) like the current one. If the fingers have closed on
nothing, which means the grasp missed, only moments that open the gripper again are offered.

**What it cannot do:**

- **It did not make the policy better.** With the memory the policy completed 98 of 100 episodes, without it 96, and a control
  trained on the same data without any memory 96. These differences are within chance.
- **The policy has to be trained for it.** A new policy must learn to read the hints (about 26 hours of training per policy on
  one GPU), and the store has to exist before that. The policy learned from hints of this store's kind. Whether a store of
  other moments would work without new training was not tested.
- **A wrong hint costs a little.** On 20 development episodes, a random moment of another skill instead of the right one cost
  2 completed episodes (20 became 18) at the last checkpoint.
- **The key is simple and hand-made.** It describes colours and where the coloured objects are in the wrist picture. It works
  for clearly coloured objects, and the picture must be the same kind of raw wrist frame the store was built from.
- **The store is only read during a run.** Nothing is added and nothing is forgotten while the robot works.
- **One hint per call.** Only the best moment is shown, never several.
- **Part of it is tied to the reference robot.** The sentences are computed with the Franka arm's kinematics, and the
  gripper thresholds are those of the reference bench.

A walkthrough of how it works: the reasoning, the code path, the store and the training data:
[detailed/README.md](detailed/README.md).

## Prerequisites

- Recorded episodes as LeRobot datasets: both cameras stored as images at their native size, the joint state with the
  gripper's closedness last, absolute joint targets as actions, and the skill as each frame's task
  (`training/convert_to_lerobot.py --no-videos --action-space absolute`, then `training/relabel_subgoals.py`). The forward
  kinematics of the Franka arm of the reference bench are built in; another arm needs its own.
- A policy that takes a third picture, `observation/memory_image`, and ignores it while the picture is missing or all zero
  (π0.5 has a third image slot with a mask), and that reads the prompt with the sentence added.
- At run time: an environment that sends the raw wrist frame (`memory/wrist_raw`) and the finger width
  (`memory/proprio_steps`), as `vla_memory.env_keys.ControlKeys` builds them, and a writer that sequences the skills with
  the sub-goal method (the VLM writer, the rules writer or the ground truth).

## How to use it in the framework

`--task task.yaml` names the task file. For the reference task it can be left out.

1. **Store and training data.** Give every dataset a source label (here `demo` and `expert`). The store is built from the
   training datasets only, and every frame of every dataset gets its hint:

   ```
   python training/relabel_memory.py --train <demos>:demo --train <corrections>:expert --val <demos val>:demo \
       --store <store dir> --root <dataset root> --repo-id <user>/<name> --task task.yaml
   ```

   It writes the store (`entries.jsonl`, the keys, the wrist pictures, `meta.json`) and the datasets `<name>_train` and
   `<name>_val`. Each has the image column `memory_image`, the prompt with the sentence, and a list of the hint every frame
   got.
2. **Training.** Two policies from `pi05_base` on the same frames, the dataset of step 1 (`training/README.md`; neither the
   dataset nor the store is distributed). One reads the hints. The control gets the same frames without them (the bare skill
   prompt, the third slot masked out), so the difference between the two is the memory and not the data.
3. **Run.** Serve a checkpoint and add the memory to the proxy:

   ```
   python -m vla_memory.proxy --writer rules --method subgoal --task task.yaml --listen-port <port> \
       --upstream-port <policy port> --with memory_policy --memory-store <store dir> [--memory-mode off|wrong]
   ```

   `--memory-mode off` sends no hint at all (the bare skill prompt, no picture). `wrong` sends a random moment of another
   skill. These two modes show whether the policy reads its memory at all. `--memory-seed` fixes the random choice and
   `--memory-cache` sets how many decoded pictures stay in memory.

Every line of the proxy's log holds `memory_mode` and, under `memory`, the hint of that call: the moment it came from
(`entry`, `episode`, `step`, `source`), its `similarity`, the `rule` that allowed it, its `sentence` and the time in ms.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim. Both policies were trained on
200 demonstrations plus 75 recorded episodes in which a scripted expert took over (87,768 frames in all); the store holds
10,966 moments of them. Checkpoint 19,999 (20,000 training steps). 100 fresh seeds of 500 control steps, a policy call every
8 steps, the rules writer switching the skills:

| policy | full tasks of 100 |
|---|---|
| memory policy, with its memory | 98 |
| the same policy without memory (`--memory-mode off`) | 96 |
| control: the same data, trained without memory | 96 |
| sub-goal policy (the reference, trained on the demonstrations only) | 94 |

- **No measurable gain.** The memory policy completed 4 episodes that the control did not, and the control 2 that the
  memory policy did not: a split that could easily be chance. Against the reference the split is 6 against 2, which
  could also be chance.
- **The policy does read its memory.** The first 8 actions it returns differ between "with the hint" and "without any hint"
  by 0.0070 rad on average per arm joint and action at the last checkpoint (the median of 20 episodes; about 0.4 degrees) and
  by 0.0165 rad at checkpoint 5,000. Two launches of the same policy differ by 0.0032 rad. On the 20 development episodes the
  policy completed 5, 2 and 1 with the right memory, a wrong memory and none at checkpoint 5,000, and 20, 18 and 18 at
  checkpoint 19,999. So it leans on the hint early in training and only a little at the end.
- **Likely reasons for the missing gain, not tested.** The reference policy already completes 94 of 100 with the same
  prompts, so little is left to fix. And the skill comes from the prompt, so the hint holds no information about the task
  that the policy lacks. It can only add the movement of a similar moment, which the policy already produces reliably.
