# RECAP-lite: learning from the policy's own rollouts

## The idea

A policy that runs in the simulator produces experience, and normally all of it is lost: when the run is over, the policy has
not changed. RECAP-lite keeps the part that worked. The policy's own **successful runs** are kept, turned into training
episodes, and the policy is fine-tuned on them (trained a little further from where it is), mixed with a few of the original
demonstrations. The experience ends up in the policy's **weights**, the numbers inside the network that hold what it has
learned. That is the memory that stays for good, and it adds nothing to a policy call when the robot runs: no store to
read, no text to add, no extra model call.

![RECAP-lite: the policy's own successful runs become its training data](detailed/recap_loop.svg)

This is the filtered-cloning core of RECAP, the method behind pi*0.6 (Physical Intelligence, 2025, arXiv 2511.14759). RECAP
trains on the robot's own runs and adds more: corrections by an expert, a value function, a label for every step that says
how good it was, and several rounds. RECAP-lite keeps only the first part and drops the rest. What remains is behaviour
cloning on the agent's own successes. It is the cheapest member of this family, and it needs no new training code, because a
successful rollout is simply another demonstration.

At run time nothing changes. The new policy replaces the old one (the **parent**) behind the same proxy, writer and method.

A scripted expert shows the task from perfect positions. The policy's own runs show what the policy itself does: its own
style and its own slips. Among the 181 runs we kept, 18 hold a retry after a failed grasp, which no demonstration contains.

**What it cannot do:**

- **It learns from successes only.** A run that fails is thrown away whole. The policy never sees what to do in the states
  that led it astray, and no expert shows it a better action there.
- **It teaches nothing new.** It makes the policy better at what it already does (here: grasping), not able to do something
  else. It is not a memory of one task, one user or one episode.
- **One round gained little here.** The parent already completed 92 of 100; RECAP-lite completed 94, within noise. The
  failures that remain are mostly bad first grasps.
- **It needs a lot around it.** A simulator or robot that records every step and knows whether a run succeeded, a training
  pipeline and a GPU (6.5 hours for the reference round), and one new training run for every update.
- **It can forget.** The 40 demonstrations in the mix are meant to prevent that. We saw no loss on the four skills of the
  task; we did not test other tasks.
- **The new policy is for the same skills as its parent,** and was measured with the ground-truth writer and the VLM writer
  only.

A walkthrough of how it works, with the data scripts: [detailed/README.md](detailed/README.md).

## Prerequisites

- A trained sub-goal policy, the parent, and its training pipeline (openpi for pi0.5; `training/`).
- A simulator or robot that records, at every control step, both camera pictures, the joint state, the gripper and the
  actions the policy executed, and the skill that was active. It must also tell whether a run completed the task.
- The scripts that turn demonstrations into training data (`training/convert_to_lerobot.py`, `training/relabel_subgoals.py`).
  The kept rollouts go through the same ones.

## How to use it in the framework

1. **Roll out** the parent on fresh seeds, with the ground-truth skill as the prompt at every call, and record every step:

   ```
   python -m vla_memory.proxy --writer oracle --method subgoal --task task.yaml --listen-port <port> --upstream-port <parent policy port>
   ```

   `--task` can be left out for the reference task.
2. **Filter:** keep only the runs that completed the whole task, and cut each of them 16 steps after the final placement (181
   of 200 in the reference round).
3. **Convert and relabel** the kept runs like demonstrations:

   ```
   python training/convert_to_lerobot.py --raw-dir <kept runs> --root <dataset root> --repo-id <user>/<name> \
       --action-space velocity --no-videos
   python training/relabel_subgoals.py --src <converted dataset> --dst <relabelled dataset> --raw-dir <kept runs> [--task task.yaml]
   ```

   `--action-space velocity` stores the actions as recorded: the policy's executed actions are already absolute joint
   targets.
4. **Mix in** original demonstrations as replay against drift (40 of the 221 episodes).
5. **Fine-tune** from the parent's weights with the same LoRA settings (small added matrices are trained), a short schedule
   and a lower learning rate (`training/README.md`, "How to train"; the reference round: 5,000 steps, learning rate 1e-5
   falling to 1e-6 on a cosine curve, 200 warm-up steps).
6. **Select** the checkpoint on development seeds and evaluate it like every other policy. It replaces the parent behind the
   proxy: `--upstream-port` names the new policy server, and the writer and method stay the same.

The framework's part is the data side; the training itself belongs to the policy's own code base. The detailed page explains
how the demonstrations and the kept runs are put into one dataset.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim, 100 fresh seeds of 500
control steps, a policy call every 8 steps. A seed fixes where the cubes start; the fresh seeds were not used to choose
anything. The numbers are **full tasks of 100**: red picked up and put back, then blue picked up and put back. Both policies
ran with the sub-goal method.

| policy | ground-truth skills (`--writer oracle`) | the VLM writer with a store of 20 episodes |
|---|---|---|
| parent (sub-goal policy, step 20000) | 92 | 90 |
| RECAP-lite, one round (step 4999) | 94 | 94 |

The VLM writer answered with the object states and no reason (`--no-verdict-only --brief`). The parent's 90 was run with an
earlier memory-vote gate that also settled the questions about the last skill; the framework always asks about it.

- **Within noise.** With the ground truth, 6 seeds were completed only by RECAP-lite and 4 only by the parent: a split that
  could easily be chance. RECAP-lite completed 6 of the parent's 8 failed seeds and failed on 4 others.
- **The grasps got better.** The parent made 0.90 failed grasp attempts per episode, RECAP-lite 0.31. Of 100 grasp
  attempts, the parent lifted the cube in 68, RECAP-lite in 86. Seed by seed, RECAP-lite had fewer failed attempts on 17
  seeds and more on 5, a split too uneven to be chance. Fewer failed attempts changed the full tasks little: the parent
  recovered from most of its failed attempts (12 of its 20 episodes with one still completed the task).
- **Most failures that remain are bad first grasps.** In 4 of RECAP-lite's 6 failed episodes the red cube was never lifted:
  the gripper closed on the edge of the cube or beside it, 2.7 to 6.4 cm off its centre (0.9 cm in the episodes that
  completed), and then closed again and again. This is what the filter leaves out: runs that stall never enter the training
  data (a likely reason, not tested).
- **Nothing is added at run time.** A policy call takes 170 ms as for the parent, and the checkpoint has the same size. The
  cost is the training round: about 6.5 hours on one GPU.
- **A test of what RECAP-lite drops.** On the 21 seeds where the parent had stalled in earlier runs (each run twice), it
  completed 36 of 42 episodes alone and 42 of 42 when a scripted expert took over at a detected stall (6 hand-overs, all
  succeeded). Paired by seed and run, the hand-over was better 6 times and worse never, which is unlikely to be chance.
  That tests the expert, not training on its corrections.
