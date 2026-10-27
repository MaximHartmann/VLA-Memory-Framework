# The head writer: the policy itself says when a skill is done

## The idea

Every method needs a writer that says when a skill is done. The other writers watch the robot from the outside: the
simulator's tracker, rules on the gripper's readings, or a vision-language model looking at the cameras. The head writer
asks the policy itself. It sees the same pictures and the same prompt, so it may already know.

![One policy call with the head writer](detailed/head_call.svg)

- At every call the policy server also hands out 2,048 numbers from inside the policy: a layer two thirds of the way up,
  averaged over the prompt's words, read during the policy's own computation by a [state tap](../../taps/README.md). The
  policy itself is not changed.
- In the proxy, the **event head**, a small formula (a weighted sum of the 2,048 numbers plus a constant), turns them into
  one score. A score above a threshold means "done".
- After two "done" in a row the skill is complete, and the method shows that to the policy from the next call on: a new
  history line, the next skill as the prompt, or a new memory picture.
- The head is fitted on recordings of the policy's own runs, in which another writer, the teacher, said when each skill was
  done.

**Why it is worth trying.** The robot needs no tracker, no extra sensors and no second model to keep track of its task.
It also answers a research question: does a VLA know inside how far its task has come? It does, and a simple formula reads
it out.

**What it cannot do well:**

- **One head per policy and method.** The numbers are the policy's own, so another policy, a retrained one or another
  method needs a new recording and fit.
- **It copies its teacher**, mistakes included.
- **It does not check which block is in the hand.** If the policy grasps the wrong block, the head still marks the skill
  done.
- **It is one call late**, because it judges the policy's answer. With the keyframe method that even helped (findings).
- **The last skill is never marked done**, as in the policy's training data.

How it works inside, why it is built this way, and the code: [detailed/README.md](detailed/README.md).

## Prerequisites

- A policy and the method it was trained for: [history](../../methods/history/README.md),
  [sub-goal](../../methods/subgoal/README.md) or [keyframe](../../methods/keyframe/README.md).
- A policy server that hands out the policy's state: `integrations/openpi/openpi_server.py` for openpi's π0.5 (JAX or
  PyTorch), `integrations/lerobot/lerobot_server.py` for LeRobot's policies, or a few lines with the
  [automatic tap](../../taps/README.md) in another PyTorch policy's own server.
- A teacher for the recording: the rules writer (on a real robot too) or the simulator's ground truth.

## How to use it in the framework

1. **Serve the policy with its state**, for example openpi's π0.5 or a LeRobot policy
   (`integrations/openpi/README.md`, `integrations/lerobot/README.md`):

   ```
   python -m openpi_server --port 8000 --config <openpi config> --checkpoint <checkpoint dir> --center-crop 0.95
   python -m lerobot_server --port 8000 --policy <hub id or folder>
   ```

2. **Record about 100 runs with the teacher** (we used 130):

   ```
   python -m vla_memory.proxy --writer rules --method subgoal --record-head runs/ --listen-port 9000 --upstream-port 8000
   ```

   At every call the proxy keeps the policy's numbers and the teacher's count of finished skills. At the 3 calls after a
   finished skill it also asks the policy once more with the old input: that is the moment the head must learn to
   recognise. The robot only acts on the real call. One file of about 250 KB per run.

3. **Fit the head** (on the CPU, about a minute):

   ```
   python training/fit_event_head.py --recordings 'runs/*.npz' --out head/
   ```

   It prints how well the head tells "done" from "not done" (the AUROC: 1.0 is perfect, 0.5 is guessing) and saves the
   head in `head/`.

4. **Run with the head as the writer**, with the method of the recording:

   ```
   python -m vla_memory.proxy --writer head --head-dir head/ --method subgoal --listen-port 9000 --upstream-port 8000
   ```

   With `--log FILE`, the proxy's log shows at every call the score, the verdict and the skill completed, if any.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim. Our three π0.5 policies,
each with a head of its own, fitted on 130 of its own runs recorded by the proxy, with the rules writer as the teacher.
100 fresh seeds of 500 control steps; for each policy only the writer differs. Full tasks of 100:

| policy | the head writer | simulator's ground truth | rules writer |
|---|---|---|---|
| sub-goal | 91 | 92 | 94 |
| history | 75 | 82 | 79 |
| keyframe | 79 | 78 | 75 |

- **About as good as the other writers.** Seeds that only one of two writers completed split 7 to 8 (sub-goal), 6 to 13
  (history) and 15 to 14 (keyframe) between the head and the simulator: splits that chance produces. Of the 55 failed
  runs, 48 had no head error, mostly failed grasps; the rest were mainly a "placed red" the history head never saw (5).
- **Late, and that helps the keyframe policy.** The head takes the "placed red" picture a call or two after the simulator
  would, with the hand higher (a median 12.8 against 9.9 cm above the cube). A picture with the hand low makes the keyframe
  policy grasp red again ([the keyframe method](../../methods/keyframe/README.md)): that never happened with the head, 10
  times with the simulator's events and 16 times with the rules writer.
- **On time.** Of the 789 skills the simulator saw completed, the head marked 733 exactly one call later, 30 in the same
  call and 14 later still; it missed 9, was early 3 times, and 4 times marked a skill the simulator never saw.
- **Almost free.** A call takes as long as without the head (171 ms), the head itself under a millisecond, the fit one or
  two minutes on the CPU, and the recordings of a policy 32 MB. In cross-validation the heads reached an AUROC of 0.994 to
  0.997.
- **The teacher matters most, not the size of the head.** Heads fitted on the scripted demonstrations said "done" at the
  grasp, before the lift; fitted on the policy's own runs, these false alarms disappeared. A small neural network did no
  better than the simple formula.
