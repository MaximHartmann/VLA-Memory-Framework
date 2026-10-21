# Learning from interventions: the robot's own mistakes, corrected by an expert

## The idea

**The problem.** A policy trained on demonstrations has only seen perfect runs. When it runs by itself it makes mistakes that
no demonstration shows, and some are mistakes of memory: it goes for the blue block before red was lifted, it lets go of a
cube it should lift, it turns to red again after putting red back. To learn to avoid them, the training data must contain
these moments together with the right action.

**The idea.** Let the policy run. When it makes such a mistake, or stops making progress, a scripted expert takes over, puts
things right and hands control back, and the run goes on. The frames of the mistake are left out of the training. The
expert's correction and everything the policy did right are kept. This is the idea of DAgger (Ross et al., 2011): let the
policy drive, and let an expert show the right action in the states it really reaches. Here the expert steps in only on
errors, as in HG-DAgger (Kelly et al., 2019).

![One run with interventions: who acts, what is trained on, what the memory sees](detailed/interventions_run.svg)

Why record the data this way:

- In demonstrations the current picture already tells what comes next, in the moments where a memory could matter. In the
  policy's own runs it does not: a held cube near the table on its way up or down, or the table after red was put back, which
  looks like the start. A policy can fit such data only with a memory. These runs are what the [built-in memory](../builtin_memory/README.md) learns from.
- The mistake itself is not imitated, but the memory still sees it happen: its history runs over the whole run.
- Once a memory policy exists, the next round is recorded with it driving, so that the corrections come from the states the
  memory policy itself reaches.

**What it cannot do:**

- **It needs a way to see the mistakes and an expert who can fix them.** In our simulator that is the privileged state (the
  true positions of the cubes) and a scripted expert. On a real robot it would be a person who takes over; we have not tried it.
- **Its rules are made by hand for this task.** Four triggers with thresholds that we set by looking at the memoryless
  policy's runs. A new task needs its own.
- **It helps only together with a memory.** The same data trained into the memoryless policy gave 0 of 100.
- **The second round gave a small gain.** 65 of 100 against 51 for the matched control, but against the policy it
  started from (60) the gain is not clear.
- **What is left is mostly grasping.** Stalls, mostly failed grasps, are 162 of the 277 hand-overs of round 2, and 27 of the
  best policy's 35 failures are grasps. Whether more data of this kind fix grasps is open.
- **The framework ships no collection tool.** It describes the procedure below; the recording belongs to the simulator.

A walkthrough of the procedure: the reasoning, the exact rules and how a run becomes training data:
[detailed/README.md](detailed/README.md).

## Prerequisites

- A simulator with the privileged state the rules read at every control step: the cubes' positions, the fingertips' position,
  the gripper's command and its width.
- A scripted expert that can complete any step of the task from wherever the policy left it (here: inverse kinematics along the
  demonstrations' waypoints, with the cube's true pose).
- The policy that drives: first the memoryless policy, then the memory policy, served with its state per episode (the
  [built-in memory's](../builtin_memory/README.md) server and episode markers).
- The built-in memory's training, which reads the runs as split episodes with feature caches.

## How to use it in the framework

The framework describes the procedure; the recording itself belongs to the simulator. Every rule reads the privileged state
once per control step.

1. **The task's phases**, followed by a tracker: pick up red, place red, pick up blue, place blue. A cube is *lifted* when its
   centre is 9 cm above its resting height while the gripper is closed. It is *placed* when the gripper is open and the cube
   rests within 1 cm of its resting height for 3 steps.
2. **The triggers** of a hand-over, checked at every step while the policy acts (the first that holds):

   | trigger | the policy |
   |---|---|
   | wrong cube | goes for blue before red was lifted: the fingertips are closer to blue than to red, and the gripper closes within 4 cm of blue, or the hand arrives above blue (within 3 cm sideways and less than 10 cm above its centre) |
   | release instead of lift | opens the gripper on the cube it should lift, after holding it for at least 3 steps (the fingers stopped by the cube, the cube within 2.5 cm of the fingertips), while the cube is below the lifting height and within 4 cm |
   | back to red | after putting red back, turns to red again: the fingertips are closer to red than to blue, and the gripper closes within 4 cm of red, or the hand comes down 2 cm from its highest point since the release while within 3 cm of red and less than 6 cm above it |
   | stall | makes no progress: no new phase for 160 steps while picking up or 120 while placing, or at least 3 grasps closed and opened again while the cube to pick moved more than 1.5 cm or turned more than 5 degrees |

3. **The expert's segments.** The expert takes over at the step after the trigger and hands back as soon as the right next step
   is unambiguous: after a wrong cube, with red lifted 5 cm and held; after a release, with the cube grasped again and lifted
   5 cm; after a turn back to red, with the hand above blue and ready to grasp it; after a stall, with the stalled step
   finished (the cube lifted, or placed and the hand moved away). It gives up after 3 attempts or 330 steps. At most 4
   hand-overs and 600 steps per episode.
4. **What is left out of the training.** At a trigger on step t, the 17 frames t-16 to t. At a stall, every policy frame since
   the stall's timer started. A completed run is cut 16 steps after the final placement. Everything else is kept, the
   policy's and the expert's frames alike.
5. **Split episodes.** Every stretch of kept frames becomes an episode of its own, so that no action chunk reaches across a
   left-out window. The feature cache maps every frame to the step it was recorded at, so the memory's call history runs over
   the whole run, the left-out steps included.
6. **Training.** The demonstrations and all rounds together, uniformly over their frames, with the built-in memory's recipe.
   Every further cycle of 10,000 steps is a warm restart from the last checkpoint: 200 warm-up steps, then a cosine decay of
   the learning rate from 1e-5 to 1e-6.
7. **The next round** is recorded with the memory policy driving. Serve it as on the built-in memory's page, with the
   environment connected through the proxy:

   ```
   python -m builtin_memory.serve --checkpoint <checkpoint dir> --port <policy port> --center-crop 0.95
   python -m vla_memory.proxy --writer off --task task.yaml --listen-port <port> --upstream-port <policy port>
   ```

   `--task` names the task file; for the reference task it can be left out. While the expert acts, the policy is still called
   at its own pace with the current observation: its actions are thrown away, but its state is updated, so at the hand-back it
   knows what happened. Nothing resets the state within an episode.

## Our findings

Reference task: 300 episodes per round (the first round's training data also hold its 10 trial episodes), then 100 fresh
seeds of 500 control steps per policy. A *full task* is the whole sequence: red picked up and put back, then blue picked up
and put back. Round 1 was recorded with the memoryless policy driving, round 2 with the memory policy driving:

| | round 1, the memoryless policy driving | round 2, the memory policy driving |
|---|---|---|
| hand-overs to the expert | 575 | 277 |
| wrong cube / release instead of lift / back to red / stall | 69 / 246 / 101 / 159 | 12 / 82 / 21 / 162 |
| episodes without a hand-over | 6 | 93 |
| full tasks with the expert's help | 254 | 279 |
| the expert's share of the recorded steps | 34 % | 19 % |
| frames kept for training (split episodes) | 127,378 (856) | 103,962 (520) |

![Two rounds: hand-overs by trigger](detailed/interventions_rounds.svg)

Full tasks of 100 for the policy with the memory and its phase head, after each training:

| policy | training data | steps | full tasks of 100 |
|---|---|---|---|
| after three cycles, the starting point | demonstrations + round 1 | 30,000 | 60 |
| a fourth cycle, the matched-steps control | demonstrations + round 1 | 40,000 | 51 |
| a fourth cycle with round 2 | demonstrations + rounds 1 and 2 | 40,000 | 65 |

- **The memory policy needs far less help:** 277 hand-overs instead of 575, and 93 of 300 episodes without any (6 in round 1).
  The mistakes a memory prevents fell in every kind (wrong cube 69 to 12, release instead of lift 246 to 82, back to red 101
  to 21). The stalls stayed (159 and 162), so the second round is mostly the expert's grasp recoveries.
- **The second round helps.** 65 against 51 for the same training on round 1 alone, a clear difference. Against the
  starting point the gain is not clear (60 to 65): on the old data a fourth cycle gives nothing, the new data add a
  little. The difference opens between red and blue: 90 episodes reached the blue cube against 76, a clear difference.
  ([the picture](detailed/interventions_results.svg))
- **Interventions are only useful with a memory.** The same round-1 data trained into the memoryless policy gave 0 of 100.
