# Learning from interventions: the robot's own mistakes, corrected by an expert

## The idea

The training data are the robot's own runs. The policy acts; when it makes a mistake that a memory would have prevented,
or stops making progress, a scripted expert takes over, puts things right and hands back, and the run goes on. This is DAgger (Ross et al., 2011) with an
expert that steps in only on errors, as in HG-DAgger (Kelly et al., 2019). The frames of the mistake are left out of the
training; the expert's correction and everything the policy did right are kept. Unlike demonstrations, such runs contain
scenes that look alike and need different actions (a held cube near the table on its way up or down; the table after the red
cube is put back, which looks like the start), so a policy fits them only with a memory: they are what the
[built-in memory](../builtin_memory/README.md) learns from.

Once a memory policy exists, the next round is recorded with it driving, so that the corrections come from the states the
memory policy itself reaches.

## Prerequisites

- A simulator with the privileged state the rules read: the cubes' positions, the fingertips, the gripper's command and
  width.
- A scripted expert that completes any step of the task from wherever the policy left it (here inverse kinematics on the
  demonstrations' waypoints, with the cube's true pose).
- The policy that drives: first the memoryless policy, then the memory policy, served with its state per episode (the
  built-in memory's server and the episode markers).
- The built-in memory's training, which reads the runs as split episodes with feature caches.

## How to use it in the framework

The framework describes the procedure; the collection itself belongs to the simulator. Every rule reads the privileged state
once per control step.

1. **The task's phases**, followed by a tracker: pick red, place red, pick blue, place blue. A cube is lifted when its centre
   is 9 cm above its resting height with the gripper closed; it is placed when the gripper is open and the cube rests within
   1 cm of its height for 3 steps.
2. **The triggers** of a hand-over, checked at every step while the policy acts (the first that holds):

   | trigger | the policy |
   |---|---|
   | wrong cube | goes for blue before red was lifted: the fingertips closer to blue than to red, and the gripper closes within 4 cm of blue, or the hand arrives above blue (within 3 cm sideways, less than 10 cm above its centre) |
   | release instead of lift | opens the gripper on the cube it should lift, after holding it for at least 3 steps (the fingers stopped by the cube, the cube within 2.5 cm of the fingertips), while the cube is below the lifting height and within 4 cm |
   | back to red | after putting red back, turns to red again: the fingertips closer to red than to blue, and the gripper closes within 4 cm of red, or the hand comes down 2 cm from its highest point since the release while within 3 cm of red and less than 6 cm above it |
   | stall | makes no progress: no new phase for 160 steps while picking or 120 while placing, or the cube to pick moved more than 1.5 cm or turned more than 5 degrees while at least 3 grasps re-opened |

3. **The expert's segments.** The expert takes over at the step after the trigger and hands back as soon as the right next
   step is unambiguous: after a wrong cube with red lifted 5 cm and held, after a release with the cube re-grasped and
   lifted 5 cm, after a turn back to red with the hand above blue, ready to grasp it, after a stall with the stalled step
   completed (the cube lifted, or placed with the hand moved away). It gives up after 3 attempts or 330 steps. At most 4
   hand-overs and 600 steps per episode.
4. **What is left out.** At a trigger on step t the 17 frames t-16 to t; at a stall every policy frame since the stall's
   timer started. A completed run is cut 16 steps after the final placement. Everything else is kept, the policy's and
   the expert's frames alike.
5. **Split episodes.** Every stretch of kept frames becomes an episode of its own, so that no action chunk reaches across a
   left-out window. The feature cache maps every frame to the step it was recorded at, so the memory's call history runs
   over the whole run, the left-out steps included: the memory sees the mistake happen, the policy does not imitate it.
6. **Training.** The demonstrations and all rounds together, uniformly over their frames, with the built-in memory's recipe;
   every further cycle of 10,000 steps is a warm restart from the last checkpoint (warm-up 200 steps, cosine 1e-5 to 1e-6).
7. **The next round** is recorded with the memory policy driving, served with its state. While the expert acts, the policy
   is still called at its own pace with the current observation; its actions are discarded, its state is updated, so at the
   hand-back the state knows what happened. Nothing resets the state within an episode.

## Our findings

Reference task: 300 episodes per round (the first round's training data also hold its 10 trial episodes), then 100 fresh
seeds of 500 control steps per policy:

| | round 1, the memoryless policy driving | round 2, the memory policy driving |
|---|---|---|
| hand-overs | 575 | 277 |
| wrong cube / release instead of lift / back to red / stall | 69 / 246 / 101 / 159 | 12 / 82 / 21 / 162 |
| episodes without a hand-over | 6 | 93 |
| full tasks with the expert's help | 254 | 279 |
| the expert's share of the recorded steps | 34 % | 19 % |
| frames kept for training (split episodes) | 127,378 (856) | 103,962 (520) |

| policy (memory with phase head) | training data | steps | full tasks of 100 |
|---|---|---|---|
| after three cycles, the starting point | demonstrations + round 1 | 30,000 | 60 |
| a fourth cycle, the matched-steps control | demonstrations + round 1 | 40,000 | 51 |
| a fourth cycle with round 2 | demonstrations + rounds 1 and 2 | 40,000 | 65 |

- The memory policy needs far less help: 277 hand-overs instead of 575, and 93 of 300 episodes without any (6 in round 1).
  The mistakes a memory prevents fell in every kind (wrong cube 69 to 12, release instead of lift 246 to 82, back to red
  101 to 21), the stalls stayed (159 and 162): the second round is mostly the expert's grasp recoveries.
- The second round helps: 65 against 51 for the same training on round 1 alone (paired sign test, p = 0.029). Against the
  starting point the gain is not significant (60 to 65, p = 0.49): on the old data a fourth cycle gives nothing, the new
  data add. They improve the step from red to blue (reached blue 90 against 76, p = 0.009).
- Interventions are only useful with a memory: the same round-1 data trained into the memoryless policy gave 0 of 100.
