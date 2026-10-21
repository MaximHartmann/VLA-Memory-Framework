# Learning from interventions in detail

The short page, [docs/methods/learning_from_interventions](../README.md), says what the method is and what we found. This page
explains why the data are recorded this way, which papers it builds on, the exact rules (every threshold), how a run becomes
training data, and what the two rounds contained. The framework ships no collection tool; this page is the description of
the procedure.

![One run with interventions: who acts, what is trained on, what the memory sees](interventions_run.svg)

## 1. The problem and the premise

### Demonstrations show only perfect runs

A policy trained by imitation learns to copy an expert's runs. Run by itself, it soon reaches states that the expert never
visited, because its own small mistakes pile up, and there it has nothing to copy. This is the old problem that DAgger
addresses (section 2).

In our reference task (pick up and put back a red and then a blue cube) the memoryless policy completes 11 of 100 tasks. Its
mistakes are of a particular kind, because the task needs a memory (the page on the
[built-in memory](../../builtin_memory/README.md) explains why). The four rules of section 3 look for them:

- it goes for the blue cube although red was not lifted yet;
- it closes the gripper on a cube and opens it again before lifting, so it lets go of the cube it should lift;
- after putting red back it turns to red again;
- it stalls: it grasps again and again and gets nowhere.

None of these is in a demonstration. And the right action in these scenes is often the opposite of the right action in a scene
that looks the same: after red is back on the table the next block is blue; at the start it is red.

### The premise: record the right action where the policy goes wrong

The idea is to let the policy drive, watch for these mistakes, and let an expert show the right continuation in the states the
policy really reaches. Four design choices:

- **The expert steps in only on errors.** It does not label every frame. The policy does everything it can by itself, and the
  expert's share of the recorded steps is 34 % in the first round and 19 % in the second.
- **The mistake is left out of the training, the correction is kept.** Imitating the mistake would teach it. The 17 frames
  before a trigger (a rule that detects one of these mistakes), or the whole stalled stretch, are not targets. The
  expert's correction and everything the policy did right are.
- **The memory still sees the mistake.** The feature cache holds every recorded step, also the left-out ones, so the
  memory's call history runs over the whole run. At the hand-back (the moment the expert gives control back to the
  policy) its state knows what happened.
- **The expert hands back as soon as the next step is unambiguous**, so the policy keeps as much of the run as possible.

Once a memory policy exists, the next round is recorded with it driving. A memoryless policy and a memory policy make
different mistakes, and the corrections should come from the states the policy being trained reaches.

### Why this is the data a memory learns from

A memory is learned only when the data require it. In demonstrations the current picture already tells the phase in the
moments that matter, so a memory trained on them stays unused. The intervention data contain look-alike scenes with different
right actions, so a policy can fit them only if it remembers. Without a memory the same data contradict each other.

## 2. Related work

- **DAgger** (Ross, Gordon and Bagnell, 2011) is the base of the loop: a model trained only on an expert's data meets states
  the expert never visited, so the learner drives and the expert labels the states the learner reaches.
- **HG-DAgger** (Kelly et al., 2019, arXiv 1810.02890) lets a human take over when needed instead of labelling every state.
  Our expert also steps in only on errors; in simulation a scripted expert plays the human.
- **Sirius** (arXiv 2211.08416), **Sirius-Fleet** (arXiv 2410.22689) and **RoboCat** (arXiv 2306.11706) run the same loop of
  rollouts, interventions and training, with a human deciding what is written into the weights.
- **π\*0.6 with RECAP** (Physical Intelligence, 2025, arXiv 2511.14759) trains a value function on autonomous rollouts and
  expert interventions and conditions the policy on "good" behaviour. We keep the interventions and drop the value function;
  the framework's [RECAP-lite](../../recap/README.md) is the other half, the successful rollouts.
- **Present but Not Remembered** (arXiv 2607.03372) shows why demonstrations alone teach no memory: when they are close to
  Markov, a policy has no reason to consult its past.
- **Grounded in Time** (arXiv 2610.04255) builds benchmark tasks from counterfactual pairs, where similar observations need
  different actions: the kind of scene our interventions produce.

## 3. How the data are collected, step by step

### The setting

The policy that drives in round 1 is the memoryless policy, with the plain instruction as the prompt at every call, also after
a hand-back. The simulator runs as in the evaluation (same cameras, same grasp physics, a policy call every 8 control steps),
with at most 600 control steps per episode instead of 500. Every rule below reads the **privileged state** once per control
step, after the step's action was applied: the fingertips' mean position (tip), the gripper's commanded value (closed if at
least 0.35), the finger width (the sum of the two finger joints in metres: 0.08 open, about 0.040 closed on the 4 cm cube,
0.032 closed on nothing), the cubes' centres, and the cube's resting height of 0.76 m.

### The tracker

The task is followed by a tracker, whose phases never go back:

| phase | name | it ends when |
|---|---|---|
| 0 | pick up red | red's centre is more than 9 cm above its resting height while the gripper is closed ("lifted") |
| 1 | place red | the gripper is open and red is within 1 cm of its resting height for 3 steps in a row ("placed") |
| 2 | pick up blue | the same for blue: lifted |
| 3 | place blue | the same for blue: placed |
| 4 | done | no more triggers; the episode ends 20 steps after the final placement |

### The four triggers

Checked at every step while the **policy** acts; the first that holds, in this order, hands over. A *close* event is the
gripper command crossing 0.35 upwards, an *open* event downwards.

![The four triggers](interventions_triggers.svg)

| trigger | it fires when |
|---|---|
| **T1, wrong cube** | In phase 0, with red never lifted, the fingertips are closer to blue than to red and either the gripper closes within 4 cm of blue's centre, or the hand has arrived above blue (within 3 cm sideways and less than 10 cm above its centre). |
| **T2, release instead of lift** | In phase 0 or 2 the gripper opens while the cube it should lift is below the lifting height (9 cm) and within 4 cm of the fingertips, after the same closing had held it for at least 3 steps. *Held* means: the gripper is closed, the finger width between 3.4 and 4.6 cm (the fingers stopped by the cube) and the cube's centre within 2.5 cm of the fingertips. Held steps count whoever acts, so a grasp made by the expert counts. |
| **T3, back to red** | From the moment red is released at its place until the end, the fingertips are closer to red than to blue and either the gripper closes within 4 cm of red's centre, or the fingertips have come down 2 cm from their highest point since the release (or since the last hand-back after it) while within 3 cm sideways of red and less than 6 cm above it. |
| **T4, stall** | Either no new phase for 160 steps while picking up or 120 while placing (a *timeout*), or, while picking up, at least 3 closings that opened again while the cube to pick moved more than 1.5 cm sideways or turned more than 5 degrees since the phase began (*chatter*). The timer starts afresh at every phase change and at every hand-back. |

### The expert's segments

The expert is a script. It knows the cubes' true poses and moves the hand along waypoints (inverse kinematics on the waypoints of
the demonstrations), with the gripper command set to 0.60 when closed, up to 3 attempts, a budget of 330 steps and the cube's
yaw matched; a retry beyond 0.72 m uses a grasp pitched by 15°. It takes over at the step after the trigger.

| trigger | the expert | it hands back when |
|---|---|---|
| T1 | picks up red: opens in place if closed, moves 10 cm up, above red, descends, closes, lifts | red is at least 5 cm above its rest, the gripper is closed, and red is within 3 cm of the fingertips |
| T2 | picks up the cube it should lift, as above | the cube is at least 5 cm up and held (same test) |
| T3 | goes for blue but stops before the descent: opens in place if closed, 10 cm up, moves above blue at the pre-grasp pose with the yaw matched | the fingertips are within 2.5 cm sideways of blue's centre and 10 ± 3 cm above it; or the move above has settled (40 settle steps): then the hand-back is marked "above_settled" |
| T4 | completes the stalled step: pick up (grasp and lift), or place (over the cube's spot, lower, open, retreat 10 cm) | for a pick-up: the phase advances (9 cm lift); for a place: the cube is placed and the fingertips are at least 6 cm above it or 8 cm away, which is where the demonstrations' labels switch from "place" |

If the expert gives up (3 attempts or 330 steps) it hands back too ("gave_up"). After every hand-back the policy gets a fresh
call, and T3's reference for "coming down" and T4's timer start afresh.

**Caps.** At most 4 hand-overs per episode. The first trigger after the fourth hand-over is logged as uncorrected and ends the
episode, with its window left out as for its type; nothing after it is recorded. At most 600 control steps. A completed run
ends 20 steps after the final placement.

### What is left out of the training

- **T1, T2, T3** at step t: the frames t-16 to t, 17 frames, whoever acted (a window can reach into the end of an earlier
  expert segment).
- **T4**, a stall at step t: every policy frame since the stall timer started, that is since the phase change or hand-back that
  started it, up to t. The frame at the restart is skipped when it is the expert's last frame of a hand-back.
- **Uncorrected triggers** (after the cap) follow the same rule by type.
- **A completed run** is cut 16 steps after the final placement.
- Everything else, the policy's frames and the expert's, is kept.

The stall rule differs on purpose. A timeout means up to 160 (picking) or 120 (placing) steps of the policy making no progress.
Leaving out only the last 16 of those would keep a long stretch of stalling in the data and teach the policy to stall. This
was decided before the main collection.

### From a run to training episodes

- **Split episodes.** Every stretch of kept frames becomes an episode of its own, so that no action chunk (the next 10
  actions) reaches across a left-out window.
- **The feature cache** holds a row for every recorded step, and maps every frame of a stretch to the step it was recorded at
  (`frame_step` in the cache's index; the built-in memory page describes the format). The memory's call history therefore
  runs over the whole run.
- **The phase labels** are the tracker's phase at each step, with "done" counted as place blue.
- **The training** reads all datasets together, uniformly over their kept frames.

### Round 2: the memory policy drives

The same triggers, expert segments, caps and exclusion rules, with the memory policy (the 30,000-step one) driving, served with
its state and called at every chunk also while the expert acts: the actions are discarded and the state is updated, so at the
hand-back it reflects what happened. The state is never reset within an episode. Episodes start from natural start layouts, on
300 new seeds.

### How the rules were checked before the collection

The rules were first run without any hand-over over recorded runs of the memoryless policy, to see when each would fire and
whether any fired in a run that succeeded. Two thresholds were changed on that evidence: T1's "arrived above blue" height from
5 to 10 cm and its radius from 2.5 to 3 cm, so that the left-out frames cover the move towards blue and not only the descent;
and T3's hand-back tolerance from 1.5 to 2.5 cm, because the arm settles about 2 cm off near its reach limit. A bug was fixed
on the way: T3's reference for coming down was armed before red's release, so it fired at the release itself. A smoke test of
10 episodes then ran the whole loop, and the trigger code replayed offline gave the same triggers as the online run.

## 4. The recorded data

300 episodes were recorded in each round, round 1 with the memoryless policy driving, round 2 with the memory policy. The
first round's training data also hold 10 trial episodes (310 runs in all).

| | round 1 | round 2 |
|---|---|---|
| hand-overs | 575 | 277 |
| the expert's share of all recorded steps | 34 % | 19 % |
| recorded frames | 159,331 | 131,592 |
| frames left out | 31,953 | 27,630 |
| frames kept: the policy's + the expert's | 72,107 + 55,271 = 127,378 | 78,255 + 25,707 = 103,962 |
| training episodes after the split | 856 | 520 |

A stretch of kept frames is a median 163 frames long. The expert's share of the kept frames is 43 % in round 1 and 25 % in
round 2.

## 5. Limitations

- **It needs a way to see mistakes and an expert.** The rules read the simulator's privileged state, and the expert is a script
  that knows the cubes' true poses. On a real robot the state would have to come from sensors and the expert would be a person.
  We have not tried that.
- **The rules are made by hand for one task.** Four triggers with thresholds that we set by looking at the memoryless policy's
  runs on the evaluation seeds, and changed once on that evidence (section 3). No policy was trained on those runs, but the
  rules are not independent of those seeds. They find the mistakes we thought of, and stalls; a mistake of another kind is
  noticed only when it stalls the run.
- **The left-out windows are rules of thumb.** 17 frames before a trigger, the whole stalled stretch for a stall. Other lengths
  were not tried.
- **One task, simulation.** The procedure was built and used on one task in simulation, with one training run per
  condition.
- **The data are tied to the driving policy.** Round 2 was recorded with the 30,000-step policy; the policy trained on it is a
  different one.
- **Grasp failures are shown, not practised.** The stalled stretches are left out of the training, so the data show the
  expert's grasp but not the policy's failed attempts; whether that limits what the policy learns is untested.
- **Without a memory the data hurt.** They are useful only for a policy that can remember: for a memoryless policy,
  look-alike scenes with different right actions contradict each other.
- **No re-weighting.** All kept frames count the same; the expert's frames are 43 % of the kept frames in round 1 and 25 % in
  round 2.
- **No collection tool in the framework.** The procedure above is the specification, the recording belongs to the simulator.

## 6. Using it with another task, robot or policy

1. **Say what the phases are and how to follow them.** The reference task has four. Write a tracker that follows them from the
   state your simulator or robot can give you.
2. **Write triggers for the mistakes that matter for your task.** Each needs a phase in which it is armed, a condition on the
   state, thresholds, and a rule for the step at which the mistake is complete. Add a stall rule.
3. **Check the triggers before collecting.** Run them without any hand-over over recorded runs of the policy, as we did: how often
   does each fire, and does any fire in a run that succeeded? One that does is too eager.
4. **Write the expert.** It must be able to finish any step of the task from wherever the policy left it, and each trigger needs
   a goal test at which it hands back. Choose the caps (hand-overs per episode, steps per segment, steps per episode).
5. **Decide what is left out.** Our rule: the frames of the mistake, never the expert's correction.
6. **Record, split and cache.** Record every step with its pictures. Convert the kept stretches to episodes, compute the feature
   cache for all recorded steps, and label every step with its phase.
7. **Train the built-in memory** on the demonstrations and the rounds together, a cycle at a time, and compare against a control
   with the same number of steps.
8. **Record the next round with the memory policy driving**, served with its state, and keep calling it while the expert acts.

With a person instead of a scripted expert the loop is HG-DAgger (section 2): the person takes over when needed and hands
back; the triggers become the person's judgement, and the phase labels and the left-out windows have to be set by some rule or
by the person. We have not tried it.
