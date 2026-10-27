# The rules writer in detail

The short page, [docs/writers/rules](../README.md), says what the rules writer is for and what we found. This page explains
how to set it up for a new robot or task, and then follows one run of the proxy through the code:

```
python -m vla_memory.proxy --writer rules --method subgoal --task vla_memory/tasks/red_blue_blocks.yaml \
    --listen-port 8001 --upstream-port 8000
```

The parts every writer shares (the server, the relay, the connection to the policy) are explained in the
[oracle walkthrough](../../oracle/detailed/README.md); this page concentrates on what the rules writer adds.

![One policy call with the rules writer and the sub-goal method](rules_call.svg)

## The pieces

| file | what it does |
|---|---|
| `vla_memory/writers/rules/writer.py` | `RulesWriter`: the writer the proxy talks to; hands each call's readings to the logic below |
| `vla_memory/writers/rules/rule_monitor.py` | `RuleMonitor`: the plan, the position in it, and what a trigger completes |
| `vla_memory/writers/rules/grasp_cycle.py` | `GraspCycle`: the four states of one grasp and the two triggers, `lift` and `set_down` |
| `vla_memory/writers/rules/identify.py` | `ColourIdentifier`: which object is in the gripper, from the wrist picture |
| `vla_memory/writers/rules/rule_spec.py` | reads and checks the task file's `history.rules` and the gripper thresholds |
| `vla_memory/writers/rules/signals.py` | turns the environment's per-step arrays into rows |
| `vla_memory/writers/rules/signal_sequence.py` | `SignalSequence`: the trigger `signals`, for skills without a grasp |
| `vla_memory/writers/rules/validate.py` | the offline check of a set of rules against recorded episodes |
| `vla_memory/writers/rules/author*.py` | lets the planning model draft a set of rules; `author_demos.py` measures demonstrations |
| `vla_memory/env_keys.py` | `ControlKeys`: the environment's helper that records the readings and builds the `memory/` entries |

## 1. Setting it up for a new robot or task

1. **The skills.** The task file lists the objects, the verbs and how a skill reads (`skill_template`: "pick up the red
   block"). The strings must be exactly those the policy was trained on.
2. **The gripper thresholds** (the task file's `proprio`), for your gripper:
   - `open_width`: wider than this, the fingers are open (6 cm in the reference task);
   - `empty_width`: narrower than this, they closed on nothing (1.5 cm);
   - `squeeze_force`: above this force the fingers hold something (8 N). For a gripper that only reports a grasped flag
     (1 or 0), use 0.5.
3. **The rules** (the task file's `history.rules`): one rule per verb, a trigger and its thresholds. The thresholds must
   reproduce the moments at which the policy's training labels switch. Measure them on labelled demonstrations, episode
   files with the readings per step (`rows`) and the number of completed skills per step (`reference`), the format of the
   validator (section 10):

   ```python
   import glob, numpy as np
   from vla_memory.task import TaskSpec
   from vla_memory.writers.rules.author_demos import demo_event_summary, demo_summary_text
   task = TaskSpec.load_or_default("task.yaml")
   episodes = [dict(np.load(path)) for path in glob.glob("demos/*.npz")]
   print(demo_summary_text(demo_event_summary(episodes, task)))
   ```

   For every skill it prints what the gripper showed when the label switched: how high the fingertips had risen above the
   grasp point (the `lift` rule's `rise`), and after a set-down the release height and how far the hand had moved up and
   away (the `set_down` rule's `release_below`, `up` and `away`), as smallest / median / largest value. Take thresholds that
   all demonstrations reach. The author (section 10) can draft the rules from these numbers or from the task's completion
   criteria.
4. **The environment.** Your simulator or robot program records the three readings at every control step and adds the
   `memory/` entries to every policy call, with `ControlKeys`:

   ```python
   from vla_memory.env_keys import ControlKeys
   keys = ControlKeys(task)
   keys.start_episode("episode-1")
   for step in range(max_steps):
       keys.record(finger_width=width, finger_effort=force, tip_xyz=fingertip)     # every control step
       if step % 8 == 0:                                                           # a policy call
           observation = keys.observation(policy_keys, instruction, step, wrist_raw=wrist_picture)
           actions = client.infer(observation)["actions"]                          # the proxy, not the policy
   ```

   The fingertip position comes from the robot's forward kinematics; most robot interfaces report the hand's position.
5. **Check the rules offline** on recorded episodes with a reference (`python -m vla_memory.writers.rules.validate`, section
   10). The writer should switch in the same policy call as the reference, never earlier.
6. **Run it:** the proxy with `--writer rules` and the policy's method.

## 2. Start-up

`cli.main()` reads the command line as for every writer (oracle walkthrough, section 1). For `--writer rules` it then calls
`RulesWriter.build(args, task)`:

1. **The object check is chosen** (`make_identifier`). The task file's `history.object_from` says how a pick learns which
   object it concerns: `plan_order` (the default) assumes the plan's object, `colour` builds a `ColourIdentifier` from the
   task file's colour rules (`fingerprint.colours`). The reference task uses `colour`.
2. **The rules are read and checked** (`parse_rules` in `rule_spec.py`). Every verb of the task must have exactly one rule,
   every rule a known trigger (`lift`, `set_down` or `signals`) and exactly the thresholds that trigger takes. A mistake stops
   the proxy at start-up with a message that names the verb.
3. **The gripper thresholds are read** (`gripper_thresholds`): `open_width`, `empty_width` and `squeeze_force` from the
   task file's `proprio` section; defaults exist for all three.
4. **The grasp cycle is built** (`GraspCycle`) with the gripper thresholds and those of the first `lift` rule and the first
   `set_down` rule. A task with a grasp verb must have both.
5. **One `SignalSequence` is built per verb with `trigger: signals`** (none in the reference task).
6. **The plan is set** (`reset`): the task's skills in the task order, `["pick up the red block", "place the red block",
   "pick up the blue block", "place the blue block"]`, the position at 0, and the grasp cycle in the state *free*.

## 3. What the environment sends

The rules writer needs the readings of **every control step**, not just of the steps with a policy call: between two calls
the robot executes several actions (8 in the reference setup), and a grasp or a release can happen at any of them.

| entry | content |
|---|---|
| `memory/proprio_steps` | one row per control step since the previous call: finger width (m, both fingers together), squeeze (N, or 1 and 0 from a grasped flag), fingertip x, y, z (m) |
| `memory/wrist_raw` | the wrist camera's picture, for the colour check (without it the writer uses the policy's wrist picture) |
| `memory/new_episode`, `memory/episode` | where an episode starts |
| `memory/objects` | the episode's objects, for a task whose episodes hold only some of its objects |
| `memory/signal_steps` | further signals the task declares, for `trigger: signals` rules |

At the very first call of an episode no steps have been recorded yet; the writer then reads the current values instead
(`memory/finger_width`, `memory/finger_effort`, `memory/tip_xyz`), see `signal_rows` in `signals.py`.

## 4. One policy call

The relay handles the call as for every writer (oracle walkthrough, section 3): it sets the `memory/` entries aside, starts a
new episode when one begins (the writer's `start_episode`: the plan for the episode's objects, the position at its first
skill, the grasp cycle at *free*), and asks the writer before the policy call. `RulesWriter.before_call` then:

1. **turns the readings into rows** (`signal_rows`): one row per control step since the previous call, five columns each
   (width, squeeze, x, y, z);
2. **feeds every row to the logic, oldest first** (`RuleMonitor.observe_rows`). For each control step:
   1. the grasp cycle takes the thresholds of the **current skill's verb** (`_use_thresholds_of_current_skill`), so that
      verbs sharing a trigger can have different thresholds;
   2. the grasp cycle advances by one step (`GraspCycle.step`, section 5) and returns `"lift"`, `"set_down"` or nothing;
   3. on a trigger, `_grasp_line` decides whether it completes the current skill: at a `lift`, the colour check names the
      held object (section 7); the trigger must fit the current skill's verb (a second `lift` while the current skill is
      "place" completes nothing: the block slipped out and was grasped again); a `set_down` completes "place" only for the
      object identified at the last lift;
   4. a completed skill (`_complete_current_skill`) is added to the call's `completed` list and the position moves on; after
      the last skill `done_all` becomes True and later triggers are ignored. At most one skill is completed per step;
   5. for a verb with `trigger: signals`, its sequence of conditions advances by one step instead (section 9);
3. **returns its record for the proxy's log**: `advanced`, `skill_index`, `completed`, `writer_state` (the grasp cycle's
   state after the last step) and `identified` (the colour check's answer, at a call with a pick).

Then the method sets the prompt from the new position (the sub-goal method's prompt is the current skill), and the policy is
called. Because all of this happens before the policy call, a skill completed at any step since the previous call shows in
the prompt of this call.

## 5. The grasp cycle

![The grasp cycle: four states and the two triggers](grasp_cycle.svg)

`GraspCycle.step(width, squeeze, fingertip)` runs once per control step. What each state waits for, with the reference
task's thresholds, and why:

**free: waiting for a grasp.** The cycle moves to *holding* when the fingers have stopped on something: the width is between
`empty_width` (1.5 cm) and `open_width` (6 cm), **and** the squeeze is above `squeeze_force` (8 N). The squeeze matters
because a gripper that closes on nothing also stops at a narrow width (3.2 cm in the simulator), but without force. The
fingertip height at this moment is remembered as the grasp point.

**holding: the fingers press on something.** The cycle fires **lift** and moves to *lifted* when the fingertips have risen
`rise` (9 cm) above the grasp point while still squeezing; 9 cm is where the demonstrations' labels switch from "pick up" to
"place". (With the optional `at_top`, the lift fires only once the hand also stops rising.) If the fingers open or close
fully before that, the grasp failed and the cycle goes back to *free*.

**lifted: carrying.** When the fingers open (width above 6 cm), the height decides what happened:

- less than `release_below` (8 cm) above the grasp point: the object was set down, or let go just above the table. The
  fingertip position is remembered as the release point, and the cycle moves to *released*;
- higher up: the object was dropped. Back to *free*, nothing fires.

If the fingers close fully instead (width below 1.5 cm), the object slipped out: back to *free*, nothing fires.

**released: open, low down.** The cycle fires **set_down** and goes back to *free* when the fingertips have moved `up`
(6 cm) above the release height or `away` (8 cm) from the release point. The demonstrations' labels switch from "place" to
the next skill only after this retreat; a writer that switched at the moment the fingers opened sent the next skill while
the fingers were still around the block, and the policy grasped the same block again. If the fingers squeeze something
again before the retreat (they touched the block just set down), the cycle returns to *holding*, but the set-down stays
pending: letting go again brings it back to *released*.

## 6. A whole episode of the reference task

| what the robot does | grasp cycle | the writer | the policy's prompt (sub-goal) |
|---|---|---|---|
| moves to the red block, closes the fingers on it | free → holding | | "pick up the red block" |
| lifts it 9 cm | holding → lifted, fires `lift`; the colour check sees red | completes "pick up the red block" | "place the red block" from the next call |
| lowers it and opens the fingers near the table | lifted → released | | "place the red block" |
| moves the hand 6 cm up (or 8 cm away) | released → free, fires `set_down` | completes "place the red block" | "pick up the blue block" |
| the same for the blue block | | completes "pick up the blue block", then "place the blue block"; `done_all` | "place the blue block" to the end |

"From the next call" means: the step at which the trigger fired lies between two policy calls, and the next call is the first
one to show the new skill. That is the earliest any writer can show it, because the policy is only asked at calls.

## 7. Which object: the colour check

The grasp cycle knows *when* a pick happened, not *which* object was picked. With `history.object_from: colour`, at every
`lift` the writer asks the `ColourIdentifier`:

- It looks at the region of the wrist picture where a held object appears (`history.held_region`: rows 30-60 % and columns
  33-67 % in the reference task), counts the pixels of each object's colour (`fingerprint.colours`), and names the colour
  that fills the largest share, if it reaches `history.held_min_share` (12 %). Otherwise it answers "unknown".
- The writer acts on the answer (`_identify_at_lift`):
  - the expected object, or "unknown": the plan's order holds;
  - another object of the episode with a pending pick: that object's skills of the current plan stage move to the front of
    the remaining plan (`_bring_forward`), so the lift completes the pick of the object really held;
  - an object without a pending pick (already picked and placed): nothing is completed.
- At the next `set_down`, "place" is completed only for the object identified at the last lift: setting down a different
  object (one was dropped, then another one carried) completes nothing, because a wrong line would mislead the policy for
  the rest of the episode.

The check runs only at lifts, two to four times per episode, and takes about 0.08 ms each time.

## 8. Thresholds per verb, and plans in stages

A task can have several verbs with the same trigger: "put the cube into the box" and "stack the cube" both end with a
`set_down`, but the cube is let go just above where it was grasped in the first case and much higher in the second. Each
verb's rule carries its own thresholds, and before every step the grasp cycle takes those of the current skill's verb.

A task file can also order the plan in stages (`plan_stages`), for example first every cube into the box, then every cube
out and onto the stack. When the colour check brings an object's skills forward, it moves only that object's skills of the
current stage.

## 9. Skills without a grasp: `trigger: signals`

Opening a drawer or pressing a button does not follow the grasp cycle. Such a verb lists phases, each a set of conditions on
the signals that must all hold at one control step; the phases are entered in order, and the skill is complete when the last
one is entered (`SignalSequence` in `signal_sequence.py`). An example for opening a drawer:

```yaml
open:
  trigger: signals
  phases:
    - when: {finger_effort: {above: 8.0}, finger_width: {above: 0.01, below: 0.06}}   # the fingers squeeze the handle
      hold: {finger_effort: {above: 8.0}}                                             # ... and keep squeezing
    - when: {tip_x: {below: 0.47}}                                                    # pulled out
    - when: {finger_width: {above: 0.06}}                                             # let go
    - when: {tip: {away: 0.03}}                                                       # moved 3 cm away from there
```

A condition is a value `above` or `below` a threshold, or a change since an earlier phase (`rise`, `fall`, `away`, `near`).
If a `hold` condition stops holding before the next phase is entered, the sequence starts over. Only the current skill's
sequence runs, and it starts afresh whenever the writer moves to the next skill. Besides the five readings, a rule may use
signals the task declares (`signals` in the task file), sent as `memory/signal_steps`.

## 10. The tools: checking and drafting rules

- **Checking** (`python -m vla_memory.writers.rules.validate --episodes 'recorded/*.npz' --task task.yaml`): replays
  recorded episodes through the rules. Each episode file holds the readings per step (`rows`) and the reference
  (`reference`: the number of completed skills at every step, from a tracker, the training labels or a person). At every
  policy call (`--call-every`, default 8) it compares the writer's completed skills with the reference and reports, per
  skill, how often the writer switches in the same call, earlier, later, not at all, or without a reason. It checks *when*
  skills switch, not *which* object they name.
- **Drafting** (`python -m vla_memory.writers.rules.author --variant easy --out authored.yaml --task task.yaml`): the
  planning model writes `history.rules` into a copy of the task file, from the task's completion criteria (`easy`) or from
  the statistics of labelled demonstrations (`--variant hard --demos 'demos/*.npz'`, the numbers of section 1). Its rules
  are a draft: check them with the validator first.

## 11. Checking a run

With `--log FILE`, every line of the proxy's log is one call. In a run of the rules writer:

- `advanced` is true at the calls where a skill was completed, and `completed` names it;
- `writer_state` shows the grasp cycle's state after the call's last step (*free*, *holding*, *lifted*, *released*);
- `identified` shows, at a call with a pick, the held object, the expected one, and whether the plan was reordered;
- the first call of an episode also carries the plan (`plan`, `plan_source`).

## 12. Limitations

- **It only knows what the gripper feels.** A block set down but fallen over, rolled away or in the wrong place still counts
  as placed.
- **Which object, only by colour.** Without the colour check it assumes the plan's object. The colour check needs objects of
  clearly different colours and a wrist camera that sees the held object; otherwise it answers "unknown" and the plan's
  order holds.
- **No planning.** It follows the task file's order, apart from picks the colour check moves forward; a new instruction
  needs the VLM writer.
- **No going back.** It never takes a completed skill back.
- **The thresholds must match the training labels.** Thresholds measured on a different robot, gripper or task make the
  policy fail even when the writer detects the event correctly.
- **Every control step is needed.** If the environment sends only the readings at policy calls, a short grasp or release
  between two calls is missed.
- **One gripper, one held object.** A task with two arms, or with objects pushed instead of grasped, needs
  `trigger: signals` or another writer.
