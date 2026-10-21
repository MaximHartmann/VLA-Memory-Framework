# The sub-goal method in detail

The short page, [docs/methods/subgoal](../README.md), says what the method does and what we found. This page explains why
it works, how the policy is trained for it, what the code does at every call, why the moment of the switch matters so much,
and how to use the method with your own policy and task.

![The sub-goal method: the policy only ever sees the current skill](subgoal_method.svg)

## 1. The problem it solves

A VLA policy decides from what it sees now: the camera pictures, the robot's state and the prompt. It has no memory of
earlier calls. In our reference task this is fatal. The robot picks up the red block and puts it back, then does the same
with the blue one. After red is back on the table, the scene looks exactly as it did at the start. A policy trained on
the full instruction cannot see that red is already done; it completed 11 of 100 tasks.

There are two ways out: give the policy a memory (the [built-in memory](../../builtin_memory/README.md) does that), or move
the memory outside the policy. The sub-goal method does the second. It splits the task into short skills, trains the
policy on those skills only, and lets someone outside the policy keep track of which skill comes next.

## 2. Where the idea comes from

This is the classic split between a planner and a library of skills. SayCan (Ahn et al., 2022, arXiv 2204.01691) had a
language model choose the next skill from a fixed set of trained skills, while a low-level policy carried each skill out.
The sub-goal method keeps that split but makes the planner exchangeable: in this framework the "planner" is whichever
writer runs. It can be the simulator's ground truth, rules on the gripper's readings, a vision-language model, or the
policy's own event head. The method does not care who decides; it only turns the decision into the prompt.

## 3. How the policy is trained for it

The policy has to know the short skills, so its training data must be labelled with them.

1. **Demonstrations.** A scripted expert performed the reference task 250 times in Isaac Sim. The script recorded, at every
   frame, which waypoint it was moving to (its phase).
2. **Relabelling** (`training/relabel_subgoals.py`). Every frame's prompt becomes the skill of its phase. The task file says
   which phases belong to which skill (`demo_phases.phase_to_skill`):

   | phases | skill | what happens |
   |---|---|---|
   | 0-3 | pick up the red block | approach, descend, close, lift |
   | 4-6 | place the red block | lower, open, move the hand back up |
   | 7-11 | pick up the blue block | the move away from red is part of it, so the direction stays continuous |
   | 12-14 | place the blue block | |

   Note that the hand's retreat after opening (phase 6) still belongs to "place". This decides when a writer may switch
   (section 5). The script works by phase, not by frame number, because the phases last different times in different
   episodes. Before it touches an episode it checks that the recording and the dataset really match, and that every
   episode goes through its skills in order. For tasks whose plan changes from episode to episode, a recording can store
   its own plan and the plan position of every frame instead of phases.
3. **Fine-tuning.** π0.5 was fine-tuned with LoRA on these relabelled demonstrations for 30,000 steps.
4. **Choosing a checkpoint.** Every saved checkpoint was tested on 20 development seeds. Several tied at the top; the tie
   was broken first by full tasks, then by the loss on held-out demonstrations. Checkpoint 20,000 was chosen.

## 4. What the code does at every call

The method is the simplest part of the framework (`vla_memory/methods/subgoal.py`, 19 lines):

1. The relay asks the writer to decide (oracle walkthrough, section 3).
2. The relay asks the method for the prompt: `SubgoalMethod.prompt` returns `writer.current_skill`, the skill the writer
   considers current.
3. The relay puts that skill into the observation's `prompt`, replacing the user's instruction, and calls the policy.

The method adds nothing else to the observation, keeps no state of its own and has no options. The proxy's log notes the
prompt of every call. One special case comes from the oracle: if the simulator sends a text that is not one of the task's
skills, the oracle passes it on as it is, and the policy receives that text.

## 5. When to switch

![When the writer may switch](subgoal_timing.svg)

The writer must switch at the moment the training labels switch.

**Switching too early.** Suppose a writer declares "placed" as soon as the fingers have opened and the cube rests on the
table. At that moment the hand is still around the cube. The policy now hears "pick up the blue block" while it sees an open
hand right above a cube, a picture that, in its training data, belongs to a pick. So it closes the fingers on the red cube
again. The simulator's tracker first had exactly this rule.

**The right moment.** "Placed" therefore also requires the hand to have moved at least 6 cm up from the cube, or 8 cm away
from it: the end of the retreat, where the demonstrations' labels switch from "place" to the next skill.

**What this means for every writer.** The rules writer's `set_down` rule (6 cm up or 8 cm away) and the VLM writer's
completion rule for "place" (the hand at least 19 cm above the table) encode the same moment. Being a little late is
harmless: the policy keeps the old skill a little longer, as it did in training just before each switch. Being early is not:
the policy then works on the next skill in a situation it has only seen during the previous one.

## 6. Limitations

- **The policy is blind to the task as a whole.** It does what the prompt says. A wrong plan, a missed switch or an early
  switch goes straight into its behaviour, and it cannot correct the writer.
- **Every skill must be trained.** A new task with new skills needs demonstrations of those skills, labelled with the
  skills' exact wording.
- **The wording is fixed.** The writer must produce exactly the prompts the policy was trained on; the task file's skill
  templates guarantee that for the framework's writers.
- **No undo.** The writers never move back. If a placed block falls off, the method keeps prompting the next skill.
- **Grasp failures are not memory failures.** When a grasp fails, the skill simply stays current and the policy tries
  again; whether it succeeds is the policy's skill, not the method's.

## 7. Using it with your own policy and task

1. Record demonstrations that note, at every frame, which skill is active (a phase table in the task file, or the plan
   position of every frame).
2. Relabel them with `training/relabel_subgoals.py --task your_task.yaml`, and train your policy on the short prompts.
3. Word the skills in the task file exactly like the training prompts (`skill_template`, `skill_templates`).
4. Pick a writer, and make sure it switches where your labels switch: check it on recorded runs before you trust it (the
   rules writer's validator does that for its rules).
