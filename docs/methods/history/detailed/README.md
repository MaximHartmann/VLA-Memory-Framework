# The history method in detail

The short page, [docs/methods/history](../README.md), says what the history method does and what we found. This page
explains how it works: what the policy sees, how the proxy builds the prompt at every call, when a new line reaches the
policy, how the policy and its training data are made, and how to use the method with your own policy and task.

![The history method: the prompt grows with what has been done](history_prompt.svg)

## 1. The problem and the idea

**The problem.** A VLA policy chooses its next movement from what it sees now: the camera pictures and the prompt. It does
not remember earlier calls. Our reference task shows what that costs. The robot picks up the red block and puts it back, then
does the same with the blue block. When the red block is back on the table, the scene looks almost exactly as it did at the
start. A policy trained on the plain instruction cannot tell the two moments apart. It completed 11 of 100 tasks. In the 31
episodes in which it had put the red block back, it went on to the blue block 17 times and grasped the red block again 13
times.

**The idea.** Tell the policy what has been done, in words. The instruction stays the full sentence, and one line is added
for every finished skill:

```
Pick up the red block with your arm, place it back, and then grasp the blue block. History: picked up the red block, placed the red block.
```

At the start and after the red block is put back, the pictures are alike, but the prompts are not: "History: none yet."
against "History: picked up the red block, placed the red block." The policy needs no new part. It reads text, as it always
does.

**Why text.**

- *It is small.* What this task needs to remember is which skills are finished. The history adds 5 to 19 tokens (pieces of
  words) to the prompt.
- *It needs no new training code.* The demonstrations get new prompts, frame by frame, and the policy is fine-tuned as
  before (section 6).
- *A person can read it.* The proxy's log holds the exact prompt of every call (section 7).
- *The writer can be exchanged.* The method only turns the writer's list of finished skills into text. The simulator's ground
  truth, rules on the gripper, a vision-language model and the policy's own event head all drive it.

**Why the policy learns to read it.** In the demonstrations, the history is the only input that tells two similar-looking
frames apart. So the training loss rewards the policy for reading it.

**What the method does not do.** It does not decide when a skill is done. That is the writer's job. It keeps nothing inside
the policy, and it does not help the policy grasp.

## 2. Related work

- **UniMem** (Osterberg, Wang and Schwager, 2026, arXiv 2608.22869) is where the idea comes from: events are appended to the
  instruction as text ("History: grabbed spoon, ..."), detected by a small event head that is trained together with the
  policy. Here the writer can be exchanged, and the policy is trained on demonstrations alone.
- **RoboMME** (Dai et al., 2026, arXiv 2603.04639) tested earlier sub-goals as text with different writers and found that the
  writer, not the form of the memory, limits the result.
- **AGM** (Gao et al., 2026, arXiv 2608.29537) writes progress only on events it has checked on the gripper, and our rules
  writer follows the same design.

## 3. What the policy sees

The prompt has three parts: the instruction as the environment sent it, the words " History: ", and the history text. In the
reference task there are four skills, and the policy sees one of four prompts. Here they are with the first of the five
trained wordings of the instruction:

| skills finished | the prompt | tokens of the history |
|---|---|---|
| none | Pick up the red block with your arm, place it back, and then grasp the blue block. History: none yet. | 5 |
| pick up red | ... History: picked up the red block. | 8 |
| pick up red, place red | ... History: picked up the red block, placed the red block. | 13 |
| pick up red, place red, pick up blue | ... History: picked up the red block, placed the red block, picked up the blue block. | 19 |

- **The instruction is never changed.** The history is added at its end. The instruction itself is 22 tokens.
- **There is no fourth line.** When the blue block is set down, the task is over. The demonstrations end during that last
  skill, so the training data never has a history of four lines, and the method never shows one (`max_lines`, below).
- **The words must be exactly the trained ones.** The policy was trained on these strings (section 6). A different wording
  at run time, even a missing comma, is a prompt it has never seen. That is why the training script and the method make the
  text with the same code (section 4.2).

The words come from the `history` section of the task file:

```yaml
history:
  prefix: " History: "
  none: "none yet"
  events: {"pick up": "picked up the {object} block", "place": "placed the {object} block"}
  joiner: ", "
  suffix: "."
  max_lines: 3
```

| key | what it is | in the reference task |
|---|---|---|
| `prefix` | the text between the instruction and the history, spaces included | " History: " |
| `none` | the history before any skill is finished | "none yet" |
| `events` | one past-tense phrase per verb; `{object}` is replaced by the skill's object | "picked up the {object} block", "placed the {object} block" |
| `joiner` | the text between two phrases | ", " |
| `suffix` | the text after the last phrase | "." |
| `max_lines` | the most lines shown: the longest history in the training data | 3 |

The first five keys have defaults, the values in this table, so a task file can leave out what it keeps. Set `max_lines`:
without it the method shows every line, while the training script stops at every skill but the last. The same section holds
settings that the method does not read: `rules`, `object_from`, `held_region` and `held_min_share` belong to the [rules
writer](../../../writers/rules/README.md).

## 4. How the method builds the prompt

![One policy call with the rules writer and the history method](history_call.svg)

The method is small: two files, 116 lines in all. This section follows one run of

```
python -m vla_memory.proxy --writer rules --method history --task vla_memory/tasks/red_blue_blocks.yaml \
    --listen-port 8001 --upstream-port 8000
```

and names the function that does each step. The parts every writer shares (the server, the relay, the connection to the
policy) are explained step by step in the [oracle walkthrough](../../../writers/oracle/detailed/README.md).

### 4.1 The pieces

| file | what it does |
|---|---|
| `vla_memory/methods/history/prompt.py` | `HistoryPrompt`, the renderer: turns an instruction and a list of finished skills into the prompt |
| `vla_memory/methods/history/method.py` | `HistoryMethod`, the method: at every call it asks the writer for the finished skills, applies `--delay-calls` and `max_lines`, and hands the list to the renderer |
| `vla_memory/writers/` | the writers; each keeps the plan and a pointer into it, and `completed()` returns the skills before the pointer |
| `vla_memory/proxy/relay.py` | `Relay.infer`: what happens at every policy call |
| `vla_memory/task.py` | `TaskSpec`: the task file; `parse_skill` splits a skill into its verb and its object |
| `training/relabel_history.py` | makes the training prompts (section 6) |

### 4.2 The renderer: from finished skills to text

`HistoryPrompt.render(instruction, completed)` works in three steps. Take the finished skills `["pick up the red block",
"place the red block"]`:

1. **One phrase per skill** (`event`). `task.parse_skill("pick up the red block")` gives the verb "pick up" and the object
   "red". The verb's phrase, "picked up the {object} block", is filled with the object: "picked up the red block". The
   second skill gives "placed the red block". A verb without a phrase in `events` keeps the skill's own words.
2. **The history text** (`history_text`). The phrases are joined with ", " and closed with ".": "picked up the red block,
   placed the red block." An empty list gives "none yet.".
3. **The prompt.** The instruction without spaces at its ends, then the prefix, then the history text: "Pick up the red
   block with your arm, place it back, and then grasp the blue block. History: picked up the red block, placed the red
   block."

The renderer knows nothing about writers or calls. The same class makes the training prompts (`frame_prompts`), the
training rows of the head writer's event head, the keyframe method's labels and the example in the rules writer's authoring
prompt. So the text the policy was trained on and the text it gets at run time come from the same lines of code.

### 4.3 Start-up

1. **The method is found.** `resolve_method("history")` finds `vla_memory.methods.history.method:HistoryMethod` in the list
   of methods and imports it. `HistoryMethod.add_arguments` adds its two options, `--max-lines` and `--delay-calls`.
2. **The task file is loaded** (`TaskSpec.load_or_default`). Without `--task`, this is the reference task.
3. **The method is built** (`HistoryMethod.build`, then `__init__`). It makes the renderer from the task file's `history`
   section, takes `max_lines` from `--max-lines` or else from the task file, takes `delay_calls` from `--delay-calls` (0 when
   it is not given), and starts an empty queue for the delay.

The method adds nothing to what the proxy tells the environment when it connects.

### 4.4 One policy call

The environment sends an observation: the camera pictures, the robot state, the full instruction as `prompt`, and the
`memory/` entries the writer needs. For the rules writer these are the gripper readings of every control step since the
last call (a control step is one tick of the robot's controller, 15 per second) and the wrist picture. The method itself only
reads the instruction. `Relay.infer` then goes through these steps:

1. **It sets the `memory/` entries aside.** The rest of the observation goes to the policy later.
2. **At a new episode**, it calls `HistoryMethod.start_episode`, which empties the delay queue, and the writer's
   `start_episode`, which starts the plan again with nothing finished.
3. **The writer decides** (`writer.before_call`). The oracle, rules and VLM writers move their pointer here. A skill
   finished since the last call therefore already counts for this call.
4. **The method makes the prompt** (`HistoryMethod.prompt(instruction, writer)`):
   - if the writer hands on a text of its own (`writer.passthrough`, below), that text is the prompt;
   - otherwise the writer's finished skills, `writer.completed()`, go through `render`: first the delay (only with
     `--delay-calls`), then the cut to `max_lines`, then the renderer.
5. **The prompt goes into the observation** in place of the instruction, and into the call's log record.
6. **The observation goes to the policy server** (`_forward`), and the proxy waits for the actions.
7. **The writer reads the reply** (`writer.after_call`). Only the head writer decides here.
8. **The actions go back to the environment**, with the log record.

### 4.5 The details

- **The finished skills.** `writer.completed()` returns the skills of the plan before the writer's pointer, in plan order;
  once the writer has marked the whole plan as done, all of them. A writer only moves its pointer forward. So a line, once in
  the history, stays until the episode ends, and no skill is listed twice: if a block slips out of the hand and is grasped
  again, "picked up" still appears once.
- **`max_lines` keeps the first lines.** The list is cut after `max_lines` skills. In the reference task this hides the
  fourth skill, "place the blue block", which the policy never saw as a line. The first lines are kept, not the newest,
  because the training data never holds a history whose first lines were dropped.
- **`--max-lines N`** replaces the task file's value. It serves tests: with 1, the history never grows past its first line,
  like a writer that misses every line after the first.
- **`--delay-calls N`** shows the history of N calls ago. The method keeps the lists of the last N + 1 calls in a queue
  (`_delayed`) and shows the oldest. With N = 2: calls 1, 2 and 3 show the list of call 1, call 4 shows the list of call 2,
  and so on. The first calls of an episode all show its first history, "none yet". This imitates a writer that is late by a
  fixed number of calls. Normal runs use 0.
- **Passthrough (oracle writer only).** The oracle writer reads the skill that the simulator reports as current. When the
  simulator sends a text that is not one of the task's skills, the writer keeps its pointer and hands the text on, and the
  method sends that text as the prompt, without a history. A simulator can use this to send a complete prompt of its own.
- **`--writer off`.** The relay never asks the method, and the policy gets the instruction unchanged. This is how the
  memoryless policy runs behind the proxy.
- **Episodes.** A new episode starts at the first call, when the environment sets `memory/new_episode`, or when
  `memory/episode` changes. The history then starts again at "none yet". Nothing is carried from one episode to the next.

## 5. When a new line reaches the policy

The method shows what the writer has decided by the moment of the call. When that is depends on the writer:

| writer | decides | the new line first appears in the prompt of |
|---|---|---|
| `oracle` | before the call, from the skill the simulator reports (`memory/oracle_prompt`) | the same call |
| `rules` | before the call, from the gripper readings of every control step since the last call | the same call |
| `vlm` | before the call, after asking the vision-language model; the robot waits for the answer | the same call |
| `head` | after the call, from the event head's verdict in the policy's reply; it needs two "done" verdicts in a row | a later call, at least one call later |

A policy call comes every 8 control steps, so "the same call" means at most 8 control steps (0.53 s) after the event. No
writer can show a line sooner.

**A whole episode with the rules writer:**

| from the first call after | `writer.completed()` | the end of the prompt |
|---|---|---|
| the start of the episode | nothing | "History: none yet." |
| the red block is 9 cm up | pick up red | "History: picked up the red block." |
| the hand has let go of the set-down red block and moved 6 cm up (or 8 cm away) from where it let go | pick up red, place red | "History: picked up the red block, placed the red block." |
| the blue block is 9 cm up | the first three skills | "History: picked up the red block, placed the red block, picked up the blue block." |
| the blue block is set down | all four, cut to the first three | unchanged |

These are the moments at which the prompts of the demonstrations change (section 6). So the history in the prompt switches
at the same points of the movement as the history the policy was trained on.

**What timing the policy needs.** The policy takes every line as true. A line written too early, such as "picked up" at
the grasp before the lift, makes it go on as if the step were done. A line that never comes leaves it with a history that
does not match the scene. A line that comes a call or two late only means that the policy sees the old history a little
longer, as it did in training just before each switch. So a writer must never be early and never miss a line; being a
little late is fine. The short page has the test.

## 6. How the policy and its training data are made

![How the training prompts are made from one recorded demonstration](history_labels.svg)

**The demonstrations.** A scripted expert performed the reference task 250 times in Isaac Sim: 200 episodes for training
(63,644 frames) and 50 for checking (15,907 frames). Each recording holds, for every frame, the camera pictures, the joint
state, the actions and the waypoint phase the expert was in (0 to 14).

**Step 1: the plain dataset.** `training/convert_to_lerobot.py --action-space absolute` writes the recordings as a LeRobot
dataset (the data format the training code reads). Every frame's prompt is the plain instruction; each episode has one of
the five wordings (`instruction_examples` in the task file; 44, 44, 38, 37 and 37 episodes).

**Step 2: the history prompts** (`training/relabel_history.py`). Every frame's prompt becomes the instruction followed by the
history that is true at that frame. For each episode the script:

1. checks that the dataset's joint state equals the recording's, so that a history is never attached to another episode's
   frames (`states_match`);
2. reads the plan and the plan position of every frame: from the waypoint phase and the task file's table
   (`demo_phases.phase_to_skill`), or, for a task whose plan varies, from the recording's own `plan` and `skill_index`
   (`episode_plan`, `episode_positions`);
3. gives each frame as many history lines as there are skills before its position, at most `history.max_lines`, and checks
   that the episode goes through the histories in order (`episode_histories`);
4. renders the prompts with the same `HistoryPrompt` that the proxy uses, and writes the new `task_index` column.

Only the prompts change. The pictures, the state and the actions are copied unchanged, so the normalisation statistics of the
data (the means and spreads that scale the numbers going into the network) stay the same. The script rewrites the dataset's
metadata and checks it again at the end.

In the reference task the phases give four histories: phases 0-3 "none yet", 4-6 "picked up the red block", 7-11 "picked up
the red block, placed the red block", 12-14 "picked up the red block, placed the red block, picked up the blue block". A line
appears where the sub-goal method's prompt would change. In all there are 20 different prompts (5 wordings times 4
histories). In the 200 training episodes the frames split like this:

| history | frames |
|---|---|
| none yet | 16,805 |
| one line | 13,200 |
| two lines | 20,439 |
| three lines | 13,200 |

**Step 3: training.** pi0.5 was fine-tuned from `pi05_base` with LoRA (small added matrices are trained, the original weights
stay frozen) on this dataset, with the same recipe as the memoryless and the sub-goal policies: 30,000 steps at batch size
16, the learning rate falling from 2.5e-5 to 2.5e-6 on a cosine curve after 1,000 warm-up steps, about 38 hours on one RTX
3090 ([training/README.md](../../../../training/README.md) has the details). The policy predicts absolute joint targets. It
reads the two camera pictures and the prompt; the reference recipe gives it no joint angles. At run time the outside picture
is centre-cropped to 95 %.

**Step 4: choosing a checkpoint.** A checkpoint was saved every 1,000 steps. Each was tried on the same 20 development seeds
(a seed fixes where the cubes start) with the simulator's ground truth as the writer, and the one that got furthest through
the task on average was taken: checkpoint 28,000. The task is scored by seven milestones, counted in order: reached red,
lifted red (more than 9 cm above its resting height), placed red, retreated, reached blue, lifted blue, placed blue.

## 7. Checking a run

With `--log FILE` the proxy writes one JSON record per policy call. The fields that matter for the history method:

| field | what it holds |
|---|---|
| `episode`, `step` | the episode and the control step of the call |
| `prompt` | the exact prompt the policy received |
| `advanced` | true at the call whose prompt is the first to show a new line |
| `skill_index` | the writer's pointer, the number of finished skills |
| `completed` | the skills finished at this call (rules writer) |
| `identified` | at a pick, the colour of the block in the hand and whether the plan was reordered (rules writer) |

Two calls of one episode with the rules writer, shortened:

```
{"episode": "seed0", "step": 24, "mode": "rules", "advanced": false, "skill_index": 0, "completed": null, "prompt": "Pick up the red block with your arm, place it back, and then grasp the blue block. History: none yet.", ...}
{"episode": "seed0", "step": 32, "mode": "rules", "advanced": true, "skill_index": 1, "completed": ["pick up the red block"], "identified": {"held": "red", "expected": "red", "reordered": false}, "prompt": "Pick up the red block with your arm, place it back, and then grasp the blue block. History: picked up the red block.", ...}
```

## 8. Limitations

- **The policy trusts its history.** A wrong line stays for the rest of the episode. An early or a missing line sends the
  policy the wrong way (section 5). The writers never take a line back, so a placed block that is moved afterwards still
  counts as placed.
- **It is only as good as the writer.** The rules writer needs thresholds measured from the training labels and the colour
  check for the object; the VLM writer needs seconds per question; the head writer needs a head fitted to this one policy.
- **The history shows the first lines only.** `max_lines` cuts the list after the first lines, not the last ones, and the
  training data must contain the longest history. A long task needs `max_lines` high enough, and the prompt grows by a line
  for every skill. We did not test a task with more than four skills.
- **Words and order are fixed.** The policy was trained on one order of skills (red first) and one wording of the lines. A
  different order was never trained, and neither was an instruction in new words. The history of a repeated skill reads the
  same each time; we did not test whether that confuses the policy.
- **No pictures, no positions.** The history lists finished skills. UniMem found that text alone does not carry tasks where
  positions matter. Our task does not need them, so this limit was not tested here; the [keyframe
  method](../../keyframe/README.md) gives a picture instead.
- **Grasping is not helped.** The history tells the policy what has been done, not how to grasp.
- **One policy, one task, simulation.** The method was built and tested with one pi0.5 policy trained on 200 scripted
  demonstrations of one task, in simulation.
- **One episode only.** The history starts again at every episode; nothing is carried over.

## 9. Using it with your own policy and task

1. Record demonstrations that note, at every frame, which skill is active: a phase table in the task file
   (`demo_phases.phase_to_skill`), or the plan and the plan position of every frame (`plan`, `skill_index`).
2. Write the task file's `history` section: `prefix`, `none`, one past-tense phrase per verb in `events` (with `{object}`),
   `joiner`, `suffix`, and `max_lines`, the number of lines in the longest history of your training data.
3. Run `training/relabel_history.py --task your_task.yaml` on the converted dataset and fine-tune your policy on it. Any VLA
   that reads a text prompt works; the method only changes the prompt.
4. Choose a writer and check its timing on recorded runs before you trust it: never early, never missing, at most about a
   second late. The rules writer has a validator for this (`python -m vla_memory.writers.rules.validate`).
5. Run the proxy with `--method history`. Use `--delay-calls` and `--max-lines` to test how much timing error your policy
   tolerates.
