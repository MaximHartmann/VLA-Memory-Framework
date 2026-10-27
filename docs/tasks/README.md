# Creating a new task

> **Work in progress.** This page is a first draft. It explains the task file, how to check that a new one loads, and
> what else a new task needs. Still missing: a second worked example (a task without pick and place, and one whose objects
> change from episode to episode), one command that checks every section of a task file at once, advice on writing the VLM
> writer's texts for a new robot, and how to build the `demo_phases` table for new demonstrations. So far only the
> reference task has run live (in simulation).

## What a task file is for

The framework knows a task only from its task file (YAML or JSON), which the proxy reads with `--task FILE`; without it,
the proxy uses the reference task. The code holds nothing task-specific. The task file says:

- **which skills there are, and in which order**: the exact prompt strings the policy was trained on, and the task order,
  which is the plan whenever nobody else plans;
- **how the writers tell that a skill is complete**: a written rule per verb and the texts for the VLM writer, triggers
  and thresholds for the rules writer;
- **what the methods show**: the words of the history lines, how many lines there are at most (also the most key moments
  of the keyframe method), and the settings of the keyframe method's later picture;
- **how recorded demonstrations are labelled**, and which colours find similar moments in the example store.

The oracle and head writers read no section of their own: the oracle follows the simulator's tracker, and the head writer
follows the policy's own event head (at most `history.max_lines` skills). On the framework's side, a new task is a new task
file and no code.

## The sections, with the reference task

The worked example is the packaged reference task, [`vla_memory/tasks/red_blue_blocks.yaml`](../../vla_memory/tasks/red_blue_blocks.yaml):
a Franka arm picks up a red cube and puts it back, then the blue one. Below is every field, its value in the reference
task, and what happens when it is left out. The fields are those of `TaskSpec` in
[`vla_memory/task.py`](../../vla_memory/task.py); keys it does not know are ignored.

### The skills

| field | reference task | what it is for | when left out |
|---|---|---|---|
| `name` | `red_blue_blocks` | the task's name, in the proxy's metadata | required |
| `objects` | `[red, blue]` | the objects; an episode may hold only some of them, and the environment then names them (`memory/objects`) | required |
| `verbs` | `["pick up", "place"]` | the verbs, in the order of one object's skills | required |
| `skill_template` | `"{verb} the {object} block"` | how a skill reads: "pick up the red block" | required |
| `skill_templates` | none | a template of its own for a verb whose skills read differently | every verb uses `skill_template` |
| `plan_stages` | none | the task order in stages, for example `[[pick up, put in box], [take out, stack]]`: each stage goes over all objects | one stage: object by object |
| `max_plan_len` | `8` | the most skills the VLM writer may put into a plan | 8 |
| `instruction_examples` | five instructions | examples; the first one is used where a tool needs an instruction (the rules author, the session store's import) | none |
| `notes` | none | free text for people; nothing reads it | empty |

The reference task's skills in the task order are "pick up the red block", "place the red block", "pick up the blue block"
and "place the blue block". They must be exactly the strings the policy was trained on, letter for letter.

### The states of the objects

The VLM writer's answers and the labels of the example store give every object a state, and the gripper one too.

| field | reference task | what it is for | when left out |
|---|---|---|---|
| `object_states` | `[on_table, held_lifted, held_at_table, unsure]` | the states an object can be in | required |
| `object_field` | `"{object}_cube"` | the name of an object's field in the answer | `"{object}_cube"` |
| `object_noun` | `cube` | how generated text calls an object ("red cube = on_table") | `cube` |
| `state_roles` | `{rest: on_table, held_low: held_at_table, held_high: held_lifted}` | which states mean resting, held at the resting height and held lifted (the labellers and the recovery step need them) | the reference task's names |
| `monitor_fields` | none | the answer's other fields: per field a list of values, `boolean`, `number` or a JSON schema | the gripper (`open`, `closed`, `unsure`) and whether it is high above the table |

### The VLM writer

| field | reference task | what it is for | when left out |
|---|---|---|---|
| `criteria` | one rule per verb ("Complete only when BOTH hold: ...") | the completion rule, put word for word into every question, with `{object}` filled in | required |
| `system_head`, `system_ext`, `system_both` | the robot, the table and the cameras | the system text of the question: the head, then the part for one camera or for both | required |
| `plan_system` | "You decompose a natural-language instruction ..." | the system text of the plan request; `{skills}` becomes the list of skills | required |
| `proprio` | `closed_threshold: 0.30`, `history: 12`, a `note` | the sentence about the gripper and the hand: closed from `closed_threshold` on, the last `history` readings, the `note` after the height | 0.3, 12, no note |
| `recovery` | `{"pick up": place}` | with `--recovery`: before a skill of the left verb, another object that is still held is put down with the right verb | a task of two verbs puts down with its second verb; other tasks never |
| `fingerprint` | a 12 x 16 thumbnail and a colour rule per object | the key that finds similar moments in the example store and the session store | no colours: the thumbnail only |
| `demo_phases` | the phase table of the demonstration generator | labels for recorded demonstrations by waypoint phase (`scripts/build_exemplar_store.py`, `training/relabel_subgoals.py`, `training/relabel_history.py`) | none: no labels by phase |

`criteria` and the four texts are required even when the VLM writer is not used, because every task file is loaded the
same way; a short placeholder text is enough then.

### The rules writer

| field | reference task | what it is for | when left out |
|---|---|---|---|
| `history.rules` | `pick up: {trigger: lift, rise: 0.09}`, `place: {trigger: set_down, release_below: 0.08, up: 0.06, away: 0.08}` | per verb the trigger that completes it, and its thresholds (in metres for a grasp) | none: the rules writer does not start |
| `proprio` | `open_width: 0.06`, `empty_width: 0.015`, `squeeze_force: 8.0` | what the gripper readings mean: open, closed on nothing, holding something | the same values |
| `history.object_from` | `colour` | `colour`: the wrist picture names the held object at every pick; `plan_order`: the plan says which | `plan_order` |
| `history.held_region`, `history.held_min_share` | `[0.30, 0.60, 0.33, 0.67]`, `0.12` | where a held object appears in the wrist picture, and how much of that region its colour must fill | the same region, 0.25 |
| `signals` | none | the robot's other signals, for rules with `trigger: signals` (sent as `memory/signal_steps`) | none |

How to measure the thresholds on demonstrations: [the rules writer in detail](../writers/rules/detailed/README.md),
section 1. The session store's labeller of live episodes also reads `proprio.high_above_table` (default 0.17 m) and
`proprio.lifted_above_grasp` (default 0.06 m).

### The methods

| field | reference task | what it is for | when left out |
|---|---|---|---|
| `history.prefix`, `history.none`, `history.joiner`, `history.suffix` | `" History: "`, `"none yet"`, `", "`, `"."` | the text around the history lines | the same |
| `history.events` | `{"pick up": "picked up the {object} block", "place": "placed the {object} block"}` | the past-tense line of each verb | a verb without one: the skill's own words |
| `history.max_lines` | `3` | the most lines: the longest history in the training data; also the most key moments of the keyframe method | every skill but the last |
| `keyframe.rest_z` | `0.76` | with `--keyframe-placed-later`: the height in metres of a resting object's centre | 0.76 |
| `keyframe.placed_verbs` | none | the verbs whose key moment waits until the hand has risen | the verbs of the `set_down` rules, else the second verb |

With the reference task the history method sends, for example, "Pick up the red block with your arm, place it back, and
then grasp the blue block. History: picked up the red block, placed the red block." The last skill of an episode's plan
never gets a line or a key moment, as in the training data, whose demonstrations end during that skill. The sub-goal method
needs no section: it sends the skills themselves.

## Checking that a new task file loads

From the repository's folder (or after `pip install -e .`):

```
python -c "from vla_memory import TaskSpec; task = TaskSpec.load_or_default('my_task.yaml'); print(*task.skills, sep='\n')"
```

It prints the skills in the task order: compare them with the policy's training prompts. A missing required field stops it
with a `TypeError` that names the field. A `skill_templates` or `recovery` entry with an unknown verb, or `plan_stages` that
do not name every verb exactly once, stop it with a `ValueError`. The sections that only one writer or method reads are
checked when that one starts, or with a few more lines:

```python
from vla_memory import TaskSpec
from vla_memory.methods.history import HistoryPrompt
from vla_memory.writers.rules.rule_spec import parse_rules
task = TaskSpec.load_or_default("my_task.yaml")
print(task.default_plan(task.objects[:1]))                                          # the plan of an episode with one object
print(HistoryPrompt(task).render(task.instruction_examples[0], task.skills[:1]))    # a prompt of the history method
print(parse_rules(task))                                                            # the rules writer's triggers
```

Whether the rules switch at the right moments is checked on recorded episodes with
`python -m vla_memory.writers.rules.validate` ([the rules writer](../writers/rules/README.md)).

## What a new task needs besides the file

- **A policy trained on the task's skills**, with exactly the task file's strings, for the method you run. The scripts in
  [`training/`](../../training/README.md) read the task file (`--task`) and write each method's training data.
- **An environment that sends the `memory/` entries** with every policy call, built with `ControlKeys` from
  [`vla_memory/env_keys.py`](../../vla_memory/env_keys.py):
  - always the episode markers (`memory/new_episode`, `memory/episode`) and the step (`memory/step`);
  - `memory/objects` when an episode holds only some of the task's objects;
  - what the chosen writer reads: for the oracle the simulator's current skill (`memory/oracle_prompt`); for the rules
    writer the gripper readings of every control step (`memory/proprio_steps`; also `memory/signal_steps` for declared
    signals and `memory/wrist_raw` for the colour check); for the VLM writer the raw camera pictures, how closed the
    gripper is and how high the hand is.
- **Thresholds or texts that fit the robot**: for the rules writer, thresholds measured where the training labels switch;
  for the VLM writer, completion rules and camera descriptions tried on recorded pictures and, for the example store,
  recorded demonstrations labelled with `demo_phases` (`scripts/build_exemplar_store.py`).
