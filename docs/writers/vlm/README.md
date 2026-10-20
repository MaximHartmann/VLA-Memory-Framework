# The VLM writer: a planner that watches the robot

## The idea

A vision-language model (VLM) is a model that understands pictures and text together. In this writer it acts as the
planner and does two jobs:

1. **Once per episode it writes the plan.** It turns the user's instruction into an ordered list of the skills the policy
   was trained on, such as "pick up the red block", "place the red block", and so on. It may only choose from the task's
   skills (and only from those of the episode's objects, if the environment names them in `memory/objects`). If its answer
   is unusable, the writer falls back to the order of the task file.
2. **At every policy call it answers one question:** is the current skill complete? It looks at the two camera pictures and
   at the robot's own readings: how closed the gripper is, and how high the hand is. A "yes" moves the writer on to the next
   skill before the policy is called, so the policy gets the new skill in the same call.

![One policy call with the VLM writer and the sub-goal method](detailed/vlm_call.svg)

With the sub-goal method, where the policy gets the current skill as its prompt, this is called the planner loop.

**What the model is shown with each question:** the plan, with the current skill marked; the completion rule of the current
skill, written in the task file (for "pick up": the gripper is closed on the cube and the hand is at least 22 cm above the
table); one sentence about the robot's readings; the two current pictures; and, if an example store is loaded, the 4 most
similar moments from earlier episodes together with the right answer for each (see "How the example store is used").

**The decision rule** (the default settings):

- One "yes" from the model is enough to move on (`--votes 1`), but never at the first call on a new skill: every skill
  lasts at least two calls (`--min-calls 2`). With `--votes 2` the model would have to say "yes" at two calls in a row.
- It never moves back.
- Once the plan is finished, it stops asking.

**Three kinds of memory are involved:**

- **Working memory** of the running episode: the plan, the current skill and the last few readings. It starts fresh with
  every episode.
- **Long-term memory**, optional: the example store. It holds moments from recorded episodes, each labelled with the skill
  that was active and the state of every object, plus a compact colour summary of its pictures that is used to find similar
  moments. During an episode the store is only read, never changed.
- **The task file**: the skills, the completion rule of each skill, the description of the cameras, and the format of the
  answers.

**How the example store is used** at a call (the default numbers; without a store, the model is simply asked at every
call):

1. The writer looks up the **6** stored moments most similar to the current pictures (`--k-vote 6`).
2. **All 6 vote.** A moment says "done" if, at that moment of its recorded episode, the robot was already working on a later
   skill than the current one. If **none** of the 6 says "done", the writer concludes "not done yet" and does not ask the
   model at all; this saves time and is called the gate. The gate can only ever say "not yet": a "done" always comes from
   the model. The last skill of the plan is always asked.
3. If **at least 1** of the 6 says "done" (`--gate-min-done 1`), the model is asked. It is shown the **4** most similar of
   these same 6 moments as examples (`--k 4`), each with its right answer.

So the 6 stored moments decide *whether* to ask, the 4 most similar of them help the model answer, and the model's single
"yes" decides that the skill is complete.

The writer builds on SayCan (a language model plans over a fixed set of trained skills), Inner Monologue (after every step,
the planner hears whether it succeeded), SuccessVQA (a VLM judges success from pictures) and RAP (similar past experiences
shown as examples).

**What it cannot do well:**

- **It is slow.** A question takes the model about 0.7 s (2.9 s with four examples), a policy call only 0.17 s, and the
  robot waits for the answer at every call.
- **It needs a large model on a GPU**: in our setup a 27-billion-parameter model on two GPUs.
- **It depends on the wording of the task file.** It judges from pictures, guided by the task file's description of the
  cameras. With a misleading description of the wrist camera it completed 69 tasks of 100 instead of 92-94.
- **A wrong "yes" cannot be taken back**, because the writer never moves back. It was rarely wrong (below 1 % of its "yes"
  answers), but each mistake skips a skill.
- **The vote of the example store assumes a fixed order of skills.** For tasks whose objects may be handled in any order,
  turn it off with `--no-gate`.

A walkthrough of the code, from the command line to the last policy call: [detailed/README.md](detailed/README.md).

## Prerequisites

- A vision-language model behind an OpenAI-compatible chat API that can force its answers into a JSON format (structured
  output). `scripts/serve_vlm_vllm.sh` starts one with vLLM; our setup is Qwen3.8-27B-INT4 on two RTX 3090.
- A task file. `vla_memory/tasks/red_blue_blocks.yaml` is the reference: objects, verbs, how a skill is worded, the
  completion rules, and the texts that describe the cameras.
- Optional: an example store, built from recorded demonstrations with `scripts/build_exemplar_store.py`. No store comes with
  the repository.
- An environment that sends the full instruction as `prompt`, and these `memory/` entries: the step, the episode, the raw
  camera pictures, how closed the gripper is and how high the hand is, plus `memory/objects` for a task whose episodes hold
  only some of its objects. `vla_memory.env_keys.ControlKeys` builds them.

## How to use it in the framework

```
python -m vla_memory.proxy --writer vlm --method subgoal --task task.yaml --listen-port <port> --upstream-port <policy port> \
    [--vlm-url http://127.0.0.1:8100/v1 --vlm-model Qwen3.8-27B-INT4] [--memory store.npz]
```

`--task` can be left out for the reference task, and `--method history` or another method can replace `subgoal`. Without
`--writer` and `--method` the proxy runs exactly this pair. The defaults are the settings of our evaluation: the gate, a
yes/no answer only, 4 examples per question, one "yes" to move on, and no switch before the second call on a skill.
`--no-gate`, `--no-verdict-only`, `--k`, `--votes` and `--min-calls` change them; `--vlm-url` and `--vlm-model` name the
model. `scripts/run_proxy.sh` wraps the command.

- **The proxy's log** holds, for every call: `queried` (the model was asked), `advanced` (a skill was completed),
  `vlm_ms` (how long the model took), `skill_index`, `vlm` (the model's answer) and `recovery`.
- **The planner's own log** (`<log>.planner.jsonl`) holds the plan and every question with everything the model was shown.
- Other memories for the planner plug in as extensions (`--with NAME`), for example the preference notebook and the
  session store.

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim, pi0.5 policies; 100 fresh
seeds of 500 control steps, a policy call every 8 steps:

| method | example store | full tasks of 100 | median episode |
|---|---|---|---|
| sub-goal | none | 94 | 84 s |
| sub-goal | 20 recorded episodes | 87 | 133 s |
| history | 20 recorded episodes, answers with object states but no reason | 77 | 143 s |
| *for comparison:* sub-goal, rules writer | none (no model) | 94 | 55 s |
| *for comparison:* sub-goal, simulator's ground truth | none (no model) | 92 | 50 s |
| *for comparison:* history, rules writer | none (no model) | 79 | 51 s |
| *for comparison:* history, simulator's ground truth | none (no model) | 82 | 51 s |

- Without a store, the planner does as well as the simulator's ground truth (92 with the sub-goal method), once the task
  file describes the wrist camera correctly. With a misleading description it completed 69, and it needed the store to get
  back to 91: the examples made up for the wrong text.
- With a correct description, the store adds nothing to the result and costs 49 s per episode, because every question then
  carries four example pairs (2.9 s per question instead of 0.7 s). The bottleneck is the question, not the policy: a policy
  call takes 0.17 s.
- The planner's "yes" answers were right in 99.4 % of the cases (98.3 % with the store), and it never declared a failed
  plan complete. The remaining failures are the policy's grasps, not the memory.
- With the history method, the planner's history matched the simulator's in 94 % of the calls (77 full tasks, against 79
  with the rules writer and 82 with the ground truth), at 1.6 s per policy call instead of 0.17 s with the rules writer.
- When the order of the skills is known in advance, the rules writer can replace the planner at no cost (94 with the
  sub-goal method, 55 s per episode). The planner is needed to turn a new instruction into a plan.
