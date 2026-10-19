# The preference notebook: a memory of the user

## The idea

The memory is a notebook about the user, kept as text and written across episodes. The VLM writer's planner reads it before
every episode; nothing in the policy or the planning model changes. It holds preferences (how the user wants the task done,
stated in words) and corrections (rules written after a failure). Every entry is one rule with a scope, which decides
where it acts, never a similarity search:

| the user says | scope | where it acts |
|---|---|---|
| "from now on, please handle the blue block first", "only the red block today" | `plan` | a paragraph of the plan request |
| "count the red block as picked up only when the hand is 22 cm up" | `skill:<skill>` | after that skill's completion criterion, in every monitor question |
| "double-check before you move on", "be quick" | `planner` | a preset of the planner's parameters: two "complete" verdicts in a row, or a question only at every second policy call |

```
User profile (apply unless the instruction says otherwise):
- Handle the blue block first.
```

Five rules keep it a profile rather than a growing log:

- **Few rules per scope.** At most three active entries per scope reach the planner, ranked by importance, then recency.
- **Supersession.** A contradicting statement replaces the old entry, which stays in the file with the time it ended; a
  repetition confirms an entry; a withdrawal ("forget what I said about the order") retires it.
- **Votes.** Importance starts at 2. After an episode an applied correction gains a point on success (up to 5) and loses one
  on failure; at 0 it is retired. Stated preferences are not voted out by the robot's own failures.
- **Validity.** An entry holds until withdrawn, for the day or for the session, as its words say ("today", "this time");
  the notebook keeps its own clock and judges validity at every episode start. A rule for the day that replaces a lasting
  one only hides it until the day is over.
- **The instruction has precedence.** An instruction that names any of the task's objects decides itself which objects are
  handled and in which order, so a plan rule that names objects is held back for that episode (logged, not applied).

The writer is the planning model. It answers three questions under JSON schemas built from the task's skills, so its answer
can only be an entry the planner understands: a user message becomes at most three rules; a failed episode's decision log
becomes at most one correction; and a proposed rule is merged with the rules of its scope (add, update, delete or no
change, as in Mem0). The votes follow ExpeL, the supersession Zep. A gold writer answers the same questions from
annotations.

## Prerequisites

- The VLM writer (`--writer vlm`): the notebook is read by its planner (the plan request, the completion criteria, the
  parameters). The task file's skills define the scopes.
- For the model writer, the planning model's server (any OpenAI-compatible chat API); the gold writer needs none.
- One JSON file per user; the writer creates it.

## How to use it in the framework

Run the proxy with the VLM writer and the notebook; it re-reads the file at the start of every episode, so a writer may
change it in between. `--profile-off` runs the same command line without it:

```
python -m vla_memory.proxy --writer vlm --method subgoal --with notebook --profile notebook.json \
    --listen-port <port> --upstream-port <policy port>
```

Write it between episodes:

```python
from vla_memory import OpenAICompatibleVLM, TaskSpec
from vla_memory.methods.notebook import PreferenceWriter, ProfileStore

writer = PreferenceWriter(OpenAICompatibleVLM(), TaskSpec.default(), ProfileStore("notebook.json"))
writer.from_message("From now on, please handle the blue block first.")
writer.from_failure(planner_records, "failed", "You moved on too early.")   # a correction after a failed episode
writer.feedback(applied_ids, success)                                       # the votes after an episode
```

`vla_memory.methods.notebook.session.run_session` runs a scripted session of messages, episodes and clock steps against any
episode runner, in four conditions: `profile` (the notebook), `none`, `first` (each message in front of the next
instruction only) and `every` (all messages so far in front of every instruction). It writes one JSON line per message,
episode, vote and correction and scores the writer against annotations. Two example sessions are in
`vla_memory/methods/notebook/sessions/`: `validity_and_precedence.yaml` and `harmful_correction.yaml`.

## Our findings

The VLM writer in its settled configuration drove the sub-goal policy in simulation (the planner loop), 20 development
seeds per condition (5 per phase in the retention test); the planning model wrote the notebook, except for the corrections,
which came from the annotations (the gold writer):

| test | with the notebook | without memory | restated in every instruction |
|---|---|---|---|
| "Please only do the red block today." | plan followed 20/20, success 20/20 | plan 0/20, success 18/20 | plan 20/20, success 20/20 |
| "Please double-check ...", then "... please be quick." | parameters in force 20/20 and 20/20; success 18/20 and 17/20 | success 17/20 | |
| a correction after switches at 19 cm, too early | success 19/20, pick switch at 0.277 m | success 12/20, 0.235 m | |
| an order rule, four unrelated messages, then a new order | plans followed 5/5 in all four phases, parameters 5/5 once stated | | plans 5/5, parameters 0/5 |

- The notebook is followed: the plan rules, the presets and the corrections reached the planner in every episode. "Be
  quick" cut the model calls per episode from 28.5 to 16.5 and the episode from 2.1 to 1.8 minutes at the same success; the
  correction removed the switches before the block was lifted (8 of 35 without it, 1 of 40 with it, in the first episode).
- Validity and precedence: in the control of the first run, whose instruction named both blocks, the written rule ("only
  the red block") was applied anyway and the blue block dropped in 5 of 5 episodes. With validity and precedence the
  rerun followed the rule in 20 of 20 episodes, held it back in all 5 controls (both blocks planned and placed), and the
  next day the rule had ended (5 of 5 with both blocks).
- Votes retire a harmful rule: an unreachable correction ("only when the hand is at least 35 cm above the table") failed
  its first two episodes and was retired after the second; 7 of the 8 episodes after it succeeded, with the pick switches
  back at 0.281 m. A rule that loosens the criterion is ignored by the monitor, causes no failure and is kept by the votes,
  which cannot tell an inert rule from a helpful one.
- The writer on 18 annotated messages with 16 rules: precision and recall 16 of 16, no rule from small talk or questions,
  16 of 16 merges and 15 of 15 validities as annotated; on 20 messages with time cues over three days 16 of 16 rules and
  validities, on 10 harder phrasings 8 of 8. Distilled corrections find the skill where the episode went wrong (14 of 17
  failures) but restate the criterion in force instead of tightening it, so the corrections above came from annotations.
- Asked for blue first, the policy lifted blue first in 18 of 20 episodes but completed both blocks in 8: its training
  data always took red first.
- Reading the notebook costs 0.5 ms at an episode start and 0.2 ms per monitor question; the writer needs about 1.5 model
  calls per message.

Ten days with one user (80 episodes, seeds 3001-3080; nine messages: double-check, later "be quick"; blue first, later
withdrawn; only red today; a harmful 35 cm correction; small talk; blue first for one session; the planning model wrote the
notebook; the sub-goal policy with the VLM writer; pre-registered):

| condition | plan followed (72 episodes from the first plan preference on) | parameters in force | task success / 80 | done the user's way | user effort |
|---|---|---|---|---|---|
| the notebook | 72/72 | 76/76 | 54 | 54 | 9 messages, 91 words |
| no memory (each message before the next episode only) | 33/72 | 0/76 | 73 | 40 | 9 messages, 91 words |
| every message restated before every episode | 72/72 | 0/76 | 58 | 58 | 280 messages, 3,376 words |

- The notebook follows the user as the restating user does (p = 1.0) and far better than no memory (p = 3.6e-12), at a
  thirtieth of the user's words, with 8.8 words added to the plan request instead of 42, and 26 model calls per episode
  instead of 34 (its "be quick" preset halves the questions from day 6 on).
- No false application in 80 episodes: the day rule ended at midnight, the session rule with the session, the withdrawal
  held for three days, and the instruction's own order won in the two control episodes.
- The harmful correction was applied in two episodes, both failed, and retired itself (votes 2 -> 1 -> 0); it was never
  applied again.
- The no-memory robot has the highest task success because it ignores the user: it runs the red-first order the policy
  executes better. On the blue-first days every condition that follows the user completes 2-4 of 8 episodes.
- With the robot's own lessons added (reflection): 53 of 80, the same as without them.
