# The preference notebook: a memory of the user

## The idea

A robot that works for one person should learn how that person wants things done, and should not need to be told again every
day. An instruction only says what to do now. It cannot say "from now on", "only today" or "be more careful". Without a
memory, the robot follows a wish only in the episode right after it was said, and a user who wants it followed every time has
to repeat every wish before every episode.

The notebook keeps the wishes as short rules about the user. They are written once and read at the start of every episode by
the planner of the VLM writer (the writer that asks a vision-language model for the plan and for whether each skill is done).
The notebook holds **preferences** (how the user wants the task done, said in words) and **corrections** (rules written after
a failure). Nothing in the policy or in the planning model changes.

Every rule has a **scope**, which decides where it acts. The scope decides, never a similarity search:

| the user says | scope | where it acts |
|---|---|---|
| "from now on, please handle the blue block first", "only the red block today" | `plan` | a paragraph in the plan request |
| "count the red block as picked up only when the hand is 22 cm up" | `skill:<skill>` | after that skill's completion rule, in every question to the planning model |
| "double-check before you move on", "be quick" | `planner` | a preset of the planner's settings: two "done" answers in a row, or a question only at every second policy call |

A plan rule reaches the planning model like this:

```
User profile (apply unless the instruction says otherwise):
- Handle the blue block first.
```

![The preference notebook: written from the user's words, read by the planner](detailed/notebook_overview.svg)

**Five rules keep it a profile and not a growing log:**

- **Few rules per scope.** At most three active rules per scope reach the planner, ranked by importance, then by how recent
  they are.
- **A newer rule replaces an older one.** A message that contradicts a rule replaces it; the old rule stays in the file with the
  time it ended. A repetition confirms a rule. A withdrawal ("forget what I said about the order") retires it.
- **Votes.** A rule's importance starts at 2. After an episode, a correction that was applied gains a point on success (up to
  5) and loses one on failure. At 0 it is retired. The robot's own failures never vote out a preference the user stated.
- **Validity.** A rule holds until it is withdrawn, for the day or for the session, as its words say ("today", "this time").
  The notebook keeps its own clock and checks validity at every episode start. A rule for the day that replaces a lasting one
  only hides it until the day is over.
- **The instruction comes first.** If the instruction names any of the task's objects, it decides which objects are handled
  and in which order. A plan rule that names objects is then held back for that episode (it is logged, not applied).

**Who writes it.** The planning model does, under JSON schemas built from the task's skills, so its answer can only be a rule
the planner understands. It answers three questions: a user message becomes at most three rules; a failed episode's decision
log becomes at most one correction; and a proposed rule is merged with the rules of its scope (add, update, delete or no
change). The merge follows Mem0, the votes follow ExpeL, and the way old rules are kept follows Zep, three memories for
language-model agents (the detailed page lists the papers). A gold writer answers the same questions from annotations, for
tests.

**What it cannot do:**

- **It acts only where the planner can.** It changes the plan, the completion rules and the planner's care. It cannot change
  how the arm moves or where a block is put down.
- **It learns from words and failures only.** It does not infer a wish from what the user does.
- **It cannot make the policy good at what the policy does badly.** Following a wish for an order the policy was not trained
  on cost success (see "Our findings").
- **Corrections written from failures are weak.** The planning model finds the skill where an episode went wrong but restates
  the rule already in force instead of tightening it. The corrections in our tests came from annotations.
- **Votes see only outcomes.** They retired a correction that made every episode fail. But a rule the planner's model simply
  ignores causes no failure and is kept: votes cannot tell a rule that does nothing from one that helps.
- **The planning model can misread a message.** Of four unusual ways to withdraw a wish, two were understood and two gave no
  rule ("Never mind the order of the blocks, any order is fine." and the bare "Forget it.").
- **One user, one task so far.** The ten-day test had a scripted user and the reference task.

How it works in the code, and how the notebook is written: [detailed/README.md](detailed/README.md).

## Prerequisites

- The VLM writer (`--writer vlm`): the notebook is read by its planner (the plan request, the completion rules, the
  settings). The task file's skills define the scopes.
- For the model writer, the planning model's server (any OpenAI-compatible chat API). The gold writer needs none.
- One JSON file per user. The writer creates it.

## How to use it in the framework

`--task task.yaml` names the task file. For the reference task it can be left out.

Run the proxy with the VLM writer and the notebook. It re-reads the file at the start of every episode, so a writer may
change it in between. `--profile-off` runs the same command line without the notebook:

```
python -m vla_memory.proxy --writer vlm --method subgoal --with notebook --profile notebook.json --task task.yaml \
    --listen-port <port> --upstream-port <policy port>
```

The proxy only reads the notebook. Your application writes it between episodes:

```python
from vla_memory import OpenAICompatibleVLM, TaskSpec
from vla_memory.methods.notebook import PreferenceWriter, ProfileStore

writer = PreferenceWriter(OpenAICompatibleVLM(), TaskSpec.load_or_default("task.yaml"), ProfileStore("notebook.json"))
writer.from_message("From now on, please handle the blue block first.")
writer.from_failure(planner_records, "failed", "You moved on too early.")   # a correction after a failed episode
writer.feedback(applied_ids, success)                                       # the votes after an episode
```

`vla_memory.methods.notebook.session.run_session` runs a scripted session of messages, episodes and clock steps against any
episode runner (a class with a `run` method that plays episodes through the proxy), in four conditions: `profile` (the
notebook), `none`, `first` (each message in front of the next instruction only) and `every` (all messages so far in front of
every instruction). It writes one JSON line per message, episode, vote and correction and scores the writer against
annotations. Two example sessions are in `vla_memory/methods/notebook/sessions/`: `validity_and_precedence.yaml` and
`harmful_correction.yaml`.

## Our findings

The VLM writer in its settled configuration drove the sub-goal policy in simulation (the planner loop). The planning
model wrote the notebook.

**Ten days with one user.** 80 episodes, 8 a day, nine messages: double-check, later "be quick"; blue first, later withdrawn;
only red today; a harmful 35 cm correction; small talk; blue first for one session. The questions and the pass criteria were
fixed in writing before the first episode. Per episode, *plan followed* means that the plan is the one the user's rules ask
for. *Settings in force* means that the planner's settings are the ones the user asked for. *Task success* means that every
block the plan picks up was placed. *Done the user's way* means success with the plan the user wants and the plan's first
block lifted first.

| condition | plan followed (72 episodes from the first plan preference on) | settings in force (76 episodes) | task success of 80 | done the user's way, of 80 | what the user typed |
|---|---|---|---|---|---|
| the notebook | 72 | 76 | 54 | 54 | 9 messages, 91 words |
| no memory (each message in front of the next episode only) | 33 | 0 | 73 | 40 | 9 messages, 91 words |
| the user repeats every message before every episode | 72 | 0 | 58 | 58 | 280 messages, 3,376 words |

- **The notebook follows the user as the repeating user does, with 91 words instead of 3,376.** Both followed the plan
  in all 72 episodes. Without memory the robot followed in 33, a gap far too large to be chance. The notebook adds 8.8
  words to the plan request instead of 42, and uses 26 model calls per episode instead of 34 (its "be quick" setting
  halves the questions from day 6 on). Settings such as "be quick" cannot be given in an instruction at all, so the
  other two score 0 on them by construction.
- **Rules ended on time, and no rule acted where it must not.** The rule for the day ended at midnight, the rule for the
  session with the session, and the withdrawal held for three days. In the two episodes whose instruction named both blocks,
  the instruction's own order won. There was no false application in the 80 episodes.
- **A harmful rule retired itself.** The correction "count the red block as picked up only when the hand is at least 35 cm up"
  cannot be met (the hand rises to about 28 cm). It was applied in two episodes, both failed, and its importance went 2, 1, 0
  and it was retired. It was never applied again.
- **The robot that ignores the user completed the most.** It runs the red-first order the policy executes best. On the four
  blue-first days the notebook completed 4, 4, 2 and 2 of 8 episodes, the repeating user 6, 3, 3 and 2 (day 5 includes the two
  episodes lost to the harmful rule). That is the policy's limit, not the memory's.

**Single sessions** (20 development seeds per condition, 5 per phase in the retention test):

| test | with the notebook | without memory | restated in every instruction |
|---|---|---|---|
| "Please only do the red block today." | plan followed 20 of 20, success 20 of 20 | plan 0 of 20, success 18 of 20 | plan 20 of 20, success 20 of 20 |
| "Please double-check ...", then "... please be quick." | settings in force 20 of 20 and 20 of 20; success 18 and 17 of 20 | success 17 of 20 | |
| the planner's rule for "picked up" asked only for a hand height of 19 cm, too early; a correction asked for 22 cm | success 19 of 20; the planner moved on when the hand was at a median 0.277 m | success 12 of 20; 0.235 m | |
| an order rule, four unrelated messages, then a new order | plans followed 5 of 5 in all four phases, settings 5 of 5 once stated | | plans 5 of 5, settings 0 of 5 |

- **The notebook is followed.** The plan rules, the presets and the corrections reached the planner in every episode. "Be
  quick" cut the model calls per episode from 28.5 to 16.5 and the episode from 2.1 to 1.8 minutes at the same success. The
  correction removed the early switches, those made before the block was 9 cm up (8 of 35 without it, 1 of 40 with it, and
  that one in the first episode, before the correction existed).
- **Validity and precedence.** In the first version of the red-only test, the written rule was applied even to a control
  whose instruction named both blocks, and the blue block was dropped in 5 of 5 episodes. With validity and precedence the
  rerun followed the rule in 20 of 20 episodes, held it back in all 5 controls (both blocks planned and placed), and the next
  day the rule had ended (5 of 5 with both blocks).
- **Votes retire a harmful rule.** An impossible correction ("only when the hand is at least 35 cm above the table") failed its
  first two episodes and was retired after the second. 7 of the 8 episodes after it succeeded, with the pick-up switches back
  at 0.281 m. A rule that loosens the criterion ("count it as picked up as soon as the gripper has closed") was ignored by the
  planner's model, caused no failure and was kept by the votes.
- **The writer.** On 18 annotated messages with 16 rules, it found all 16 and wrote no other (precision and recall 16 of 16,
  no rule from small talk or questions), merged all 16 as annotated and gave all 15 validities as annotated (all "until
  withdrawn": these messages had no time words). On 20 messages with time cues over three days it was right on 16 of 16
  rules and validities, on 10 harder phrasings on 8 of 8. Distilled corrections found the skill where the episode went wrong
  in 14 of 17 failures, but restated the rule in force.
- **The policy limits blue first.** Asked for blue first, the policy lifted blue first in 18 of 20 episodes and completed both
  blocks in 8. A likely reason, not tested: its demonstrations always took red first.
- **Cost.** Reading the notebook takes 0.5 ms at an episode start and 0.2 ms per question to the planning model (measured with
  10 entries). The writer needs about 1.5 model calls per message (27 calls for 18 messages).
