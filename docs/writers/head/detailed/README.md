# The head writer in detail

The short page, [docs/writers/head](../README.md), says what the head writer does, how to use it and what we found. This
page explains where the numbers come from and what the head does with them, why it is built this way, how it is trained,
and how the code works.

![One policy call with the head writer](head_call.svg)

## 1. The question behind it

A memory method needs someone to say when a skill is done. All the other writers watch the robot from the outside:

- the **oracle** asks the simulator, which knows the true state of the world;
- the **rules writer** reads the gripper's own readings and applies hand-written thresholds;
- the **VLM writer** shows the camera pictures to a second, large model and asks it.

But the policy itself already looks at the same pictures and the same prompt at every call. To choose its next movement, it
must in some sense know whether the cube is still on the table, held, or already lifted. So the question is: **is the
progress of the task already written somewhere inside the policy, and can we read it out?** If so, the policy can keep its
own memory: no tracker, no extra sensors, no second model.

**What others found.** Several groups have read things out of the inside of VLA policies:

- **UniMem** (Osterberg, Wang and Schwager, 2026, arXiv 2608.22869) is closest to us. It trains an "event head", a small
  multilayer perceptron on π0.5's last-layer representation, *together with* the policy. When the head detects an event, the
  event is appended to the prompt as text ("History: grabbed spoon, ..."). It reported 93.4 % against 68.2 % for a variant
  with frames at fixed intervals in simulation, and 80.0 % against 43.5 % on hardware.
- **A progress probe on π0.5** (Bhardwaj et al., 2026, arXiv 2608.13474): a *linear* read-out of the averaged activations of
  a fine-tuned π0.5 predicts how much of the task is left, with R² of about 0.95 on familiar tasks.
- **Failure detectors on VLA internals:** SAFE (Gu et al., 2025, arXiv 2506.09937) reads a VLA's internal features with an
  LSTM or a two-layer perceptron; Hide-and-Seek (Park et al., 2026, arXiv 2605.30834) reads the action embeddings with an
  LSTM.
- **Linear probes** as a research tool go back to Alain and Bengio (2016, arXiv 1610.01644): if a simple linear classifier
  can read a property from a layer of a network, the network represents that property there.

**Our choice: a probe on the frozen policy.** UniMem retrains the policy with its head. We tried the cheapest thing first:
leave the policy exactly as it is, record what it computes, and fit a small read-out afterwards. That costs no retraining,
cannot harm the policy's movements, and can be tried and checked in hours.

**What the head looks for.** The head does not have to judge the scene alone; it notices when the scene and the method's
input no longer agree. With the [history method](../../../methods/history/README.md): "the cube is lifted, but the history
does not say 'picked up the red block' yet". With the [sub-goal method](../../../methods/subgoal/README.md): the prompt
still says "pick up the red block" while the cube is already up. With the
[keyframe method](../../../methods/keyframe/README.md): the memory picture still shows the previous key moment. That
mismatch is exactly the moment to complete the skill.

## 2. Where the numbers come from and what the head does

![Where the numbers come from and what the head does with them](head_inside.svg)

π0.5 has two parts. A language-model **backbone** (18 blocks) reads the input: 256 tokens for each of its three picture
slots (the outside camera, the wrist camera, and a third slot that only the keyframe policy uses, for its memory picture),
then the tokens of the prompt. An **action expert** reads the backbone's inner results at every block and produces the next
movements.

**In the policy server**, a state tap reads the backbone after **block 12** (counting from 0) while the policy computes its
actions. Inside the backbone everything is numbers: each piece of input, a token, is a list of 2,048 numbers. A picture is
cut into 256 small patches, each one token; the prompt is cut into words and parts of words, each one token too. Every block
mixes information between the tokens, so after block 12 each token has taken in something of all the others. The tap
averages the vectors of the prompt's tokens, number by number, and leaves out the padding that fills the prompt to its fixed
length. The result is one list of **2,048 numbers**, sent with the actions in float16 (`reply["policy_state"]`). Because
the prompt's tokens have taken in the pictures inside the backbone, their average sums up both. The actions come from the
same pass and stay exactly what they are without the tap.

**In the proxy**, the event head turns the 2,048 numbers into a verdict:

1. **Standardise:** subtract each number's mean over the training rows and divide by its spread, so that all 2,048 numbers
   are on the same scale.
2. **Score:** `score = w · x + b`, a weighted sum of the 2,048 numbers plus one constant. The higher, the surer the head is
   that the skill is done; 0 means 50/50.
3. **Decide:** "done" if the score is at least the **threshold** (section 3 says where it comes from).
4. **Two in a row:** after two "done" verdicts in a row the skill is complete.

Every server reads by the same rule: the backbone two thirds of the way up, averaged over the prompt's words, named
`block12:prompt` for π0.5. The servers of other models use the [automatic tap](../../../taps/README.md), which finds all
three by itself: the backbone (the stack of blocks as wide as the model's word table), the block two thirds
up, and the prompt's words (the tokens that come from the word table; the padding is left out). On our π0.5 its numbers are
the same as the hand-written tap's.

## 3. Why it is built this way

### Trained on the policy's own runs, not on demonstrations

The first heads were trained on the **demonstrations** the policy itself was trained on. We tried 6 layers, 6 ways of
pooling and two kinds of head. All of them told "done" from "not done" almost perfectly on held-out demonstrations, and much
worse on the policy's own runs. They said "done" at the grasp: while the cube was still on the table, squeezed by the
fingers, or after a failed close. In the scripted demonstrations the lift always follows the close within a few frames, so
"a cube between the fingers" almost always means "about to be lifted". In the policy's own runs slow and failed grasps are
common, and from the wrist camera a cube squeezed on the table looks almost the same as a lifted one.

No head, however large, can learn from demonstrations what a slow or failed grasp looks like, because the demonstrations
never show one. This is the mismatch DAgger (Ross, Gordon and Bagnell, 2011) addresses for policies: a model trained only on
an expert's data meets states the expert never visited. So the head is trained on the **policy's own runs**. They contain
the slow and failed grasps, and the rules writer labels them for free: it knows from the gripper readings when a block was
lifted and when it was set down. Trained this way, the false alarms at the grasp disappeared.

### A logistic regression, not a multilayer perceptron

The second kind of head was a **multilayer perceptron** (one hidden layer of 256 units). On the policy's own runs it was no
better than the logistic regression: the problem had been the training data, not the size of the head. We kept the
logistic regression:

- **A cleaner answer to the research question.** A linear read-out shows that the policy itself has organised the
  information so that a straight line separates "done" from "not done". A perceptron could compute part of the answer
  itself, and we would not know whether the policy or the probe did the work.
- **Less risk of overfitting.** The logistic regression learns 2,049 numbers; the perceptron about 525,000
  (2,048 × 256 + 256 + 256 + 1). Rows from one run are very similar, so the real amount of data is small.
- **A fit with one exact answer.** The loss has a single minimum, found with Newton's method in a few iterations, without a
  random start, a learning rate or a number of epochs to choose. The same rows always give the same head.
- **A meaningful score and almost no cost.** The score is a logit, so a threshold has a clear meaning; at run time the head
  is one dot product of 2,048 numbers.

### An event, not the progress

The progress probe on π0.5 (section 1) reads out how much of the task is left. We read out an event instead, because the
writer needs a yes-or-no decision at one moment. A progress estimate would still need a threshold to turn it into that
decision, and "how much is left" depends on how the episode goes, while "the skill is done" is one clear moment that
matches the policy's training labels.

### Block 12 and the text tokens, the threshold, two verdicts in a row

**Which block, and which average.** The head can read the policy after any of its 18 blocks. The blocks are numbered from
0, as in the code: block 0 is the first, block 17 the last. It can also sum up the block's output in different ways: the
vector of the prompt's **last token**, the average over the **picture tokens**, or the average over the prompt's **text
tokens**. Seven choices were compared on 29 runs of the history policy, with 5-fold cross-validation: each run is scored by
a head trained on the other runs. The runs held 402 "done" rows and 2,223 "not done" rows.

**The AUROC** measures how well a head tells the two apart. Take one "done" row and one "not done" row at random: a good
head gives the "done" row the higher score. The AUROC is how often that happens, over all such pairs. At 1.0 every "done"
row scores above every "not done" row, so one threshold separates them without a mistake; at 0.5 the head is guessing,
like a coin. The 0.991 in the last row of the table means: in 99.1 % of the pairs the "done" row scored higher. Unlike the
share of right answers, the AUROC needs no threshold, and the many "not done" rows cannot fool it: a head that always says
"not done" would be right 85 % of the time, but its AUROC is 0.5. (The name stands for "area under the ROC curve", a
plot from statistics that gives the same number.)

In the table, "caught" counts the "done" rows above the threshold that only 1 % of the "not done" rows reach. The last four
columns play each head through the 29 runs as a writer, against the rules writer's 67 lines.

| block | summary | AUROC | "done" rows caught | lines one call late | two or more calls late | missed | false |
|---|---|---|---|---|---|---|---|
| 6 | last token | 0.987 | 327 of 402 | 53 | 8 | 6 | 1 |
| 9 | last token | 0.989 | 339 of 402 | 56 | 7 | 4 | 1 |
| 12 | last token | 0.988 | 308 of 402 | 47 | 13 | 7 | 1 |
| 15 | last token | 0.987 | 302 of 402 | 43 | 13 | 11 | 0 |
| 17 | last token | 0.987 | 313 of 402 | 50 | 7 | 10 | 1 |
| 12 | picture tokens | 0.973 | 233 of 402 | 34 | 18 | 15 | 2 |
| **12** | **text tokens** | **0.991** | **351 of 402** | **57** | **5** | **4** | **0** |

(With block 12 and the text tokens, one more line came in the same call as the rules writer's.)

- **The block made little difference.** With the last token, every block from 6 to 17 reached an AUROC of 0.987 to 0.989.
- **The summary made more.** At block 12 the average over the text tokens caught the most "done" rows with the fewest false
  alarms (8 of the 2,223 "not done" rows, against 26 with the last token and 41 with the picture tokens), and it was the
  only setting that wrote no early and no false line.
- **The demonstrations did not help.** Trained on the runs plus the 200 demonstrations, every setting scored a little lower
  (AUROC 0.963 to 0.989).
- **Not tried:** the text-token average at the other blocks.
- **Chosen after looking.** Block 12 with the text-token average was picked from these results, on only 29 runs. So the
  head was then fixed and tested once on 49 fresh runs: it wrote no early and no false line there.

**The threshold.** A false completion is worse than a late one: once written, it stays for the rest of the episode and
misleads the policy. So the threshold is the score that only **1 % of the "not done" moments** reach in the
cross-validation (0.571 for the history policy's head).

**Two "done" verdicts in a row.** Single false alarms are rare, but they happen. Asking for two in a row removes almost all
of them, at the price of one call of delay, and of missing events that last only one call (a block dropped right after the
lift).

## 4. How the head is trained

![How the training examples are recorded](head_rows.svg)

`training/fit_event_head.py` fits the head on the proxy's recordings (`--record-head`), for any policy whose server hands
out its state.

1. **The training examples.** Each one is the policy's state for one input, with a label:
   - **label 0** ("not done"): every policy call, with the input the policy got: the history with a line for each finished
     skill, the current skill as the prompt, or the picture of the last key moment (none before the first).
   - **label 1** ("done, not shown yet"): at the 3 calls after the teacher counted a skill as done, the proxy asks the
     policy once more, with the same pictures but the input of the call before the switch: the history still lacks the new
     line, the prompt still names the finished skill, or the memory picture is still the previous one. The answer to this
     extra question goes nowhere; the robot acts on the real call only.

   The same pictures appear twice: with the new input as label 0, with the old one as label 1. So the head cannot decide
   from the pictures alone; it has to compare them with what the method shows. Once the input shows every skill the head
   can complete, the calls are left out: the last skill writes no line.
2. **The fit** (`vla_memory/writers/head/fit.py`, on the CPU):
   - **balanced class weights**: there are many more label-0 rows than label-1 rows, so each class gets the same total
     weight; otherwise the head could score well by always saying "not done";
   - an **L2 penalty** keeps the weights small. Its strength is chosen from 1e-4, 1e-3, 1e-2 and 1e-1 by **5-fold
     cross-validation over runs**: each fifth of the runs is predicted by a head trained on the other four, and the strength
     with the best average AUROC wins. Whole runs stay together, because frames of the same run are so similar that
     splitting them would make the test far too easy;
   - the **threshold** comes from these out-of-fold scores, the scores of heads that did not see the run;
   - the final head is fitted on all rows; when the server hands out several summaries, this is done for each one and the
     best is kept.
3. **The head files** (`vla_memory/writers/head/file.py`): `head.npz` holds the weights, the bias, the mean and spread of
   the numbers, the threshold, the number of verdicts in a row and the name of the summary the head reads (for example
   `block12:prompt`); `head.json` notes how it was fitted; `head.sha256` holds checksums, checked whenever the head is
   loaded.

For openpi's π0.5 there is a second way in: the pictures of recorded episodes (`--episodes`; the rows:
`vla_memory/writers/head/rows.py`). Every 4th frame becomes a label-0 row, and those among the first 24 frames after a
finished skill also a label-1 row, built after the fact; the policy, loaded as `openpi_server.py` serves it, computes each row's
numbers on a GPU. Our first heads were fitted that way; the script's description lists the arrays an episode file needs.

## 5. The code at run time

**The policy server** hands out the state with every reply (`vla_memory/taps/reply.py`):

```
reply["policy_state"] = {"names": ["block12:prompt"], "values": (1, 2048) float16}
```

- openpi's π0.5 (`integrations/openpi/openpi_server.py`): `openpi_tap.py` taps block 12. For a JAX policy,
  `sample_actions` becomes a copy of openpi's whose pass over the pictures and the prompt also returns the block's average
  (the built-in memory's `pooling.py`); for a PyTorch policy a forward hook on block 12 does the same.
- any LeRobot policy (`integrations/lerobot/lerobot_server.py`): the automatic tap, with the same rule.
- any other PyTorch policy: its own server with the automatic tap added:

  ```python
  tap = AutoTap(model)                   # once, after loading (vla_memory/taps/auto.py)
  tap.start_call()                       # before each call
  values = tap.read()                    # after each call
  reply["policy_state"] = state_entry(tap.names, values)
  ```

**The writer** (`vla_memory/writers/head/writer.py`, `HeadWriter`, in the proxy):

- `start_episode`: the plan is the task's order of skills (of the episode's objects, if the environment names them), the
  position starts at the first skill, the "done" counter at 0.
- `after_call`: it finds the summary its head was fitted on in the reply, by its name, scores it and reads the verdict. A
  "done" raises the counter, a "not done" sets it back to 0. When the counter reaches `--consecutive` (default: the head's,
  2), the current skill is complete and the counter starts again. Without that summary in the reply it stops with an error
  that says what the server hands out.
- `before_call`: a skill completed after the previous call reaches the method at this call. Because the head decides
  *after* a call, its decision reaches the policy one call later than with the writers that decide before the call.
- At most `max_lines` skills are completed this way (the method's own limit, `--max-lines` or `--keyframe-max-events`,
  else the task file's `history.max_lines`, 3 here): the last skill has no line.

**The recording** (`vla_memory/writers/head/record.py`, in the proxy with `--record-head DIR`): at every call it takes the
policy's state from the reply and the teacher's count of finished skills, and after a finished skill it asks the extra
questions (section 4). When an episode ends it writes one file: the summaries of every row, the rows (the step, the
finished skills the input showed, the label), the summaries' names, and the episode's instruction, plan, writer and method.

The head scores the numbers of the policy's own pass, exactly the kind of numbers the recording keeps, so it is fitted on
the very numbers it scores.

## 6. Does it depend on the policy? Yes, completely

- **The numbers are the policy's own.** Another model, another fine-tune, even another checkpoint of the same fine-tune
  has a different inner representation. Each of our heads belongs to one checkpoint: 28000 of the history policy, 20000 of
  the sub-goal policy, 29000 of the keyframe policy.
- **The training data are the policy's own runs.** Section 3 shows why: a head trained on someone else's behaviour (the
  scripted demonstrations) fails on the policy's.
- **The method is part of the input.** The examples show the policy what its method shows it. A head fitted for one method
  does not fit another.
- **The server must reach inside the model.** openpi's PyTorch port of π0.5 gives slightly different numbers than the JAX
  policy (about 3 % apart), so a head fitted on one is not exact on the other: fit the head on the policy that will run.

To use it with another VLA: make its server hand out `reply["policy_state"]` (the LeRobot server does it for LeRobot's
policies, the automatic tap for any PyTorch policy; a JAX model has to return the numbers itself), record runs with a
teacher and `--record-head`, fit, check the cross-validation's AUROC, and run the proxy with `--writer head --head-dir`.

## 7. Limitations

- **It copies its teacher.** Where the teacher is wrong, the head learns the mistake. Where the rules writer and the
  simulator disagree, for example a cube let go a little too high (the rules writer calls it a drop, the simulator a
  placement), the head has to side with one of them.
- **It does not check which block is held.** If the policy grasps the wrong block, the head still completes the skill.
- **Rare events:** drops right after a lift, unusual movements after a release, and events shorter than two calls are where
  it misses.
- **One call of delay**, because it decides after the call.
- **No "task finished":** the last skill has no line, as in the policy's training data.
- **Narrow testing:** three π0.5 policies of one task, one instruction, one camera setup, in simulation; other VLAs were
  checked for the mechanics only.
- **Speed is not a limit:** a call takes as long as without the head, and the head itself takes well under a millisecond.
