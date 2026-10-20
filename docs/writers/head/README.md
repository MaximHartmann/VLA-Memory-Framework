# The head writer: the policy's own event head

## The idea

Every method needs a writer that says when a skill is complete. Here the policy is its own writer. At every policy call
the policy server reads the policy's internal representation of that call, the residual stream after block 12 (counted
from 0) of its 18 Gemma blocks averaged over the prompt's text tokens, and a logistic regression on it (2,048 weights, the
event head) scores whether the current skill is complete. The verdict comes back with the actions. When the score is above
the threshold in two consecutive calls, the writer moves its pointer; since the verdict arrives after the call's prompt was
sent, the method shows it from the next call on. With the history method, which appends a line per completed skill:

```
call t      "... History: picked up the red block."  ->  policy  ->  actions, and the head: score 1.3 >= 0.57, done
call t + 1  the same skill, done again               ->  the writer completes it: "placed the red block"
call t + 2  "... History: picked up the red block, placed the red block."
```

The writer's plan is the task order (of the episode's objects, when the environment names them as `memory/objects`).
The policy stays frozen and nothing outside it looks at the robot or the cameras. The head is fitted afterwards on the
policy's own rollouts, labelled by another writer, here the rules writer. The idea follows the event head of UniMem
(Osterberg et al., 2026, arXiv 2608.22869), here as a probe on a frozen policy instead of a head trained with it.

## Prerequisites

- A policy served with openpi (pi0.5: three image slots of 256 tokens, then the prompt), here the history policy
  ([the history method](../../methods/history/README.md)).
- Recorded rollouts of that policy, not demonstrations: the camera frames of every control step and either the signals the
  rules writer reads (finger width, squeeze, fingertip position) or a reference count of completed skills per step.
- The openpi environment and one GPU for the features of the fit and for serving; the fit itself runs on the CPU with numpy.

## How to use it in the framework

1. **Record rollouts** of the history policy with another writer, for example `--writer rules`, and keep the frames. One
   `.npz` per episode: `exterior`, `wrist`, `instruction`, and `rows` (the rules writer labels them) or `reference`
   (`training/fit_event_head.py` describes the arrays).
2. **Fit the head:**

   ```
   python training/fit_event_head.py --episodes 'rollouts/*.npz' --config <policy config> \
       --checkpoint <checkpoint dir> --center-crop 0.95 --features rows.npz --out <head dir> [--jax-cache DIR]
   ```

   `--config` is the policy's openpi training config and `--center-crop` the centre crop of the exterior picture that the
   reference policy runs at inference (0.95), the same at the fit and at the server. The rows are every 4th frame: negatives
   show the skill in progress with the correct history, positives the first 24 frames after a skill was completed with the
   history that still lacks its line. The frozen policy computes their features; the fit is a logistic regression with
   balanced class weights, its L2 penalty chosen by a 5-fold cross-validation over episodes, and the threshold that 1 % of
   the out-of-fold negatives reach. The head directory holds `head.npz` (weights, bias, the features' mean and standard
   deviation, threshold, N, layer, pooling), `head.json` (how it was fitted) and `head.sha256` (checked whenever the head is
   loaded). No fitted head and no policy checkpoint is distributed with the repository; these two steps produce them.
3. **Serve the policy with its head** (openpi environment); every reply then carries
   `event_head: {score, done, threshold, ms}`:

   ```
   python -m vla_memory.writers.head.server --port <server port> --config <policy config> \
       --checkpoint <checkpoint dir> --center-crop 0.95 --head-dir <head dir> [--head-batch 8] [--jax-cache DIR]
   ```

4. **Run the proxy with the head as the writer:**

   ```
   python -m vla_memory.proxy --writer head --method history --listen-port <port> --upstream-port <server port> \
       [--consecutive 2]
   ```

   Every call's log record holds, under `event_head`, the verdict, the counter of consecutive "done" verdicts and the skill
   completed, if any (`wrote`). The writer reads no control keys.

The server scores a call as 8 copies of itself, the batch shape of the fit's rows. With the same compiled kernels as the
fit (a copy of its JAX compilation cache, `--jax-cache`) it computes the training features bit for bit; without them, or
with `--head-batch 1`, its scores differ from the training numerics by about 0.08 logits on average.

## Our findings

Reference task, the history policy (checkpoint 28000) and a head fitted on 129 of its own rollouts labelled by the rules
writer (11,791 rows; L2 1e-3; threshold 0.571, which 1 % of the out-of-fold negatives reach, recall 0.965). 100 fresh seeds
of 500 control steps, the same policy, only the writer differs:

| writer of the history | full tasks of 100 | policy call, round trip | 500-step episode |
|---|---|---|---|
| the policy's event head | 78 | 714 ms | 86 s |
| simulator's ground truth | 82 | 169 ms | 51 s |
| rules writer | 79 | 172 ms | 51 s |
| VLM monitor | 77 | 1,630 ms | 145 s |

- No difference is significant (paired sign tests against the event head: p = 0.54, 1.00 and 1.00). 20 of the 22 failed
  episodes have no writer error, mostly failed grasps; 2 failed after a missed line.
- Against the simulator's tracker the head wrote 250 of 253 lines: 236 one policy call after the tracker, 7 in the same
  call, 6 later (up to 11 calls), 1 two calls early while the cube was rising (the episode succeeded), 3 missed and none
  false. It said "done" on 19 of the 3,744 calls on which the skill was not complete (0.5 %); two in a row turned one of
  them into a line.
- The misses: a block dropped one call after the lift (one call above the threshold is not enough), and a hand that slid
  along the table after the release instead of rising.
- The teacher matters. Heads fitted on the demonstrations separate the classes on held-out demonstrations (AUROC 0.999 to
  1.000) but not on the policy's own rollouts (0.85 to 0.93): they fire at the grasp, where a cube squeezed on the table
  looks from the wrist camera like a lifted one, since in the scripted demonstrations the lift follows the close within a
  few frames. Fitted on the policy's own rollouts, the head wrote no early and no false line on 49 fresh rollouts: 3 of
  119 lines missed, every written line one call after the tracker (frame-level AUROC 0.991).
- The price is time. The exact head pass adds about 545 ms per call (716 against 170 ms for the plain server), more than
  an 8-step chunk at 15 Hz lasts (533 ms). With `--head-batch 1` it adds about 100 ms (280 ms per call), with scores a
  mean 0.08 logits off the training numerics (at most 0.44) and no verdict changed in 585 rows.
- One policy, one instruction and one camera setup, in simulation; the last skill has no line, so the head never
  reports that the task is finished.
