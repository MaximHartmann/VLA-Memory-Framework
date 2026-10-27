# The frame window in detail

The window's code is not in the framework, because it is a negative result. This page describes how both versions worked, so
that the mechanism can be understood and, if someone wants, rebuilt on top of the built-in memory's code. What we measured is
on the short page, [docs/methods/frame_window](../README.md).

![The frame window next to the recurrent state](window_how.svg)

## 1. The idea

The task needs a memory (the [built-in memory](../../builtin_memory/README.md) page explains why): after red is put back the
scene looks like the start, and a held cube looks the same going up or down. The most direct way to give a policy a past is to
show it its last frames. A person would solve the task that way, and it is the baseline that every memory claim needs.

To keep the comparison clean, the window differs from the recurrent memory in one place only. It reads the same summaries of
the policy's own backbone, learns from the same data (the demonstrations and the intervention episodes), writes into the same
place of the action expert, with the same frozen backbone and the same training recipe. If the two policies differ, the
difference is the window against the recurrent state.

## 2. What the window holds

The window has X **slots** and a spacing of Y seconds. The simulator and the demonstrations run at 15 control steps a second,
so Y seconds are 15 × Y control steps. At a policy call, slot k (k from 1 to X) holds **the most recent earlier call at or
before k × Y seconds ago**:

- if k × Y seconds ago lies before the start of the episode, the slot is empty (masked);
- the current call is never in its own window: the backbone sees the current frame anyway;
- in training the same rule is applied to the sample's call history (the walk over earlier calls that the built-in memory's
  `sequences.py` makes, with the same random draws), so the training windows are those of a policy that is called every 6 to
  10 steps. The history cap was raised from 96 to 128 calls so that 32 seconds fit.

Four settings were trained:

| X (slots) | Y (seconds apart) | the history covered |
|---|---|---|
| 8 | 1 | 8 s |
| 8 | 2 | 16 s |
| 16 | 1 | 16 s |
| 16 | 2 | 32 s |

In the demonstrations the events fall at about 5.6 s (red picked up), 10 s (red placed) and 16.8 s (blue picked up); the
robot's own runs take longer, up to 40 s with interventions. 8 seconds reach from the moment before going for blue back to the
red placement; 32 seconds cover almost a whole episode.

## 3. What the module does with the slots (version 1)

1. **Each slot's summary:** the same 6,144 numbers as the recurrent memory reads (the backbone's block-12 means over the
   prompt, the outside picture and the wrist picture), z-scored with the same fixed statistics (each number minus its mean,
   divided by its spread).
2. **Embedding:** a linear layer from 6,144 to 256 numbers, a LayerNorm, and a learned **age embedding** (256 numbers per slot,
   so that the module knows how old a slot is).
3. **Attention pooling.** A single learned **query** (256 numbers, initialised small) asks every slot a question through 4
   heads of 64 numbers each. Keys and values are linear layers of the slots' embeddings; the score of a slot is the query
   times its key (divided by 8); a softmax over the slots turns the scores into weights that add up to 1. Empty slots get
   weight 0. The four heads' weighted sums are put side by side: 256 numbers. The query does not depend on the current call.
4. **Read-out:** a linear layer from 256 to 1,024 numbers, starting at exactly zero, added to the action expert's conditioning
   as the recurrent state's read-out is. With no past frame at all the module adds exactly 0.

The module has about 2.0 million numbers to learn (1.57 million of them in the linear layer of step 2).

## 4. Training and run time

- **Training:** as the recurrent memory with the intervention data: 10,000 steps from the memoryless policy at batch 16, the
  backbone frozen, the same data (the demonstrations and round 1 of the interventions, uniformly), 200 warm-up steps, a cosine
  decay of the learning rate from 1e-5 to 1e-6, ten times that for the module, a checkpoint every 1,000 steps and the same
  choice of checkpoint on development seeds.
- **Run time:** the policy server keeps, for the episode, every call's summary with its number, builds the window from them at
  every call (the same rule as in training) and resets at the episode markers. The summaries come from the same forward pass
  as the actions.
- **The blank window**, the window's version of the reset test: the window is emptied at every call, and the module then adds
  exactly 0. If the policy uses its window, it must get worse.

## 5. Version 2: a query from the current frame, and a phase head

In version 1 the query is one fixed vector that does not depend on the current call, and the read-out starts at zero, so early
in training no gradient (no learning signal) reaches the query. Version 2 changed two things, together, for the 16 × 2 s
setting:

1. **The query is computed from the current call.** The current call's own summary goes through the module's input layer,
   the LayerNorm and a new linear layer (256 to 256) and gives the query of the four heads. The module now asks which past
   frames matter for what the policy sees now.
2. **A phase head on the pooled window vector.** A linear layer from 256 to 4 numbers predicts the current call's phase (pick
   up red, place red, pick up blue, place blue); its cross-entropy, weighted 0.1, is added to the loss, as in the recurrent
   memory with the phase head. It gives the selection a training signal from the first step.

Everything else was unchanged.

## 6. Rebuilding a window on the built-in memory's code

The built-in memory's integration ([integrations/openpi](../../../../integrations/openpi/README.md)) is the base:

| file | what changes for a window |
|---|---|
| `pooling.py` | nothing: it already gives every call's summary from the forward pass that computes the actions |
| `sequences.py` | nothing in the format; the window picks its slots from the call history that `HistoryDataset` builds (raise `max_calls` if the window reaches further back than 96 calls) |
| `model.py` | the window module replaces the GRU step (`memory_scan`); its read-out enters the conditioning in `embed_suffix`, at the same place |
| `policy.py` | the server keeps every call's summary of the episode with its call number, instead of one state, and builds the window at every call |
| the proxy | nothing: it forwards the episode markers, and `--writer off` leaves the instruction alone |

Pick the slots by the same rule in training and at run time. A rule that differs between the two would show the policy
windows it was not trained on.

To test a rebuilt window, do what we did: run it with the window blank at every call (it must get worse), look at the
attention at the moment of the decision (does it select the decisive frames?), and compare it against the recurrent memory
with the same data and steps.

**Not tried yet,** and the places to start if you want a window to work: a regulariser that predicts past tokens, as RoboMME
uses; frames chosen at events instead of at regular spacing; max pooling over the window; differences between consecutive
summaries instead of the summaries themselves; training the window's read-out with the recurrent memory's objective, or
feeding the recurrent state through the window's read-out.

## 7. Limits of what was tried

- **One task, one policy, one data mix, one training per setting.**
- **Only 10,000 steps.** The recurrent memory improved with more training cycles; the window was not given any. At equal steps
  the comparison is fair, but whether more cycles would help the window is not known.
- **Version 2 changed two things at once.** A follow-up without the phase head would tell them apart.
- **The window is uniformly spaced and fully learned.** Frames chosen by events, or a window with a regulariser as in
  RoboMME, are different designs. Frame windows that worked in other papers were on other benchmarks with other designs.

## Where the idea comes from

- **RoboMME** (Dai et al., 2026, arXiv 2603.04639) reports a frame window working on π₀.₅: 2 to 8 past frames, each compressed
  to one token and fed through the action expert's modulation, with a regulariser that predicts past tokens against copycat
  behaviour. Our window is fed the same way but has no such regulariser.
- **NativeMEM** (Wang et al., 2026, arXiv 2607.06678) compresses every past frame of π₀.₅ to one token with the policy's own
  vision encoder and appends it to the input: another window that worked, with a different entry point.
- **MEM** (Torne et al., 2026, arXiv 2603.03596) keeps a video memory of frames about one second apart, close to our spacing,
  plus a language memory. **SimpleMemVLA** (Yin et al., 2026, arXiv 2609.05533) shows how long windows can be: the last 15 to
  126 seconds.
- **UniMem** (arXiv 2608.22869) and **KEMO** (arXiv 2606.23589) found that keyframes chosen at events beat uniformly spaced
  frames. Our window is uniform; the [keyframe policy](../../keyframe/README.md) picks its moment by an event.
- **SmoLSTM** (arXiv 2609.22854) argues that a recurrent state beats a wider observation window, because the window's length
  becomes a setting to tune and its cost grows with it. We tried exactly this comparison.
- **The copycat problem** (Wen et al., 2020, arXiv 2010.14876): a policy that sees its past can learn to copy its own earlier
  actions instead of looking at the scene.
- **Present but Not Remembered** (arXiv 2607.03372): policies trained on demonstrations in which the present already tells
  what to do ignore their past. That is why the window is trained on the intervention data, as the recurrent memory is.
