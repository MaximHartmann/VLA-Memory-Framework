# The frame window: tried and not adopted

**Status: evaluated in two versions, both without effect; no code in the framework.** This page records what was tried and
what the measurements showed, so that the attempt is not repeated.

## The idea

Give the policy a memory of its own recent past as a compressed window of frames: the last X policy calls at a spacing of Y
seconds, each frame reduced to the summary the policy itself computes for it (the layer-12 features of its language
backbone, averaged over the text tokens and over each camera's image tokens: 6,144 numbers instead of two pictures). A small
module pools the X summaries into one vector and adds it to the action expert's conditioning signal. It is the frame-history
baseline that every memory comparison needs, and it keeps everything else identical to the recurrent memory (the same
features, data, entry point into the action expert, frozen backbone and training recipe), so window and recurrent state can
be compared cleanly. Related work had reported frame windows working for pi0.5 (RoboMME, arXiv 2603.04639, with an extra
regulariser; NativeMEM, 2607.06678), while SmoLSTM (2609.22854) argued that a recurrent state beats a wider observation window.

## What was built

Two versions, each trained for 10,000 steps from the memoryless policy on the demonstrations plus the correction data that
made the recurrent memory work, and each evaluated on the same 100 fresh seeds as every other method, with a blank-window run
(the window empty at every call) as the control of whether the window is used at all:

1. **Attention pooling with a fixed learned query** over the X frame summaries, each with an embedding of its age; four
   settings: 8 frames x 1 s, 8 x 2 s, 16 x 1 s, 16 x 2 s (8 to 32 s of history).
2. **Query computed from the current frame, plus a phase head** on the pooled window vector (the four task phases as
   labels, loss weight 0.1), for the 16 x 2 s setting. The first version's attention had stayed uniform; this version was
   built to make it select, and to give the selection a training signal from the first step.

## Results

Full tasks out of 100 fresh seeds (pick up and put back red, then blue; strict sequence):

| policy | full tasks | with the window blank |
|---|---|---|
| memoryless policy (reference) | 11 | |
| corrections without any memory (reference) | 0 | |
| recurrent memory, GRU (reference, same data and recipe) | 34 | 0 |
| recurrent memory with a phase head (reference) | 39 | 3 |
| window v1, 8 x 1 s / 8 x 2 s / 16 x 1 s / 16 x 2 s | 12 / 8 / 10 / 11 | 5 / 0 / 7 / 2 |
| window v2, 16 x 2 s, query from the current frame + phase head | 6 | 6 |

- Version 1 beat the corrections-without-memory policy (all four settings, Holm-corrected p < 0.01) and was used (blanking
  the window lowered the sequence score in all four, p < 5e-5), but stayed at the memoryless policy's level and far below the
  recurrent memory (16 x 2 s vs GRU: 11 vs 34, p = 6.6e-5). Window length and spacing made no difference.
- Its attention was uniform: entropy 98-99.6 % of the maximum, flat over slot age. At the moment the policy has to decide
  that red is done, the frames showing red held aloft were in the window in 93-100 % of the calls and received exactly their
  uniform share of attention, in the episodes that then went for blue and in those that went back to red alike. The module
  acted as a plain average of the last X frames.
- Version 2 moved the attention away from uniform (entropy 0.83 of the maximum) but still gave the decisive frames only
  1.6 times their uniform share, regardless of the outcome; its phase head learned the task phase from the pooled window
  (accuracy 0.99 in training). The policy completed 6 tasks with the window and 6 without it (p = 1.0): the action expert did
  not use the signal.
- Both versions grasped at the wrong place more often than the recurrent memory (the gripper closing with no cube within
  6 cm in 21-40 episodes against 6-11), and both lifted red more often than the memoryless policy (37-61 against 31) without
  finishing more tasks. Latency was unchanged (about 175 ms per call).

## Reading

- On this task a frame window is not a substitute for a recurrent state, even when the right frames are in it, even when
  they are selected, and even when the window's output encodes the task phase. The information was present; the controller
  did not act on it.
- The difference to the recurrent memory is therefore not where the information is but how it reaches the actions: the
  GRU's state is updated once per call and trained through the whole episode, and its read-out stays consistent from call to
  call; the window's read-out is recomputed at every call from whichever frames fall into its slots. This is a hypothesis,
  not a measured cause; a direct test would train the window's read-out with the recurrent memory's objective.
- Memory written from outside (the history line, the keyframe picture) did not need any of this: a writer chooses one
  moment and the policy reads it (78-82 full tasks). Choosing the moment is what the window never learned.
