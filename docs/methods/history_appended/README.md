# The history-appended prompt

**Status: in development.** The policy for this method is being trained; the numbers below come from screening its
early checkpoints with a ground-truth event tracker.

The second method keeps the memory inside the policy's own input. The instruction stays the full sentence, exactly as
a memoryless policy would receive it, and a short text is appended that says what has already been done:

```
Pick up the red block, place it back, and then grasp the blue block. History: picked up the red block, placed the red block.
```

The policy is fine-tuned on frames labelled this way, so at run time it can tell apart phases that look alike from the
cameras, such as the start and the moment after the first cube was put back. The idea follows UniMem (Osterberg et
al., 2026, arXiv 2608.22869), which appends text events to the instruction of a pi0.5 policy. Their second ingredient,
cached keyframe features, is not implemented here.

## How it differs from the planner loop

| | planner loop | history-appended prompt |
|---|---|---|
| what the policy sees | one short skill prompt at a time | the full instruction plus the completed skills as text |
| who sequences the task | the proxy: the planner's plan and a forward-only pointer | the policy itself, from instruction and history |
| what the external part must do | plan the decomposition and judge completion | recognise events only |
| policy training data | frames labelled with the active skill prompt | frames labelled with instruction plus history |

## How the framework realises it

The method reuses the planner loop unchanged as its event detector: the planner still writes a plan, monitors every
policy call with the exemplar memory and advances its pointer. Only the prompt handed to the policy changes.
`HistoryProxy` replaces `PlannerProxy` and renders instruction plus history from the skills before the pointer.
`HistoryPrompt` does the rendering from the `history` section of the task file (prefix, one past-tense phrase per
verb, joiner, suffix) and also produces the per-frame prompts of the training data, so run-time strings and training
strings are identical.

```
python -m vla_memory.history_appended.proxy --listen-port <port> --upstream-port <policy port> --memory <store.npz>
```

Prerequisites: a policy trained on instruction-plus-history prompts (for the reference task, the monolith
demonstrations relabelled per frame with the four history states, at the same event boundaries as the sub-goal
labels), and the same task file, store and planning model as the planner loop.

## Early results (reference task, 20 dev seeds, ground-truth event tracker)

| training step | history-appended: full task | sub-prompt policy with oracle | memoryless monolith |
|---|---|---|---|
| 10000 | 2/20 | 11/20 | 2/20 |
| 14000 | 4/20 | 15/20 | 5/20 |

Training runs to 30000 steps. The checkpoint is then selected by the same rule as for the other policies and
evaluated on the 100 fresh seeds, first with the ground-truth tracker and then with the planner loop's monitor as the
event detector.
