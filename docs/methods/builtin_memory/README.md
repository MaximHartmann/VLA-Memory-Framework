# The built-in memory: a recurrent state inside the policy

## The idea

The memory lives in the policy's network, and nothing outside writes it. At every policy call the frozen backbone already
computes a summary of what it sees: its residual stream after block 12 (of 18), averaged over the prompt's tokens and over
each camera's tokens, 6,144 numbers. A small recurrent cell, a GRU with a state of 256 numbers, takes this summary once per
call and keeps what matters across the episode. Its state is projected into the action expert's conditioning signal (pi0.5's
adaRMS conditioning, which already carries the flow-matching time and modulates every norm of the action expert), so every
action chunk is computed with the episode's past in view:

```
call t   cameras + instruction -> backbone (frozen) -> layer-12 summary x_t
         h_t = GRU(h_t-1, x_t)                       the state, zero at the episode start
         action expert, conditioned on time + W h_t  -> the actions of call t
```

The read-out W starts at zero, so the untrained memory adds nothing and the policy starts as the memoryless one. An optional
phase head, a linear layer on the state, learns the task's phase (pick red, place red, pick blue, place blue) as an auxiliary
loss. The policy is trained on sequences: every training frame comes with the summaries of the calls before it in its
episode. The policy server keeps the state per episode; the framework's part is the episode contract (the proxy forwards
the environment's episode markers) and the reset test, which clears the state before every call.

The memory learns only from data that require it. Demonstrations of a scripted expert can be imitated from the current
frame alone, so a memory trained on them stays unused. It learns from the robot's own runs with an expert's interventions
([learning from interventions](../learning_from_interventions/README.md)), where scenes that look alike need different
actions. A window of the same summaries instead of a recurrent state was tried and not adopted
([the frame window](../frame_window/README.md)).

## Prerequisites

- A policy with the memory and its server: the reference implementation for pi0.5 on openpi in
  [integrations/openpi](../../../integrations/openpi/README.md), or another VLA built along the five decisions below.
- Training data that require memory: the demonstrations plus intervention episodes, each dataset with a feature cache of
  the backbone's summaries (the integration's `sequences.py` describes the format).
- At run time: an environment that marks its episodes, `memory/new_episode` at the first call (and `memory/episode`, an id),
  as `vla_memory.env_keys.ControlKeys` sends them.

## How to use it in the framework

1. **Train** the memory with the integration's launcher (`python -m builtin_memory.train`, its README): 10,000 steps from
   the memoryless policy with the backbone frozen, on the demonstrations and the intervention episodes; further cycles of
   10,000 steps continue from the last checkpoint.
2. **Serve** a checkpoint and put the proxy in front of it. The policy reads the plain instruction, so no writer is needed:

   ```
   python -m builtin_memory.serve --checkpoint <checkpoint dir> --port <policy port> --center-crop 0.95
   python -m vla_memory.proxy --writer off --listen-port <port> --upstream-port <policy port> [--memory-reset every-call]
   ```

   The proxy forwards `memory/new_episode` and `memory/episode` unchanged, and the server resets its state at them. Every
   reply carries `memory`: the call in the episode, whether it reset, the state's norm and the phase head's phase.
   `--memory-reset every-call` marks every call as a new episode, so the server clears the state before every call: if the
   policy really uses its memory, its gain must vanish.

### Another VLA

Five decisions carry the method to another policy:

1. **The layer and its pooling.** A middle block of the frozen backbone, here 12 of 18, averaged per modality (the text,
   each camera), taken from the prefix pass that also feeds the actions, so the memory costs no extra pass.
2. **Where the action head takes the state.** A conditioning vector that reaches every layer of the action head: pi0.5's
   adaRMS conditioning; another head needs an equivalent (a modulation of its norms, or an extra token).
3. **Zero initialisation.** The read-out starts at exactly zero, so training starts from the memoryless policy's function;
   the phase head gives the cell a gradient from the first step, the action loss from the first update on.
4. **Sequence training and the history length.** Every training sample is a frame with the summaries of its episode's
   calls up to it, at the run-time call spacing (gaps of 6-10 frames, with repeated calls as pauses), and the history cap
   must cover whole episodes: 96 calls for episodes of up to 600 steps (64 cut 21 % of the intervention histories short).
   With the backbone frozen the summaries are computed once and cached.
5. **The memory's learning rate.** Ten times the action expert's (peak 1e-4 against 1e-5): the memory starts from scratch,
   the action expert continues from a converged checkpoint.

Without intervention data the memory learns nothing: trained on the demonstrations alone it completed exactly as many
tasks as the memoryless policy, and clearing it made no significant difference (11 and 9 of 100).

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim; the memoryless pi0.5
policy and the policies trained on from it with the backbone frozen (their steps count from there); interventions recorded
in the memoryless policy's runs (round 1) and in the memory policy's runs (round 2); 100 fresh seeds of 500 control steps, a
policy call every 8 steps:

| policy | memory | training data | steps | full tasks of 100 | memory cleared before every call |
|---|---|---|---|---|---|
| memoryless policy | none | demonstrations | 27,000 | 11 | |
| interventions without memory | none | demonstrations + round 1 | 4,000 | 0 | |
| memory, demonstrations only | GRU + phase head | demonstrations | 10,000 | 11 | 9 |
| memory | GRU | demonstrations + round 1 | 10,000 | 34 | 0 |
| memory with phase head | GRU + phase head | demonstrations + round 1 | 10,000 | 39 | 3 |
| the same, two more cycles | GRU + phase head | demonstrations + round 1 | 20,000 / 30,000 | 51 / 60 | 2 / 0 |
| the same, a fourth cycle (control) | GRU + phase head | demonstrations + round 1 | 40,000 | 51 | 0 |
| the same, with the second round | GRU + phase head | demonstrations + rounds 1 and 2 | 40,000 | 65 | 0 |

- Memory and interventions need each other: the memory trained on demonstrations stays at the memoryless policy's 11, the
  interventions without a memory sink the policy to 0, together they reach 34-39 at 10,000 steps.
- The gain lives in the state: cleared before every call, every memory policy trained on interventions falls to 0-3 of 100.
  In the best one a linear probe reads the task phase from the state with a balanced accuracy of 0.99, from the current
  call's summary with 0.78.
- More training cycles on the same data raised the policy with the phase head from 39 to 51 and 60; a fourth cycle gave 51
  (not significantly different). A second round of interventions recorded with the memory policy driving gave 65 against
  51 for that matched-steps control (paired sign test, p = 0.029); the difference opens between red and blue (reached blue
  90 against 76).
- The memory costs a few milliseconds per policy call: a median of 173-179 ms with it, 170-174 ms without it, on the same
  setup.
- A window of the last 8-16 calls' summaries in place of the recurrent state, with the same data and recipe, completed 8-12
  of 100 (a second version 6; [the frame window](../frame_window/README.md)). Memory written from outside the policy still
  does better on this task (history 77-82, keyframe 75-78); 27 of the best memory policy's 35 failures are grasps.
