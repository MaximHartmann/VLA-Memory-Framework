# The planner loop

## The idea

The VLA policy is trained on a few short skills ("pick up the red block", "place the red block", ...). A vision-language
model (VLM) acts as the planner: it turns the user's long instruction into an ordered list of those skills once, and at
every policy call it judges from the camera images and the robot's proprioception whether the current skill is complete.
A proxy between the environment and the policy server holds the plan and the current step, consults a store of labelled
past observations, and hands the policy only the prompt of the current skill. The policy stays unchanged and memoryless.

![The planner loop](planner_loop.png)

([planner_loop.pdf](planner_loop.pdf): the loop, the prerequisites, and what is stored where.)

Three kinds of memory are involved:

- **Working memory** of the running episode: the plan, the current step and the last few proprioception values (gripper
  closedness, hand height). It is reset at the start of every episode.
- **Long-term memory**: the exemplar store. Frames of recorded episodes are labelled with the active skill and the object
  states, each frame gets a compact colour fingerprint, and at every question the most similar stored frames are shown to
  the planner as labelled examples. Six nearest stored moments also vote: when none of them says the skill is done, the
  model is not asked (the memory-vote gate). The store is read-only during an episode.
- **Symbolic state**: the task file. It lists the skills, the completion rule of each skill as text, the answer schemas and
  how the proprioception is summarised.

The decision rule: one "done" answer moves the plan pointer, never before the second question about a skill, and the pointer
never moves back. The method draws on SayCan (a language model plans over a fixed library of trained skills), Inner Monologue
(success detection fed back to the planner after every step), SuccessVQA (a VLM judges completion from images) and RAP
(retrieved past experience as in-context examples).

## Prerequisites

- A VLA policy trained on the skills of the task, served over the openpi websocket protocol (another server: `--transport`).
- A vision-language model behind an OpenAI-compatible chat API with JSON-schema structured output; `scripts/serve_vlm_vllm.sh`
  serves one with vLLM (the reference setup: Qwen3.8-27B-INT4 on two RTX 3090).
- A task file; `vla_memory/tasks/red_blue_blocks.yaml` is the reference: objects, verbs, skill template, completion criteria,
  the system texts that describe the cameras.
- Optional: an exemplar store built from recorded demonstrations (`scripts/build_exemplar_store.py`).
- An environment that sends the full instruction as `prompt` and the control keys under `memory/`: step, episode, the raw
  camera frames, gripper closedness and hand height. `vla_memory.env_keys.ControlKeys` builds them.

## How to use it in the framework

```
python -m vla_memory.planner_loop.proxy --listen-port <port> --upstream-port <policy port> [--task task.yaml] [--memory store.npz]
```

The environment connects to `<port>` instead of the policy server and otherwise runs unchanged. `--mode` chooses who decides
the prompt: `vlm` (default, the planner), `rules` (the history method's rules writer sequences the skills from the gripper
signals, no model), `oracle` (the environment's ground truth, for evaluation), `off` (passthrough). The defaults are the
settled configuration of the evaluation: the memory-vote gate, a verdict-only answer, 4 examples per question, one "done"
answer to advance, no switch before the second question; `--no-gate`, `--no-verdict-only`, `--k`, `--votes`, `--min-calls`
change them. `scripts/run_proxy.sh` wraps the command. Optional memories plug in as extensions (`--with module:Class`,
`vla_memory/planner_loop/extensions.py`).

## Our findings

Reference task: pick up and put back a red and then a blue cube with a Franka arm in Isaac Sim; pi0.5 policy; 100 fresh seeds
of 500 control steps; a policy call every 8 steps.

| setup | full tasks of 100 | median episode |
|---|---|---|
| planner loop, no store | 94 | 84 s |
| planner loop, 20 stored episodes | 87 | 133 s |
| ground-truth switching (upper bound) | 92 | 50 s |
| the rules writer as sequencer (`--mode rules`, no model) | 94 | 55 s |
| memoryless policy trained on the full instruction | 11 | |

- The planner reaches the ground truth without a store once the task file describes the wrist camera correctly. With a
  misleading description it completed 69 and needed the store to reach 91: the examples compensated for the text.
- The store adds nothing to the result and costs 49 s per episode, because every question carries four example pairs
  (2.9 s per question against 0.7 s without). The policy is not the bottleneck, the question is: a policy call takes 0.17 s.
- The planner's "done" answers were right in 99.4 % of the cases (98.3 % with the store), and no failed plan was declared
  complete. The remaining failures are grasps, not the memory.
- For a task whose skill order is known, the rules writer replaces the VLM monitor at no cost; the planner is needed to turn
  a new instruction into a plan.
