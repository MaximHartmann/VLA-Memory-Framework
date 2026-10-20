# VLA-Memory-Framework

A framework to incorporate different kinds of knowledge bases and memory into a robotic AI agent built on a
vision-language-action (VLA) policy.

A VLA policy maps the current camera images and a language instruction to robot actions. On its own it is
memoryless: it does not know what it has already done in the current task, it does not learn from earlier tasks, and
it cannot adapt to a user over time. This framework adds that missing part. It defines the places where memory can
live in a VLA agent: the working memory of the running task, long-term memory kept outside the models in a store,
knowledge written into the models' weights, and the symbolic task state. It provides the interfaces to plug a memory
method into the control loop and collects reference implementations, so that different memory methods and knowledge
bases can be built, exchanged and compared against the same policy, the same task and the same metrics.

The framework is model-agnostic: the VLA policy, the language or vision-language model, the task and the memory
method are replaceable components behind small interfaces.

## Two layers: writers and methods

A proxy sits between the environment and the VLA policy server and relays every observation. Two exchangeable components
run in it. A **writer** signals that a step of the task is done: it keeps the plan, the task's skills in order, and a pointer
that it moves forward when it decides that the current skill is complete. A **method** turns that signal into what the
policy receives. Every writer drives every method:

```
python -m vla_memory.proxy --writer NAME --method NAME --listen-port <port> --upstream-port <policy port>
```

The environment connects to `<port>` instead of the policy server and otherwise runs unchanged: it sends the full
instruction as the prompt and its control keys under `memory/` (`vla_memory/env_keys.py` builds them). A writer that decides
from the call's inputs moves its pointer before the policy call, so the policy's input of the same call shows the decision;
a writer that decides from the policy's reply shows it from the next call on. The skills, their completion rules and the
prompt texts come from a task file (`vla_memory/tasks/red_blue_blocks.yaml`).

| writer | who decides that a skill is complete | real robot | page |
|---|---|---|---|
| `oracle` | the simulator's ground truth, the reference | no | [docs/writers/oracle](docs/writers/oracle/README.md) |
| `rules` | rules on the robot's gripper and hand signals, declared in the task file, and a colour identifier for the object; no model, real time | yes | [docs/writers/rules](docs/writers/rules/README.md) |
| `vlm` | a vision-language model that plans the skills once and judges completion from the cameras and the proprioception at every call, with a store of labelled past observations | yes | [docs/writers/vlm](docs/writers/vlm/README.md) |
| `head` | the policy itself: an event head on its internal representation of each call | yes | [docs/writers/head](docs/writers/head/README.md) |

| method | what the policy receives | memory | page |
|---|---|---|---|
| `subgoal` | the prompt of the current skill; the policy is trained on the short skills | the writer's plan and pointer, held by the proxy | [docs/methods/subgoal](docs/methods/subgoal/README.md) |
| `history` | the full instruction with a line per completed skill; the policy is trained on such prompts | the completed skills as text in the policy's input | [docs/methods/history](docs/methods/history/README.md) |
| `keyframe` | the plain instruction, and in the third image slot the wrist picture of the moment the last skill was completed | a picture of the last key moment in the policy's input | [docs/methods/keyframe](docs/methods/keyframe/README.md) |

`--writer off` hands the policy the instruction unchanged.

### Measured results

Full tasks out of 100 fresh seeds on the reference task: pick up and put back a red and then a blue cube with a Franka arm
in Isaac Sim; a pi0.5 policy trained for each method; 500 control steps, a policy call every 8 steps. An empty cell: the
combination is supported but was not measured.

| method \ writer | `oracle` | `rules` | `vlm` | `head` |
|---|---|---|---|---|
| `subgoal` | 92 | 94 | 94 | |
| `history` | 82 | 79 | 77 | 78 |
| `keyframe` | 78 | 75 | | |

For each method the writers are within noise of each other; the memoryless policy, trained on the full instruction, completes
11. The VLM writer ran without a store with the sub-goal method (87 with a store of 20 recorded episodes) and with that store
with the history method. The pages give the conditions, the timing and what failed.

## What the policy must accept

Any VLA policy works behind the proxy if it was trained on the inputs its method and extensions produce. The writer only
decides when a skill is complete, so what the policy reads follows from the method and the extensions. Each row below names
the script in `training/` that builds the matching training dataset, in LeRobot format, from recorded demonstrations
(`training/README.md`). The strings in the training data are the ones the proxy sends at run time, which the task file defines.

| in the run | the policy must accept | training data |
|---|---|---|
| `--writer off` | the full instruction as the prompt (the memoryless policy) | `convert_to_lerobot.py` |
| `--method subgoal` | short skill prompts, one at a time ("pick up the red block") | `relabel_subgoals.py` |
| `--method history` | the full instruction followed by " History: <completed skills>" ("... History: picked up the red block.") | `relabel_history.py` |
| `--method keyframe` | the plain instruction and a third picture, `observation/memory_image`: the wrist picture of the last key moment, masked out until the first event | `relabel_keyframe.py` |
| `--with memory_policy` (with `--method subgoal`) | the third picture, the wrist picture of a retrieved moment, and the skill prompt with ". memory: <sentence>" appended | `relabel_memory.py` |
| `--with notebook` | nothing extra: it changes which of the task's skills the VLM writer plans and when it moves on | |
| `--with session_store` | nothing extra: it is the memory of the VLM writer's monitor | |
| `--writer oracle`, `rules`, `vlm` | nothing from the policy | |
| `--writer head` | its internal features go to the head: the head's server reads them and returns a verdict with every reply; as shipped it reads those of an openpi pi0.5 policy, so it runs on openpi policies only | `fit_event_head.py` fits the head on the policy's rollouts |

The proxy reaches the policy through the openpi serving protocol: msgpack-encoded numpy arrays over a websocket, through the
`openpi_client` library. It reads the server's metadata once and then sends one observation per policy call: the environment's
own keys (the reference: `observation/image`, `observation/wrist_image`, `observation/state`) with the `prompt` the method
decided and, for a third picture, `observation/memory_image`. It expects a dict with `actions` back. A server that speaks this
protocol, such as openpi's `serve_policy.py`, needs nothing else; for another protocol, `--transport` names a class with an
`infer(observation)` method that returns `{"actions": ...}` (`vla_memory/transport.py`).

## Other methods

| method | memory used | how it plugs in | page |
|---|---|---|---|
| RECAP-lite | parametric long-term memory: the policy's own successful rollouts written into its weights by fine-tuning; one round completes 94 of 100 against 92 for the parent (ground-truth sub-goals), within noise | a new policy behind any writer and method | [docs/methods/recap](docs/methods/recap/README.md) |
| Memory policy | long-term memory read by the policy: the stored moment of recorded episodes most similar to the current one (its wrist picture and the next movement as a sentence) in the policy's input at every call; the policy reads it, but it does not improve it (98 of 100 with memory, 96 without) | `--with memory_policy` | [docs/methods/memory_policy](docs/methods/memory_policy/README.md) |
| Preference notebook | long-term memory of the user: short rules with a scope (the plan, a skill's completion rule, the planner's strictness), written from the user's words and after failures, read by the VLM writer's planner before every episode; "only the red block today" gave the red-only plan in 20 of 20 episodes (0 of 20 without), a correction raised the success from 12 to 19 of 20 | `--with notebook`, with `--writer vlm` | [docs/methods/notebook](docs/methods/notebook/README.md) |
| Session store | long-term memory of the VLM writer on an embedded vector database (LanceDB): labelled moments of past episodes, retrieved by a choice of keys, with payload filters, provenance and a size cap, growing from the robot's own episodes; the colour key finds the right skill more often than DINOv2 and SigLIP (recall@1 0.47 against 0.19 with the camera moved 5 cm), and a cap of 1,000 moments costs no task success (91 of 100 with use-and-recency eviction, 91 without a cap) | `--with session_store`, with `--writer vlm` | [docs/methods/session_store](docs/methods/session_store/README.md) |

## Tried and not adopted

| idea | result | page |
|---|---|---|
| Frame window: the last 8-16 policy calls as compressed frame summaries, pooled into the action expert's conditioning | 6-12 of 100 in two versions (the recurrent memory on the same data: 34-39); the information was in the window, selected in the second version, and not used by the controller | [docs/methods/frame_window](docs/methods/frame_window/README.md) |

Each further writer or method gets its own page under `docs/writers/` or `docs/methods/` and a row in these tables; an idea
that was evaluated and did not work gets a page and a row here.
