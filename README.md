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

## Getting started

A proxy sits between the environment (a simulator or a robot) and the VLA policy server and relays every observation. Two
exchangeable components run in it. A **writer** decides when a step of the task is done: it keeps the plan, the task's
skills in order (for example pick up the red block, place it, pick up the blue block, place it), and moves on to the next
skill when it judges the current one complete. A **method** turns that decision into what the policy receives, for example
the current skill as the prompt.

### Install

Python 3.10 or newer. The proxy talks to the policy server through openpi's client library, which is installed from the
openpi repository; then install this repository with the proxy's extra:

```
pip install -e <openpi checkout>/packages/openpi-client
pip install -e ".[proxy]"
```

What else a run needs:

- a policy server for the VLA: any policy served over openpi's websocket protocol (such as openpi's `serve_policy.py`), or
  another protocol through `--transport`. The policy must be trained for the chosen method ([What the policy must
  accept](#what-the-policy-must-accept));
- the environment that would otherwise talk to the policy server. It connects to the proxy instead and adds a few entries
  for the proxy to every observation (below);
- for `--writer vlm`, a vision-language model behind an OpenAI-compatible API, such as vLLM (`--vlm-url`, `--vlm-model`);
- for the session store, `pip install -e ".[session_store]"` (and `".[session_store_encoders]"` for its image-embedding keys).

### Run: a writer and a method

```
python -m vla_memory.proxy --writer NAME --method NAME --task FILE --listen-port <port> --upstream-port <policy port>
```

`--task` names the task file (below); for the reference task it can be left out. The proxy listens on `<port>` and forwards to the policy server on `<policy port>`. The environment connects to `<port>`
instead of the policy server and otherwise runs as before: it sends the full task instruction as the prompt, plus entries
for the proxy whose names start with `memory/`, for example that a new episode starts or the gripper's readings since the
previous call. `vla_memory/env_keys.py` builds these entries from the environment's raw signals. The proxy removes them
before it forwards the observation, except the two that mark a new episode, which a policy with its own memory needs.

`--writer` chooses who decides that a skill is complete:

| writer | who decides that a skill is complete | real robot | page |
|---|---|---|---|
| `oracle` | the simulator's ground truth, the reference | no | [docs/writers/oracle](docs/writers/oracle/README.md) |
| `rules` | rules on the robot's gripper and hand signals, declared in the task file, and a colour identifier for the object; no model, real time | yes | [docs/writers/rules](docs/writers/rules/README.md) |
| `vlm` | a vision-language model that plans the skills once and judges completion from the cameras and the proprioception at every call, with a store of labelled past observations | yes | [docs/writers/vlm](docs/writers/vlm/README.md) |
| `head` | the policy itself: an event head on its internal representation of each call | yes | [docs/writers/head](docs/writers/head/README.md) |

`--method` chooses what the policy receives:

| method | what the policy receives | memory | page |
|---|---|---|---|
| `subgoal` | the prompt of the current skill; the policy is trained on the short skills | the writer's plan and its current skill, held by the proxy | [docs/methods/subgoal](docs/methods/subgoal/README.md) |
| `history` | the full instruction with a line per completed skill; the policy is trained on such prompts | the completed skills as text in the policy's input | [docs/methods/history](docs/methods/history/README.md) |
| `keyframe` | the plain instruction, and in the third image slot the wrist picture of the moment the last skill was completed | a picture of the last key moment in the policy's input | [docs/methods/keyframe](docs/methods/keyframe/README.md) |

Every writer drives every method. `--writer off` hands the policy the instruction unchanged; without `--writer` and
`--method` the proxy runs `vlm` with `subgoal`. `--with NAME` adds an extension ([other methods](#other-methods)), and
`--help` together with a choice lists that choice's options. The measured results of the writer and method combinations on
the reference task are on [their own page](docs/results/README.md).

### Tasks

Everything about the task comes from one task file in YAML: the objects, the skills, how to tell that each skill is
complete, and the prompt texts. Without `--task FILE` the proxy uses the reference task,
[`vla_memory/tasks/red_blue_blocks.yaml`](vla_memory/tasks/red_blue_blocks.yaml). On the framework's side a new task needs no
code, only a new task file: [Creating a new task](docs/tasks/README.md). The policy behind the proxy must of course be able
to do the new task's skills. If the objects change from episode to episode, the environment names each episode's objects
(`memory/objects`) and the writers plan for those.

## What the policy must accept

Any VLA policy works behind the proxy if it was trained on the inputs its method and extensions produce. The writer only
decides when a skill is complete, so what the policy reads follows from the method and the extensions. Each row below names
the script in `training/` that builds the matching training dataset, in LeRobot format, from recorded demonstrations
(`training/README.md`). The strings in the training data are the ones the proxy sends at run time, which the task file defines.

| in the run | the policy must accept | training data |
|---|---|---|
| `--writer off` | the full instruction as the prompt (the memoryless policy) | `convert_to_lerobot.py` |
| `--writer off` with a policy that keeps its own memory (the built-in memory) | the full instruction and the episode markers, which the proxy forwards unchanged: the policy keeps a state per episode and resets it at `memory/new_episode` (True at an episode's first call) or a new `memory/episode`; `--memory-reset every-call` marks every call as a new episode, so the state is cleared before every call (the reset test) | the demonstrations and intervention episodes with their feature caches, trained with the launcher in `integrations/openpi` |
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
decided, the episode markers `memory/new_episode` and `memory/episode` as the environment sent them and, for a third picture,
`observation/memory_image`. It expects a dict with `actions` back. A server that speaks this protocol, such as openpi's
`serve_policy.py`, needs nothing else; for another protocol, `--transport` names a class with an `infer(observation)` method
that returns `{"actions": ...}` (`vla_memory/transport.py`).

## Other methods

| method | memory used | how it plugs in | page |
|---|---|---|---|
| RECAP-lite | parametric long-term memory: the policy's own successful rollouts written into its weights by fine-tuning; one round completes 94 of 100 against 92 for the parent (ground-truth sub-goals), within noise | a new policy behind any writer and method | [docs/methods/recap](docs/methods/recap/README.md) |
| Memory policy | long-term memory read by the policy: the stored moment of recorded episodes most similar to the current one (its wrist picture and the next movement as a sentence) in the policy's input at every call; the policy reads it, but it does not improve it (98 of 100 with memory, 96 without) | `--with memory_policy` | [docs/methods/memory_policy](docs/methods/memory_policy/README.md) |
| Preference notebook | long-term memory of the user: short rules with a scope (the plan, a skill's completion rule, the planner's strictness), written from the user's words and after failures, read by the VLM writer's planner before every episode; "only the red block today" gave the red-only plan in 20 of 20 episodes (0 of 20 without), a correction raised the success from 12 to 19 of 20 | `--with notebook`, with `--writer vlm` | [docs/methods/notebook](docs/methods/notebook/README.md) |
| Session store | long-term memory of the VLM writer on an embedded vector database (LanceDB): labelled moments of past episodes, retrieved by a choice of keys, with payload filters, provenance and a size cap, growing from the robot's own episodes; the colour key finds the right skill more often than DINOv2 and SigLIP (recall@1 0.47 against 0.19 with the camera moved 5 cm), and a cap of 1,000 moments costs no task success (91 of 100 with use-and-recency eviction, 91 without a cap) | `--with session_store`, with `--writer vlm` | [docs/methods/session_store](docs/methods/session_store/README.md) |
| Built-in memory | working memory inside the network: a recurrent state over the policy's own layer-12 features, kept per episode by the policy server and read by the action expert; learned only from intervention data: 65 of 100 (the memoryless policy 11), 0 with the state cleared before every call | a policy server with its own state behind `--writer off`; the proxy forwards the episode markers, `--memory-reset every-call` runs the reset test; pi0.5 on openpi in `integrations/openpi` | [docs/methods/builtin_memory](docs/methods/builtin_memory/README.md) |
| Learning from interventions | parametric memory of the robot's own mistakes: its runs with a scripted expert taking over on memory errors, trained into a policy with the built-in memory; a second round recorded with the memory policy driving gave 65 of 100 against 51 for the same training on the first round's data (p = 0.029) | a data recipe for the built-in memory's training | [docs/methods/learning_from_interventions](docs/methods/learning_from_interventions/README.md) |

## Tried and not adopted

| idea | result | page |
|---|---|---|
| Frame window: the last 8-16 policy calls as compressed frame summaries, pooled into the action expert's conditioning | 6-12 of 100 in two versions (the [built-in memory](docs/methods/builtin_memory/README.md) on the same data: 34-39); the information was in the window, selected in the second version, and not used by the controller | [docs/methods/frame_window](docs/methods/frame_window/README.md) |

Each further writer or method gets its own page under `docs/writers/` or `docs/methods/` and a row in these tables; an idea
that was evaluated and did not work gets a page and a row here.
