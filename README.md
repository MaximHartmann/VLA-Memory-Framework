# VLA-Memory-Framework

A framework to add different kinds of knowledge bases and memory to a robotic AI agent built on a vision-language-action
(VLA) policy.

A **VLA policy** is a neural network that turns the current camera pictures and a written instruction into robot
movements. On its own it has no memory: it does not know what it has already done in the current task, it does not learn
from earlier tasks, and it cannot adapt to a user over time. This framework adds the missing part. It names the places
where memory can live in an agent built on a VLA policy:

- the **working memory** of the running task;
- a **long-term memory**, kept outside the models in a store;
- **knowledge in the models' weights**, written there by training;
- the **symbolic task state**: which steps of the task are done.

It provides the interfaces to plug a memory method into the robot's control loop, and reference implementations of
several methods. So different memory methods and knowledge bases can be built, exchanged and compared against the same
policy, the same task and the same measures.

The framework is model-agnostic: the VLA policy, the language or vision-language model, the task and the memory method are
replaceable parts behind small interfaces.

## How it works

The framework runs as a **proxy**: a small server between the environment (a simulator or a robot program) and the server
of the VLA policy. The environment sends its observations to the proxy instead of the policy server. The proxy changes
what the policy sees, forwards the observation, and returns the policy's actions.

The robot does a task as a sequence of **skills**, short steps the policy was trained on: for example "pick up the red
block", "place the red block", "pick up the blue block", "place the blue block". The ordered list of skills is the
**plan**. An **episode** is one attempt at the whole task, and a **policy call** is one request to the policy: one
observation in, a short chunk of actions out. Two exchangeable parts run in the proxy at every call:

- a **writer** decides when a step of the task is done: it keeps the plan and moves on to the next skill when it judges the
  current one complete;
- a **method** turns that decision into what the policy receives, for example the current skill as the prompt.

## Getting started

### Install

You need Python 3.10 or newer. The proxy talks to the policy server through openpi's client library, which is installed
from the openpi repository. Then install this repository with the proxy's extra:

```
pip install -e <openpi checkout>/packages/openpi-client
pip install -e ".[proxy]"
```

A run also needs:

- **a policy server** for the VLA: any policy served over openpi's websocket protocol (such as openpi's
  `serve_policy.py`), or another protocol through `--transport`. The policy must be trained for the chosen method
  ([What the policy must accept](#what-the-policy-must-accept));
- **the environment** that would otherwise talk to the policy server. It connects to the proxy instead and adds a few
  entries for the proxy to every observation (below);
- for `--writer vlm`, **a vision-language model** behind an OpenAI-compatible API, such as vLLM (`--vlm-url`,
  `--vlm-model`);
- for the session store, `pip install -e ".[session_store]"` (and `".[session_store_encoders]"` for its image-embedding
  keys).

### Run: a writer and a method

```
python -m vla_memory.proxy --writer NAME --method NAME --task FILE --listen-port <port> --upstream-port <policy port>
```

The proxy listens on `<port>` and forwards every call to the policy server on `<policy port>`. `--task` names the task file
(below); for the reference task you can leave it out. After the install, `vla-memory-proxy` is the same command.

The environment connects to `<port>` instead of the policy server and otherwise runs as before. It sends the full task
instruction as the prompt, plus entries for the proxy whose names start with `memory/`: for example that a new episode
starts, or the gripper's readings since the previous call. `vla_memory/env_keys.py` builds these entries from the
environment's raw signals. The proxy removes them before it forwards the observation, except the two that mark a new
episode (`memory/new_episode`, `memory/episode`), which a policy with its own memory needs.

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

- Every writer drives every method. `--writer off` hands the policy the instruction unchanged (the memoryless baseline).
  Without `--writer` and `--method` the proxy runs `vlm` with `subgoal`.
- `--with NAME` adds an extension, an optional memory that plugs into the proxy ([other methods](#other-methods)).
- `--help` together with a choice lists that choice's options, for example
  `python -m vla_memory.proxy --writer vlm --method subgoal --help`.
- `--log FILE` writes one JSON line per policy call: what the writer decided and what the policy received.
- Ctrl-C stops the proxy cleanly: it closes the open connections, lets the extensions save the last episode,
  and closes its log.

The measured results of the writer and method combinations on the reference task are on
[their own page](docs/results/README.md).

### Tasks

Everything about the task comes from one task file in YAML: the objects, the skills, how to tell that each skill is
complete, and the prompt texts. Without `--task FILE` the proxy uses the reference task,
[`vla_memory/tasks/red_blue_blocks.yaml`](vla_memory/tasks/red_blue_blocks.yaml): a Franka arm picks up a red block and
puts it back, then the blue one. On the framework's side a new task needs no code, only a new task file:
[Creating a new task](docs/tasks/README.md). The policy behind the proxy must of course be able to do the new task's
skills. If the objects change from episode to episode, the environment names each episode's objects (`memory/objects`),
and the writers plan for those.

## What the policy must accept

Any VLA policy works behind the proxy if it was trained on the inputs its method and extensions produce. The writer only
decides when a skill is complete, so what the policy reads depends on the method and the extensions alone. Each row below
names the script in `training/` that builds the matching training dataset, in LeRobot format, from recorded
demonstrations ([training/README.md](training/README.md)). The strings in the training data must be the ones the proxy
sends at run time, which the task file defines.

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
| `--writer head` | its server hands out a summary of the policy's internal state with every reply (`policy_state`), which the head in the proxy scores; the framework's servers do this for openpi's π0.5 (JAX or PyTorch) and for any LeRobot policy, another VLA's server needs a [state tap](docs/taps/README.md) | the proxy records the policy's own runs (`--record-head`), `fit_event_head.py --recordings` fits the head on them |

**How the proxy reaches the policy.** It uses the openpi serving protocol: msgpack-encoded numpy arrays over a websocket,
through the `openpi_client` library. It reads the server's metadata once and then sends one observation per policy call:
the environment's own keys (in the reference task `observation/image`, `observation/wrist_image`, `observation/state`)
with the `prompt` the method decided, the episode markers `memory/new_episode` and `memory/episode` as the environment sent
them, and, for a third picture, `observation/memory_image`. It expects a dict with `actions` back. A server that speaks
this protocol, such as openpi's `serve_policy.py`, needs nothing else. For another protocol, `--transport` names a class
with an `infer(observation)` method that returns `{"actions": ...}` (`vla_memory/transport.py`).


## Other methods

These memories go beyond one writer and one method: some plug into the proxy as extensions (`--with NAME`), others are
training recipes or live inside the policy.

| method | what it is | what it is good for | page |
|---|---|---|---|
| RECAP-lite | the policy's own successful runs, trained back into its weights | learning from the robot's own experience without new demonstrations; nothing extra is needed at run time | [docs/methods/recap](docs/methods/recap/README.md) |
| Memory policy | at every call, the most similar moment from recorded episodes (its wrist picture and the next movement as a sentence) is added to the policy's input | showing the policy how a similar situation was handled before | [docs/methods/memory_policy](docs/methods/memory_policy/README.md) |
| Preference notebook | short rules from the user's own words ("blue first", "only red today"), kept across sessions together with how long they hold | adapting to one user over days: following preferences, changes and corrections without the user repeating them | [docs/methods/notebook](docs/methods/notebook/README.md) |
| Session store | labelled moments of the robot's past episodes in a small vector database, found again by similarity | a growing long-term memory for the VLM writer: it answers many of the monitor's checks by itself and saves model calls | [docs/methods/session_store](docs/methods/session_store/README.md) |
| Built-in memory | a recurrent state inside the policy, updated at every call and kept for the whole episode | keeping track of the task inside the policy itself, without a writer outside it | [docs/methods/builtin_memory](docs/methods/builtin_memory/README.md) |
| Learning from interventions | the robot's own runs, with a scripted expert taking over at its mistakes, used as training data | teaching the built-in memory what to remember: without such data it learns nothing | [docs/methods/learning_from_interventions](docs/methods/learning_from_interventions/README.md) |

## Tried and not adopted

| idea | result | page |
|---|---|---|
| Frame window: the last 8-16 policy calls as compressed frame summaries, pooled into the action expert's conditioning | 6-12 of 100 in two versions (the [built-in memory](docs/methods/builtin_memory/README.md) on the same data: 34-39); the information was in the window, but the attention barely picked it out (in the second version the decisive frames got only 1.6 times their equal share) and the action expert did not use it | [docs/methods/frame_window](docs/methods/frame_window/README.md) |
