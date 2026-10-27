# LeRobot policies behind the framework

## What it is

`lerobot_server.py` serves any policy of LeRobot, Hugging Face's robot-learning library (SmolVLA, π0, ACT and the other
policies of LeRobot 0.6.1), in openpi's serving protocol: msgpack (a compact binary format) over a websocket, the protocol
that the framework's proxy and openpi's clients speak. So a LeRobot policy runs behind the proxy like an openpi policy,
with any writer and any method.

Every reply also holds the policy's state for the [head writer](../../docs/writers/head/README.md), read by the rule every
server of the framework follows: the language backbone two thirds of the way up, averaged over the prompt's words. The
[automatic tap](../../docs/taps/README.md) finds the backbone and the words by itself and hands out this one summary as
`reply["policy_state"]`, named like `"block11:prompt"`. Nothing in `vla_memory` imports LeRobot; the server imports the
framework.

## Prerequisites

- A LeRobot environment: Python, torch and `lerobot` with the policy's extras (for SmolVLA `lerobot[smolvla]==0.6.1`),
  plus `websockets` and `msgpack` for the network part.
- The framework, this folder and openpi's client library on `PYTHONPATH`:

  ```
  PYTHONPATH=<framework>:<framework>/integrations/lerobot:<openpi>/packages/openpi-client/src
  ```

- The policy's files: a hub id (already downloaded; with `HF_HUB_OFFLINE=1` nothing is fetched) or a folder with its
  `config.json`, `model.safetensors` and pre- and post-processor files. A policy whose config would load its backbone's
  weights separately (`load_vlm_weights`, as SmolVLA's does) takes them from its own weights file instead, so the
  backbone's own weights are not needed; the backbone's small files (config, tokenizer, processor) are.

## How to use it

```
python -m lerobot_server --policy lerobot/smolvla_base --port 18951 [--device cuda] \
    [--camera observation/image=observation.images.camera1 --camera observation/wrist_image=observation.images.camera2] \
    [--state observation/state=observation.state] [--default-prompt TEXT]
python -m vla_memory.proxy --writer off --method subgoal --listen-port 19951 --upstream-port 18951
```

The server loads the policy and its pre- and post-processors (LeRobot's conversions of the inputs and the outputs) with
LeRobot's own functions, for any policy type (`load_policy`). Then, at every call (`LeRobotPolicy.infer`):

- **The observation is renamed** to the policy's inputs. By default openpi's pictures `observation/image` and
  `observation/wrist_image` become the policy's cameras in order, and `observation/state` becomes its state; `--camera`
  and `--state` (repeatable, `KEY=FEATURE`) choose by hand. A camera that is not mapped is missing for the policy: SmolVLA
  then leaves it out, or uses a black picture if its config asks for empty cameras.
- **The inputs are converted** as LeRobot's own policy server does it: a picture (uint8, height x width x 3) becomes a
  float tensor in [0, 1], resized to the policy's size; the state is cut or padded with zeros to the policy's size; the
  prompt becomes the task.
- **The policy computes its action chunk**, and the reply holds it with the state:

  ```
  {"actions": (steps, action size) float32,
   "policy_state": {"names": ["block11:prompt"], "values": (1, width) float16}}      # vla_memory/taps/reply.py
  ```

- **A new episode** (`memory/new_episode` True, which the proxy passes on from the environment) resets the policy.

At start-up the server makes one call with black pictures and a short prompt. In this call the tap picks the backbone and
its block, and the metadata then names the summary and the backbone. `LeRobotPolicy` (the policy part: `infer(observation)`
and `metadata`) is separate from `serve()` (the network part), so it can be tested without a network.

## What was checked

With SmolVLA (`lerobot/smolvla_base`) on one RTX 3090, on our recorded pictures (the tap's own checks are on the
[state taps page](../../docs/taps/README.md)):

- **Every weight loads** with `load_vlm_weights` off: no missing and no unexpected keys.
- **The tap finds the language backbone:** SmolVLA's text model (as wide as the word table, 960; 177 tokens), not the
  vision encoder (1,024 tokens per picture, width 768) or the action expert (50 tokens, width 720). SmolVLA's own loop calls
  the parts of each block one by one, so the tap reads block 11 of 16 at the next block's norm.
- **It finds the prompt's words:** the summary averages the 9 real words of our task, in every call, and neither the
  pictures nor the state token.
- **The summary is the policy's own numbers:** the tap's state after a block is exactly that of SmolVLA's own loop cut
  after that block (blocks 8, 11 and 13 checked), and a median 0.2 % away from the text model run by `transformers` on the
  same input.
- **The actions do not change:** identical with and without the tap in 30 of 30 calls; a call took 296 ms without the tap
  and 297 ms with it.
- **Over the network**, directly with openpi's client and through the proxy (`--writer off --method subgoal`), every reply
  had 50 actions of size 6 and the summary `block11:prompt`.
- **On the CPU, with a small fake LeRobot policy:** the renaming of the observation, the post-processed action chunk, the
  state's names, the reset at a new episode, and a policy without a stack of blocks.

**Not covered yet:**

- Only SmolVLA was run. Another policy type is expected to work through LeRobot's own functions, but is untested.
- A policy without a stack of transformer blocks (a diffusion policy's U-Net) has no state to hand out: the server then
  sends its actions only.
- The server calls `predict_action_chunk` and hands out the whole chunk; LeRobot's real-time chunking is not used.
- A state of the wrong size is cut or padded without a warning, so a wrong `--state` shows only in the actions.
- Calls run one after another, at batch 1.
- A policy whose backbone input shows no words (it adds position vectors to them first) gets the last real token
  instead, named so (`block11:last`).
- A policy that keeps past observations (`n_obs_steps` above 1) would count the extra questions of the proxy's recorder
  (`--record-head`) as calls.
