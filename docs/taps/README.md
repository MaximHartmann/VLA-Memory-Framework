# State taps: reading the policy's internal state during its own pass

## The idea

The [head writer](../writers/head/README.md) judges from numbers inside the policy whether a skill is done. A VLA's
language backbone is a stack of **blocks**; each block takes one vector per input **token** (a piece of a picture or of the
prompt) and hands on a new vector per token. The head reads a summary of one block's output, for example its average over
the prompt's tokens. Computing it in a second pass after every call would cost about as much time as a large part of the
policy itself. A **state tap** reads it while the policy computes its actions: in the same **forward pass** (the policy's
computation from input to actions), with nothing computed twice, and the actions stay what they are.

The policy server hands the numbers out with every reply, as `reply["policy_state"]` (`vla_memory/taps/reply.py`): a few
named summaries, `{"names": [...], "values": (n, width) float16}`. A name says which block and which tokens:
`"block12:prompt"` is block 12 (counted from 0, as in the code), averaged over the prompt's words. The head runs in the proxy and takes the summary it was
fitted on, by its name.

The framework has one interface for both kinds of models (`vla_memory/taps/`):

- **PyTorch** (`TorchStateTap`): a **forward hook**, a function PyTorch calls with a layer's output, on a layer chosen by
  its name, for example `"model.layers.12"`. The hook only reads the output.
- **JAX** (`JaxStateTap`): JAX has no hooks. A JAX model is a function, and only what it returns is visible, so the model's
  own pass has to return the layer's state and hand it to the tap. For flax linen models, `linen_tapped` does this for any
  submodule. openpi's π0.5 runs its blocks inside a loop (a scan) in which a block is not a submodule of its own, so
  `integrations/openpi/openpi_tap.py` takes the state from a copy of openpi's backbone that also returns it
  (`integrations/openpi/builtin_memory/pooling.py`, the same code the built-in memory reads its input with).

Either way, a tap turns the layer's output, one vector per token, into one vector per call: the mean over the valid tokens
of a span (for π0.5 the prompt's tokens, which come after the 3 x 256 picture tokens), or the last valid token. "Valid"
leaves out the padding that fills the prompt to its fixed length. With `trainable=True` a PyTorch tap keeps the gradient,
so that a head on it could be trained together with the policy; the head writer does not need that, its head is fitted
afterwards on the unchanged policy.

**The automatic tap** (`AutoTap`, PyTorch, `vla_memory/taps/auto.py`). `TorchStateTap` has to be told which layer to read
and which tokens to average, so someone has to know the model's insides. `AutoTap` finds both by itself in the policy's
first call, and reads every policy by one rule, the one the openpi server follows for π0.5: **the language backbone two
thirds of the way up, averaged over the prompt's words** (`"block12:prompt"` for π0.5, `"block11:prompt"` for SmolVLA). So
the head writer works the same way with every server.

- **The language backbone.** A transformer keeps its blocks in a list of at least 4 modules of one class, and a VLA has
  several such stacks: a vision encoder, the language backbone, often an action expert. The language backbone is as wide
  as the model's **word table** (its largest table of token vectors, an `nn.Embedding`), and it reads the pictures and the
  words together, so it sees many tokens. The tap keeps the stack of that width that saw the most tokens (if no stack has
  that width, the stack with the most tokens).
- **The block and the words.** It reads the block two thirds of the way up the stack. It finds the words through the word
  table: the vectors the table gives out in a call show up again in the backbone's input, at the prompt's places (compared
  by their direction, since Gemma scales them). The attention mask the block gets marks the padding, and the tap averages
  the block's output over the remaining words.
- **Two ways to read a block.** Normally with a forward hook on the block. Some models call a block's parts one by one in
  a loop of their own, so that this hook never fires. Then the tap reads the input of the next block's first norm (the
  normalisation layer a block starts with), which holds the same numbers; `tap.how` says which way it uses. The norm gets no
  attention mask, so the tap takes the mask the model last gave to a module on the way to the stack. A policy that runs its
  backbone several times in one call (an autoregressive policy, once per new token) is read in the pass with the most
  tokens.

If the backbone's input shows no word in a call (a model that adds position vectors to the words first), the tap hands out
the last real token instead, named so (`"block12:last"`), and warns the first time. `blocks=[...]` asks for more blocks,
and `kinds=("prompt", "last", "mean")` for the other two summaries: the last real token and the mean over all real tokens.

**What the automatic tap cannot do:** reading at the next norm assumes blocks that start with their norm, as recent
language models do. A stack whose blocks are of different classes is not found; name it with `stack=`. The words are found
only where the backbone's input holds the word table's vectors unchanged, up to their length.

## Prerequisites

- **PyTorch models:** `torch`. For `TorchStateTap` the tapped layer must be called through its own forward: a hook on a
  layer whose parts a custom loop calls one by one never fires (the automatic tap reads such a layer at the next layer's
  norm).
- **JAX models:** `jax`, and `flax` for `linen_tapped`. A model in another JAX library has to return the layer's state
  itself.
- **openpi's π0.5:** openpi's environment, with the framework and `integrations/openpi` on `PYTHONPATH`. The PyTorch
  version of π0.5 also needs openpi's patch of `transformers` 4.53.2 (openpi's README) and a converted checkpoint:

  ```
  python integrations/openpi/convert_to_pytorch.py --checkpoint <JAX checkpoint dir> --config <openpi config> --out <dir>
  ```

  This is openpi's converter with one step added. openpi's converter ignores LoRA weights (the small extra matrices a LoRA
  fine-tune trains), so a LoRA fine-tune would silently lose them; here they are merged into the base weights first.
- **LeRobot policies:** a LeRobot environment with `websockets` and `msgpack` (`integrations/lerobot/README.md`).

## How to use it in the framework

With the head writer, the policy server hands out the state, and the proxy does the rest:

1. **Record the policy's own runs** while another writer runs the robot: the proxy's `--record-head DIR` keeps every
   call's state with the writer's count of completed skills (`vla_memory/writers/head/record.py`).
2. **Fit the head** on the CPU: `python training/fit_event_head.py --recordings 'DIR/*.npz' --out <head dir>`. It tries
   every summary the server hands out and keeps the one with the best cross-validated score.
3. **Run with the head:** `python -m vla_memory.proxy --writer head --head-dir <head dir> --method <method> ...`.

The policy servers that hand out the state:

- **openpi's π0.5,** JAX or PyTorch: `integrations/openpi/openpi_server.py`, with `openpi_tap.py`:

  ```
  python -m openpi_server --port <port> --config <openpi config> --checkpoint <checkpoint dir> [--center-crop 0.95] \
      [--layer 12] [--pool text]
  ```

- **Any LeRobot policy:** `integrations/lerobot/lerobot_server.py`, with the automatic tap:

  ```
  python -m lerobot_server --policy <hub id or folder> --port <port>
  ```

- **Another policy:** its own server adds `reply["policy_state"] = state_entry(names, values)` to every reply
  (`vla_memory/taps/reply.py`), with one of the taps below. `AutoTap` gives the names (`tap.names`). With `TorchStateTap`
  or `JaxStateTap`, name the summary yourself by the same rule, for example `state_entry(["block12:prompt"], tap.read()[:1])`.

The automatic tap, on any PyTorch policy:

```python
from vla_memory.taps.auto import AutoTap

tap = AutoTap(model)            # the policy's torch module; stack=, blocks= and kinds= choose by hand
tap.start_call()                # before each call
actions = policy(...)           # the policy's own forward pass, unchanged
values = tap.read()             # (1, width) float32, named in tap.names: ["block12:prompt"] for π0.5
```

A tap on one chosen layer of a PyTorch model:

```python
from vla_memory.taps.torch_tap import TorchStateTap

tap = TorchStateTap(model, "model.layers.12", start=768)    # the mean over the tokens from position 768 on
tap.valid = token_mask                                      # (batch, tokens), False for padding
outputs = model(...)                                        # the model's own forward pass
features = tap.read()                                       # (batch, width) float32
tap.close()                                                 # removes the hook
```

A flax linen model:

```python
from vla_memory.taps.jax_tap import JaxStateTap, linen_tapped

run = jax.jit(linen_tapped(model, "layers_12"))             # the model's apply, plus the output of layers_12
outputs, hidden = run(variables, inputs)
tap = JaxStateTap(start=768)
tap.valid = token_mask
features = tap.capture(hidden)
```

openpi's π0.5, JAX or PyTorch (the policy as `policy_config.create_trained_policy` returns it):

```python
from openpi_tap import attach

tap = attach(policy, layer=12)          # after block 12, the mean over the prompt's valid tokens
outputs = policy.infer(observation)     # the actions as without the tap
feature = tap.read()[0]                 # (2048,)
```

## Our findings

We checked the taps with our history policy (checkpoint 28,000) on 1,442 recorded policy calls of the event head's test
runs: every call with its real prompt, plus the 3 calls after each new history line with the old prompt, where the head
must say "done" (192 calls). Every variant got the same observations and the same noise.

**openpi's JAX π0.5 (the original):**

- **The head's input** from the policy's own pass is bit for bit that of the old second pass at batch 1. The reference head
  was fitted on the second pass at batch 8, and the two differ by rounding (a median 0.16 % of the feature, a mean 0.086 in
  the score): 2 of 1,442 verdicts flipped, both close to the threshold.
- **The actions** stay the same up to rounding. The changed program compiles a little differently, and the first action of
  a chunk moved by at most 0.0032 rad.
- **Time per call,** the policy alone on one RTX 3090: 165 ms without the tap, 157 ms with it. The old second pass added
  524 ms.

**openpi's PyTorch port of π0.5** (our policy, converted):

- **The hook changes nothing:** with and without the tap the actions were identical in 1,442 of 1,442 calls.
- **In a compiled model** (openpi compiles `sample_actions` with `torch.compile`, PyTorch's compiler) the first version of
  the tap captured nothing: code compiled before the hook was attached skipped it. The tap now drops the compiled code and
  switches on the compiler's check for hooks, so it captures at every call; the actions then move by rounding (at most
  0.018).
- **Time per call,** the policy alone: 353 ms without the tap and 354 ms with it; compiled, 114 ms and 126 ms. The first
  call after attaching compiles once more (28 s).
- **openpi's converter dropped our LoRA weights.** Converted as it is, the PyTorch policy's features were 8 % away from the
  JAX policy's, and 50 of 1,442 verdicts flipped. With the LoRA weights merged first (`convert_to_pytorch.py`), the
  features are 3.1 % away and 17 verdicts flip. openpi's PyTorch port is close to the JAX policy but does not give the same
  numbers, so fit a head on recordings of the policy it will be served with.

**The automatic tap on openpi's PyTorch π0.5** (the same policy and calls):

- **It picked the language model:** as wide as the word table (2,048) and 968 tokens in a call (three picture slots of
  256 tokens and the prompt's 200), over the vision encoder (256 tokens, one picture at a time, width 1,152) and the action
  expert (10 tokens, width 1,024). It reads block 12 of 18 with the block hook.
- **It gives exactly the head's feature:** its `block12:prompt` was bit for bit `openpi_tap`'s average over the prompt's
  tokens in 60 of 60 calls. It found the word table's vectors at all 200 prompt places and at none of the 768 picture
  tokens; the mask left the prompt's 27 to 41 real tokens.
- **The hooks change nothing:** the actions were identical with and without the tap in every call, and a call took 360 ms
  either way. Compiled (measured with the first version of the tap), a call took 127 ms against 114 ms without it, and the
  actions moved by rounding (at most 0.0045).
- **The next-norm way works:** openpi's training forward is a loop that calls the blocks' parts itself. There the tap read
  the next blocks' norms and got the same blocks as in the policy's call, a median 0.2 to 1 % apart.
