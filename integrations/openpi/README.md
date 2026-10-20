# The built-in memory on openpi

The reference implementation of the built-in memory ([docs/methods/builtin_memory](../../docs/methods/builtin_memory/README.md))
for pi0.5 on stock openpi, pinned to openpi commit `215abfb`. It extends openpi from the outside, by subclasses and
composition: no file of an openpi checkout is changed, and nothing in `vla_memory` imports this folder, so `import vla_memory`
needs no jax. The package `builtin_memory` runs in openpi's environment with this folder and the framework on the path:

```
PYTHONPATH=<openpi>/src:<openpi>/packages/openpi-client/src:<framework>:<framework>/integrations/openpi
```

## What it adds to openpi

| file | what it does | what of openpi it builds on |
|---|---|---|
| `pooling.py` | the memory's input: the residual stream after block 12, averaged over the text, the exterior picture and the wrist picture, from the prefix pass that also fills the KV cache | `gemma.Block` (subclass); `gemma.Module` (subclass; `setup` and `__call__` copied to pass the pooling mask into the scanned blocks) |
| `model.py` | the model: the GRU (state 256), its zero-initialised read-out into the action expert's adaRMS conditioning, the optional phase head; the observation with the call history | `Pi0Config`, `Pi0`, `Observation` (subclasses); `compute_loss` and `sample_actions` copied, since their suffix embedding cannot take the memory's conditioning otherwise |
| `policy.py` | the stateful policy: one state per episode, reset at the episode markers, `reply["memory"]` | `Policy` (subclass; `infer` copied with the state); the JAX part of `create_trained_policy` copied, which restores every parameter in bfloat16, while the memory stays float32 |
| `serve.py` | the policy server | `WebsocketPolicyServer` |
| `sequences.py` | the call histories of the training samples from a feature cache (numpy only) | |
| `data.py` | the training data: several LeRobot datasets in a row, the frame filter, the call histories | `create_torch_dataset`, `transform_dataset`, `TorchDataLoader`, `DataLoaderImpl` (subclass), `DataConfigFactory` |
| `training.py` | the frozen backbone, the memory's 10x learning rate, the weight loader for checkpoints without the memory | `OptimizerConfig`, `CheckpointWeightLoader` (subclass) |
| `config.py` | the reference recipe: the model, the task's data config, the training, the serving config | `LeRobotLiberoDataConfig` (subclass: 8 action numbers), `TrainConfig` |
| `train.py` | the training launcher | `scripts/train.py`: `init_train_state`, `train_step`, the checkpoints; its `main()` copied with the data loader replaced, since the stock one reads one dataset and drops the call histories |

The parameters are named as in the checkpoints of the measured results (`memory_*`), which load unchanged. On the same
observations and random numbers the code gives the same actions, memory states and phase probabilities as the
implementation the results were measured with, to the bit when both run the same GPU kernels (XLA picks them per process),
across episode boundaries and with the memory cleared before every call; in training it gives the same batches, loss and
gradients, and its training run logs the numbers of the original one.

## Serving

```
python -m builtin_memory.serve --checkpoint <checkpoint dir> --port <policy port> --center-crop 0.95 [--reset-every-call]
python -m vla_memory.proxy --writer off --listen-port <port> --upstream-port <policy port> [--memory-reset every-call]
```

The policy keeps its state per episode. It resets the state to zero at `memory/new_episode` (True at the first call of an
episode) or at a new `memory/episode`; the environment sends both (`vla_memory.env_keys.ControlKeys`) and the proxy forwards
them unchanged. `--memory-reset every-call` (proxy) and `--reset-every-call` (server) clear the state before every call: the
test of whether the policy uses its memory. Every reply carries `memory`: the episode, the call, whether it reset, the
state's norm and, with a phase head, the phase and its probabilities. The checkpoint's phase head and norm stats are read
from it; `--center-crop 0.95` is the centre crop of the exterior picture that the reference policies run at inference.

## Training

```
python -m builtin_memory.train --exp-name <name> --init-checkpoint <checkpoint dir> \
    --dataset <demonstrations repo id> <its feature index> --dataset <interventions repo id> <its feature index> \
    [--phases 4] [--feature-stats <stats.npz>] [--steps 10000] [--checkpoint-dir <dir>] [--overwrite | --resume] [--no-wandb]
```

It trains from `--init-checkpoint` (its weights and its norm stats) on the datasets in the given order, uniformly over their
frames: 10,000 steps at batch 16, the backbone frozen; the action expert's LoRA, the action and time projections, the memory
and the phase head train; cosine schedule from 1e-5 to 1e-6 after 200 warm-up steps, the memory at ten times that; call
histories of at most 96 calls. A checkpoint without the memory (the memoryless policy) needs `--feature-stats`, an `.npz`
with the features' `mean` and `std` (6,144 numbers each) for the memory's fixed z-scoring; a checkpoint with the memory
brings its own (a further training cycle).

Every dataset needs a feature cache: one row per recorded step with the policy's own layer-12 features, as `pooling.py`
computes them in the prefix pass at run time, and an index `.npz` with the episodes' rows and phase labels. The format,
including the mapping of split episodes to the recorded steps of their run, is in `sequences.py`. The task's own transforms
come from `config.RedBlueDataConfig`; another robot brings its own `DataConfigFactory` there.
