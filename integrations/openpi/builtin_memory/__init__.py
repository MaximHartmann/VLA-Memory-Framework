"""The built-in memory for pi0.5 on stock openpi: a recurrent state inside the policy (docs/methods/builtin_memory).

  pooling       PoolingModule: openpi's Gemma module whose prefix pass also returns the pooled input of every block
  model         MemoryPi0Config, MemoryPi0: a GRU over the policy's own layer-12 features, its read-out added to the action
                expert's adaRMS conditioning (zero at the start), and an optional phase head; MemoryObservation
  policy        MemoryPolicy: one memory state per episode, reset at the framework's episode markers; create_policy
  serve         python -m builtin_memory.serve: the policy server behind the proxy
  sequences     the call histories of the training samples from a feature cache (numpy only)
  data          the training data: several LeRobot datasets, the frame filter, the call histories
  training      the frozen backbone, the memory's learning rate, the weight loader
  config        the reference recipe (the red/blue task)
  train         python -m builtin_memory.train: the training launcher
Everything but `sequences` needs openpi (commit 215abfb) and jax; nothing in vla_memory imports this package.
"""
