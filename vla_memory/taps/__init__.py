"""State taps: one layer's state of a model, read during the model's own forward pass, for PyTorch and JAX models alike. A
head on the tapped state needs no second pass.

  base.pool_weights         the mean (or last token) over the valid tokens of a span, as weights every backend uses
  base.StateTap             the interface: valid, value, read(), close()
  torch_tap.TorchStateTap   PyTorch: a forward hook on a module chosen by name; frozen, or trainable with the model
  jax_tap.JaxStateTap       JAX: the model's pass hands its state over; flax linen models via jax_tap.linen_tapped
  auto.AutoTap              PyTorch, without knowing the model: its language backbone 2/3 up, averaged over the words

The backends import torch or jax: import them from their modules.
"""
from .base import StateTap, pool_weights

__all__ = ["StateTap", "pool_weights"]
