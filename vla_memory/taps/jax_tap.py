"""The JAX state tap. JAX has no hooks: a model is a function, and only what it returns is visible. So the model's own pass
returns the layer's state, and the tap pools it (capture) or takes it already pooled (tap.value = ...).

  flax linen     linen_tapped(module, path) is the module's own apply that also returns the output of the submodule at
                 `path` ("layers_12", "encoder/layers_12"), through flax's capture_intermediates; it can be jitted.
  other models   the model is changed to return the state. A layer inside a scan is not a submodule of its own: openpi's
                 Gemma runs its blocks in one, and integrations/openpi/builtin_memory/pooling.py returns their pooled states
                 from the same pass (integrations/openpi/openpi_tap.py hands them to a tap).
"""
from __future__ import annotations

import jax.numpy as jnp

from .base import StateTap


class JaxStateTap(StateTap):
    """A tap that the model's pass hands its state to."""

    def capture(self, hidden):
        """Pool hidden (B, S, W) and keep it; returns the pooled (B, W)."""
        self.value = jnp.einsum("bs,bsw->bw", self.weights(*hidden.shape[:2]), hidden.astype(jnp.float32))
        return self.value


def linen_tapped(module, path: str):
    """(variables, *args, **kwargs) -> (the module's outputs, the output of the submodule at `path`) for a flax linen module."""
    names = tuple(part for part in path.split("/") if part)

    def wanted(submodule, method_name) -> bool:
        return method_name == "__call__" and submodule.scope is not None and submodule.scope.path[-len(names):] == names

    def run(variables, *args, **kwargs):
        outputs, state = module.apply(variables, *args, capture_intermediates=wanted, mutable=["intermediates"], **kwargs)
        node = state["intermediates"]
        for name in names:
            node = node[name]
        hidden = node["__call__"][0]
        return outputs, hidden[0] if isinstance(hidden, tuple) else hidden

    return run
