"""The PyTorch state tap: a forward hook on one module, chosen by name.

    tap = TorchStateTap(model, "model.layers.12", start=768)    # the prompt's tokens after the 768 picture tokens
    tap.valid = mask; model(...); features = tap.read()

The hook only reads, so the model computes what it computes without it. It also works in a torch.compile'd model: the hook
itself runs as plain Python, and attaching a tap drops the code torch.compile has compiled so far and turns on its check for
hooks (skipped by default), so that compiled code recompiles once with the hook instead of skipping it. Two more details: a
module that is not called through its forward (a layer whose parts a custom loop calls one by one) never fires its hook;
and a few modules take a fused fast path only while no hook is registered (nn.TransformerEncoderLayer in eval mode), so
there the outputs move by rounding. trainable=True keeps the gradient, for a head trained together with the model.
"""
from __future__ import annotations

import torch

from .base import StateTap


class TorchStateTap(StateTap):
    """A forward hook on model.get_submodule(module) that pools the module's output (its first entry if it is a tuple)."""

    def __init__(self, model: torch.nn.Module, module: str, start: int = 0, stop: int | None = None, kind: str = "mean",
                 trainable: bool = False):
        super().__init__(start, stop, kind)
        self.trainable = trainable
        torch._dynamo.config.skip_nnmodule_hook_guards = False      # compiled code checks its hooks (module docstring)
        torch.compiler.reset()
        self._handle = model.get_submodule(module).register_forward_hook(self._hook)

    @torch.compiler.disable          # runs as plain Python inside a torch.compile'd model (openpi compiles sample_actions)
    def _hook(self, module, inputs, output) -> None:
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        weights = torch.as_tensor(self.weights(*hidden.shape[:2]), device=hidden.device)
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            self.value = torch.einsum("bs,bsw->bw", weights, hidden.float())

    def read(self):
        if self.value is None:
            return super().read()                   # raises: nothing captured
        return self.value.detach().float().cpu().numpy()

    def close(self) -> None:
        self._handle.remove()
