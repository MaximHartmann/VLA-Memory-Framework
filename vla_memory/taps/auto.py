"""The automatic state tap: one rule for every PyTorch policy, without knowing the model's insides.

    tap = AutoTap(model)                    # the policy's torch module, before its first call
    tap.start_call()                        # before each call (optional: a read() also ends a call)
    actions = policy(...)                   # the policy's own forward pass; its outputs stay the same
    values = tap.read()                     # (1, width) float32, named in tap.names: "block12:prompt" for pi0.5

The rule, as the head writer's openpi server reads pi0.5 (block 12 of 18, the prompt's tokens): the language backbone two
thirds of the way up, averaged over the prompt's words. A VLA keeps the blocks of its vision encoder, language backbone and
action expert in stacks (nn.ModuleLists of modules of one class); the language backbone is the stack as wide as the word
table (the largest nn.Embedding) that sees the most tokens in the first call (else the one with the most tokens; stack=...
names one). Its words are the positions of its input whose vector, up to its length, the word table gave out in this call
(cosine 0.999: Gemma scales its word vectors, Llama does not), less the padding: what no query may attend to by the
attention mask the block gets. Without words (a model that adds position vectors to them first) a call gives the last real
token, named so ("block12:last"), with a warning the first time. blocks=[...] and kinds=("prompt", "last", "mean") ask for
more.

A block is read by a forward hook or, if that never fires (a model whose own loop calls a block's parts one by one), by a
pre-hook on the next block's first norm, whose input is the same residual stream (tap.how): for pre-norm blocks, with the
attention mask last given to a module on the way to the stack, whose forwards are wrapped (SmolVLA calls its backbone's
forward directly, past any hook). A backbone run several times in a call (an autoregressive policy) is read in its pass
with the most tokens. The hooks only read; in a torch.compile'd model they work as TorchStateTap's (torch_tap.py), and the
code compiled around the first call's many hooks is dropped, so the next call compiles once more.
"""
from __future__ import annotations

import functools
import itertools
import logging

import numpy as np
import torch

from .base import pool_weights

log = logging.getLogger("vla_memory.taps")
KINDS = ("prompt", "last", "mean")      # the summaries: the mean over the real words, the last real token, all real tokens
DEPTH = 2 / 3                           # the default block, as a fraction of the stack's depth
WORD = 0.999                            # the cosine from which a vector of the stack's input is a word's


def block_stacks(model: torch.nn.Module) -> dict:
    """{name: nn.ModuleList} of every list of at least 4 modules of one class: the model's stacks of transformer blocks."""
    return {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.ModuleList)
            and len(module) >= 4 and len({type(block) for block in module}) == 1}


def word_table(model: torch.nn.Module):
    """The model's word table: its nn.Embedding with the most entries, or None."""
    embeddings = [module for module in model.modules() if isinstance(module, torch.nn.Embedding)]
    return max(embeddings, key=lambda module: module.num_embeddings) if embeddings else None


def first_norm(modules):
    """The first of the modules whose class name contains "Norm" (input_layernorm, ln_1, layer_norm1), or None."""
    return next((module for module in modules if "Norm" in type(module).__name__), None)


def final_norm(model: torch.nn.Module, stack: str):
    """The first norm after the stack among its parent's children (the backbone's final norm), or None."""
    parent, _, name = stack.rpartition(".")
    after = itertools.dropwhile(lambda child: child[0] != name, model.get_submodule(parent).named_children())
    return first_norm(module for _, module in after)


def candidate_blocks(depth: int, blocks=None) -> list[int]:
    """The blocks to read in a stack of `depth` blocks: those of `blocks` it has, by default the one 2/3 of the way up."""
    wanted = [round(depth * DEPTH)] if blocks is None else blocks
    return sorted({index for index in wanted if 0 <= index < depth})


def hidden_of(args, kwargs, output=None):
    """The (batch, tokens, width) tensor a block returns (a tuple's first entry) or, for a pre-hook, its first input."""
    if output is None:
        output = next((value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)), None)
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    return hidden if isinstance(hidden, torch.Tensor) and hidden.dim() == 3 else None


def real_tokens(mask, batch: int, length: int):
    """(batch, length) bool from an attention mask, True where some query may attend; None without a usable mask.
    A bool mask is True there, a (batch, keys) mask 1, a float mask added to the scores 0 (blocked: very negative). With a
    cache, the pass's own tokens are the last `length` keys."""
    if not (isinstance(mask, torch.Tensor) and mask.dim() > 1 and mask.shape[0] in (1, batch) and mask.shape[-1] >= length):
        return None
    if mask.dtype != torch.bool:
        mask = mask > 0 if mask.dim() == 2 else mask > -1
    mask = mask[..., -length:]
    return (mask if mask.dim() == 2 else mask.flatten(1, -2).any(1)).expand(batch, length).cpu().numpy()


def word_positions(hidden: torch.Tensor, words: list) -> np.ndarray:
    """(batch, tokens) bool: the positions of hidden whose vector is, up to its length, one of the word vectors `words`."""
    table = torch.cat([vectors.reshape(-1, hidden.shape[-1]) for vectors in words]).float()
    cosine = torch.nn.functional.normalize(hidden.detach().float(), dim=-1) @ torch.nn.functional.normalize(table, dim=-1).T
    return cosine.amax(-1).ge(WORD).cpu().numpy()


class AutoTap:
    """A PyTorch model's language backbone after a block, averaged over the prompt's words, read during its own pass.

    width: the word table's; tokens, widths: what each stack's first block saw in the first call; stack, blocks, how: set
    when read() or start_call() follows that call; names: what read() gave, "block<index>:<kind>". read() gives the batch's
    first row (servers run batch 1)."""

    def __init__(self, model: torch.nn.Module, stack: str | None = None, blocks: list[int] | None = None,
                 kinds: tuple = ("prompt",)):
        if not set(kinds) <= set(KINDS):
            raise ValueError(f"kinds {kinds}: of {KINDS}")
        torch._dynamo.config.skip_nnmodule_hook_guards = False      # compiled code checks its hooks, as in TorchStateTap
        torch.compiler.reset()
        stacks = block_stacks(model) if stack is None else {stack: model.get_submodule(stack)}
        if not stacks:
            raise ValueError("the model has no stack of blocks (an nn.ModuleList of at least 4 modules of one class)")
        table = word_table(model)
        self.width, self.kinds = (None if table is None else table.embedding_dim), tuple(kinds)
        self.tokens, self.widths = dict.fromkeys(stacks, 0), {}
        self.names, self.stack, self.blocks, self.how = [], None, [], None
        self._kept, self._read = {}, False      # (stack, block, way) -> (tokens, kinds read, (k, width)) of the longest pass
        self._masks = {}                        # stack -> the attention mask last given on the way to it
        self._words, self._is_word = [], {}     # this call's word vectors; stack -> the word positions of its latest pass
        self._hooks = {}                        # (stack, block, way) -> hook handle; block None: the stack's input
        self._wrapped = {}                      # module -> (its own forward attribute or None, the stacks below it)
        self._warned = False
        if table is not None and "prompt" in self.kinds:
            self._hooks[None, None, "words"] = table.register_forward_hook(self._word_table)
        for name, layers in stacks.items():
            entry = first_norm(layers[0].children()) or layers[0]
            self._hooks[name, None, "input"] = entry.register_forward_pre_hook(self._input(name), with_kwargs=True)
            for index in candidate_blocks(len(layers), blocks):
                self._hooks[name, index, "block"] = layers[index].register_forward_hook(
                    self._capture(name, index, "block"), with_kwargs=True)
                norm = first_norm(layers[index + 1].children()) if index + 1 < len(layers) else final_norm(model, name)
                if norm is not None:
                    self._hooks[name, index, "next norm"] = norm.register_forward_pre_hook(
                        self._capture(name, index, "next norm"), with_kwargs=True)
            parts = name.split(".")
            for depth in range(len(parts)):                 # the modules on the way to the stack
                self._wrap(model.get_submodule(".".join(parts[:depth])), name)

    def _wrap(self, module: torch.nn.Module, stack: str) -> None:
        """Wrap the module's forward to note the attention mask it is given for the stack (once per module)."""
        if module not in self._wrapped:
            self._wrapped[module] = (vars(module).get("forward"), set())
            noting = functools.partial(self._noting, self._wrapped[module][1], module.forward)
            module.forward = functools.wraps(module.forward)(noting)
        self._wrapped[module][1].add(stack)

    def _noting(self, stacks: set, forward, *args, **kwargs):
        """A wrapped forward: note the attention mask for the stacks below, then run the module's own forward."""
        if isinstance(kwargs.get("attention_mask"), torch.Tensor):
            self._masks.update(dict.fromkeys(stacks, kwargs["attention_mask"]))
        return forward(*args, **kwargs)

    def _unwrap(self, module: torch.nn.Module) -> None:
        """Give the module its own forward back."""
        forward = self._wrapped.pop(module)[0]
        if forward is None:
            del module.forward
        else:
            module.forward = forward

    def _new_call(self) -> None:
        """At the first hook after a read(): the call before is over."""
        if self._read:
            self._kept, self._words, self._read = {}, [], False

    @torch.compiler.disable
    def _word_table(self, module, args, output) -> None:
        """A forward hook on the word table: the word vectors of this call."""
        self._new_call()
        self._words.append(output.detach())

    def _input(self, stack: str):
        """A pre-hook on the stack's first block: its width and tokens, and which positions of the pass are words."""
        @torch.compiler.disable
        def enter(module, args, kwargs):
            hidden = hidden_of(args, kwargs)
            if hidden is not None:
                self._new_call()
                self.tokens[stack] = max(self.tokens[stack], hidden.shape[1])
                self.widths[stack] = hidden.shape[2]
                usable = self._words and hidden.shape[2] == self.width
                self._is_word[stack] = word_positions(hidden, self._words) if usable else None
        return enter

    def _capture(self, stack: str, index: int, way: str):
        """A hook that pools block `index`'s output when the pass has more tokens than the call's earlier ones."""
        @torch.compiler.disable          # runs as plain Python inside a torch.compile'd model
        def capture(module, args, kwargs, output=None):
            hidden = hidden_of(args, kwargs, output)
            if hidden is None:
                return
            self._new_call()
            key = (stack, index, way)
            if key in self._kept and self._kept[key][0] >= hidden.shape[1]:
                return
            batch, length = hidden.shape[:2]
            mask = kwargs.get("attention_mask") if way == "block" else self._masks.get(stack)
            kinds, weights = self._weights(stack, real_tokens(mask, batch, length), batch, length)
            pooled = torch.einsum("kbs,bsw->kbw", torch.as_tensor(weights, device=hidden.device), hidden.detach().float())
            self._kept[key] = (length, kinds, pooled[:, 0])
        return capture

    def _weights(self, stack: str, real, batch: int, length: int):
        """(the kinds read, their (k, batch, length) pooling weights); "prompt" without real words becomes "last"."""
        kinds, weights = [], []
        for kind in self.kinds:
            valid = real
            if kind == "prompt":
                words = self._is_word.get(stack)
                words = words if words is not None and words.shape == (batch, length) else np.zeros((batch, length), bool)
                valid = words if real is None else words & real
                kind, valid = ("prompt", valid) if valid.any() else ("last", real)
            kinds.append(kind)
            weights.append(pool_weights(valid, batch, length, kind="mean" if kind == "prompt" else kind))
        return tuple(kinds), np.stack(weights)

    def _settle(self) -> None:
        """After the first call: keep the stack and the way that fired there; remove the other hooks and wrappers."""
        same = [name for name in self.tokens if self.tokens[name] and self.widths.get(name) == self.width]
        stack = max(same or self.tokens, key=self.tokens.get)
        fired = {(index, way) for name, index, way in self._kept if name == stack}
        how = "block" if any(way == "block" for _, way in fired) else "next norm"
        blocks = sorted(index for index, way in fired if way == how)
        if not blocks:
            raise RuntimeError("the tap has read no block yet: did the model run?")
        self.stack, self.how, self.blocks = stack, how, blocks
        keep = {(stack, index, how) for index in blocks}
        if "prompt" in self.kinds:
            keep |= {(stack, None, "input"), (None, None, "words")}
        for key in [key for key in self._hooks if key not in keep]:
            self._hooks.pop(key).remove()
        for module in [module for module, (_, stacks) in self._wrapped.items() if how == "block" or stack not in stacks]:
            self._unwrap(module)
        torch.compiler.reset()      # drop the code compiled around the first call's many hooks: it would run partly eager
        self._kept = {key: value for key, value in self._kept.items() if key in self._hooks}
        self.names = [f"block{index}:{kind}" for index in blocks for kind in self.kinds]

    def start_call(self) -> None:
        """Begin a new call (a server calls it before each inference): the tap forgets the last one."""
        if self.stack is None and any(self.tokens.values()):
            self._settle()
        self._kept, self._read, self._masks, self._words, self._is_word = {}, False, {}, [], {}

    def read(self) -> np.ndarray:
        """(n, width) float32: the last call's summaries, from its pass with the most tokens, in the order of names."""
        if self.stack is None:
            self._settle()
        rows = [self._kept.get((self.stack, index, self.how)) for index in self.blocks]
        if any(row is None for row in rows):
            raise RuntimeError("the tap has captured nothing in this call: did the model run since start_call()?")
        self._read = True
        self.names = [f"block{index}:{kind}" for index, (_, kinds, _) in zip(self.blocks, rows) for kind in kinds]
        if not self._warned and any(kinds != self.kinds for _, kinds, _ in rows):
            log.warning("no prompt words found in the input of %s: its last real token instead", self.stack)
            self._warned = True
        return torch.cat([pooled for _, _, pooled in rows]).cpu().numpy()

    def close(self) -> None:
        """Remove the tap's hooks and wrappers from the model."""
        for handle in self._hooks.values():
            handle.remove()
        for module in list(self._wrapped):
            self._unwrap(module)
        self._hooks = {}
