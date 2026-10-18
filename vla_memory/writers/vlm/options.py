"""The VLM writer's command line: the planning model's client and the planner's decision rule, and their builders.

The planner's defaults are the settled configuration of the reference evaluation: the memory-vote gate (with --memory; the
6 nearest stored moments vote, one "done" vote asks the model, the last skill is always asked), a verdict-only answer, 4
examples per question, votes 1, min-calls 2. --no-gate, --no-verdict-only (with --brief: no reason field), --k, --k-vote,
--gate-min-done, --votes and --min-calls change them.

--vlm-backend NAME picks the planning model's client (default openai: OpenAICompatibleVLM on --vlm-url and --vlm-model; or a
registered name, an entry point of vla_memory.vlm_backends, or module:Class), with --vlm-arg KEY=VALUE for its own options
and --vlm-api-key (default: the environment variable VLM_API_KEY).
"""
from __future__ import annotations

import argparse
import logging
from typing import Sequence

from ...fingerprint import fingerprint_from_task
from ...plugins import construct, key_value
from ...proxy.extensions import Extension, memory_from
from ...task import TaskSpec
from ...vlm import OpenAICompatibleVLM, VLMBackend, resolve_api_key, resolve_vlm_backend
from .exemplar import ExemplarStore

log = logging.getLogger("vla_memory.writers.vlm")


def add_model_options(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("the planning model (--writer vlm)")
    group.add_argument("--vlm-url", default="http://127.0.0.1:8100/v1")
    group.add_argument("--vlm-model", default="Qwen3.8-27B-INT4")
    group.add_argument("--vlm-backend", default="openai", metavar="NAME",
                       help="the planning model's client: openai (default, any OpenAI-compatible chat API), a registered "
                            "name, an entry point of vla_memory.vlm_backends, or module:Class (a VLMBackend; gets base_url, "
                            "model, api_key and chat_template_kwargs if it takes them)")
    group.add_argument("--vlm-arg", action="append", type=key_value, default=[], metavar="KEY=VALUE",
                       help="a keyword argument of the backend's constructor, e.g. timeout=60 (repeatable; read as JSON if "
                            "it parses)")
    group.add_argument("--vlm-api-key", default=None, metavar="KEY",
                       help="API key of the VLM server (sent as a bearer token); default: the environment variable "
                            "VLM_API_KEY, which keeps the key out of the process list")
    group.add_argument("--thinking", action="store_true", help="enable the model's reasoning mode (Qwen3 chat template)")
    group.add_argument("--no-chat-template-kwargs", action="store_true", help="for servers that reject chat_template_kwargs")


def add_planner_options(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("the planner's decision rule (--writer vlm)")
    group.add_argument("--memory", default=None, help="exemplar store .npz (scripts/build_exemplar_store.py)")
    group.add_argument("--k", type=int, default=4, help="stored examples shown to the planning model per question")
    group.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True,
                       help="memory-vote gate (with --memory): skip the question when fewer than --gate-min-done of the "
                            "--k-vote nearest stored moments are labelled with a later skill; the last skill of the plan is "
                            "always asked (default: on; --no-gate asks at every monitor call)")
    group.add_argument("--k-vote", type=int, default=6, help="stored moments consulted by the gate (the model sees --k of them)")
    group.add_argument("--gate-min-done", type=int, default=1, help="ask the model from this many 'done' votes on (1 = any)")
    group.add_argument("--verdict-only", action=argparse.BooleanOptionalAction, default=True,
                       help="the monitor answers only current_skill_done (default: on; --no-verdict-only: object states, "
                            "gripper, reason and verdict)")
    group.add_argument("--brief", action="store_true", help="with --no-verdict-only: the state answer without the free-text reason")
    group.add_argument("--votes", type=int, default=1, help="consecutive 'done' answers needed to advance")
    group.add_argument("--min-calls", type=int, default=2, help="a skill cannot end before its N-th monitor call")
    group.add_argument("--call-every", type=int, default=1, help="query the planning model every N policy calls")
    group.add_argument("--recovery", action="store_true", help="put down a wrongly held object first (needs --no-verdict-only)")
    group.add_argument("--no-wrist", action="store_true", help="show the planning model the external camera only")
    group.add_argument("--upscale", type=int, default=1, help="enlarge the pictures by this factor before sending them")


def build_vlm(args) -> VLMBackend:
    """The planning model's client of --vlm-backend with base_url, model, chat_template_kwargs and api_key (if set) as far
    as its constructor takes them, and --vlm-arg. --no-chat-template-kwargs sends none."""
    name = getattr(args, "vlm_backend", None) or "openai"
    factory = OpenAICompatibleVLM if name == "openai" else resolve_vlm_backend(name)
    if args.no_chat_template_kwargs:
        template_kwargs = {}
    else:
        template_kwargs = {"enable_thinking": bool(args.thinking)}
    standard = {"base_url": args.vlm_url, "model": args.vlm_model, "chat_template_kwargs": template_kwargs}
    key = resolve_api_key(getattr(args, "vlm_api_key", None))
    if key:
        standard["api_key"] = key
    return construct(factory, standard, dict(getattr(args, "vlm_arg", None) or []), f"VLM backend {name!r}")


def build_memory(args, task: TaskSpec, extensions: Sequence[Extension] = ()):
    """The planner's long-term memory: an extension's store, else the exemplar store of --memory, else None."""
    store = memory_from(extensions)
    if store is not None:
        return store
    if not args.memory:
        return None
    store = ExemplarStore(fingerprint_from_task(task), task, args.memory)
    log.info("exemplar memory: %d entries from %s", len(store), args.memory)
    return store


def log_planner_configuration(args, store):
    if args.gate and store is not None:
        gate = (f"gate on (the {args.k_vote} nearest stored moments vote; asked from {args.gate_min_done} 'done' vote(s); "
                f"last skill always asked)")
    elif args.gate:
        gate = "gate off (no memory store)"
    else:
        gate = "gate off"
    if args.verdict_only:
        answer = "verdict only"
    elif args.brief:
        answer = "states without reason"
    else:
        answer = "states + reason + verdict"
    log.info("planner: %s; answer: %s; %d example(s) per question, votes %d, min-calls %d, call-every %d",
             gate, answer, args.k if store is not None else 0, args.votes, args.min_calls, args.call_every)
