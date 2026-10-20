"""The VLM planner as a writer: it plans once per episode and monitors every policy call (planner.Planner), with the
exemplar store as its long-term memory, and moves its pointer before the policy call, so the prompt of the same call shows
it. Other memories of the planner plug in as extensions (their planner hooks, and before_decision / after_decision
around every monitor call).

    --writer vlm [--vlm-url URL] [--vlm-model NAME] [--memory store.npz] [--k N] [--no-gate] [--votes N] ...  (options.py)

The proxy's log record of a call: queried, advanced, vlm_ms (the question's time), skill_index, vlm (the answer),
recovery (a put-down skill inserted before a pick).
"""
from __future__ import annotations

from typing import Sequence

from ..base import Call, Writer
from .options import add_model_options, add_planner_options, build_memory, build_vlm, log_planner_configuration
from .planner import Planner
from ...proxy.extensions import memory_from
from ...vlm import resolve_vlm_backend


class VlmWriter(Planner, Writer):
    """The Planner behind the writer interface: before_call hands it the pictures and the proprioception of the call."""

    name = "vlm"
    settings: dict | None = None    # the planning model, the exemplar store and k, for the proxy's metadata (set by build)

    @classmethod
    def add_arguments(cls, parser):
        """The planning model's and the planner's options (options.py)."""
        add_model_options(parser)
        add_planner_options(parser)

    @classmethod
    def build(cls, args, task, extensions: Sequence = ()):
        """The writer with the model client, the memory store (an extension's or --memory) and the decision rule."""
        if args.recovery and args.verdict_only:
            raise ValueError("--recovery reads the object states from the monitor's answer: add --no-verdict-only")
        if (args.vlm_backend or "openai") != "openai":
            resolve_vlm_backend(args.vlm_backend)       # an unknown backend is a usage error before anything is built
        store = build_memory(args, task, extensions)
        writer = cls(build_vlm(args), task, memory=store, k=args.k, votes_to_advance=args.votes, call_every=args.call_every,
                     min_calls_per_skill=args.min_calls, recovery=args.recovery, use_wrist=not args.no_wrist,
                     upscale=args.upscale, brief=args.brief, gate=args.gate, k_vote=args.k_vote,
                     gate_min_done=args.gate_min_done, verdict_only=args.verdict_only,
                     log=(args.log + ".planner.jsonl") if args.log else None,
                     log_memory=memory_from(extensions) is not None, extensions=extensions)
        writer.settings = {"vlm": args.vlm_model, "memory": args.memory, "k": args.k}
        log_planner_configuration(args, store)
        return writer

    def metadata(self) -> dict:
        """The planning model (--vlm-model), the exemplar store file (--memory) and the examples per question (--k)."""
        return dict(self.settings or {})

    def before_call(self, call: Call, extensions: Sequence = ()) -> dict:
        """One policy call: observe the pictures and the proprioception, with the extensions' hooks around the decision."""
        return observe_call(self, call, extensions)


def observe_call(planner, call: Call, extensions: Sequence = ()) -> dict:
    """One policy call of a planner (anything with observe(images, step, proprio) and the pointer): it looks at the pictures
    and the proprioception and keeps or advances its pointer; the extensions' hooks run around it. Returns the log fields."""
    images = call.images()
    proprio = {key: float(call.control_keys[key]) for key in ("gripper_closedness", "hand_height") if key in call.control_keys}
    for extension in extensions:
        extension.before_decision(call.control_keys, images, planner)
    record = planner.observe(images, call.step, proprio=proprio or None)
    for extension in extensions:
        extension.after_decision(call.control_keys, images, record, planner)
    return dict(queried=record["queried"], advanced=record["advanced"], vlm_ms=record.get("ms", 0.0),
                skill_index=planner.skill_index, vlm=record.get("vlm"), recovery=record.get("recovery_inserted"))
