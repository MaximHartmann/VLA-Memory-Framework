"""The rules author: the planning model writes the rules writer's `history.rules` from the task description.

Code-as-Monitor's idea (Zhou et al., 2024, arXiv 2412.04455) applied to the history writer: a model writes the monitor once,
offline, and the monitor then runs without the model, in real time. Here the monitor is fixed code (rules.GraspCycle); the
model only chooses, per verb of the task, which of the engine's two triggers completes the verb and the trigger's thresholds.
The answer is constrained by a JSON schema (author_prompt.answer_schema); rules_from_answer() converts it to the task file's
metres and checks it with the rules writer; author_yaml.write_task_yaml() writes it into a copy of the task file. The offline
validator (validate.py) then decides whether the rule set is used: a model-written rule set is never trusted without it.

Three variants of the prompt (author_prompt.build_messages): `easy` gives the model the completion criteria of the task
file, `hard` gives statistics of the writer's signals at the label switches of a few labelled demonstrations
(author_demos), `none` gives the skill names and history phrases only.

    python -m vla_memory.history_appended.author --variant easy --out authored.yaml [--task task.yaml]
    python -m vla_memory.history_appended.author --variant hard --demos 'demos/*.npz' --squeeze-force 0.5 --out authored.yaml
    python -m vla_memory.history_appended.validate --task authored.yaml --episodes 'recorded/*.npz'
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys
from typing import Any

import numpy as np

from ..plugins import key_value
from ..task import TaskSpec
from ..vlm import VLMBackend
from .author_demos import demo_event_summary, demo_summary_text, pick_evenly
from .author_prompt import EASY_HEAD, HARD_HEAD, SYSTEM, answer_schema, build_messages
from .author_rules_spec import GRIPPER_DEFAULTS, PARAMS, REASON_MAX, VARIANTS, check_rules, gripper_thresholds, with_rules
from .author_yaml import rules_yaml_lines, write_task_yaml
from .rules import TRIGGERS

__all__ = ["VARIANTS", "PARAMS", "REASON_MAX", "GRIPPER_DEFAULTS", "SYSTEM", "EASY_HEAD", "HARD_HEAD", "gripper_thresholds",
           "build_messages", "answer_schema", "with_rules", "check_rules", "rules_from_answer", "author_rules",
           "rules_yaml_lines", "write_task_yaml", "demo_event_summary", "demo_summary_text", "pick_evenly", "main"]


# ---------------------------------------------------------------------------------------------------------------- answer
def rules_from_answer(answer: Any, task: TaskSpec) -> tuple[dict | None, dict, list[str]]:
    """The model's answer -> (history.rules in the task file's metres, the reason per verb, errors). The rules are None when
    the answer cannot be used: not an object, a verb missing or unknown, an unknown trigger, a required parameter missing,
    not a number or out of bounds (a backend without structured output can return any of these), or a rule set the writer
    rejects (e.g. both verbs on the trigger lift). Optional parameters that are null are left out."""
    if not isinstance(answer, dict):
        return None, {}, [f"the answer is not a JSON object: {answer!r}"[:300]]
    rules = {}
    reasons = {}
    errors: list[str] = []
    for verb in task.verbs:
        item = answer.get(verb)
        if not isinstance(item, dict):
            errors.append(f"no rule for verb {verb!r}")
            continue
        reasons[verb] = str(item.get("reason") or "")
        spec = _rule_from_item(verb, item, errors)
        if spec is not None:
            rules[verb] = spec
    unknown = [key for key in answer if key not in task.verbs]
    if unknown:
        errors.append(f"unknown verb(s) in the answer: {unknown}")
    if not errors:
        try:
            check_rules(task, rules)
        except ValueError as error:
            errors.append(str(error))
    return (None if errors else rules), reasons, errors


def _rule_from_item(verb: str, item: dict, errors: list[str]) -> dict | None:
    """One verb's answer -> its rule in the task file's units; None for an unknown trigger. Problems go to errors."""
    rule = item["rule"] if isinstance(item.get("rule"), dict) else item   # tolerate a flat answer
    trigger = rule.get("trigger")
    if trigger not in PARAMS:
        errors.append(f"{verb!r}: trigger must be one of {TRIGGERS}, got {trigger!r}")
        return None
    spec: dict[str, Any] = {"trigger": trigger}
    for name, parameter in PARAMS[trigger].items():
        value = _parameter_value(verb, trigger, rule, parameter, errors)
        if value is not None:
            spec[name] = value
    return spec


def _parameter_value(verb: str, trigger: str, rule: dict, parameter: tuple, errors: list[str]) -> float | None:
    """One parameter of the answer converted to the task file's metres; None when it is absent or unusable."""
    field, factor, low, high, optional = parameter
    value = rule.get(field)
    if value is None:
        if not optional:
            errors.append(f"{verb!r}: {field} is required for the trigger {trigger}")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        errors.append(f"{verb!r}: {field} must be a number, got {value!r}")
        return None
    if not low <= value <= high:
        errors.append(f"{verb!r}: {field} = {value} is outside [{low:g}, {high:g}]")
        return None
    return round(float(value) * factor, 6)


def author_rules(vlm: VLMBackend, task: TaskSpec, variant: str = "easy", demo_summary: str | None = None,
                 max_tokens: int = 1000, **prompt_kw) -> dict:
    """One authoring request. Returns {"variant", "rules" (None if unusable), "reasons", "errors", "answer" (the model's
    JSON), "text", "ms", "tokens", "messages"}. A failed call is reported in errors, not raised."""
    messages = build_messages(task, variant, demo_summary=demo_summary, **prompt_kw)
    out: dict[str, Any] = {"variant": variant, "rules": None, "reasons": {}, "errors": [], "answer": None, "text": None,
                           "ms": None, "tokens": None, "messages": messages}
    try:
        reply = vlm.chat(messages, schema=answer_schema(task), name="history_rules", max_tokens=max_tokens)
    except Exception as error:  # server down, timeout, a JSON the server could not finish
        out["errors"] = [f"model call failed: {error!r}"]
        return out
    out.update(answer=reply.get("json"), text=reply.get("text"), ms=reply.get("ms"),
               tokens=[reply.get("prompt_tokens"), reply.get("completion_tokens")])
    out["rules"], out["reasons"], out["errors"] = rules_from_answer(reply.get("json"), task)
    return out


# ---------------------------------------------------------------------------------------------------------------- CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=None, help="task YAML (default: the packaged reference task)")
    parser.add_argument("--variant", choices=VARIANTS, default="easy")
    parser.add_argument("--out", required=True, help="where to write the copy of the task file with the authored rules")
    parser.add_argument("--demos", default=None, help="glob of labelled episode .npz files (rows, reference) for --variant hard")
    parser.add_argument("--n-demos", type=int, default=5, help="how many demonstrations, spread evenly over the sorted files")
    parser.add_argument("--squeeze-force", type=float, default=None, help="for the demos: 0.5 if they carry a grasped flag")
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--call-every", type=int, default=None, help="the policy's call period in control steps")
    _add_model_options(parser)
    return parser


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    """The --vlm-* options: which model client answers, and how it is called."""
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--vlm-model", default="Qwen3.8-27B-INT4")
    parser.add_argument("--vlm-backend", default="openai", metavar="NAME",
                        help="the model's client, as in the proxies: openai (default), a registered name, an entry point or module:Class")
    parser.add_argument("--vlm-arg", action="append", type=key_value, default=[], metavar="KEY=VALUE",
                        help="a keyword argument of the backend's constructor (repeatable)")
    parser.add_argument("--vlm-api-key", default=None, metavar="KEY", help="default: the environment variable VLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)


def _demo_summary_for(args, task: TaskSpec, parser: argparse.ArgumentParser) -> str | None:
    """Variant hard: the statistics of the demonstrations as prompt text; the other variants need none."""
    if args.variant != "hard":
        return None
    if not args.demos:
        parser.error("--variant hard needs --demos")
    files = pick_evenly(glob.glob(args.demos), args.n_demos)
    episodes = [dict(np.load(file)) for file in files]
    return demo_summary_text(demo_event_summary(episodes, task, squeeze_force=args.squeeze_force))


def _make_model(args) -> VLMBackend:
    """The model client from the --vlm-* options."""
    from ..vlm import make_vlm, resolve_api_key
    key = resolve_api_key(args.vlm_api_key)
    extra_body = {} if args.seed is None else {"seed": args.seed}
    options = {"base_url": args.vlm_url, "model": args.vlm_model, "temperature": args.temperature, "extra_body": extra_body}
    if key:
        options["api_key"] = key
    return make_vlm(args.vlm_backend, dict(args.vlm_arg), **options)


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.task:
        src = pathlib.Path(args.task)
    else:
        src = pathlib.Path(__file__).resolve().parents[1] / "tasks" / "red_blue_blocks.yaml"
    task = TaskSpec.load(src)
    summary = _demo_summary_for(args, task, parser)
    vlm = _make_model(args)
    result = author_rules(vlm, task, args.variant, demo_summary=summary, control_hz=args.control_hz, call_every=args.call_every)
    for verb, why in result["reasons"].items():
        print(f"{verb}: {why}")
    if result["rules"] is None:
        print("unusable answer:", "; ".join(result["errors"]), file=sys.stderr)
        return 1
    comment = [f"history.rules written by {args.vlm_model} (author.py, variant {args.variant}); validate before use"]
    write_task_yaml(src, args.out, result["rules"], comment=comment)
    print("\n".join(rules_yaml_lines(result["rules"], "")), f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
