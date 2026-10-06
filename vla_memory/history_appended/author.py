"""The rules author: the planning model writes the rules writer's `history.rules` from the task description.

Code-as-Monitor's idea (Zhou et al., 2024, arXiv 2412.04455) applied to the history writer: a model writes the monitor once,
offline, and the monitor then runs without the model, in real time. Here the monitor is fixed code (rules.GraspCycle); the
model only chooses, per verb of the task, which of the engine's two triggers completes the verb and the trigger's thresholds.
The answer is constrained by a JSON schema (structured output): a trigger of the engine and numbers in centimetres within
physical bounds. rules_from_answer() converts it to the task file's metres and checks it with parse_rules() and RuleMonitor();
write_task_yaml() writes it into a copy of the task file. The offline validator (validate.py) then decides whether the rule
set is used: a model-written rule set is never trusted without it.

What the model is told (build_messages):
    system   what the history writer does, the robot's signals, the grasp cycle and its two triggers, every parameter with
             its unit and physical meaning, the gripper thresholds of the task file's proprio section, and what a good rule
             set does (never early, never missed; up to about a second late is harmless)
    user     the task: objects, the plan's skills, the history phrases, and, by variant,
               easy   the completion criteria of the task file (texts with numbers, e.g. "the cube lifted about 9 cm")
               hard   no criteria; statistics of the writer's signals around each label switch of a few labelled
                      demonstrations (demo_event_summary, demo_summary_text)
               none   nothing more: skill names and history phrases only

    python -m vla_memory.history_appended.author --variant easy --out authored.yaml [--task task.yaml]
    python -m vla_memory.history_appended.author --variant hard --demos 'demos/*.npz' --squeeze-force 0.5 --out authored.yaml
    python -m vla_memory.history_appended.validate --task authored.yaml --episodes 'recorded/*.npz'
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import glob
import json
import pathlib
import re
import statistics
import sys
from typing import Any, Iterable, Sequence

import numpy as np

from ..plugins import key_value
from ..task import TaskSpec
from ..vlm import VLMBackend
from .prompt import HistoryPrompt
from .rules import TRIGGERS, RuleMonitor, parse_rules

VARIANTS = ("easy", "hard", "none")

# The answer's parameters: rule-engine name -> (answer field, factor to the task file's metres, minimum, maximum, optional).
# The bounds are physical sanity limits, wider than any plausible answer: a server with structured output enforces them while
# it generates the digits, which truncates a number instead of clamping it (asked for 50 below a maximum of 30, Qwen on vLLM
# wrote 5). A value at a bound is therefore suspicious; the validator, not the schema, is the safety check.
PARAMS: dict[str, dict[str, tuple[str, float, float, float, bool]]] = {
    "lift": {"rise": ("rise_cm", 0.01, 1.0, 40.0, False),
             "at_top": ("at_top_cm_per_step", 0.01, 0.01, 2.0, True)},
    "set_down": {"release_below": ("release_below_cm", 0.01, 1.0, 40.0, False),
                 "up": ("up_cm", 0.01, 1.0, 40.0, False),
                 "away": ("away_cm", 0.01, 1.0, 40.0, True)},
}
REASON_MAX = 600
GRIPPER_DEFAULTS = {"open_width": 0.06, "empty_width": 0.015, "squeeze_force": 8.0}   # as RuleMonitor


def _cm(x_m: float) -> str:
    return f"{100 * x_m:.1f}".rstrip("0").rstrip(".")


def gripper_thresholds(task: TaskSpec, **overrides) -> dict:
    """open_width, empty_width, squeeze_force: the task file's proprio section over RuleMonitor's defaults."""
    p = task.proprio or {}
    return {**GRIPPER_DEFAULTS, **{k: float(p[k]) for k in GRIPPER_DEFAULTS if k in p},
            **{k: float(v) for k, v in overrides.items() if v is not None}}


# ---------------------------------------------------------------------------------------------------------------- prompt
SYSTEM = """\
You configure the history writer of a robot arm that is driven by a learned policy (a vision-language-action model). The \
policy receives the task instruction plus a short history of the skills completed so far, for example "<instruction> \
History: picked up the red block." The history writer adds a line when a skill is complete. It uses no model and no camera: \
it is a fixed rule engine on the robot's own gripper and hand signals, and you write its rules for one task. For each verb \
of the task you choose which of the engine's two triggers completes it, and the trigger's thresholds.

SIGNALS, one row per control step{rate}:
- finger width: the distance between the two fingers.
- squeeze: whether the fingers press on something: the finger force, or 1/0 from a gripper that only reports "grasped".
- fingertip position x, y, z (z points up): the point between the fingertips, computed from the measured joint angles.

GRASP CYCLE, a state machine the engine follows at every control step. The gripper thresholds are fixed for this robot: \
OPEN = {open_cm} cm, EMPTY = {empty_cm} cm, SQUEEZE = {force}.
1. free -> holding: the fingers stopped between EMPTY and OPEN apart and the squeeze is above SQUEEZE. The fingertip \
position at this step is the GRASP POINT. A gripper that closes on nothing stops at its commanded width without squeeze, so \
an empty grasp never starts a cycle.
2. holding -> lifted: still squeezing, the fingertips are at least RISE above the grasp point. This fires the trigger \
"lift". With AT_TOP set, the trigger also waits until the fingertips rise less than AT_TOP in one control step, i.e. for the \
top of the lift. If the fingers open, or close fully, before that, the cycle returns to free without a trigger.
3. lifted -> released: the fingers open (width at least OPEN) while the fingertips are less than RELEASE_BELOW above the \
grasp point: the object was set down, or let go just above the surface. The fingertip position at this step is the RELEASE \
POINT. Opening higher up is a drop: the cycle returns to free without a trigger.
4. released -> free: the fingertips are at least UP above the release point, or at least AWAY from it in any direction \
(straight-line distance). This fires the trigger "set_down". A touch after the release (squeeze again, then open again \
without a lift) keeps the set-down pending.

FROM TRIGGERS TO HISTORY LINES:
- Every verb of the task is completed by one trigger, "lift" or "set_down". The rule set needs at least one verb with each \
trigger.
- The writer follows the plan skill by skill. A trigger completes the current skill only if the skill's verb uses that \
trigger; otherwise nothing is written. The plan pointer only moves forward, so an object that slipped out and is lifted \
again does not produce a second line.
- The new line appears in the policy's prompt at its next call{call}.

PARAMETERS (lengths in cm, measured with the fingertip position):
- rise_cm (lift, required): how far the fingertips must rise above the grasp point. Too small: the line can appear before \
the object is really lifted. Too large: a lift that ends lower never fires.
- at_top_cm_per_step (lift, optional, null = off): fire only once the fingertips rise less than this in one control \
step, i.e. at the top of the lift{per_s}. It never fires earlier than rise_cm alone, only later.
- release_below_cm (set_down, required): an opening of the fingers counts as a set-down only while the fingertips are less \
than this above the grasp point. Too small: an object put down a little higher than where it was grasped counts as dropped \
and its line is never written. Too large: an object dropped from a height counts as set down.
- up_cm (set_down, required): how far the fingertips must rise above the release point. Too small: the line can appear \
while the hand is still at the object. Too large: a retreat that ends lower never fires.
- away_cm (set_down, optional, null = off): the alternative straight-line distance from the release point. It is never \
smaller than the rise, so an away_cm below up_cm makes up_cm irrelevant; use it to catch a sideways retreat.

WHAT A GOOD RULE SET DOES:
- The policy was trained on prompts whose history changes at fixed moments of each skill. At run time it follows its \
history: as soon as a line appears, it moves on to the next skill.
- A line that appears too early is the worst error: the policy treats the skill as done and moves on, for example it lets \
go of an object it has not lifted yet. A line that never appears is just as bad: the policy waits for it.{late}
- So every threshold must be reached only once the skill is complete, and it must be reached in every execution, also in \
imperfect ones. A learned policy executes less precisely than its demonstrations: a lift may end lower, an object may be let \
go slightly above the surface, the hand may retreat sideways. When in doubt, choose the later moment, but never one that \
some executions do not reach.

ANSWER: JSON only. For each verb of the task: "reason", one or two short sentences on how you derived the numbers, then \
"rule" with the trigger and its parameters in cm."""

EASY_HEAD = ("COMPLETION CRITERIA, the definitions the policy's training labels follow (<object> stands for the object of the "
             "skill). They were written for a camera monitor that sees the gripper's closedness (0 = open, 1 = closed) and the "
             "hand's height above the table; the history writer reads finger width, squeeze and fingertip position instead.")
HARD_HEAD = ("MEASUREMENTS FROM DEMONSTRATIONS, the recordings the policy was trained on. Their labels switch to the next "
             "history line at the moments described below. The values are the history writer's own signals, relative to the "
             "grasp and release points of the grasp cycle (min / median / max over the demonstrations; one control step = "
             "{step}).")


def build_messages(task: TaskSpec, variant: str = "easy", demo_summary: str | None = None, control_hz: float | None = None,
                   call_every: int | None = None, late_ok_s: float | None = 1.0, **gripper_overrides) -> list[dict]:
    """The chat messages of one authoring request. control_hz and call_every (the policy's call period in control steps)
    only add timing sentences; late_ok_s says how late a line may appear without cost (None: not said)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if variant == "hard" and not demo_summary:
        raise ValueError("variant 'hard' needs demo_summary (demo_summary_text of a few labelled demonstrations)")
    g = gripper_thresholds(task, **gripper_overrides)
    rate = f" (the controller runs at {control_hz:g} Hz)" if control_hz else ""
    call = ""
    if call_every:
        call = f", every {call_every} control steps" + (f" (about {call_every / control_hz:.2f} s)" if control_hz else "")
    per_s = f" (at {control_hz:g} Hz, 0.1 cm per step is {0.1 * control_hz:g} cm/s)" if control_hz else ""
    late = f" A line that appears up to about {late_ok_s:g} s late costs nothing." if late_ok_s else ""
    system = SYSTEM.format(rate=rate, open_cm=_cm(g["open_width"]), empty_cm=_cm(g["empty_width"]),
                           force=f"{g['squeeze_force']:g} (a force in N, or 0.5 for a grasped flag)", call=call,
                           per_s=per_s, late=late)
    hp = HistoryPrompt(task)
    phrase = {v: hp.events.get(v, task.skill_template.format(verb=v, object="{object}")).format(object="<object>")
              for v in task.verbs}
    phrases = "; ".join(f'"{v}" -> "{phrase[v]}"' for v in task.verbs)
    plan = "; ".join(f"{i + 1}. {s}" for i, s in enumerate(task.skills))
    instr = (task.instruction_examples or ["<instruction>"])[0]
    example = hp.render(instr, list(task.skills[:2]))
    user = [f'TASK "{task.name}". Objects: {", ".join(task.objects)}. Verbs: {", ".join(task.verbs)}.',
            f"The plan, in order (the skills the policy was trained on): {plan}.",
            f"History line written when a skill is complete: {phrases}.",
            f'Example prompt after the first two skills: "{example}"']
    if variant == "easy":
        user.append(EASY_HEAD)
        user += [f'- "{v}": ' + " ".join(str(task.criteria[v]).replace("{object}", "<object>").split())
                 for v in task.verbs if v in (task.criteria or {})]
    elif variant == "hard":
        user.append(HARD_HEAD.format(step=f"{1000 / control_hz:.0f} ms" if control_hz else "one row of the signals"))
        user.append(demo_summary.strip())
    user.append("Write one rule for each verb: " + ", ".join(f'"{v}"' for v in task.verbs) + ".")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user)}]


def answer_schema(task: TaskSpec) -> dict:
    """Per verb: a short reason, then the rule: one trigger of the engine with its parameters in cm, within PARAMS' bounds
    (optional parameters may be null)."""
    def alternative(trig: str) -> dict:
        props: dict[str, Any] = {"trigger": {"type": "string", "enum": [trig]}}
        for field, _, lo, hi, optional in PARAMS[trig].values():
            props[field] = {"type": ["number", "null"] if optional else "number", "minimum": lo, "maximum": hi}
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    verb = {"type": "object", "properties": {"reason": {"type": "string", "maxLength": REASON_MAX},
                                             "rule": {"anyOf": [alternative(t) for t in TRIGGERS]}},
            "required": ["reason", "rule"], "additionalProperties": False}
    return {"type": "object", "properties": {v: copy.deepcopy(verb) for v in task.verbs}, "required": list(task.verbs),
            "additionalProperties": False}


# ---------------------------------------------------------------------------------------------------------------- answer
def with_rules(task: TaskSpec, rules: dict) -> TaskSpec:
    """A copy of the task with history.rules replaced."""
    t = copy.deepcopy(task)
    t.history = {**(t.history or {}), "rules": copy.deepcopy(rules)}
    return t


def check_rules(task: TaskSpec, rules: dict) -> None:
    """Raises ValueError if the rules writer would reject the rule set (parse_rules, RuleMonitor)."""
    t = with_rules(task, rules)
    parse_rules(t)
    RuleMonitor(t)


def rules_from_answer(answer: Any, task: TaskSpec) -> tuple[dict | None, dict, list[str]]:
    """The model's answer -> (history.rules in the task file's metres, the reason per verb, errors). The rules are None when
    the answer cannot be used: not an object, a verb missing or unknown, an unknown trigger, a required parameter missing,
    not a number or out of bounds (a backend without structured output can return any of these), or a rule set the writer
    rejects (e.g. both verbs on the trigger lift). Optional parameters that are null are left out."""
    if not isinstance(answer, dict):
        return None, {}, [f"the answer is not a JSON object: {answer!r}"[:300]]
    rules, reasons, errors = {}, {}, []
    for verb in task.verbs:
        item = answer.get(verb)
        if not isinstance(item, dict):
            errors.append(f"no rule for verb {verb!r}")
            continue
        reasons[verb] = str(item.get("reason") or "")
        rule = item["rule"] if isinstance(item.get("rule"), dict) else item   # tolerate a flat answer
        trig = rule.get("trigger")
        if trig not in PARAMS:
            errors.append(f"{verb!r}: trigger must be one of {TRIGGERS}, got {trig!r}")
            continue
        spec: dict[str, Any] = {"trigger": trig}
        for name, (field, factor, lo, hi, optional) in PARAMS[trig].items():
            v = rule.get(field)
            if v is None:
                if not optional:
                    errors.append(f"{verb!r}: {field} is required for the trigger {trig}")
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                errors.append(f"{verb!r}: {field} must be a number, got {v!r}")
            elif not lo <= v <= hi:
                errors.append(f"{verb!r}: {field} = {v} is outside [{lo:g}, {hi:g}]")
            else:
                spec[name] = round(float(v) * factor, 6)
        rules[verb] = spec
    unknown = [k for k in answer if k not in task.verbs]
    if unknown:
        errors.append(f"unknown verb(s) in the answer: {unknown}")
    if not errors:
        try:
            check_rules(task, rules)
        except ValueError as e:
            errors.append(str(e))
    return (None if errors else rules), reasons, errors


def author_rules(vlm: VLMBackend, task: TaskSpec, variant: str = "easy", demo_summary: str | None = None,
                 max_tokens: int = 1000, **prompt_kw) -> dict:
    """One authoring request. Returns {"variant", "rules" (None if unusable), "reasons", "errors", "answer" (the model's
    JSON), "text", "ms", "tokens", "messages"}. A failed call is reported in errors, not raised."""
    msgs = build_messages(task, variant, demo_summary=demo_summary, **prompt_kw)
    out: dict[str, Any] = {"variant": variant, "rules": None, "reasons": {}, "errors": [], "answer": None, "text": None,
                           "ms": None, "tokens": None, "messages": msgs}
    try:
        r = vlm.chat(msgs, schema=answer_schema(task), name="history_rules", max_tokens=max_tokens)
    except Exception as e:  # server down, timeout, a JSON the server could not finish
        out["errors"] = [f"model call failed: {e!r}"]
        return out
    out.update(answer=r.get("json"), text=r.get("text"), ms=r.get("ms"),
               tokens=[r.get("prompt_tokens"), r.get("completion_tokens")])
    out["rules"], out["reasons"], out["errors"] = rules_from_answer(r.get("json"), task)
    return out


# ---------------------------------------------------------------------------------------------------------------- task file
_PLAIN = re.compile(r"[A-Za-z][A-Za-z0-9 _-]*")


def _key(s: str) -> str:
    return s if _PLAIN.fullmatch(s) and not s.endswith(" ") else json.dumps(s)


def _num(v: float) -> str:
    return f"{float(v):.6f}".rstrip("0").rstrip(".")


def rules_yaml_lines(rules: dict, indent: str = "    ") -> list[str]:
    """history.rules as flow-style YAML lines, one per verb (as in the reference task file)."""
    width = max(len(_key(v)) for v in rules) + 1
    out = []
    for verb, spec in rules.items():
        items = [f"trigger: {spec['trigger']}"] + [f"{k}: {_num(v)}" for k, v in spec.items() if k != "trigger" and v is not None]
        out.append(f"{indent}{(_key(verb) + ':').ljust(width)} {{{', '.join(items)}}}")
    return out


def write_task_yaml(src, dst, rules: dict, comment: Sequence[str] | None = None) -> pathlib.Path:
    """A copy of the task file `src` with history.rules replaced by `rules`; every other line is copied unchanged. With
    `comment`, the comment lines directly above `rules:` are replaced by these lines (without '#'). The copy is loaded back
    and checked: same task apart from history.rules, and the rules writer accepts it."""
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    check_rules(TaskSpec.load(src), rules)
    lines = src.read_text().splitlines()
    h = next((i for i, l in enumerate(lines) if re.match(r"history:\s*(#.*)?$", l)), None)
    if h is None:
        raise ValueError(f"{src}: no top-level history: section")
    end_h = next((i for i in range(h + 1, len(lines)) if lines[i].strip() and not lines[i][0].isspace()
                  and not lines[i].lstrip().startswith("#")), len(lines))
    r = next((i for i in range(h + 1, end_h) if re.match(r"\s+rules:\s*(#.*)?$", lines[i])), None)
    if r is None:   # no rules yet: append them at the end of the history section, at the indentation of its keys
        body = [i for i in range(h + 1, end_h) if lines[i].strip() and not lines[i].lstrip().startswith("#")]
        indent = re.match(r"\s*", lines[body[0]]).group(0) if body else "  "
        last = body[-1] if body else h
        lines[last + 1:last + 1] = [f"{indent}rules:"]
        r = last + 1
    indent = re.match(r"\s*", lines[r]).group(0)
    stop = r + 1
    while stop < len(lines) and (not lines[stop].strip() or len(re.match(r"\s*", lines[stop]).group(0)) > len(indent)):
        stop += 1
    while stop > r + 1 and not lines[stop - 1].strip():   # keep blank lines after the block
        stop -= 1
    new = [lines[r]] + rules_yaml_lines(rules, indent + "  ")
    top = r
    if comment is not None:
        while top > h + 1 and re.match(rf"{indent}#", lines[top - 1]):
            top -= 1
        new = [f"{indent}# {c}".rstrip() for c in comment] + new
    lines[top:stop] = new
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.tmp{dst.suffix}")   # same suffix (TaskSpec.load reads by suffix); checked before it replaces dst
    tmp.write_text("\n".join(lines) + "\n")
    try:
        a, b = dataclasses.asdict(TaskSpec.load(src)), dataclasses.asdict(TaskSpec.load(tmp))
        got = b["history"].pop("rules", None); a["history"].pop("rules", None)
        if a != b or parse_rules(with_rules(TaskSpec.load(src), got or {})) != parse_rules(with_rules(TaskSpec.load(src), rules)):
            raise RuntimeError(f"{dst}: the written copy does not reproduce the task with the new rules")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


# ---------------------------------------------------------------------------------------------------------------- demos
def _cycles(rows: np.ndarray, open_width: float, empty_width: float, squeeze_force: float) -> list[dict]:
    """Grasp cycles from the gripper signals alone (no rules): start of the squeeze, end (fingers opened or closed fully)."""
    out, cur = [], None
    for t, (w, e) in enumerate(rows[:, :2]):
        if cur is None:
            if empty_width < w < open_width and e > squeeze_force:
                cur = {"grasp": t}
        elif w >= open_width or w <= empty_width:
            cur.update(end=t, opened=bool(w >= open_width)); out.append(cur); cur = None
    if cur is not None:
        out.append(cur)
    return out


def demo_event_summary(episodes: Iterable[dict], task: TaskSpec, window: Sequence[int] = (-16, -8, -4, 0, 4, 8),
                       **gripper_overrides) -> list[dict]:
    """Statistics of the writer's signals at the label switches of labelled episodes ({"rows", "reference"}, as for
    validate.py): per history line k, at the first step where the label count reaches k, relative to the grasp cycle that
    the gripper signals show (no rules involved). If the fingers still hold: the fingertip rise above the grasp point, the
    highest rise of that hold, the upward speed, the steps since the grasp. If they opened after a set-down: the release
    height above the grasp point, the fingertip rise above and the distance from the release point, the steps since the
    release. Plus the median course of the rise around the switch (`window`, control steps). Lengths in cm."""
    g = gripper_thresholds(task, **gripper_overrides)
    cap = (task.history or {}).get("max_lines"); cap = len(task.skills) if cap is None else cap
    per: dict[int, dict[str, list]] = {}
    for ep in episodes:
        rows = np.asarray(ep["rows"], float); ref = np.minimum(np.asarray(ep["reference"], int), cap)
        z, p = 100 * rows[:, 4], 100 * rows[:, 2:5]
        cyc = _cycles(rows, g["open_width"], g["empty_width"], g["squeeze_force"])
        for k in range(1, int(ref.max(initial=0)) + 1):
            t = int(np.flatnonzero(ref >= k)[0]); s = per.setdefault(k, {"n": [], "state": []})
            s["n"].append(1)
            c = next((c for c in reversed(cyc) if c["grasp"] <= t), None)
            if c is None:
                s["state"].append("no grasp yet"); continue
            gp, around = c["grasp"], [t + o for o in window]
            if c.get("end") is None or c["end"] > t:   # still holding at the switch
                s["state"].append("holding")
                end = c.get("end", len(rows) - 1)
                for key, val in (("rise", z[t] - z[gp]), ("max_rise", z[gp:end + 1].max() - z[gp]),
                                 ("speed", z[t] - z[t - 1] if t > 0 else 0.0), ("steps_since_grasp", t - gp)):
                    s.setdefault(key, []).append(float(val))
                s.setdefault("rise_around", []).append([z[min(max(i, 0), len(z) - 1)] - z[gp] for i in around])
            elif c["opened"]:                       # opened after the grasp: a release
                s["state"].append("released")
                rel = c["end"]
                for key, val in (("release_height", z[rel] - z[gp]), ("up", z[t] - z[rel]),
                                 ("away", float(np.linalg.norm(p[t] - p[rel]))), ("steps_since_release", t - rel)):
                    s.setdefault(key, []).append(float(val))
                s.setdefault("up_around", []).append([z[min(max(i, 0), len(z) - 1)] - z[rel] for i in around])
            else:
                s["state"].append("closed on nothing")
    plan = list(task.skills); hp = HistoryPrompt(task); out = []
    for k in sorted(per):
        s = per[k]; d = {"line": k, "phrase": hp.event(plan[k - 1]) if k <= len(plan) else None, "n": len(s["n"]),
                         "states": {x: s["state"].count(x) for x in dict.fromkeys(s["state"])}, "window": list(window)}
        for key, vals in s.items():
            if key in ("n", "state"):
                continue
            if key.endswith("_around"):
                d[key] = [round(float(statistics.median(col)), 1) for col in zip(*vals)]
            else:
                d[key] = [round(float(f(vals)), 2) for f in (min, statistics.median, max)]
        out.append(d)
    return out


_STATE_TEXT = {"holding": "still holding the object", "released": "open again after a set-down",
               "closed on nothing": "closed on nothing", "no grasp yet": "not yet grasping"}


def demo_summary_text(summary: list[dict]) -> str:
    """The prompt text of demo_event_summary (variant hard)."""
    def num(x, nd=1):
        s = f"{x:.{nd}f}"
        return s[1:] if re.fullmatch(r"-0\.?0*", s) else s   # no "-0.0"

    def mmm(v, unit, nd=1):
        return " / ".join(num(x, nd) for x in v) + unit
    lines = []
    for d in summary:
        states = ", ".join(f"{_STATE_TEXT.get(x, x)} in {c} of {d['n']}" for x, c in d["states"].items())
        lines.append(f'History line {d["line"]}, "{d["phrase"]}" ({d["n"]} demonstrations): at the switch the gripper was '
                     f'{states}.')
        if "rise" in d:
            lines += [f"  fingertip rise above the grasp point at the switch: {mmm(d['rise'], ' cm')}",
                      f"  highest rise during that hold: {mmm(d['max_rise'], ' cm')}",
                      f"  upward speed at the switch: {mmm(d['speed'], ' cm per step', 2)}",
                      f"  control steps since the grasp began: {mmm(d['steps_since_grasp'], '', 0)}",
                      "  rise above the grasp point around the switch (median): " +
                      ", ".join(f"{o:+d} steps {num(v)} cm" for o, v in zip(d["window"], d["rise_around"]))]
        if "up" in d:
            lines += [f"  release height above the grasp point: {mmm(d['release_height'], ' cm')}",
                      f"  fingertip rise above the release point at the switch: {mmm(d['up'], ' cm')}",
                      f"  straight-line distance from the release point at the switch: {mmm(d['away'], ' cm')}",
                      f"  control steps since the release: {mmm(d['steps_since_release'], '', 0)}",
                      "  rise above the release point around the switch (median): " +
                      ", ".join(f"{o:+d} steps {num(v)} cm" for o, v in zip(d["window"], d["up_around"]))]
    return "\n".join(lines)


def pick_evenly(files: Sequence[str], n: int) -> list[str]:
    """n files spread evenly over a sorted list (first and last included)."""
    files = sorted(files)
    if n >= len(files):
        return list(files)
    return [files[int(round(i))] for i in np.linspace(0, len(files) - 1, n)]


# ---------------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=None, help="task YAML (default: the packaged reference task)")
    ap.add_argument("--variant", choices=VARIANTS, default="easy")
    ap.add_argument("--out", required=True, help="where to write the copy of the task file with the authored rules")
    ap.add_argument("--demos", default=None, help="glob of labelled episode .npz files (rows, reference) for --variant hard")
    ap.add_argument("--n-demos", type=int, default=5, help="how many demonstrations, spread evenly over the sorted files")
    ap.add_argument("--squeeze-force", type=float, default=None, help="for the demos: 0.5 if they carry a grasped flag")
    ap.add_argument("--control-hz", type=float, default=None)
    ap.add_argument("--call-every", type=int, default=None, help="the policy's call period in control steps")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--vlm-model", default="Qwen3.8-27B-INT4")
    ap.add_argument("--vlm-backend", default="openai", metavar="NAME",
                    help="the model's client, as in the proxies: openai (default), a registered name, an entry point or module:Class")
    ap.add_argument("--vlm-arg", action="append", type=key_value, default=[], metavar="KEY=VALUE",
                    help="a keyword argument of the backend's constructor (repeatable)")
    ap.add_argument("--vlm-api-key", default=None, metavar="KEY", help="default: the environment variable VLM_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    from ..vlm import make_vlm, resolve_api_key
    src = pathlib.Path(args.task) if args.task else pathlib.Path(__file__).resolve().parents[1] / "tasks" / "red_blue_blocks.yaml"
    task = TaskSpec.load(src)
    summary = None
    if args.variant == "hard":
        if not args.demos:
            ap.error("--variant hard needs --demos")
        files = pick_evenly(glob.glob(args.demos), args.n_demos)
        eps = [dict(np.load(f)) for f in files]
        summary = demo_summary_text(demo_event_summary(eps, task, squeeze_force=args.squeeze_force))
    key = resolve_api_key(args.vlm_api_key)
    vlm = make_vlm(args.vlm_backend, dict(args.vlm_arg), base_url=args.vlm_url, model=args.vlm_model, temperature=args.temperature,
                   extra_body={} if args.seed is None else {"seed": args.seed}, **({"api_key": key} if key else {}))
    res = author_rules(vlm, task, args.variant, demo_summary=summary, control_hz=args.control_hz, call_every=args.call_every)
    for verb, why in res["reasons"].items():
        print(f"{verb}: {why}")
    if res["rules"] is None:
        print("unusable answer:", "; ".join(res["errors"]), file=sys.stderr)
        return 1
    write_task_yaml(src, args.out, res["rules"], comment=[f"history.rules written by {args.vlm_model} (author.py, variant "
                                                           f"{args.variant}); validate before use"])
    print("\n".join(rules_yaml_lines(res["rules"], "")), f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
