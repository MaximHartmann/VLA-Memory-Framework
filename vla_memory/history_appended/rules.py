"""The rules writer: a monitor without a model. It decides from the robot's own gripper and hand signals when the current
skill is done, by rules declared in the task file, and advances a pointer through the plan exactly as the planner does.

Signals, one row per control step:  finger width (m, both fingers together), squeeze (finger force in N in the simulator,
or 1/0 from a gripper that reports a grasped flag), fingertip position x, y, z (m, forward kinematics of the measured
joints). All of them exist on a real robot; nothing here looks at the objects. A task may declare further signals of the
robot (`signals` in the task file, e.g. the contact force at the hand), sent as memory/signal_steps.

Two kinds of triggers; the task file maps each verb to one of them (`history.rules`), and a trigger completes the current
skill only if its verb matches, so a second grasp of a block that slipped out does not write "picked up" twice.

1. The grasp cycle, for pick and place (triggers lift and set_down), followed at every step:
    free      -> holding   the fingers stopped between empty_width and open_width and squeeze above squeeze_force
    holding   -> lifted    the fingertips rose `rise` above the point where the grasp began, still squeezing   => trigger "lift"
                           (with `at_top`: only once they also rise less than at_top per step, i.e. at the top of the lift)
    lifted    -> released  the fingers opened (width >= open_width) less than `release_below` above the grasp point
                           (set down, or let go just above the table); opening higher up is a drop and ends the cycle
    released  -> free      the fingertips moved `up` above or `away` from the release point                     => trigger "set_down"
   A touch after the release (squeeze, then open again without a lift) keeps the set-down pending.

2. Signal thresholds, for any other skill (trigger signals: push, open or close a drawer, press a button): the task file
   lists phases, each a set of conditions on the signals that must all hold at one control step; the phases are entered in
   order and the skill is complete when the last one is entered. A phase's `hold` conditions must keep holding until the next
   phase is entered, else the sequence starts over (the grip on a handle lost before the pull). One phase per control step.
   Only the current skill's verb is followed, and its sequence starts afresh whenever the pointer moves.
       open:
         trigger: signals
         phases:
           - when: {finger_effort: {above: 8.0}, finger_width: {above: 0.01, below: 0.06}}   # the fingers squeeze the handle
             hold: {finger_effort: {above: 8.0}}
           - when: {tip_x: {below: 0.47}}                    # pulled out: an absolute position (the environment's frame)
           - when: {finger_width: {above: 0.06}}             # let go
           - when: {tip: {away: 0.03}}                       # 3 cm from where it let go (relative to the previous phase)
   Signals: finger_width, finger_effort, tip_x, tip_y, tip_z (the columns of memory/proprio_steps), the point tip (the
   fingertip position, for away / near), and the task's declared signals. Conditions: above / below a value; rise / fall by
   at least d since a reference phase was entered; away (distance at least d) / near (less than d) from the position there.
   The reference is the previous phase (for a hold, its own phase); `from: k` names another earlier phase (1-based). A signal
   the environment does not send is NaN, and a condition on NaN never holds.

The thresholds must reproduce the moments at which the training labels change: a writer that is right about the event but
switches earlier than the labels loses success (docs/methods/history_appended/README.md). The same rules label recorded
episodes (label_rows), so training labels and run-time history can come from one definition.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..env_keys import POINTS, PROPRIO_COLUMNS

TRIGGERS = ("lift", "set_down")             # the grasp cycle's triggers
SIGNALS = "signals"                         # the trigger of signal thresholds (task-declared phases)
ALL_TRIGGERS = TRIGGERS + (SIGNALS,)
_PARAMS = {"lift": ("rise", "at_top"), "set_down": ("release_below", "up", "away")}
_OPTIONAL = ("away", "at_top")
ABSOLUTE_OPS = ("above", "below")           # the value against a threshold
RELATIVE_OPS = ("rise", "fall", "away", "near")   # the change since a reference phase was entered
_NAN_ROW = np.full(5, np.nan)


def _conditions(spec, where: str, names: set, ref_default: int | None, ref_max: int) -> dict:
    """One `when` or `hold` mapping {signal: {op: value, [from: k]}} -> the normalised mapping (floats; `from` as given)."""
    if not isinstance(spec, dict) or not spec:
        raise ValueError(f"{where}: a non-empty mapping {{signal: {{condition: value}}}} expected, got {spec!r}")
    out = {}
    for sig, ops in spec.items():
        if sig not in names:
            raise ValueError(f"{where}: unknown signal {sig!r} (signals: {', '.join(sorted(names))}; declare a robot signal "
                             f"of your own under `signals` in the task file)")
        if not isinstance(ops, dict) or not ops:
            raise ValueError(f"{where}.{sig}: a mapping of conditions expected, e.g. {{above: 5.0}}, got {ops!r}")
        ops = dict(ops); ref = ops.pop("from", None); point = sig in POINTS
        allowed = RELATIVE_OPS[2:] if point else ABSOLUTE_OPS + RELATIVE_OPS
        bad = [op for op in ops if op not in allowed]
        if bad or not ops:
            raise ValueError(f"{where}.{sig}: condition(s) {bad or 'none'}; {sig} takes {list(allowed)}")
        for op, value in ops.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{where}.{sig}.{op}: a number expected, got {value!r}")
        relative = [op for op in ops if op in RELATIVE_OPS]
        if ref is not None:
            if not relative:
                raise ValueError(f"{where}.{sig}: `from` applies to the relative conditions {list(RELATIVE_OPS)} only")
            if isinstance(ref, bool) or not isinstance(ref, int) or not 1 <= ref <= ref_max:
                raise ValueError(f"{where}.{sig}: from must be a phase number from 1 to {ref_max}, got {ref!r}")
        elif relative and ref_default is None:
            raise ValueError(f"{where}.{sig}: {relative} measure from an earlier phase, and the first phase has none")
        out[sig] = {**{op: float(v) for op, v in ops.items()}, **({"from": ref} if ref is not None else {})}
    return out


def parse_signal_phases(spec: dict, where: str, names: set) -> dict:
    """The parameters of a `trigger: signals` rule -> {"phases": [{"when": {...}, "hold": {...}}, ...]} (checked; see the
    module docstring)."""
    unknown = [k for k in spec if k != "phases"]
    phases = spec.get("phases")
    if unknown or not isinstance(phases, list) or not phases:
        raise ValueError(f"{where}: trigger signals takes `phases`, a non-empty list (unknown: {unknown})")
    out = []
    for i, ph in enumerate(phases, 1):
        w = f"{where}.phases[{i}]"
        if not isinstance(ph, dict) or "when" not in ph or any(k not in ("when", "hold") for k in ph):
            raise ValueError(f"{w}: a mapping with `when` and optionally `hold` expected, got {ph!r}")
        p = {"when": _conditions(ph["when"], f"{w}.when", names, i - 1 if i > 1 else None, i - 1)}
        if ph.get("hold") is not None:
            p["hold"] = _conditions(ph["hold"], f"{w}.hold", names, i, i)
        out.append(p)
    return {"phases": out}


def rule_signals(task) -> set:
    """The signal names a `trigger: signals` rule may use: the columns of memory/proprio_steps, the point tip, and the task's
    declared signals."""
    return set(PROPRIO_COLUMNS) | set(POINTS) | set(getattr(task, "signals", None) or ())


def parse_rules(task) -> dict[str, tuple[str, dict]]:
    """history.rules of the task file -> {verb: (trigger, params)}; checks triggers, parameters and verbs. params: the
    thresholds of a grasp trigger (lift, set_down), or {"phases": [...]} of a signals trigger."""
    rules = dict((task.history or {}).get("rules") or {})
    if not rules:
        raise ValueError("the task file has no history.rules section; the rules writer needs one")
    out = {}
    for verb, spec in rules.items():
        if verb not in task.verbs:
            raise ValueError(f"history.rules: unknown verb {verb!r} (task verbs: {task.verbs})")
        spec = dict(spec); trig = spec.pop("trigger", None)
        if trig not in ALL_TRIGGERS:
            raise ValueError(f"history.rules[{verb!r}]: trigger must be one of {ALL_TRIGGERS}, got {trig!r}")
        if trig == SIGNALS:
            out[verb] = (trig, parse_signal_phases(spec, f"history.rules[{verb!r}]", rule_signals(task)))
            continue
        missing = [p for p in _PARAMS[trig] if p not in spec and p not in _OPTIONAL]
        unknown = [p for p in spec if p not in _PARAMS[trig]]
        if missing or unknown:
            raise ValueError(f"history.rules[{verb!r}]: missing {missing}, unknown {unknown} (trigger {trig} takes {_PARAMS[trig]})")
        out[verb] = (trig, {k: (None if v is None else float(v)) for k, v in spec.items()})
    uncovered = [v for v in task.verbs if v not in out]
    if uncovered:
        raise ValueError(f"history.rules: no rule for verb(s) {uncovered}; every skill of a plan needs a trigger")
    return out


class SignalSequence:
    """The per-step state machine of one `trigger: signals` rule (module docstring); step() returns True when the last phase
    is entered, and the sequence starts over."""

    def __init__(self, phases: Sequence[dict]):
        self.phases = [(self._compile(p["when"], i - 1), self._compile(p.get("hold") or {}, i))
                       for i, p in enumerate(phases)]   # reference: the previous phase (when), the phase itself (hold); 0-based
        self.signals = {s for p in phases for part in ("when", "hold") for s in (p.get(part) or {})}
        self.reset()

    @staticmethod
    def _compile(conds: dict, ref_default: int) -> list:
        return [(sig, op, v, ops["from"] - 1 if "from" in ops else ref_default)
                for sig, ops in conds.items() for op, v in ops.items() if op != "from"]

    def reset(self):
        self.k = 0; self.anchors: list[dict] = []   # phases entered; the signals at the step each was entered

    @property
    def progress(self) -> str:
        return f"{self.k}/{len(self.phases)}"

    def _holds(self, conds: list, now: dict) -> bool:
        for sig, op, v, ref in conds:
            x = now[sig]
            if op == "above":
                ok = x > v
            elif op == "below":
                ok = x < v
            else:
                a = self.anchors[ref][sig]
                if op == "rise":
                    ok = x - a >= v
                elif op == "fall":
                    ok = a - x >= v
                else:   # away / near: the distance from the position at the reference phase (|x - a| for a scalar)
                    d = float(np.linalg.norm(np.subtract(x, a)))
                    ok = d >= v if op == "away" else d < v
            if not ok:   # also a NaN reading
                return False
        return True

    def step(self, now: dict) -> bool:
        """One control step; now: every signal's value at this step (signal -> float; tip -> (3,))."""
        if self.k and not self._holds(self.phases[self.k - 1][1], now):
            self.reset()   # a hold condition broke: start over (from the next step)
            return False
        if not self._holds(self.phases[self.k][0], now):
            return False
        self.anchors.append(now); self.k += 1
        if self.k < len(self.phases):
            return False
        self.reset()
        return True


class GraspCycle:
    """The per-step state machine of one grasp cycle; step() returns "lift", "set_down" or None."""

    def __init__(self, open_width: float, empty_width: float, squeeze_force: float, rise: float,
                 release_below: float, up: float, away: float | None, at_top: float | None = None):
        self.open_width, self.empty_width, self.squeeze_force = open_width, empty_width, squeeze_force
        self.rise, self.release_below, self.up, self.away, self.at_top = rise, release_below, up, away, at_top
        self.reset()

    def reset(self):
        self.state = "free"; self.z_grasp = self.z_rel = None; self.p_rel = None; self.pending = False; self.z_prev = None

    def _squeezing(self, w, e):
        return self.empty_width < w < self.open_width and e > self.squeeze_force

    def step(self, w: float, e: float, p: Sequence[float]) -> str | None:
        p = np.asarray(p, float); z = float(p[2]); trig = None
        if self.state == "free":
            if self._squeezing(w, e):
                self.z_grasp, self.state = z, "holding"
        elif self.state == "holding":
            if w >= self.open_width or w <= self.empty_width:
                if self.pending:   # touched the block that was just set down, and let go again
                    self.z_rel, self.p_rel, self.state = z, p, "released"
                else:
                    self.state = "free"
            elif (z - self.z_grasp >= self.rise and e > self.squeeze_force
                  and (self.at_top is None or (self.z_prev is not None and z - self.z_prev < self.at_top))):
                trig, self.state, self.pending = "lift", "lifted", False
        elif self.state == "lifted":
            if w >= self.open_width:
                if z - self.z_grasp < self.release_below:
                    self.z_rel, self.p_rel, self.state, self.pending = z, p, "released", True
                else:
                    self.state = "free"   # dropped from height: no set-down
            elif w <= self.empty_width:
                self.state = "free"
        elif self.state == "released":
            away = (self.away is not None and self.p_rel is not None and np.all(np.isfinite(p)) and np.all(np.isfinite(self.p_rel))
                    and float(np.linalg.norm(p - self.p_rel)) >= self.away)
            if z - self.z_rel >= self.up or away:
                trig, self.state, self.pending = "set_down", "free", False
            elif self._squeezing(w, e):
                self.z_grasp, self.state = z, "holding"
        self.z_prev = z
        return trig


def signal_rows(ctrl: dict) -> np.ndarray:
    """The environment's signals of one policy call as rows (n, 5): width, squeeze, x, y, z.
    memory/proprio_steps carries every control step since the last call; the first call of an episode may only have the
    current values (memory/finger_width, memory/finger_effort, memory/tip_xyz or memory/tip_z)."""
    steps = ctrl.get("proprio_steps")
    if steps is not None and np.size(steps):
        return np.asarray(steps, float).reshape(-1, 5)
    if "finger_width" not in ctrl:
        return np.zeros((0, 5))
    tip = ctrl.get("tip_xyz")
    tip = np.asarray(tip, float).reshape(3) if tip is not None else np.array([np.nan, np.nan, float(ctrl.get("tip_z", np.nan))])
    return np.array([[float(ctrl["finger_width"]), float(ctrl.get("finger_effort", np.nan)), *tip]])


def signal_columns(signals, names: Sequence[str]) -> dict[str, np.ndarray]:
    """The task's declared signals of one policy call as columns {name: (n,) float}: memory/signal_steps ({name: values per
    control step since the previous call}) or the same mapping from a recorded episode. A name it does not carry is NaN."""
    given = {k: np.asarray(v, float).reshape(-1) for k, v in (signals or {}).items() if k in names}
    n = max((len(v) for v in given.values()), default=0)
    if any(len(v) != n for v in given.values()):
        raise ValueError(f"memory/signal_steps: the signals have different lengths {({k: len(v) for k, v in given.items()})}")
    return {k: given.get(k, np.full(n, np.nan)) for k in names} if given else {}


class RuleMonitor:
    """Drop-in writer for the history-appended prompt with the planner's pointer attributes (plan, idx, done_all,
    current_skill). The plan is the task's skill order unless one is given. Gripper thresholds come from the task's
    proprio section (open_width, empty_width, squeeze_force); keyword arguments override them (e.g. squeeze_force=0.5
    for a gripper that reports a grasped flag).

    With an identifier (identify.py), a pick names the object the wrist camera shows in the gripper instead of assuming
    the plan's: if another object of the task is held, its pending skills move to the front of the remaining plan and the
    line names it; if the held object has no pending pick, nothing is written; if the identifier cannot tell, the plan
    order holds. A set-down writes "placed" only for the object identified at the last lift; a different one (a block
    that was dropped, then another one carried) writes nothing, because a false line misleads the policy for the rest of
    the episode.

    Verbs with `trigger: signals` follow their phases (SignalSequence) while their skill is current, in the plan's order (no
    identifier); signal_names are the task's declared signals the rules use, which observe_rows() takes as `signals`. A task
    without grasp verbs keeps an idle grasp cycle (never stepped, always "free") for code that reads monitor.cycle, and its
    writer_state is the current skill's progress, "<verb> <phases entered>/<phases>"."""

    def __init__(self, task, plan: Sequence[str] | None = None, identifier=None, **gripper_overrides):
        self.task = task; self.rules = parse_rules(task); self.identifier = identifier
        g = {**{"open_width": 0.06, "empty_width": 0.015, "squeeze_force": 8.0},
             **{k: task.proprio[k] for k in ("open_width", "empty_width", "squeeze_force") if k in (task.proprio or {})},
             **gripper_overrides}
        lift = next((p for t, p in self.rules.values() if t == "lift"), None)
        down = next((p for t, p in self.rules.values() if t == "set_down"), None)
        self.grasp = lift is not None or down is not None   # the grasp cycle runs for a task with grasp verbs only
        if self.grasp and (lift is None or down is None):
            raise ValueError("history.rules needs one verb with trigger lift and one with trigger set_down")
        if self.grasp:
            self.cycle = GraspCycle(g["open_width"], g["empty_width"], g["squeeze_force"], lift["rise"],
                                    down["release_below"], down["up"], down.get("away"), lift.get("at_top"))
        else:
            self.cycle = GraspCycle(g["open_width"], g["empty_width"], g["squeeze_force"], math.inf, 0.0, math.inf, None)
        self.sequences = {v: SignalSequence(p["phases"]) for v, (t, p) in self.rules.items() if t == SIGNALS}
        used = set().union(*(s.signals for s in self.sequences.values()))
        self.signal_names = tuple(n for n in (getattr(task, "signals", None) or ()) if n in used)
        self.default_plan = list(plan) if plan else list(task.skills)
        for s in self.default_plan:
            task.parse_skill(s)   # raises for strings outside the vocabulary
        self.reset()

    def reset(self):
        self.plan = list(self.default_plan); self.idx = 0; self.done_all = False; self.events: list[tuple] = []
        self.held_obj = None   # the object the identifier saw at the last lift (None: unknown or no identifier)
        self.cycle.reset()
        for s in self.sequences.values():
            s.reset()

    @property
    def current_skill(self) -> str:
        return self.plan[min(self.idx, len(self.plan) - 1)]

    @property
    def writer_state(self) -> str:
        """The grasp cycle's state, or, for a task without grasp verbs, the current skill's progress through its phases."""
        if self.grasp or not self.sequences:
            return self.cycle.state
        verb = self.task.parse_skill(self.current_skill)[0]
        return f"{verb} {self.sequences[verb].progress}" if verb in self.sequences else self.cycle.state

    def start_episode(self, instruction: str = "", episode_id: str = "") -> dict:
        self.reset()
        return {"plan": list(self.plan), "source": "task order (rules writer)", "ms": 0.0}

    def _bring_forward(self, obj: str) -> bool:
        """Another object than planned was picked up: move its pending skills to the front of the remaining plan."""
        rest = self.plan[self.idx:]
        mine = [s for s in rest if self.task.parse_skill(s)[1] == obj]
        if not mine or self.rules[self.task.parse_skill(mine[0])[0]][0] != "lift":
            return False   # no pending pick of this object
        self.plan = self.plan[:self.idx] + mine + [s for s in rest if self.task.parse_skill(s)[1] != obj]
        return True

    def _write(self, t, done: list):
        """The current skill is complete at control step t: a line, and the pointer moves on (a new skill's sequence starts afresh)."""
        done.append(self.plan[self.idx]); self.events.append((t, self.plan[self.idx]))
        if self.idx + 1 >= len(self.plan):
            self.done_all = True; self.idx = len(self.plan) - 1
        else:
            self.idx += 1
        for s in self.sequences.values():
            s.reset()

    def observe_rows(self, rows, step: int | None = None, image=None, signals=None) -> dict:
        """Feed the control steps since the last call (and the current wrist image, for the identifier); returns the
        decision record. signals: the task's declared signals over the same steps ({name: (n,)}, memory/signal_steps); when
        both are given, rows and signals must have one entry per step each."""
        rows = np.asarray(rows, float).reshape(-1, 5); done = []; ident = None
        extra = signal_columns(signals, self.signal_names) if self.signal_names else {}
        m = len(next(iter(extra.values()))) if extra else 0
        if len(rows) and m and len(rows) != m:
            raise ValueError(f"{len(rows)} control steps in memory/proprio_steps but {m} in memory/signal_steps: every step "
                             f"must appear in both")
        n = max(len(rows), m)
        for i in range(n):
            r = rows[i] if len(rows) else _NAN_ROW
            t = None if step is None else int(step) - n + i
            if self.grasp:
                trig = self.cycle.step(r[0], r[1], r[2:5])
                if trig is not None and not self.done_all:
                    seen: dict = {}; line = self._grasp_line(trig, image, seen); ident = seen.get("ident", ident)
                    if line:
                        self._write(t, done)
                        continue   # one line per control step
            if self.sequences and not self.done_all:
                seq = self.sequences.get(self.task.parse_skill(self.plan[self.idx])[0])
                if seq is not None and seq.step(self._values(r, extra, i)):
                    self._write(t, done)
        return {"advanced": bool(done), "completed": done, "skill_after": self.current_skill, "writer_state": self.writer_state,
                "plan_complete": self.done_all, "identified": ident}

    def _grasp_line(self, trig: str, image, out: dict) -> bool:
        """Whether a grasp trigger completes the current skill (identifier, reordering and set-down guard of the class
        docstring); an identification goes to out["ident"]."""
        verb, obj = self.task.parse_skill(self.plan[self.idx])
        if trig == "lift" and self.identifier is not None and image is not None:
            held = self.identifier.identify(image); out["ident"] = ident = {"held": held, "expected": obj, "reordered": False}
            self.held_obj = held
            if held is not None and held != obj:
                if self.rules[verb][0] != "lift" or not self._bring_forward(held):
                    return False   # the held object has no pending pick: not a step of the plan, write nothing
                ident["reordered"] = True; verb, obj = self.task.parse_skill(self.plan[self.idx])
        if self.rules[verb][0] != trig:
            return False   # e.g. a second lift of a block that slipped out: the pick is already written
        if trig == "set_down":
            if self.held_obj is not None and self.held_obj != obj:
                return False   # another object than the one this place step is about was set down: write nothing
            self.held_obj = None
        return True

    def _values(self, r, extra: dict, i: int) -> dict:
        """Every signal's value at one step: the five columns, the point tip and the declared signals (NaN if not sent)."""
        now = dict(zip(PROPRIO_COLUMNS, (float(x) for x in r)))
        now["tip"] = np.array(r[2:5], float)
        for k in self.signal_names:
            now[k] = float(extra[k][i]) if k in extra else math.nan
        return now

    def completed(self) -> list[str]:
        """The skills completed so far, in order (what the history lists)."""
        return list(self.plan) if self.done_all else list(self.plan[:self.idx])

    def label_rows(self, rows, signals=None) -> np.ndarray:
        """Training labels of a recorded episode: the number of completed skills after each control step (signals: the
        task's declared signals per step, {name: (T,)})."""
        self.reset(); rows = np.asarray(rows, float).reshape(-1, 5)
        extra = signal_columns(signals, self.signal_names) if self.signal_names else {}
        T = max(len(rows), len(next(iter(extra.values()))) if extra else 0)
        out = np.zeros(T, dtype=np.int16)
        for i in range(T):
            self.observe_rows(rows[i:i + 1], signals={k: v[i:i + 1] for k, v in extra.items()} or None)
            out[i] = len(self.completed())
        return out
