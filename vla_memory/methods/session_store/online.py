"""The session store's write path: the calls of an episode, labelled in hindsight when the episode ends, and written into the
store.

    EpisodeRecorder.record(control_keys, images, record)    after every planner decision: the two pictures, the gripper rows
                                                            since the previous call, the closedness and the hand height, and
                                                            the simulator's prompt (if sent)
    EpisodeRecorder.finish(store, episode, context)         at the episode's end: label every call (RulesLabeller: the rules
                                                            writer replayed over the robot's own signals, in the task order;
                                                            or OracleLabeller), leave out the calls without a label and those
                                                            after the task's last skill (dropped as "terminal"), leave out
                                                            near-duplicates (an episode keeps at most two calls with the same
                                                            labels and a colour-key cosine of 0.9999 or more to each other),
                                                            insert the rest with the store's keys, evict above the cap and
                                                            write the counters; returns the store_update record for the
                                                            planner's log

The colour key decides the near-duplicates whichever key retrieves, so which calls are admitted does not depend on the
retrieval key.
"""
from __future__ import annotations

import dataclasses
import time
from collections import Counter
from typing import Sequence

import numpy as np

from ...env_keys import PROPRIO_COLUMNS
from ...transport import text_of
from ...writers.rules.identify import make_identifier
from ...writers.rules.rule_monitor import RuleMonitor
from ...writers.rules.signals import signal_rows

HIGH_ABOVE_TABLE = 0.17     # m, how high above the table "high" is (the task's proprio.high_above_table overrides it)
LIFTED_ABOVE_GRASP = 0.06   # m, how far above the grasp "lifted" is (the task's proprio.lifted_above_grasp overrides it)
DEDUP_COS = 0.9999          # colour-key cosine from which two calls with the same labels are near-duplicates
DEDUP_COPIES = 2            # calls with the same labels an episode keeps
# the columns of memory/proprio_steps the held-object label reads
WIDTH_COLUMN, EFFORT_COLUMN, TIP_Z_COLUMN = (PROPRIO_COLUMNS.index(name) for name in ("finger_width", "finger_effort", "tip_z"))


@dataclasses.dataclass
class Call:
    """One monitored policy call as the recorder keeps it."""

    step: int
    images: list                  # exterior, wrist
    rows: np.ndarray              # the gripper signals of the control steps since the previous call (memory/proprio_steps)
    closedness: float | None = None
    hand_height: float | None = None
    oracle: str | None = None     # the simulator's prompt at this call, when it was sent

    @property
    def wrist(self):
        """The wrist picture (None when the call had one camera)."""
        return self.images[1] if len(self.images) > 1 else None


def finite_number(value) -> float | None:
    """The value as a float; None when it is missing, not a number or not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


class TaskOrderMonitor(RuleMonitor):
    """The rules writer in the store's label space: a lift of an object the task order does not expect next writes nothing,
    where the writer itself would move that object's skills to the front of the plan."""

    def _bring_forward(self, held: str) -> bool:
        return False


class RulesLabeller:
    """Labels the calls of an episode from the robot's own signals: the rules writer replayed over them gives the skill
    pointer, the grasp cycle and a colour identification of the held object give the state of every object, and the
    closedness and hand height give the gripper's."""

    name = "rules"

    def __init__(self, task, high_above: float | None = None, lifted_above: float | None = None):
        self.task = task
        self.identifier = make_identifier(task)
        self.monitor = TaskOrderMonitor(task, identifier=self.identifier)
        proprio = task.proprio or {}
        self.closed_threshold = float(proprio.get("closed_threshold", 0.3))
        self.high_above_table = float(high_above if high_above is not None else proprio.get("high_above_table", HIGH_ABOVE_TABLE))
        self.lifted_above_grasp = float(lifted_above if lifted_above is not None
                                        else proprio.get("lifted_above_grasp", LIFTED_ABOVE_GRASP))

    def label(self, calls: Sequence[Call]):
        """(the label of every call, whether the task is done at every call, the number of skills done at the end)."""
        labels, done = self.replay(calls)
        return labels, done, len(self.monitor.completed())

    def replay(self, calls: Sequence[Call]):
        """(labels, task done) per call, from the rules writer replayed over the calls' signals."""
        monitor, task = self.monitor, self.task
        monitor.start_episode()
        labels, done = [], []
        for call in calls:
            monitor.observe_rows(call.rows, call.step, image=call.wrist)
            label = {task.field(name): "on_table" for name in task.objects}
            held = self._held(call)
            if held is not None:
                label[task.field(held[0])] = held[1]
            label["gripper"] = self._gripper(call)
            label["gripper_high_above_table"] = bool(call.hand_height is not None and call.hand_height > self.high_above_table)
            label["subgoal"] = task.skills.index(monitor.current_skill)
            labels.append(label)
            done.append(bool(monitor.done_all))
        return labels, done

    def _gripper(self, call: Call) -> str:
        """"closed" from the closedness threshold on, else "open"; "unsure" when the call has no closedness."""
        if call.closedness is None:
            return "unsure"
        return "closed" if call.closedness >= self.closed_threshold else "open"

    def _held(self, call: Call):
        """(object, "held_lifted" or "held_at_table") when the gripper holds an object at this call, else None. The object is
        the one the colour identifier sees in the wrist picture, else the current skill's."""
        cycle = self.monitor.cycle
        if not len(call.rows) or cycle.state not in ("holding", "lifted") or cycle.grasp_height is None:
            return None
        width, effort, height = (float(call.rows[-1][column]) for column in (WIDTH_COLUMN, EFFORT_COLUMN, TIP_Z_COLUMN))
        if not (cycle.empty_width < width < cycle.open_width and effort > cycle.squeeze_force and np.isfinite(height)):
            return None
        identified = self.identifier.identify(call.wrist) if self.identifier is not None and call.wrist is not None else None
        held_object = identified or self.task.parse_skill(self.monitor.current_skill)[1]
        return held_object, ("held_lifted" if height - cycle.grasp_height > self.lifted_above_grasp else "held_at_table")


class OracleLabeller(RulesLabeller):
    """The skill of a call is the simulator's prompt at that call (simulation only); a call without a prompt that is one of
    the task's skills has no label. The task is done when the rules writer says so at the last skill."""

    name = "oracle"

    def label(self, calls: Sequence[Call]):
        """As RulesLabeller.label, with the skills from the simulator's prompts."""
        labels, writer_done = self.replay(calls)
        done, last_skill, skill_index = [], len(self.task.skills) - 1, 0
        for position, (label, call, finished) in enumerate(zip(labels, calls, writer_done)):
            if call.oracle in self.task.skills:
                label["subgoal"] = skill_index = self.task.skills.index(call.oracle)
            else:
                labels[position] = None
            done.append(bool(finished) and labels[position] is not None and label["subgoal"] == last_skill)
        return labels, done, skill_index + int(bool(done and done[-1]))


LABELLERS = {"rules": RulesLabeller, "oracle": OracleLabeller}


class Clock:
    """The milliseconds of each step of one write, and of the whole."""

    def __init__(self):
        self.started = self.last = time.perf_counter()
        self.ms: dict = {}

    def lap(self, name: str) -> None:
        """The time since the previous lap (or the start) belongs to `name`."""
        now = time.perf_counter()
        self.ms[name] = round(1000 * (now - self.last), 1)
        self.last = now

    def report(self) -> dict:
        """The laps and their total."""
        return {**self.ms, "total": round(1000 * (self.last - self.started), 1)}


class EpisodeRecorder:
    """The calls of the running episode, and their labelled admission into the store when it ends. labeller: a name of
    LABELLERS, built with `labeller_options`; dedup and dedup_copies: the near-duplicate rule (module docstring)."""

    def __init__(self, task, labeller: str = "rules", dedup: float = DEDUP_COS, dedup_copies: int = DEDUP_COPIES,
                 **labeller_options):
        self.task = task
        self.labeller = LABELLERS[labeller](task, **labeller_options)
        self.dedup, self.dedup_copies = float(dedup), int(dedup_copies)
        self.calls: list[Call] = []
        # the labels two calls must share to be near-duplicates
        self.fields = [task.field(name) for name in task.objects] + ["gripper", "gripper_high_above_table", "subgoal"]

    def record(self, control_keys: dict, images, record: dict):
        """Keep one monitored call: its pictures and signals; `record` is the planner's record of the call (its step)."""
        step = record.get("step", control_keys.get("step"))
        self.calls.append(Call(step=int(step) if step is not None else -1,
                               images=[np.array(picture, copy=True) for picture in images],
                               rows=signal_rows(control_keys), closedness=finite_number(control_keys.get("gripper_closedness")),
                               hand_height=finite_number(control_keys.get("hand_height")),
                               oracle=text_of(control_keys.get("oracle_prompt"))))

    def finish(self, store, episode: str, context: dict) -> dict | None:
        """Label, admit and insert the recorded calls; evict; returns the store_update record (None: no calls)."""
        calls, self.calls = self.calls, []
        if not calls:
            return None
        clock = Clock()
        labels, done, skills_done = self.labeller.label(calls)
        clock.lap("label")
        update = int(store.info.get("updates", 0)) + 1
        admitted, dropped = self._admissible(calls, labels, done)
        keys = {name: store.encoder.encode(name, [calls[n].images[:2] for n in admitted]) for name in store.keys}
        clock.lap("keys")
        kept = self._without_duplicates(keys["colour"], [labels[n] for n in admitted])
        inserted = [admitted[position] for position in kept]
        complete = bool(done and done[-1])
        payloads = [self._payload(calls[n], labels[n], episode, update, context, complete, skills_done) for n in inserted]
        ids = store.insert([calls[n].images[:2] for n in inserted], payloads, {name: keys[name][kept] for name in store.keys})
        clock.lap("insert")
        removed = store.evict()
        clock.lap("evict")
        store.end_update()
        clock.lap("counters")
        return {"event": "store_update", "episode": episode, "calls": len(calls), "added": len(ids),
                "rejected_duplicates": len(admitted) - len(kept), "dropped": dict(dropped), "removed": len(removed),
                "removed_ids": removed[:2000], "size": len(store), "complete": complete, "skills_done": int(skills_done),
                "session": context.get("session"), "instruction": context.get("instruction"), "update": update,
                "oracle_agreement": self._agreement(calls, labels), "ms": clock.report()}

    def _admissible(self, calls: list[Call], labels: list, done: list) -> tuple[list[int], Counter]:
        """(the positions of the calls that may be stored, why the others are not): no label, after the task's last skill
        ("terminal"), or fewer than two pictures ("camera_mismatch")."""
        admitted, dropped = [], Counter()
        for position, (call, label, finished) in enumerate(zip(calls, labels, done)):
            if label is None:
                dropped["unlabelled"] += 1
            elif finished:
                dropped["terminal"] += 1
            elif len(call.images) < 2:
                dropped["camera_mismatch"] += 1
            else:
                admitted.append(position)
        return admitted, dropped

    def _without_duplicates(self, colour_keys: np.ndarray, labels: list) -> list[int]:
        """The positions (in call order) to keep: a call is left out when the episode already keeps `dedup_copies` calls with
        the same labels and a colour-key cosine of at least `dedup` to it."""
        if not len(labels) or self.dedup <= 0:
            return list(range(len(labels)))
        numbers: dict = {}                           # each distinct combination of the compared labels gets a number
        codes = [numbers.setdefault(tuple(label.get(field) for field in self.fields), len(numbers)) for label in labels]
        similarity = colour_keys @ colour_keys.T
        kept: list[int] = []
        for position in range(len(labels)):
            copies = sum(1 for other in kept if codes[other] == codes[position] and similarity[position, other] >= self.dedup)
            if copies < self.dedup_copies:
                kept.append(position)
        return kept

    def _payload(self, call: Call, label: dict, episode: str, update: int, context: dict, complete: bool,
                 skills_done: int) -> dict:
        """The store's payload of one call: where it came from, what was true then, how the episode ended."""
        state = {name: value for name, value in label.items() if name != "subgoal"}
        subgoal = int(label["subgoal"])
        return {"session": context.get("session"), "task": self.task.name, "instruction": context.get("instruction", ""),
                "plan": context.get("plan", ""), "skill": self.task.skills[subgoal], "subgoal": subgoal, "phase": -1,
                "labels": state, "time": time.time(), "episode": f"{context.get('session')}/{episode}@{update}",
                "step": int(call.step), "outcome": "complete" if complete else "incomplete", "skills_done": int(skills_done),
                "provenance": "self", "labeller": self.labeller.name, "update_no": update}

    def _agreement(self, calls: list[Call], labels: list) -> dict | None:
        """The labels against the simulator's prompts at the calls, when it sent them (a check of the labels' quality)."""
        pairs = [(label["subgoal"], self.task.skills.index(call.oracle)) for call, label in zip(calls, labels)
                 if label is not None and call.oracle in self.task.skills]
        if not pairs:
            return None
        return {"calls": len(pairs), "same": sum(a == b for a, b in pairs), "ahead": sum(a > b for a, b in pairs),
                "behind": sum(a < b for a, b in pairs)}
