"""The planner: plans once per episode, monitors on every policy call, and holds the working memory of the episode.

    start_episode(instruction)      a text-only call: the model returns the ordered skill list under a JSON schema that
                                    admits only the task's skills. If the call fails or a skill is outside the vocabulary,
                                    the plan falls back to the task's skills in their listed order (logged as the source).
    observe(images, step, proprio)  on every policy call. With a memory store, the memory-vote gate may settle the call from
                                    the store alone; otherwise the model sees the plan, the completion rule of the current
                                    skill, the proprioception sentence, the retrieved examples and the current images, and
                                    answers whether the CURRENT skill is complete. The plan pointer only ever moves forward,
                                    after `votes_to_advance` consecutive "done" answers and never within the first
                                    `min_calls_per_skill` calls of a skill.

Default configuration (settled 2026-09-30 on the reference task; every part can be switched off):
    gate           the memory-vote gate: the k_vote (6) stored moments nearest to the observation vote; a moment votes "done"
                   when its stored skill is later in the task's skill order than the current skill. A full vote with fewer
                   than gate_min_done (1) "done" votes skips the question and counts as "not done". The last skill of the plan,
                   and the task's last skill, are always asked. Without a memory store the gate is inactive.
    verdict_only   the model answers {"current_skill_done": bool} only. verdict_only=False adds the object states, the
                   gripper fields and a reason (brief=True drops the reason); `recovery` needs those states.
    k = 4 examples per question, votes_to_advance = 1, min_calls_per_skill = 2, call_every = 1.

Working memory = plan + pointer + recent proprioception, rebuilt at every start_episode. Long-term memory = the MemoryStore,
read at every question and never written within an episode. Optional memories plug in as extensions (extensions.py): they
add text to the plan request and the question, fields to the records, and may veto a switch.

Every monitor record carries the step, the skill, the latest proprioception, whether the model was asked (`queried`), whether
the gate settled the call (`gated`), the vote, the answer (`vlm`) and whether the pointer moved (`advanced`); the plan record
carries the configuration (`config`). log_event() appends records of other components to the same log.
"""
from __future__ import annotations

import json
import time
from typing import Any, Sequence

import numpy as np

from .extensions import Extension, join_texts
from .gate import gate_skips, vote
from .proprio import ProprioSummary
from .question import ask_text, plan_messages, question_text
from ..task import TaskSpec
from ..vlm import VLMBackend, encode_png

HELD_STATES = ("held_lifted", "held_at_table")


def proprio_summary_of(task: TaskSpec) -> ProprioSummary:
    """The proprioception sentence builder with the task file's thresholds."""
    settings = task.proprio or {}
    return ProprioSummary(settings.get("closed_threshold", 0.3), settings.get("history", 12), settings.get("note", ""))


def recovery_fix(task: TaskSpec, answer: dict, current_skill: str) -> str | None:
    """The put-down skill to insert before a pick when the answer says another object is still held, else None.
    Only for a task with exactly two verbs (pick, place)."""
    verb, obj = task.parse_skill(current_skill)
    if len(task.verbs) != 2 or verb != task.verbs[0]:
        return None
    for other in task.objects:
        if other != obj and answer.get(task.field(other)) in HELD_STATES:
            return task.skill(task.verbs[1], other)
    return None


class Planner:
    def __init__(self, vlm: VLMBackend, task: TaskSpec, memory=None, k: int = 4, votes_to_advance: int = 1,
                 call_every: int = 1, min_calls_per_skill: int = 2, recovery: bool = False, log=None,
                 use_wrist: bool = True, upscale: int = 1, brief: bool = False, exclude_episode: Any = None,
                 gate: bool = True, k_vote: int = 6, gate_min_done: int = 1, verdict_only: bool = True,
                 log_memory: bool = False, extensions: Sequence[Extension] = ()):
        if recovery and verdict_only:
            raise ValueError("recovery reads the object states from the monitor's answer: use verdict_only=False with it")
        self.vlm = vlm
        self.task = task
        self.memory = memory
        self.k = k
        self.exclude_episode = exclude_episode
        self.log_memory = log_memory                 # store size and retrieval time in every monitor record
        self.base_parameters = (votes_to_advance, call_every, min_calls_per_skill)
        self.votes, self.call_every, self.min_calls = self.base_parameters
        self.recovery = recovery
        self.use_wrist = use_wrist
        self.upscale = upscale
        self.brief = brief
        self.gate = gate
        self.k_vote = k_vote
        self.gate_min_done = gate_min_done
        self.verdict_only = verdict_only
        self.extensions = list(extensions)
        self.schema = self._answer_schema()
        self.proprio = proprio_summary_of(task)
        self._log = open(log, "a") if log else None
        self.history: list[dict] = []
        self._reset(None)

    def _answer_schema(self) -> dict:
        """The monitor's answer schema: the task's, changed by the extensions that need more fields."""
        schema = self.task.monitor_schema(brief=self.brief, verdict_only=self.verdict_only)
        for extension in self.extensions:
            schema = extension.answer_schema(schema)
        return schema

    def _reset(self, episode_id):
        """The working memory of a new episode."""
        self.episode = episode_id
        self.plan: list[str] = []
        self.idx = 0
        self.calls = 0
        self.calls_in_skill = 0
        self.consec_done = 0
        self.history = []
        self.done_all = False
        self.proprio.reset()
        self.votes, self.call_every, self.min_calls = self.base_parameters

    def config(self) -> dict:
        """The decision parameters in effect, as logged with every plan."""
        cfg = {"gate": self.gate, "k_vote": self.k_vote, "gate_min_done": self.gate_min_done, "verdict_only": self.verdict_only,
               "brief": self.brief, "k": self.k, "votes_to_advance": self.votes, "min_calls_per_skill": self.min_calls,
               "call_every": self.call_every, "memory_entries": len(self.memory) if self.memory is not None else 0}
        for extension in self.extensions:
            extension_config = extension.config()
            if extension_config:
                cfg[extension.name] = extension_config
        return cfg

    @property
    def current_skill(self) -> str:
        return self.plan[min(self.idx, len(self.plan) - 1)]

    # ------------------------------------------------------------------ the plan
    def start_episode(self, instruction: str, episode_id: str) -> dict:
        """Ask the model for the plan of this episode. Returns the plan record (also written to the log)."""
        self._reset(episode_id)
        extra_text, fields = self._extension_plan_input(instruction)
        record = {"event": "plan", "episode": episode_id, "instruction": instruction, **fields, "config": self.config()}
        plan = self._ask_for_plan(instruction, extra_text, record)
        self._set_plan(plan, record)
        self._write(record)
        return record

    def _extension_plan_input(self, instruction: str) -> tuple[str, dict]:
        """What the extensions add to the plan request: parameter overrides, text, record fields."""
        texts = []
        fields: dict = {}
        for extension in self.extensions:
            overrides = extension.parameters()
            self.votes = overrides.get("votes_to_advance", self.votes)
            self.call_every = overrides.get("call_every", self.call_every)
            self.min_calls = overrides.get("min_calls_per_skill", self.min_calls)
            text, extension_fields = extension.plan_request(instruction)
            if text:
                texts.append(text)
            fields.update(extension_fields)
        return "\n".join(texts), fields

    def _ask_for_plan(self, instruction: str, extra_text: str, record: dict):
        """The model's plan, or None when the call failed (the record keeps the error)."""
        messages = plan_messages(self.task, instruction, extra_text)
        try:
            reply = self.vlm.chat(messages, schema=self.task.plan_schema(), name="plan", max_tokens=256)
        except Exception as error:   # server down, timeout, bad JSON
            record["error"] = repr(error)
            return None
        plan = (reply.get("json") or {}).get("plan")
        record.update(vlm_plan=plan, ms=reply.get("ms", 0.0), reasoning=reply.get("reasoning"),
                      tokens=[reply.get("prompt_tokens"), reply.get("completion_tokens")])
        return plan

    def _set_plan(self, plan, record: dict):
        """The model's plan if every skill is in the vocabulary, else the task's skill order."""
        if plan and all(skill in self.task.skills for skill in plan):
            self.plan = list(plan)
            record["source"] = "vlm"
        else:
            self.plan = list(self.task.skills)
            record["source"] = "fallback_skill_order"
        record["plan"] = self.plan

    # ------------------------------------------------------------------ the monitor call
    def observe(self, images: Sequence[np.ndarray], step: int, proprio: dict | None = None) -> dict:
        """One policy call: images = (exterior, wrist) or (exterior,). Returns the decision record (also logged)."""
        self.calls += 1
        self.calls_in_skill += 1
        self.proprio.push(proprio)
        pictures = self._pictures(images)
        record = self._new_record(step)
        for extension in self.extensions:
            record.update(extension.begin_call(self.episode, step, pictures, self.proprio.latest, self.current_skill))
        if self.done_all or not self._is_monitor_call():
            return self._finish(record, plan_complete=self.done_all)
        hits = self._retrieve(pictures, record)
        if record["gated"]:
            self.consec_done = 0
            record["consecutive_done"] = 0
            return self._finish(record)
        messages = self._question(pictures, hits, record)
        answer = self._ask(messages, record)
        if self._recovery_step(answer, record):
            return self._finish(record)
        done = self._count_vote(answer, record)
        if done:
            self._advance_if_ready(record)
        record["skill_after"] = self.current_skill
        for extension in self.extensions:
            extension.end_call(record)
        self._write(record)
        return record

    def _is_monitor_call(self) -> bool:
        return (self.calls - 1) % self.call_every == 0

    def _pictures(self, images) -> list:
        """The pictures the model sees: both cameras, or the external camera only."""
        return list(images) if self.use_wrist else list(images[:1])

    def _label(self) -> int:
        """The current skill's index in the task's skill order (the store's label space)."""
        return self.task.skills.index(self.current_skill)

    def _new_record(self, step: int) -> dict:
        record = {"event": "monitor", "episode": self.episode, "step": step, "call": self.calls, "skill_index": self.idx,
                  "skill": self.current_skill, "queried": False, "gated": False, "advanced": False,
                  "proprio": self.proprio.latest}
        if self.log_memory:
            record["memory_entries"] = len(self.memory) if self.memory is not None else 0
        return record

    def _finish(self, record: dict, plan_complete=None) -> dict:
        """A call without a question (not a monitor call, or settled by the gate): close and log the record."""
        record["skill_after"] = self.current_skill
        if plan_complete is not None:
            record["plan_complete"] = plan_complete
        self._write(record)
        return record

    def _retrieve(self, pictures, record: dict) -> list:
        """The stored moments most similar to the pictures; the gate may settle the call (record["gated"])."""
        if self.memory is None or not len(self.memory):
            return []
        label = self._label()
        use_gate = self.gate and self.k_vote > 0
        k = max(self.k, self.k_vote) if use_gate else self.k
        started = time.perf_counter()
        hits = self.memory.retrieve(pictures, k=k, exclude_episode=self.exclude_episode)
        if self.log_memory:
            record["retrieval_ms"] = round(1000 * (time.perf_counter() - started), 3)
        if use_gate:
            verdict = vote(self.memory, hits[:self.k_vote], label)
            record["vote"] = verdict
            if gate_skips(verdict, self.k_vote, self.gate_min_done, self.idx >= len(self.plan) - 1,
                          label >= len(self.task.skills) - 1):
                record["gated"] = True
                return []
            hits = hits[:self.k]                 # the model sees the k nearest of them
        record["exemplars"] = [{"i": j, "sim": round(sim, 3), **self.memory.describe_hit(j)} for j, sim in hits]
        return hits

    def _question(self, pictures, hits, record: dict) -> list[dict]:
        """The chat messages of the monitor question: examples, the extensions' content, the text, the current pictures."""
        skill = self.current_skill
        items: list[dict] = []
        n_pictures = 0
        if hits:
            items = self.memory.fewshot_content(hits, self._label(), skill, encode_png)
            n_pictures = sum(1 for item in items if item.get("type") == "image_url")
        extension_items = self._extension_examples(n_pictures, len(pictures), record)
        items += extension_items
        n_pictures += sum(1 for item in extension_items if item.get("type") == "image_url")
        criterion_extra = self._extension_criterion_text(skill, record)
        two_cameras = len(pictures) >= 2
        text = question_text(self.task, self.plan, self.idx, criterion_extra, self.proprio.text(),
                             join_texts(self.extensions, "state_text"), n_pictures + 1, two_cameras,
                             ask_text(self.task, self.verdict_only) + join_texts(self.extensions, "ask_text"),
                             bool(hits) or bool(extension_items))
        items.append({"type": "text", "text": text})
        for picture in pictures:
            items.append({"type": "image_url", "image_url": {"url": self._encode(picture)}})
        return [{"role": "system", "content": self.task.system_text(two_cameras)}, {"role": "user", "content": items}]

    def _extension_examples(self, pictures_before: int, pictures_after: int, record: dict) -> list:
        """Content the extensions show after the retrieved examples (their record fields go into the record)."""
        items: list = []
        for extension in self.extensions:
            extension_items, n_pictures, fields = extension.examples(pictures_before, pictures_after, self._encode)
            items += extension_items
            pictures_before += n_pictures
            record.update(fields)
        return items

    def _extension_criterion_text(self, skill: str, record: dict) -> str:
        text = ""
        for extension in self.extensions:
            extension_text, fields = extension.criterion_text(skill)
            text += extension_text
            record.update(fields)
        return text

    def _ask(self, messages: list[dict], record: dict) -> dict:
        """The model's answer as a dict ({} when the call failed; the record keeps the error)."""
        record["queried"] = True
        try:
            reply = self.vlm.chat(messages, schema=self.schema, name="monitor")
        except Exception as error:
            record["error"] = repr(error)
            return {}
        answer = reply.get("json") or {}
        record.update(vlm=answer, ms=reply.get("ms", 0.0), reasoning=reply.get("reasoning"),
                      tokens=[reply.get("prompt_tokens"), reply.get("completion_tokens")])
        return answer

    def _recovery_step(self, answer: dict, record: dict) -> bool:
        """Optional recovery: a pick is active but another object is still held, so its put-down comes first."""
        if not (self.recovery and answer):
            return False
        fix = recovery_fix(self.task, answer, self.current_skill)
        if fix is None:
            return False
        self.plan.insert(self.idx, fix)
        record["recovery_inserted"] = fix
        self.consec_done = 0
        self.calls_in_skill = 0
        return True

    def _count_vote(self, answer: dict, record: dict) -> bool:
        """Whether the answer says "done" (a string "false" from a backend without schema enforcement is not done)."""
        done = answer.get("current_skill_done") is True
        self.consec_done = self.consec_done + 1 if done else 0
        record["consecutive_done"] = self.consec_done
        return done

    def _advance_if_ready(self, record: dict):
        """Move the pointer after enough "done" answers and calls, if every extension allows it."""
        if self.consec_done < self.votes or self.calls_in_skill < self.min_calls:
            return
        skill = self.current_skill
        if not all(extension.allow_switch(skill, record) for extension in self.extensions):
            return
        self.idx += 1
        self.consec_done = 0
        self.calls_in_skill = 0
        record["advanced"] = True
        if self.idx >= len(self.plan):
            self.done_all = True
            self.idx = len(self.plan) - 1
            record["plan_complete"] = True

    # ------------------------------------------------------------------ pictures and the log
    def _encode(self, image) -> str:
        return encode_png(self._prepare_image(image))

    def _prepare_image(self, image):
        image = np.asarray(image)
        if not self.upscale or self.upscale == 1:
            return image
        from PIL import Image
        pil = Image.fromarray(np.ascontiguousarray(image))
        return np.asarray(pil.resize((pil.width * self.upscale, pil.height * self.upscale), Image.BICUBIC))

    def _write(self, record: dict):
        self.history.append(record)
        if self._log:
            self._log.write(json.dumps(record, default=str) + "\n")
            self._log.flush()

    def log_event(self, record: dict):
        """Append an event of another component to the decision log (e.g. an extension's end-of-episode record)."""
        self._write(dict(record))

    def close(self):
        if self._log:
            self._log.close()
