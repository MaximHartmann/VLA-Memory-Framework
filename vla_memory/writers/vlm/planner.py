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

Default configuration (the reference evaluation's; every part can be switched off):
    gate           the memory-vote gate: the k_vote (6) stored moments nearest to the observation vote; a moment votes "done"
                   when its stored skill is later in the task's skill order than the current skill. A full vote with fewer
                   than gate_min_done (1) "done" votes skips the question and counts as "not done". The last skill of the plan,
                   and the task's last skill, are always asked. Without a memory store the gate is inactive.
    verdict_only   the model answers {"current_skill_done": bool} only. verdict_only=False adds the object states, the
                   gripper fields and a reason (brief=True drops the reason); `recovery` needs those states.
    k = 4 examples per question, votes_to_advance = 1, min_calls_per_skill = 2, call_every = 1.

Working memory = plan + pointer + recent proprioception, rebuilt at every start_episode. Long-term memory = the MemoryStore,
read at every question and never written within an episode. Optional memories plug in as extensions (proxy/extensions.py): they
add text to the plan request and the question, fields to the records, and may veto a switch.

Every monitor record carries the step, the skill, the latest proprioception, whether the model was asked (`queried`), whether
the gate settled the call (`gated`), the vote, the answer (`vlm`) and whether the pointer moved (`advanced`); the plan record
carries the configuration (`config`). log_event() appends records of other components to the same log. The plan request
lives in plan.py, the question in question.py, the retrieval and the gate in gate.py.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from ...proxy.extensions import Extension
from .gate import retrieve_with_gate
from .plan import ask_for_plan, chosen_plan, extension_plan_input, recovery_fix
from .proprio import ProprioSummary
from .question import monitor_messages
from ...task import TaskSpec
from ...vlm import VLMBackend, encode_png


class Planner:
    """The VLM planner's state and decision loop (module docstring). memory: a MemoryStore or None; log: the path of the
    decision log (one JSON record per plan and per monitor call); exclude_episode: a past episode the store must not
    retrieve (leave-one-out evaluation); log_memory: note the store size and the retrieval time in every record."""

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
        self.log_memory = log_memory
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
        self.proprio = ProprioSummary.from_task(task)
        self._log_file = open(log, "a") if log else None
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
        self.skill_index = 0
        self.calls = 0
        self.calls_in_skill = 0
        self.consecutive_done = 0
        self.history = []
        self.done_all = False
        self.proprio.reset()
        self.votes, self.call_every, self.min_calls = self.base_parameters

    def config(self) -> dict:
        """The decision parameters in effect, as logged with every plan."""
        config = {"gate": self.gate, "k_vote": self.k_vote, "gate_min_done": self.gate_min_done, "verdict_only": self.verdict_only,
                  "brief": self.brief, "k": self.k, "votes_to_advance": self.votes, "min_calls_per_skill": self.min_calls,
                  "call_every": self.call_every, "memory_entries": len(self.memory) if self.memory is not None else 0}
        for extension in self.extensions:
            extension_config = extension.config()
            if extension_config:
                config[extension.name] = extension_config
        return config

    @property
    def current_skill(self) -> str:
        """The skill the pointer is on (the last one once the plan is complete)."""
        return self.plan[min(self.skill_index, len(self.plan) - 1)]

    def current_label(self) -> int:
        """The current skill's index in the task's skill order (the store's label space)."""
        return self.task.skills.index(self.current_skill)

    # ------------------------------------------------------------------ the plan
    def start_episode(self, instruction: str, episode_id: str) -> dict:
        """Ask the model for the plan of this episode. Returns the plan record (also written to the log)."""
        self._reset(episode_id)
        extra_text, fields = extension_plan_input(self, instruction)
        record = {"event": "plan", "episode": episode_id, "instruction": instruction, **fields, "config": self.config()}
        plan = ask_for_plan(self.vlm, self.task, instruction, extra_text, record)
        self.plan = chosen_plan(self.task, plan, record)
        self._log_record(record)
        return record

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
        hits = retrieve_with_gate(self, pictures, record)
        if record["gated"]:
            self.consecutive_done = 0
            record["consecutive_done"] = 0
            return self._finish(record)
        messages = monitor_messages(self, pictures, hits, record)
        answer = self._ask(messages, record)
        if self._recovery_step(answer, record):
            return self._finish(record)
        done = self._count_vote(answer, record)
        if done:
            self._advance_if_ready(record)
        record["skill_after"] = self.current_skill
        for extension in self.extensions:
            extension.end_call(record)
        self._log_record(record)
        return record

    def _is_monitor_call(self) -> bool:
        """Whether this call asks the model: every call_every-th call, counted from the first."""
        return (self.calls - 1) % self.call_every == 0

    def _pictures(self, images) -> list:
        """The pictures the model sees: both cameras, or the external camera only."""
        return list(images) if self.use_wrist else list(images[:1])

    def _new_record(self, step: int) -> dict:
        """The monitor record of a call before any decision."""
        record = {"event": "monitor", "episode": self.episode, "step": step, "call": self.calls, "skill_index": self.skill_index,
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
        self._log_record(record)
        return record

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
        put_down = recovery_fix(self.task, answer, self.current_skill)
        if put_down is None:
            return False
        self.plan.insert(self.skill_index, put_down)
        record["recovery_inserted"] = put_down
        self.consecutive_done = 0
        self.calls_in_skill = 0
        return True

    def _count_vote(self, answer: dict, record: dict) -> bool:
        """Whether the answer says "done" (a string "false" from a backend without schema enforcement is not done)."""
        done = answer.get("current_skill_done") is True
        self.consecutive_done = self.consecutive_done + 1 if done else 0
        record["consecutive_done"] = self.consecutive_done
        return done

    def _advance_if_ready(self, record: dict):
        """Move the pointer after enough "done" answers and calls, if every extension allows it."""
        if self.consecutive_done < self.votes or self.calls_in_skill < self.min_calls:
            return
        skill = self.current_skill
        if not all(extension.allow_switch(skill, record) for extension in self.extensions):
            return
        self.skill_index += 1
        self.consecutive_done = 0
        self.calls_in_skill = 0
        record["advanced"] = True
        if self.skill_index >= len(self.plan):
            self.done_all = True
            self.skill_index = len(self.plan) - 1
            record["plan_complete"] = True

    # ------------------------------------------------------------------ pictures and the log
    def encode_picture(self, image) -> str:
        """A picture as the data URL the chat API takes, enlarged by `upscale` first."""
        return encode_png(self._prepare_picture(image))

    def _prepare_picture(self, image):
        """The image as an array, enlarged by the upscale factor (bicubic) when one is set."""
        image = np.asarray(image)
        if not self.upscale or self.upscale == 1:
            return image
        from PIL import Image
        picture = Image.fromarray(np.ascontiguousarray(image))
        return np.asarray(picture.resize((picture.width * self.upscale, picture.height * self.upscale), Image.BICUBIC))

    def _log_record(self, record: dict):
        """Keep the record in the episode's history and append it to the decision log."""
        self.history.append(record)
        if self._log_file:
            self._log_file.write(json.dumps(record, default=str) + "\n")
            self._log_file.flush()

    def log_event(self, record: dict):
        """Append an event of another component to the decision log (e.g. an extension's end-of-episode record)."""
        self._log_record(dict(record))

    def close(self):
        """Close the decision log."""
        if self._log_file:
            self._log_file.close()
