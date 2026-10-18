"""The relay between an environment client and the VLA policy server, with a writer and a method.

    environment  --ws-->  Relay (:LISTEN)  --ws-->  policy server (:UPSTREAM)
                            |  a writer signals that a step of the task is done (writers/)
                            |  a method turns that into what the policy receives (methods/)

The environment sends its ordinary observation with the FULL instruction as `prompt`, plus control keys under the
`memory/` prefix (vla_memory/env_keys.py). The relay strips them and, at every call, in this order:

    1. a new episode: the extensions, the method and the writer (its plan) start it
    2. writer.before_call      a writer that decides from this call's inputs
    3. method.prompt           the prompt of this call shows that decision (with the writer "off": the instruction)
    4. method.before_policy    keys added to the observation (a picture)
       extensions' before_policy   keys added by the extensions (a retrieved memory)
    5. the policy's call
    6. the extensions' on_reply, then writer.after_call: a writer that decides from the policy's reply; its decision shows
       from the next call on
    7. the reply goes back with a `planner` info dict; one JSON record per call goes to the log

Neither the environment nor the policy server needs to know about the writer or the method. The command line is in
cli.py:  python -m vla_memory.proxy --writer rules --method history --listen-port <port> --upstream-port <policy port>
"""
from __future__ import annotations

import json
import logging
import time
from typing import Sequence

from .extensions import Extension, collect
from ..task import TaskSpec
from ..transport import PolicyTransport
from ..writers.base import Call

log = logging.getLogger("vla_memory.proxy")
CTRL = "memory/"


def split_control_keys(observation: dict) -> tuple[dict, dict]:
    """The memory/ control keys without their prefix, and the observation without them."""
    ctrl = {}
    policy_obs = {}
    for key, value in observation.items():
        if isinstance(key, str) and key.startswith(CTRL):
            ctrl[key[len(CTRL):]] = value
        else:
            policy_obs[key] = value
    return ctrl, policy_obs


def text_of(value):
    """A string that may have arrived as bytes over the wire."""
    return value.decode() if isinstance(value, bytes) else value


class Relay:
    def __init__(self, transport: PolicyTransport, writer, method, log_path=None,
                 image_keys=("observation/image", "observation/wrist_image"), task: TaskSpec | None = None,
                 extensions: Sequence[Extension] = ()):
        self.transport = transport
        self.writer = writer               # keeps the plan and its pointer
        self.method = method               # turns them into what the policy receives
        self.image_keys = tuple(image_keys)
        self.task = task if task is not None else getattr(writer, "task", None)
        self.extensions = list(extensions)
        self.episode = None
        self.episode_count = 0
        self._log = open(log_path, "a") if log_path else None
        log.info("upstream metadata: %s", transport.metadata())

    @property
    def mode(self) -> str:
        """The writer's name, as every log record says it."""
        return self.writer.name

    # ------------------------------------------------------------------ one policy call
    def infer(self, observation: dict) -> dict:
        """Relay one policy call: decide the prompt, forward the observation, return the reply with the `planner` info."""
        started = time.perf_counter()
        ctrl, obs = split_control_keys(observation)
        instruction = text_of(obs.get("prompt", ""))
        info = {"mode": self.writer.name, "queried": False, "advanced": False, "vlm_ms": 0.0}
        if self._is_new_episode(ctrl):
            self._begin_episode(ctrl, instruction, info)
        info.update(self.writer.before_call(Call(instruction, ctrl, obs, self.image_keys), self.extensions))
        prompt = self.method.prompt(instruction, self.writer) if self.writer.drives_prompt else instruction
        info.update(collect(self.extensions, "log_fields"))
        obs["prompt"] = prompt
        info.update(self.method.before_policy(obs, ctrl, self.writer))
        for extension in self.extensions:
            info.update(extension.before_policy(obs, ctrl, prompt, self.writer.monitor))
        info["prompt"] = obs["prompt"]
        info["oracle_prompt"] = ctrl.get("oracle_prompt")
        reply = self._forward(obs, info)
        self._reply_hooks(obs, ctrl, reply, prompt, info)
        info.update(self.writer.after_call(reply))
        info["total_ms"] = 1000 * (time.perf_counter() - started)
        reply["planner"] = info
        self._log_call(ctrl, info)
        return reply

    # ------------------------------------------------------------------ episodes
    def _is_new_episode(self, ctrl: dict) -> bool:
        episode_id = text_of(ctrl.get("episode"))
        if ctrl.get("new_episode") or self.episode is None:
            return True
        return episode_id is not None and str(episode_id) != self.episode

    def _begin_episode(self, ctrl: dict, instruction: str, info: dict):
        """Close the episode that just ended, name the new one, and let the writer plan it."""
        self._end_episode()
        episode_id = text_of(ctrl.get("episode"))
        self.episode = str(episode_id) if episode_id is not None else f"ep{self.episode_count}"
        self.episode_count += 1
        for extension in self.extensions:
            extension.start_episode(self.episode, ctrl)
        self.method.start_episode(self.episode, ctrl)
        record = self.writer.start_episode(instruction, self.episode)
        if record is not None:
            info.update(plan=record["plan"], plan_source=record["source"], plan_ms=record.get("ms"))
            log.info("episode %s plan (%s): %s", self.episode, record["source"], record["plan"])

    def _end_episode(self):
        """The extensions take the finished episode; their records go to the writer's log."""
        for extension in self.extensions:
            record = extension.end_episode(self.writer.monitor)
            if record is not None:
                self._log_event(record)

    def _log_event(self, record: dict):
        log_event = getattr(self.writer.monitor, "log_event", None)
        if log_event is not None:
            log_event(record)

    # ------------------------------------------------------------------ the policy's reply
    def _forward(self, obs: dict, info: dict) -> dict:
        started = time.perf_counter()
        reply = self.transport.infer(obs)
        info["upstream_ms"] = 1000 * (time.perf_counter() - started)
        return reply

    def _reply_hooks(self, obs: dict, ctrl: dict, reply: dict, prompt: str, info: dict):
        """Extensions that look at (or change) the policy's reply; each one's record goes into the info."""
        skill = self.current_skill(prompt)
        for extension in self.extensions:
            record = extension.on_reply(obs, ctrl, reply, skill, self.episode)
            if record is not None:
                info[extension.name] = record

    def current_skill(self, prompt: str) -> str | None:
        """The skill the policy works on now: the prompt if it is a skill, else the monitoring writer's current skill
        (not the oracle's or off's), else None."""
        skills = list(self.task.skills) if self.task is not None else []
        if prompt in skills:
            return prompt
        monitor = self.writer.monitor
        plan = getattr(monitor, "plan", None)
        if plan:
            return plan[min(getattr(monitor, "idx", 0), len(plan) - 1)]
        return None

    def _log_call(self, ctrl: dict, info: dict):
        if self._log:
            self._log.write(json.dumps({"episode": self.episode, "step": ctrl.get("step"), **info}, default=str) + "\n")
            self._log.flush()

    def close(self):
        """End of the run: the extensions take the last episode and release their files; the log is closed."""
        self._end_episode()
        for extension in self.extensions:
            extension.close()
        self.method.close()
        if self._log:
            self._log.close()
            self._log = None
