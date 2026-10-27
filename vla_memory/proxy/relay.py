"""The relay between an environment client and the VLA policy server, with a writer and a method.

    environment  --ws-->  Relay (:LISTEN)  --ws-->  policy server (:UPSTREAM)
                            |  a writer signals that a step of the task is done (writers/)
                            |  a method turns that into what the policy receives (methods/)

The environment sends its ordinary observation with the FULL instruction as `prompt`, plus control keys under the
`memory/` prefix (vla_memory/env_keys.py). The relay strips them (only the episode markers also reach the policy, step
4) and, at every call, in this order:

    1. a new episode: the extensions, the method and the writer (its plan, for the objects the environment declares as
       memory/objects) start it
    2. writer.before_call      a writer that decides from this call's inputs
    3. method.prompt           the prompt of this call shows that decision (with the writer "off": the instruction)
    4. method.before_policy    keys added to the observation (a picture)
       extensions' before_policy   keys added by the extensions (a retrieved memory)
       the episode markers     memory/new_episode and memory/episode go to the policy as the environment sent them, so that
                               a policy that keeps its own memory resets it when an episode starts; with --memory-reset
                               every-call every call is marked as a new episode (the reset test)
    5. the policy's call
    6. the extensions' on_reply, then writer.after_call: a writer that decides from the policy's reply; its decision shows
       from the next call on
    7. the reply goes back with a `planner` info dict; one JSON record per call goes to the log
With a recorder (--record-head, writers/head/record.py) the policy's state in every reply is kept for fitting an event head,
together with the writer's count of completed skills; after a completed skill the recorder asks the policy once more with
the input of the call before.

Neither the environment nor the policy server needs to know about the writer or the method. The command line is in
cli.py:  python -m vla_memory.proxy --writer rules --method history --listen-port <port> --upstream-port <policy port>
"""
from __future__ import annotations

import json
import logging
import time
from typing import Sequence

from .extensions import Extension, collect
from ..env_keys import CONTROL_PREFIX
from ..task import TaskSpec
from ..transport import PolicyTransport, text_of
from ..writers.base import Call

log = logging.getLogger("vla_memory.proxy")
EPISODE_MARKERS = ("new_episode", "episode")    # the control keys that also go to the policy, unchanged
MEMORY_RESETS = ("episode", "every-call")       # when a policy that keeps its own memory is told to reset it


def split_control_keys(observation: dict) -> tuple[dict, dict]:
    """The memory/ control keys without their prefix, and the observation without them."""
    control_keys = {}
    policy_observation = {}
    for key, value in observation.items():
        if isinstance(key, str) and key.startswith(CONTROL_PREFIX):
            control_keys[key[len(CONTROL_PREFIX):]] = value
        else:
            policy_observation[key] = value
    return control_keys, policy_observation


class Relay:
    """One writer, one method and the extensions around a policy transport; infer() relays one policy call (module
    docstring). log_path: the proxy's log, one JSON record per call, appended. memory_reset: "episode" forwards the
    environment's episode markers, "every-call" marks every call as a new episode."""

    def __init__(self, transport: PolicyTransport, writer, method, log_path=None,
                 image_keys=("observation/image", "observation/wrist_image"), task: TaskSpec | None = None,
                 extensions: Sequence[Extension] = (), memory_reset: str = "episode", recorder=None):
        if memory_reset not in MEMORY_RESETS:
            raise ValueError(f"memory_reset {memory_reset!r}: one of {MEMORY_RESETS}")
        self.transport = transport
        self.writer = writer               # keeps the plan and its pointer
        self.method = method               # turns them into what the policy receives
        self.image_keys = tuple(image_keys)
        self.task = task if task is not None else getattr(writer, "task", None)
        self.extensions = list(extensions)
        self.memory_reset = memory_reset
        self.recorder = recorder           # keeps the policy's state for fitting an event head (writers/head/record.py)
        self.episode = None
        self.episode_count = 0
        self._log_file = open(log_path, "a") if log_path else None
        log.info("upstream metadata: %s", transport.metadata())

    @property
    def mode(self) -> str:
        """The writer's name, as every log record says it."""
        return self.writer.name

    # ------------------------------------------------------------------ one policy call
    def infer(self, observation: dict) -> dict:
        """Relay one policy call: decide the prompt, forward the observation, return the reply with the `planner` info."""
        started = time.perf_counter()
        control_keys, observation = split_control_keys(observation)
        environment_keys = set(observation)
        instruction = text_of(observation.get("prompt", ""))
        info = {"mode": self.writer.name, "queried": False, "advanced": False, "vlm_ms": 0.0}
        if self._is_new_episode(control_keys):
            self._begin_episode(control_keys, instruction, info)
        call = Call(instruction, control_keys, observation, self.image_keys)
        info.update(self.writer.before_call(call, self.extensions))
        shown = len(self.writer.completed())
        prompt = self._decide_prompt(instruction, observation, control_keys, info)
        inputs = {key: value for key, value in observation.items() if key == "prompt" or key not in environment_keys}
        observation.update(self.episode_markers(control_keys))
        reply = self._forward(observation, info)
        if self.recorder is not None:
            info["recorded"] = self.recorder.on_call(observation, inputs, reply, shown, control_keys.get("step"),
                                                     self.transport.infer)
        self._reply_hooks(observation, control_keys, reply, prompt, info)
        info.update(self.writer.after_call(reply))
        info["total_ms"] = 1000 * (time.perf_counter() - started)
        reply["planner"] = info
        self._log_call(control_keys, info)
        return reply

    def _decide_prompt(self, instruction: str, observation: dict, control_keys: dict, info: dict) -> str:
        """Set the observation's prompt from the writer's state, then let the method and the extensions add their keys.
        Returns the prompt as the method decided it (an extension may still change the observation's prompt)."""
        prompt = self.method.prompt(instruction, self.writer) if self.writer.drives_prompt else instruction
        info.update(collect(self.extensions, "log_fields"))
        observation["prompt"] = prompt
        info.update(self.method.before_policy(observation, control_keys, self.writer))
        for extension in self.extensions:
            info.update(extension.before_policy(observation, control_keys, prompt, self.writer.monitor))
        info["prompt"] = observation["prompt"]
        info["oracle_prompt"] = control_keys.get("oracle_prompt")
        return prompt

    # ------------------------------------------------------------------ episodes
    def episode_markers(self, control_keys: dict) -> dict:
        """The episode markers the policy receives with the observation: memory/new_episode and memory/episode as the
        environment sent them. With memory_reset "every-call" every call is marked as a new episode, so that a policy that
        keeps its own memory clears it before every call; the proxy's own episodes stay the environment's."""
        markers = {CONTROL_PREFIX + key: control_keys[key] for key in EPISODE_MARKERS if key in control_keys}
        if self.memory_reset == "every-call":
            markers[CONTROL_PREFIX + "new_episode"] = True
        return markers

    def _is_new_episode(self, control_keys: dict) -> bool:
        """Whether this call starts an episode: the first call, memory/new_episode, or a changed episode id."""
        episode_id = text_of(control_keys.get("episode"))
        if control_keys.get("new_episode") or self.episode is None:
            return True
        return episode_id is not None and str(episode_id) != self.episode

    def _begin_episode(self, control_keys: dict, instruction: str, info: dict):
        """Close the episode that just ended, name the new one, and let the writer plan it."""
        self._end_episode()
        episode_id = text_of(control_keys.get("episode"))
        self.episode = str(episode_id) if episode_id is not None else f"ep{self.episode_count}"
        self.episode_count += 1
        for extension in self.extensions:
            extension.start_episode(self.episode, control_keys)
        self.method.start_episode(self.episode, control_keys)
        objects = control_keys.get("objects")      # the objects of this episode, when the environment declares them
        if objects is None:
            record = self.writer.start_episode(instruction, self.episode)
        else:
            record = self.writer.start_episode(instruction, self.episode, objects=objects)
        if record is not None:
            info.update(plan=record["plan"], plan_source=record["source"], plan_ms=record.get("ms"))
            log.info("episode %s plan (%s): %s", self.episode, record["source"], record["plan"])
        if self.recorder is not None:
            self.recorder.start_episode(self.episode, {"instruction": instruction, "plan": list(self.writer.plan),
                                                       "writer": self.writer.name, "method": self.method.name})

    def _end_episode(self):
        """The extensions take the finished episode; their records go to the writer's log. The recorder writes it. Before
        the first episode there is nothing to end."""
        if self.episode is None:
            return
        if self.recorder is not None:
            self.recorder.end_episode()
        for extension in self.extensions:
            record = extension.end_episode(self.writer.monitor)
            if record is not None:
                self._log_event(record)

    def _log_event(self, record: dict):
        """Write a record to the writer's own log, when the writer keeps one."""
        log_event = getattr(self.writer.monitor, "log_event", None)
        if log_event is not None:
            log_event(record)

    # ------------------------------------------------------------------ the policy's reply
    def _forward(self, observation: dict, info: dict) -> dict:
        """The policy's reply to the observation; the round trip goes into the info as upstream_ms."""
        started = time.perf_counter()
        reply = self.transport.infer(observation)
        info["upstream_ms"] = 1000 * (time.perf_counter() - started)
        return reply

    def _reply_hooks(self, observation: dict, control_keys: dict, reply: dict, prompt: str, info: dict):
        """Extensions that look at (or change) the policy's reply; each one's record goes into the info."""
        skill = self.current_skill(prompt)
        for extension in self.extensions:
            record = extension.on_reply(observation, control_keys, reply, skill, self.episode)
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
            return plan[min(getattr(monitor, "skill_index", 0), len(plan) - 1)]
        return None

    def _log_call(self, control_keys: dict, info: dict):
        """One JSON record per call: the episode, the step and the info."""
        if self._log_file:
            self._log_file.write(json.dumps({"episode": self.episode, "step": control_keys.get("step"), **info}, default=str) + "\n")
            self._log_file.flush()

    def close(self):
        """End of the run: the extensions take the last episode and release their files; the log is closed."""
        self._end_episode()
        for extension in self.extensions:
            extension.close()
        self.method.close()
        if self._log_file:
            self._log_file.close()
            self._log_file = None
