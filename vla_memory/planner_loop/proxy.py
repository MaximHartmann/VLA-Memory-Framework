"""Planner proxy: the relay between an environment client and the VLA policy server.

    environment  --ws-->  PlannerProxy (:LISTEN)  --ws-->  policy server (:UPSTREAM)
                                |  HTTP (OpenAI-compatible chat API)
                                v
                          planner model (vLLM, ...)

The environment sends its ordinary observation with the FULL instruction as `prompt`, plus control keys under the
`memory/` prefix (docs/ENVIRONMENT_CONTRACT.md). The proxy strips them, decides the prompt of this call, forwards the
observation to the policy and returns the policy's reply with an extra `planner` info dict. Neither the environment nor
the policy server needs to know about the planner.

Modes:  vlm      the planner decides the prompt (plan + monitor; planner.Planner)
        rules    the history method's rules writer (history_appended.rules.RuleMonitor) keeps the plan pointer from the
                 robot's own gripper and hand signals (memory/proprio_steps, the wrist image for the identifier); no model
        oracle   forwards memory/oracle_prompt (a ground-truth reference planner, for evaluation)
        off      passthrough

A method's proxy changes what the policy receives by overriding prompt_for and oracle_prompt_for: the history-appended
proxy renders instruction + history instead of the current skill. Optional memories plug in as extensions (extensions.py,
command line --with). The command line and the builders are in cli.py; run it as

    python -m vla_memory.planner_loop.proxy --listen-port <port> --upstream-port <policy port> [--mode vlm|rules|oracle|off]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Sequence

import numpy as np

from .extensions import Extension, collect
from ..task import TaskSpec
from ..transport import PolicyTransport

log = logging.getLogger("vla_memory.planner_loop.proxy")
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


class PlannerProxy:
    MODES = ("vlm", "rules", "oracle", "off")

    def __init__(self, transport: PolicyTransport, mode: str, planner, log_path=None,
                 image_keys=("observation/image", "observation/wrist_image"), task: TaskSpec | None = None,
                 extensions: Sequence[Extension] = ()):
        self.transport = transport
        self.mode = mode
        self.planner = planner             # keeps the plan and its pointer: the VLM planner, or the rules writer
        self.image_keys = tuple(image_keys)
        self.task = task if task is not None else getattr(planner, "task", None)
        self.extensions = list(extensions)
        self.episode = None
        self.episode_count = 0
        self._warned: set[str] = set()
        self._log = open(log_path, "a") if log_path else None
        log.info("upstream metadata: %s", transport.metadata())

    # ------------------------------------------------------------------ what a method's proxy may change
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Command-line options of a method's proxy (none here)."""

    def configure(self, args):
        """Apply those options after construction (nothing here)."""

    @classmethod
    def build_monitor(cls, args, task: TaskSpec, extensions: Sequence[Extension] = ()):
        """The monitor of the chosen mode: the rules writer, the VLM planner, or None (the prompt needs no monitor)."""
        if args.mode == "rules":
            from ..history_appended.identify import make_identifier
            from ..history_appended.rules import RuleMonitor
            return RuleMonitor(task, identifier=make_identifier(task))
        if args.mode == "vlm":
            from .cli import build_planner
            return build_planner(args, task, extensions)
        return None

    def prompt_for(self, instruction: str, record: dict) -> str:
        """The prompt handed to the policy after a decision: here the current skill. Other methods override."""
        return record.get("skill_after", self.planner.current_skill)

    def oracle_prompt_for(self, instruction: str, oracle_prompt: str) -> str:
        """The prompt in oracle mode: the environment's ground-truth current skill, forwarded as it is."""
        return oracle_prompt

    # ------------------------------------------------------------------ one policy call
    def infer(self, observation: dict) -> dict:
        """Relay one policy call: decide the prompt, forward the observation, return the reply with the `planner` info."""
        started = time.perf_counter()
        ctrl, obs = split_control_keys(observation)
        instruction = text_of(obs.get("prompt", ""))
        info = {"mode": self.mode, "queried": False, "advanced": False, "vlm_ms": 0.0}
        if self._is_new_episode(ctrl):
            self._begin_episode(ctrl, instruction, info)
        prompt, fields = self.decide(instruction, ctrl, obs)
        info.update(fields)
        info.update(collect(self.extensions, "log_fields"))
        obs["prompt"] = prompt
        info["prompt"] = prompt
        info["oracle_prompt"] = ctrl.get("oracle_prompt")
        reply = self._forward(obs, info)
        self._reply_hooks(obs, ctrl, reply, prompt, info)
        info["total_ms"] = 1000 * (time.perf_counter() - started)
        reply["planner"] = info
        self._log_call(ctrl, info)
        return reply

    def decide(self, instruction: str, ctrl: dict, obs: dict) -> tuple[str, dict]:
        """The prompt of this call and the fields it adds to the log record, by mode."""
        if self.mode == "vlm":
            return self._decide_vlm(instruction, ctrl, obs)
        if self.mode == "rules":
            return self._decide_rules(instruction, ctrl, obs)
        if self.mode == "oracle":
            return self._decide_oracle(instruction, ctrl)
        return instruction, {}

    def _decide_vlm(self, instruction: str, ctrl: dict, obs: dict) -> tuple[str, dict]:
        """The planner looks at the pictures (and the proprioception) and keeps or advances its pointer."""
        images = self._images(ctrl, obs)
        proprio = {key: float(ctrl[key]) for key in ("gripper_closedness", "hand_height") if key in ctrl}
        for extension in self.extensions:
            extension.before_decision(ctrl, images, self.planner)
        record = self.planner.observe(images, int(ctrl.get("step", -1)), proprio=proprio or None)
        for extension in self.extensions:
            extension.after_decision(ctrl, images, record, self.planner)
        fields = dict(queried=record["queried"], advanced=record["advanced"], vlm_ms=record.get("ms", 0.0),
                      skill_index=self.planner.idx, vlm=record.get("vlm"), recovery=record.get("recovery_inserted"))
        return self.prompt_for(instruction, record), fields

    def _decide_rules(self, instruction: str, ctrl: dict, obs: dict) -> tuple[str, dict]:
        """The rules writer reads the control steps since the last call (and the wrist image at a pick)."""
        from ..history_appended.rules import signal_rows
        record = self.planner.observe_rows(signal_rows(ctrl), int(ctrl.get("step", -1)), image=self._wrist(ctrl, obs),
                                           **self._declared_signals(ctrl))
        fields = {"advanced": record["advanced"], "skill_index": self.planner.idx, "completed": record["completed"] or None,
                  "writer_state": record["writer_state"], "identified": record["identified"]}
        return self.prompt_for(instruction, record), fields

    def _decide_oracle(self, instruction: str, ctrl: dict) -> tuple[str, dict]:
        """The environment's ground truth decides."""
        if "oracle_prompt" not in ctrl:
            self._warn_once("oracle", "oracle mode without memory/oracle_prompt: forwarding the instruction")
        oracle_prompt = text_of(ctrl.get("oracle_prompt", instruction))
        return self.oracle_prompt_for(instruction, oracle_prompt), {}

    def _images(self, ctrl: dict, obs: dict) -> list:
        """The pictures of this call: the raw frames the environment sent, else the policy's own image keys."""
        exterior = ctrl.get("exterior_raw", obs.get(self.image_keys[0]))
        images = [np.asarray(exterior)]
        wrist = self._wrist(ctrl, obs)
        if wrist is not None:
            images.append(np.asarray(wrist))
        return images

    def _wrist(self, ctrl: dict, obs: dict):
        fallback = obs.get(self.image_keys[1]) if len(self.image_keys) > 1 else None
        return ctrl.get("wrist_raw", fallback)

    def _declared_signals(self, ctrl: dict) -> dict:
        """The task's declared signals (memory/signal_steps) for a rules writer whose rules use them."""
        names = getattr(self.planner, "signal_names", ())
        if not names:
            return {}
        steps = ctrl.get("signal_steps")
        if steps is None:
            self._warn_once("signals", f"rules mode without memory/signal_steps: the rules on {', '.join(names)} never fire")
        return {"signals": steps}

    def _warn_once(self, topic: str, message: str):
        if topic not in self._warned:
            log.warning(message)
            self._warned.add(topic)

    # ------------------------------------------------------------------ episodes
    def _is_new_episode(self, ctrl: dict) -> bool:
        episode_id = text_of(ctrl.get("episode"))
        if ctrl.get("new_episode") or self.episode is None:
            return True
        return episode_id is not None and str(episode_id) != self.episode

    def _begin_episode(self, ctrl: dict, instruction: str, info: dict):
        """Close the episode that just ended, name the new one, and let the monitor plan it."""
        self._end_episode()
        episode_id = text_of(ctrl.get("episode"))
        self.episode = str(episode_id) if episode_id is not None else f"ep{self.episode_count}"
        self.episode_count += 1
        for extension in self.extensions:
            extension.start_episode(self.episode, ctrl)
        if self.planner is not None:
            record = self.planner.start_episode(instruction, self.episode)
            info.update(plan=record["plan"], plan_source=record["source"], plan_ms=record.get("ms"))
            log.info("episode %s plan (%s): %s", self.episode, record["source"], record["plan"])

    def _end_episode(self):
        """The extensions take the finished episode; their records go to the planner's log."""
        for extension in self.extensions:
            record = extension.end_episode(self.planner)
            if record is not None:
                self._log_event(record)

    def _log_event(self, record: dict):
        log_event = getattr(self.planner, "log_event", None)
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
        """The skill the policy works on now: the prompt if it is a skill, else the monitor's current skill, else None."""
        skills = list(self.task.skills) if self.task is not None else []
        if prompt in skills:
            return prompt
        plan = getattr(self.planner, "plan", None)
        if plan:
            return plan[min(getattr(self.planner, "idx", 0), len(plan) - 1)]
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
        if self._log:
            self._log.close()
            self._log = None


if __name__ == "__main__":
    from .cli import main
    main()
