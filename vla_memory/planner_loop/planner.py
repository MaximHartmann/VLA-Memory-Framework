"""The planner: plans once per episode, monitors on every policy call, and holds the working memory of the episode.

    start_episode(instruction)   text-only call: the model returns the ordered skill list under a JSON schema that
                                 admits only the task's skills (at least one, at most max_plan_len). If the call
                                 fails or a skill is outside the vocabulary, the plan falls back to the task's skills
                                 in their listed order; both cases are logged with their source.
    observe(images, step, proprio)   on every policy call (every `call_every`-th one queries the model): the model
                                 sees the completion rule of the current skill, the proprioception sentence, the
                                 retrieved exemplars (if a memory store is attached) and the current images, and
                                 answers whether the CURRENT skill is complete. The plan index only ever advances
                                 (monotone), after `votes_to_advance` consecutive "done" answers, and never within
                                 the first `min_calls_per_skill` calls of a skill.

Working memory = plan + index + recent proprioception; it is rebuilt at every start_episode and restated in every
query. Long-term memory = the MemoryStore, read at every query, never written during a run.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from .proprio import ProprioSummary
from ..task import TaskSpec
from ..vlm import VLMBackend, encode_png


class Planner:
    def __init__(self, vlm: VLMBackend, task: TaskSpec, memory=None, k: int = 4, votes_to_advance: int = 2,
                 call_every: int = 1, min_calls_per_skill: int = 2, recovery: bool = False, log=None,
                 use_wrist: bool = True, upscale: int = 1, brief: bool = False, exclude_episode: Any = None):
        self.vlm = vlm; self.task = task; self.memory = memory; self.k = k; self.exclude_episode = exclude_episode
        self.votes = votes_to_advance; self.call_every = call_every; self.min_calls = min_calls_per_skill
        self.recovery = recovery; self.use_wrist = use_wrist; self.upscale = upscale; self.brief = brief
        self.schema = task.monitor_schema(brief=brief)
        p = task.proprio or {}
        self.proprio = ProprioSummary(p.get("closed_threshold", 0.3), p.get("history", 12), p.get("note", ""))
        self._log = open(log, "a") if log else None
        self.episode = None; self.plan: list[str] = []; self.idx = 0; self.calls = 0; self.calls_in_skill = 0
        self.consec_done = 0; self.history: list[dict] = []; self.done_all = False

    # ------------------------------------------------------------------ episode
    def start_episode(self, instruction: str, episode_id: str) -> dict:
        self.episode = episode_id; self.idx = 0; self.calls = 0; self.calls_in_skill = 0
        self.consec_done = 0; self.history = []; self.done_all = False; self.proprio.reset()
        msgs = [{"role": "system", "content": self.task.plan_system_text()},
                {"role": "user", "content": f"Instruction: \"{instruction}\"\nReturn the ordered skill list as JSON."}]
        rec = {"event": "plan", "episode": episode_id, "instruction": instruction}
        try:
            r = self.vlm.chat(msgs, schema=self.task.plan_schema(), name="plan", max_tokens=256)
            plan = (r.get("json") or {}).get("plan")
            rec.update(vlm_plan=plan, ms=r["ms"], tokens=[r.get("prompt_tokens"), r.get("completion_tokens")],
                       reasoning=r.get("reasoning"))
        except Exception as e:  # server down, timeout, bad JSON
            plan = None; rec["error"] = repr(e)
        if plan and all(s in self.task.skills for s in plan):
            self.plan = list(plan); rec["source"] = "vlm"
        else:   # call failed, or a backend without schema enforcement returned a skill the policy never saw
            self.plan = list(self.task.skills); rec["source"] = "fallback_skill_order"
        rec["plan"] = self.plan
        self._write(rec)
        return rec

    @property
    def current_skill(self) -> str:
        return self.plan[min(self.idx, len(self.plan) - 1)]

    # ------------------------------------------------------------------ monitor
    def observe(self, images: Sequence[np.ndarray], step: int, proprio: dict | None = None) -> dict:
        """images = (exterior, wrist) or (exterior,). Returns the decision record (also appended to the log)."""
        self.calls += 1; self.calls_in_skill += 1
        self.proprio.push(proprio)
        rec = {"event": "monitor", "episode": self.episode, "step": step, "call": self.calls,
               "skill_index": self.idx, "skill": self.current_skill, "queried": False, "advanced": False}
        if self.done_all or (self.calls - 1) % self.call_every:
            rec["skill_after"] = self.current_skill; rec["plan_complete"] = self.done_all
            self._write(rec); return rec
        t = self.task; skill = self.current_skill; verb, obj = t.parse_skill(skill)
        plan_txt = "\n".join(f"  {i + 1}. {s}{'  <- CURRENT' if i == self.idx else ('  (done)' if i < self.idx else '')}"
                             for i, s in enumerate(self.plan))
        user, n0, hits = [], 0, []
        imgs = list(images) if self.use_wrist else list(images[:1])
        if self.memory is not None and len(self.memory):
            hits = self.memory.retrieve(imgs, k=self.k, exclude_episode=self.exclude_episode)
            fs = self.memory.fewshot_content(hits, self.idx, skill, encode_png)
            user += fs; n0 = sum(1 for it in fs if it.get("type") == "image_url")
            rec["exemplars"] = [{"i": j, "sim": round(sim, 3), **self.memory.describe_hit(j)} for j, sim in hits]
        pics = (f"Picture {n0 + 1} (external camera) and Picture {n0 + 2} (wrist camera) show" if self.use_wrist
                else f"Picture {n0 + 1} (external camera) shows")
        now = "Now the CURRENT observation. " if hits else ""
        ptxt = self.proprio.text(); rec["proprio"] = self.proprio.latest
        objs = " and ".join(f"{o} {t.object_noun}" for o in t.objects) if len(t.objects) <= 2 else "every " + t.object_noun
        user.append({"type": "text", "text": f"{now}Plan:\n{plan_txt}\n\nCurrent skill: \"{skill}\".\nCompletion criterion: "
                                             f"{t.criterion(skill)}\n\n" + (ptxt + "\n\n" if ptxt else "") +
                                             f"{pics} the scene NOW. Combine the proprioception with what you see: the "
                                             f"cameras tell you where the objects are, the sensors tell you whether the "
                                             f"gripper is closed on something and how high the hand is. Report the state "
                                             f"of the {objs} and the gripper, then decide whether the current skill is complete."})
        for im in imgs:
            user.append({"type": "image_url", "image_url": {"url": encode_png(self._prep(im))}})
        msgs = [{"role": "system", "content": t.system_text(self.use_wrist)}, {"role": "user", "content": user}]
        rec["queried"] = True
        try:
            r = self.vlm.chat(msgs, schema=self.schema, name="monitor")
            j = r.get("json") or {}
            rec.update(vlm=j, ms=r["ms"], tokens=[r.get("prompt_tokens"), r.get("completion_tokens")], reasoning=r.get("reasoning"))
            done = bool(j.get("current_skill_done"))
        except Exception as e:
            rec["error"] = repr(e); done = False; j = {}
        # optional recovery: a later skill is active but another object is still held -> put it down first
        if self.recovery and j and len(t.verbs) == 2 and verb == t.verbs[0]:
            for other in t.objects:
                if other != obj and j.get(t.field(other)) in ("held_lifted", "held_at_table"):
                    fix = t.skill(t.verbs[1], other)
                    if self.plan[self.idx] != fix:
                        self.plan.insert(self.idx, fix); rec["recovery_inserted"] = fix
                        self.consec_done = 0; self.calls_in_skill = 0
                        rec["skill_after"] = self.current_skill; self._write(rec); return rec
        self.consec_done = self.consec_done + 1 if done else 0
        rec["consecutive_done"] = self.consec_done
        if done and self.consec_done >= self.votes and self.calls_in_skill >= self.min_calls:
            self.idx += 1; self.consec_done = 0; self.calls_in_skill = 0; rec["advanced"] = True
            if self.idx >= len(self.plan):
                self.done_all = True; self.idx = len(self.plan) - 1; rec["plan_complete"] = True
        rec["skill_after"] = self.current_skill
        self._write(rec)
        return rec

    def _prep(self, img):
        img = np.asarray(img)
        if self.upscale and self.upscale != 1:
            from PIL import Image
            im = Image.fromarray(np.ascontiguousarray(img))
            img = np.asarray(im.resize((im.width * self.upscale, im.height * self.upscale), Image.BICUBIC))
        return img

    def _write(self, rec):
        self.history.append(rec)
        if self._log:
            self._log.write(json.dumps(rec, default=str) + "\n"); self._log.flush()

    def close(self):
        if self._log:
            self._log.close()
