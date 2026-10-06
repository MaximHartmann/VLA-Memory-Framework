"""The planner: plans once per episode, monitors on every policy call, and holds the working memory of the episode.

    start_episode(instruction)   text-only call: the model returns the ordered skill list under a JSON schema that
                                 admits only the task's skills (at least one, at most max_plan_len). If the call
                                 fails or a skill is outside the vocabulary, the plan falls back to the task's skills
                                 in their listed order; both cases are logged with their source.
    observe(images, step, proprio)   on every policy call (every `call_every`-th one is a monitor call). With a memory
                                 store, the memory-vote gate may settle the call from the store alone; otherwise the model
                                 sees the completion rule of the current skill, the proprioception sentence, the retrieved
                                 exemplars (if a memory store is attached) and the current images, and answers whether the
                                 CURRENT skill is complete. The plan index only ever advances (monotone), after
                                 `votes_to_advance` consecutive "done" answers, and never within the first
                                 `min_calls_per_skill` calls of a skill.

Default configuration (settled 2026-09-30 on the reference task; every part can be switched off):
    gate           the memory-vote gate. The k_vote (6) stored moments nearest to the current observation vote; a moment
                   votes "done" when its stored skill is later in the task's skill order than the current skill, the same
                   rule that labels the shown examples. A full vote with fewer than gate_min_done (1) "done" votes skips the
                   question and counts as a "not done" answer; any "done" vote asks the model. The last skill of the plan,
                   and the task's last skill wherever it stands in the plan, are always asked: no stored moment is labelled
                   later than the task's last skill. Without a memory store the gate is inactive.
                   The store's labels are indices into the task's skill order (the order the recordings followed); votes and
                   example labels use the current skill's index in that order, which equals the plan pointer when the plan
                   follows the task order and stays right for a reordered plan (e.g. blue first) or an inserted recovery step.
    verdict_only   the model answers {"current_skill_done": bool} only (task.monitor_schema(verdict_only=True)); it still
                   receives the full observation. verdict_only=False adds the object states, the gripper fields and a reason
                   (brief=True drops the reason); `recovery` needs those states.
    k = 4 examples shown per question, votes_to_advance = 1, min_calls_per_skill = 2, call_every = 1.

Every monitor record carries the step, the skill, the latest proprioception, whether the model was asked (`queried`),
whether the gate settled the call (`gated`), the vote (`vote`: done votes, of how many, lowest similarity), the answer (`vlm`)
and whether the pointer moved (`advanced`); the plan record carries the configuration (`config`). With log_memory (set for
a store that grows during the run) every monitor record also carries the store's size (`memory_entries`) and, when it
retrieved, the retrieval time (`retrieval_ms`); log_event() appends other events to the same log (the store's
`store_update` records).

Working memory = plan + index + recent proprioception; it is rebuilt at every start_episode and restated in every
query. Long-term memory = the MemoryStore, read at every query and never written within an episode; the online store
(online.py, proxy --online-store) adds the finished episode between episodes. An optional preference memory
(`profile`, vla_memory.preference.ProfileStore) is re-read at every start_episode: its plan-scope rules are added to the plan
request (except those held back because the instruction names the task's objects itself: the instruction has precedence,
logged as `profile_plan_held`), its skill-scope rules to that skill's completion criterion, its planner-scope parameters
override votes/cadence.
An optional keyframe memory (`keyframes`, keyframes.KeyframeMemory; proxy --keyframes N) shows labelled frames of earlier
moments of the running episode (start, switches, ...) in every question; the record then lists them (`keyframes`).
An optional world memory (`world`, vla_memory.world.WorldMemory; proxy --world-state / --world-verify / --world-store), which
the proxy writes at every call before the decision, adds its state block to every question (--world-state) and is asked
before every switch: with --world-verify a "done" moves the pointer only if the world state agrees that the skill is
complete. Every record then carries the world state (`world`); a would-be switch carries the verdict (`world_check`) and,
when it was blocked, `world_blocked`.
"""
from __future__ import annotations

import json
import time
from typing import Any, Sequence

import numpy as np

from .proprio import ProprioSummary
from ..task import TaskSpec
from ..vlm import VLMBackend, encode_png


class Planner:
    def __init__(self, vlm: VLMBackend, task: TaskSpec, memory=None, k: int = 4, votes_to_advance: int = 1,
                 call_every: int = 1, min_calls_per_skill: int = 2, recovery: bool = False, log=None,
                 use_wrist: bool = True, upscale: int = 1, brief: bool = False, exclude_episode: Any = None, profile=None,
                 gate: bool = True, k_vote: int = 6, gate_min_done: int = 1, verdict_only: bool = True,
                 log_memory: bool = False, keyframes=None, world=None):
        if recovery and verdict_only:
            raise ValueError("recovery reads the object states from the monitor's answer: use verdict_only=False with it")
        self.vlm = vlm; self.task = task; self.memory = memory; self.k = k; self.exclude_episode = exclude_episode
        self.log_memory = log_memory   # store size and retrieval time in every monitor record (a store that grows)
        self.votes = votes_to_advance; self.call_every = call_every; self.min_calls = min_calls_per_skill
        self.profile = profile; self._base_params = (votes_to_advance, call_every, min_calls_per_skill)   # preference memory (notebook), optional
        self.recovery = recovery; self.use_wrist = use_wrist; self.upscale = upscale; self.brief = brief
        self.gate = gate; self.k_vote = k_vote; self.gate_min_done = gate_min_done; self.verdict_only = verdict_only
        self.schema = task.monitor_schema(brief=brief, verdict_only=verdict_only)
        self.keyframes = keyframes   # keyframe memory (keyframes.KeyframeMemory): frames of earlier moments of this episode, optional
        if keyframes is not None:
            self.schema = keyframes.schema(self.schema)   # the model-selected policy adds keep_frame; the others change nothing
        self.world = world   # structured world memory (vla_memory.world.WorldMemory): state block and switch verifier, optional
        p = task.proprio or {}
        self.proprio = ProprioSummary(p.get("closed_threshold", 0.3), p.get("history", 12), p.get("note", ""))
        self._log = open(log, "a") if log else None
        self.episode = None; self.plan: list[str] = []; self.idx = 0; self.calls = 0; self.calls_in_skill = 0
        self.consec_done = 0; self.history: list[dict] = []; self.done_all = False

    def config(self) -> dict:
        """The decision parameters in effect (after the notebook's overrides), as logged with every plan."""
        cfg = {"gate": self.gate, "k_vote": self.k_vote, "gate_min_done": self.gate_min_done, "verdict_only": self.verdict_only,
               "brief": self.brief, "k": self.k, "votes_to_advance": self.votes, "min_calls_per_skill": self.min_calls,
               "call_every": self.call_every, "memory_entries": len(self.memory) if self.memory is not None else 0}
        if self.keyframes is not None:
            cfg["keyframes"] = self.keyframes.config()
        if self.world is not None:
            cfg["world"] = self.world.config()
        return cfg

    # ------------------------------------------------------------------ episode
    def start_episode(self, instruction: str, episode_id: str) -> dict:
        self.episode = episode_id; self.idx = 0; self.calls = 0; self.calls_in_skill = 0
        self.consec_done = 0; self.history = []; self.done_all = False; self.proprio.reset()
        self.votes, self.call_every, self.min_calls = self._base_params
        profile_txt, profile_rec = "", {}
        if self.profile is not None:   # the notebook: re-read, apply strictness overrides, render the plan-scope rules
            self.profile.reload(); ov = self.profile.planner_overrides()
            self.votes = ov.get("votes_to_advance", self.votes); self.call_every = ov.get("call_every", self.call_every)
            self.min_calls = ov.get("min_calls_per_skill", self.min_calls)
            plan_rules, held = self.profile.plan_rules_for(instruction, self.task.objects)   # the instruction has precedence
            profile_txt = self.profile.plan_text(plan_rules)
            planner_rules = self.profile.planner_rules() if ov else []   # strictness entries that set these parameters
            profile_rec = {"profile_plan_rules": [e.id for e in plan_rules], **({"profile_plan_held": held} if held else {}),
                           "profile_planner_rules": [e.id for e in planner_rules], "profile_params": ov}
            self.profile.note_applied(plan_rules + planner_rules)
        msgs = [{"role": "system", "content": self.task.plan_system_text()},
                {"role": "user", "content": f"Instruction: \"{instruction}\"\n" + (profile_txt + "\n" if profile_txt else "")
                                            + "Return the ordered skill list as JSON."}]
        rec = {"event": "plan", "episode": episode_id, "instruction": instruction, **profile_rec, "config": self.config()}
        try:
            r = self.vlm.chat(msgs, schema=self.task.plan_schema(), name="plan", max_tokens=256)
            plan = (r.get("json") or {}).get("plan")
            rec.update(vlm_plan=plan, ms=r.get("ms", 0.0), tokens=[r.get("prompt_tokens"), r.get("completion_tokens")],
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

    # ------------------------------------------------------------------ memory-vote gate
    def _vote(self, hits, label: int) -> dict | None:
        """The store's own verdict on the current skill (index `label` in the task's skill order): each retrieved moment votes
        "done" when its stored skill index is later (the rule that labels the shown examples' current_skill_done). None when
        the store does not label its entries with a skill index (MemoryStore.describe_hit(...)["subgoal"]): the gate then
        cannot decide."""
        labels = [self.memory.describe_hit(j).get("subgoal") for j, _ in hits]
        if not hits or any(s is None for s in labels):
            return None
        return {"done": sum(int(s) > label for s in labels), "of": len(labels), "min_sim": round(min(s for _, s in hits), 3)}

    def _gate_skips(self, vote: dict | None, label: int) -> bool:
        """A full vote (k_vote moments) with fewer than gate_min_done "done" votes settles the call as "not done". Always asked:
        the last skill of the plan, and the task's last skill (no stored moment is labelled later, its vote is never "done")."""
        return (vote is not None and vote["of"] >= self.k_vote and vote["done"] < self.gate_min_done
                and self.idx < len(self.plan) - 1 and label < len(self.task.skills) - 1)

    # ------------------------------------------------------------------ monitor
    def observe(self, images: Sequence[np.ndarray], step: int, proprio: dict | None = None) -> dict:
        """images = (exterior, wrist) or (exterior,). Returns the decision record (also appended to the log)."""
        self.calls += 1; self.calls_in_skill += 1
        self.proprio.push(proprio)
        rec = {"event": "monitor", "episode": self.episode, "step": step, "call": self.calls, "skill_index": self.idx,
               "skill": self.current_skill, "queried": False, "gated": False, "advanced": False, "proprio": self.proprio.latest}
        if self.log_memory:
            rec["memory_entries"] = len(self.memory) if self.memory is not None else 0
        if self.world is not None:   # written by the proxy for this call before the decision
            rec["world"] = self.world.summary()
        kf = self.keyframes   # keyframe memory: sees every call (start, uniform), is shown in the question, kept after it
        if kf is not None:
            kf.begin(self.episode, step, list(images) if self.use_wrist else list(images[:1]), self.proprio.latest, self.current_skill)
        if self.done_all or (self.calls - 1) % self.call_every:
            rec["skill_after"] = self.current_skill; rec["plan_complete"] = self.done_all
            self._write(rec); return rec
        t = self.task; skill = self.current_skill; verb, obj = t.parse_skill(skill)
        label = t.skills.index(skill)   # the store's label space; = self.idx for a plan in the task's order
        user, n0, hits = [], 0, []
        imgs = list(images) if self.use_wrist else list(images[:1])
        two = len(imgs) >= 2   # a wrist image may be missing even when use_wrist is set
        if self.memory is not None and len(self.memory):
            gate = self.gate and self.k_vote > 0; t0 = time.perf_counter()
            hits = self.memory.retrieve(imgs, k=max(self.k, self.k_vote) if gate else self.k, exclude_episode=self.exclude_episode)
            if self.log_memory:
                rec["retrieval_ms"] = round(1000 * (time.perf_counter() - t0), 3)
            if gate:
                rec["vote"] = vote = self._vote(hits[:self.k_vote], label)
                if self._gate_skips(vote, label):   # too few "done" votes: no question, the call counts as "not done"
                    rec["gated"] = True; self.consec_done = 0; rec["consecutive_done"] = 0
                    rec["skill_after"] = self.current_skill; self._write(rec); return rec
                hits = hits[:self.k]   # the model sees the k nearest of them
            if hits:
                fs = self.memory.fewshot_content(hits, label, skill, encode_png)
                user += fs; n0 = sum(1 for it in fs if it.get("type") == "image_url")
            rec["exemplars"] = [{"i": j, "sim": round(sim, 3), **self.memory.describe_hit(j)} for j, sim in hits]
        shown = []
        if kf is not None:   # the keyframes after the examples; the current pictures are numbered after them
            items, n_kf, info = kf.content(n0, len(imgs), lambda im: encode_png(self._prep(im)))
            user += items; n0 += n_kf; shown = info["shown"]; rec["keyframes"] = shown
            if info["dropped"]:
                rec["keyframes_dropped"] = info["dropped"]
        plan_txt = "\n".join(f"  {i + 1}. {s}{'  <- CURRENT' if i == self.idx else ('  (done)' if i < self.idx else '')}"
                             for i, s in enumerate(self.plan))
        pics = (f"Picture {n0 + 1} (external camera) and Picture {n0 + 2} (wrist camera) show" if two
                else f"Picture {n0 + 1} (external camera) shows")
        now = "Now the CURRENT observation. " if hits or shown else ""
        ptxt = self.proprio.text()
        wtxt = self.world.text() if self.world is not None else ""   # the world memory's state block (--world-state)
        objs = " and ".join(f"{o} {t.object_noun}" for o in t.objects) if len(t.objects) <= 2 else "every " + t.object_noun
        ask = ("Decide whether the current skill is complete." if self.verdict_only else
               f"Report the state of the {objs} and the gripper, then decide whether the current skill is complete.")
        if kf is not None:
            ask += kf.ask_text()
        rules_txt = ""
        if self.profile is not None:   # the notebook: extra completion rules the user gave for this skill
            rules = self.profile.rules_for_skill(skill)
            if rules:
                rules_txt = "\n" + self.profile.rules_text(rules); rec["profile_rules"] = [e.id for e in rules]; self.profile.note_applied(rules)
        user.append({"type": "text", "text": f"{now}Plan:\n{plan_txt}\n\nCurrent skill: \"{skill}\".\nCompletion criterion: "
                                             f"{t.criterion(skill)}{rules_txt}\n\n" + (ptxt + "\n\n" if ptxt else "") +
                                             (wtxt + "\n\n" if wtxt else "") +
                                             f"{pics} the scene NOW. Combine the proprioception with what you see: the "
                                             f"cameras tell you where the objects are, the sensors tell you whether the "
                                             f"gripper is closed on something and how high the hand is. {ask}"})
        for im in imgs:
            user.append({"type": "image_url", "image_url": {"url": encode_png(self._prep(im))}})
        msgs = [{"role": "system", "content": t.system_text(two)}, {"role": "user", "content": user}]
        rec["queried"] = True
        try:
            r = self.vlm.chat(msgs, schema=self.schema, name="monitor")
            j = r.get("json") or {}
            rec.update(vlm=j, ms=r.get("ms", 0.0), tokens=[r.get("prompt_tokens"), r.get("completion_tokens")], reasoning=r.get("reasoning"))
            done = j.get("current_skill_done") is True   # a string "false" from a backend without schema enforcement is not done
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
        if (done and self.consec_done >= self.votes and self.calls_in_skill >= self.min_calls
                and (self.world is None or self.world.check(skill, rec))):   # --world-verify: the world state must agree
            self.idx += 1; self.consec_done = 0; self.calls_in_skill = 0; rec["advanced"] = True
            if self.idx >= len(self.plan):
                self.done_all = True; self.idx = len(self.plan) - 1; rec["plan_complete"] = True
        rec["skill_after"] = self.current_skill
        if kf is not None:
            kf.end(rec)   # a switch (or the model's keep_frame) keeps this call's pictures for the later questions
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

    def log_event(self, rec: dict):
        """Append an event of another component to the decision log (e.g. the online store's store_update record)."""
        self._write(dict(rec))

    def close(self):
        if self._log:
            self._log.close()
