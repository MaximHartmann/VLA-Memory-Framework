"""The notebook's writer: user messages and failed episodes become entries, merged into the profile Mem0-style.

    extract(message)        the planning model turns a user message into at most `max_entries` proposed entries, under a
                            JSON schema that admits only the store's fields: text, kind, a scope from the task's vocabulary
                            (plan, planner, skill:<skill>), validity (until withdrawn, today, this session: from the
                            message's words), condition, and a strictness preset as params
    distil(records, ...)    after a failed episode: the planner's own decision log (plus the outcome and the user's remark,
                            if any) becomes at most one correction, keyed by the skill where it went wrong
    merge(proposal)         the proposal is compared with the active entries of its scope; the model chooses ADD,
                            UPDATE <id> (the new entry supersedes <id>), DELETE <id> (retire it) or NOOP <id> (a repetition:
                            upvote <id>). A scope without entries means ADD, without a model call. A withdrawal ("forget
                            what I said about ...") is never stored: it can only retire an entry (DELETE) or change nothing
    feedback(ids, success)  ExpeL-style votes after an episode: applied corrections gain on success and lose on failure
                            (retired at 0); stated preferences are not voted out by the robot's own failures

from_message() and from_failure() chain these and return what was proposed and done, with every model call. The schemas
and texts are in writer_schema.py, the failed episode's log in writer_log.py; gold.GoldWriter answers the same questions
from annotations (the oracle writer, and the dry run).
"""
from __future__ import annotations

from typing import Any

from .entry import STRICTNESS_PRESETS, VALIDITY
from .store import ProfileStore
from .writer_log import query_key, query_line, render_log
from .writer_schema import (KINDS, OPS, ROLE, WITHDRAWAL, answer_entries, entry_schema, merge_schema, scopes, show,
                            vocabulary)

__all__ = ["PreferenceWriter", "entry_schema", "merge_schema", "scopes", "KINDS", "WITHDRAWAL", "OPS", "ROLE"]


class PreferenceWriter:
    """vlm: the planning model (VLMBackend); store: the notebook it writes. At most max_entries rules per message, at most
    max_candidates entries shown to the merge, about log_lines monitor queries shown to the distiller; keep_prompts stores
    every prompt in the call records; reasoning_tokens adds answer room for a model that thinks before it answers (its
    reasoning is kept in the call records too), capped by what the model's context_tokens leave after the prompt. The
    instructions are class attributes, so a variant can be tried without copying the methods."""

    EXTRACT_TASK = ("Turn the user's message into at most {n} rules. text: the rule as one short imperative sentence, "
                    "without a time limit; kind: preference (how the user wants things done), correction (something that "
                    "went wrong and must change) or withdrawal (the user takes back an earlier wish without stating a "
                    "new one; scope: where that wish acted; text: what is taken back); validity: how long the rule "
                    "holds, from the message's words: session (only this time, just this once, for this session), day "
                    "(today, for the rest of the day) or until_withdrawn (from now on, always, until the user says "
                    "otherwise, and whenever the message sets no time limit); condition: when the rule applies, as a "
                    "clause starting with 'when', 'if' or 'unless', or empty if always (a time limit is the validity, "
                    "not a condition); params: a preset for scope planner, otherwise none. A new wish that replaces an "
                    "old one is a preference, not a withdrawal. A message without a lasting wish (thanks, small talk, a "
                    "question, a remark about the scene) gives no rules; a wish for this time or for today is a rule "
                    "too, with that validity.")
    DISTIL_TASK = ("An episode failed. Read the planner's decision log and the user's remark, if any, find where it went "
                   "wrong and write at most one correction that would have prevented it: usually skill:<skill> with a "
                   "stricter completion condition when the planner moved on too early, or plan when the order or the "
                   "choice of skills was wrong. step: the step where it went wrong. Give no rule if the log does not "
                   "show the cause.")
    MERGE_TASK = ("A new rule is proposed. Compare it with the notebook's rules of the same scope ({scope}) and choose "
                  "one operation: ADD (a new rule that leaves the others as they are; id none), UPDATE <id> (it changes, "
                  "refines or contradicts rule <id> and replaces it), DELETE <id> (it only withdraws rule <id>), NOOP "
                  "<id> (it says the same as rule <id>) or NOOP none (nothing worth keeping).")
    DISTIL_MAX = 1   # corrections per failed episode
    WITHDRAW_TASK = ("The user takes back an earlier wish of the scope {scope}. Choose DELETE <id> for the notebook rule "
                     "that is taken back, or NOOP none if none of them is meant.")

    def __init__(self, vlm, task, store: ProfileStore, max_entries: int = 3, max_candidates: int = 8, log_lines: int = 40,
                 presets: dict | None = None, keep_prompts: bool = False, reasoning_tokens: int = 0,
                 context_tokens: int | None = None):
        self.vlm = vlm
        self.task = task
        self.store = store
        self.max_entries = max_entries
        self.reasoning_tokens = reasoning_tokens
        self.max_candidates = max_candidates
        self.log_lines = log_lines
        self.keep_prompts = keep_prompts
        self.presets = dict(presets or STRICTNESS_PRESETS)
        self.scopes = scopes(task)
        self.calls: list[dict] = []
        self.context_tokens = context_tokens

    # ------------------------------------------------------------------ write paths
    def from_message(self, text: str, gold=None) -> dict:
        """A user message -> proposed entries -> merged into the notebook."""
        self.calls = []
        self.store.reload()   # the planner may have saved its applied-counters since
        props = self.extract(text, gold)
        return {"proposals": props, "ops": [self.merge(p) for p in props], "calls": self.calls}

    def from_failure(self, records, outcome: str = "", complaint: str = "", gold=None) -> dict:
        """A failed episode's planner records (+ the outcome, + the user's remark) -> at most one correction, merged."""
        self.calls = []
        self.store.reload()
        props = self.distil(records, outcome, complaint, gold)
        return {"proposals": props, "ops": [self.merge(p) for p in props], "calls": self.calls}

    def feedback(self, applied_ids, success: bool, kinds=("correction",), cap: int = 5) -> dict:
        """+1 for every applied entry of these kinds after a success, up to `cap` (so that a rule which later harms is
        retired within a few failures), -1 after a failure (retired at 0)."""
        self.store.reload()
        out = {"up": [], "down": [], "retired": []}
        for i in dict.fromkeys(applied_ids):
            e = self.store.get(i)
            if e is None or not e.active or e.kind not in kinds:
                continue
            if success and e.importance < cap:
                self.store.upvote(i)
                out["up"].append(i)
            elif not success:
                self.store.downvote(i)
                out["down"].append(i)
                if not self.store.get(i).active:
                    out["retired"].append(i)
        return out

    # ------------------------------------------------------------------ the three questions
    def extract(self, text: str, gold=None) -> list[dict]:
        system = f"{ROLE}\n{self._vocab()}\n" + self.EXTRACT_TASK.format(n=self.max_entries)
        schema = entry_schema(self.task, self.max_entries, self.presets)
        ans = self._ask("extract", system, f'Message: "{text}"', schema, 400, {"gold": gold})
        items = answer_entries(ans)[: self.max_entries]
        proposals = (self._proposal(it, source="user", evidence=f'"{text}"') for it in items)
        return [p for p in proposals if p]

    def distil(self, records, outcome: str = "", complaint: str = "", gold=None) -> list[dict]:
        system = f"{ROLE}\n{self._vocab()}\n" + self.DISTIL_TASK
        user = self.render_log(records, outcome, complaint)
        schema = entry_schema(self.task, 1, self.presets, distil=True)
        ans = self._ask("distil", system, user, schema, 300, {"gold": gold})
        ep = next((r.get("episode") for r in records if r.get("episode") is not None), "?")
        out = []
        for it in answer_entries(ans)[: self.DISTIL_MAX]:
            ev = f"episode {ep} step {it.get('step')}" + (f'; user: "{complaint}"' if complaint else "")
            p = self._proposal(it, kind="correction", source="distilled", evidence=ev)
            if p:
                out.append(p)
        return out

    def merge(self, p: dict) -> dict:
        """Compare a proposal with the active entries of its scope, choose the operation and apply it to the store."""
        cands = self.store.active(p["scope"])[: self.max_candidates]
        ids = [e.id for e in cands]
        back = p["kind"] == WITHDRAWAL
        ops = ("DELETE", "NOOP") if back else OPS
        op = "NOOP" if back else "ADD"
        target = None
        rec = {"scope": p["scope"], "text": p["text"], "candidates": ids}
        if cands:
            op, target = self._merge_answer(p, cands, ops, op, rec)
        if target is None and op != "ADD":   # an update of nothing is a new rule; deleting or repeating nothing does nothing
            op = "ADD" if op == "UPDATE" else "NOOP"
        rec.update(op=op, target=target, entry=self._apply(p, op, target))
        return rec

    def _merge_answer(self, p: dict, cands: list, ops: tuple, op: str, rec: dict) -> tuple:
        """Ask the merge question; returns the chosen operation and its target id (None for none)."""
        ids = [e.id for e in cands]
        back = p["kind"] == WITHDRAWAL
        system = f"{ROLE}\n" + (self.WITHDRAW_TASK if back else self.MERGE_TASK).format(scope=p["scope"])
        rules = "\n".join(f"- [{e.id}] {self._show(e.text, e.condition, e.params, e.validity)}" for e in cands)
        if back:
            user = "Notebook rules:\n" + rules + f"\nTaken back: {p['text']}"
        else:
            proposed = self._show(p["text"], p["condition"], p["params"], p["validity"])
            user = "Notebook rules:\n" + rules + f"\nProposed rule: {proposed}"
        ans = self._ask("merge", system, user, merge_schema(ids, ops), 64, {"proposal": p, "candidates": cands})
        if ans and ans.get("op") in ops:
            return ans["op"], (ans.get("id") if ans.get("id") in ids else None)
        rec["fallback"] = True   # failed call: keep a statement rather than lose it; a withdrawal then changes nothing
        return op, None

    # ------------------------------------------------------------------ helpers
    def render_log(self, records, outcome: str = "", complaint: str = "") -> str:
        """The failed episode as the distiller reads it (writer_log.py)."""
        return render_log(self.task, records, outcome, complaint, self.log_lines)

    _query_key = staticmethod(query_key)
    _query_line = staticmethod(query_line)

    def _vocab(self) -> str:
        return vocabulary(self.task, self.presets)

    def _show(self, text: str, condition: str, params: dict, validity: str = "until_withdrawn") -> str:
        return show(text, condition, params, validity, self.presets)

    def _proposal(self, item: dict, kind: str | None = None, source: str = "user", evidence: str = "") -> dict | None:
        """Schema answer -> proposed entry. None outside the vocabulary (backends without schema enforcement) and for a
        planner rule without a preset, since the planner reads only the parameters of that scope."""
        scope = item.get("scope")
        text = str(item.get("text") or "").strip()
        kind = kind or item.get("kind")
        presetted = scope == "planner" and kind != WITHDRAWAL and item.get("params") in self.presets
        preset = item.get("params") if presetted else ""
        missing_preset = scope == "planner" and kind != WITHDRAWAL and not preset
        if scope not in self.scopes or not text or kind not in (*KINDS, WITHDRAWAL) or missing_preset:
            return None
        validity = item.get("validity") if item.get("validity") in VALIDITY else "until_withdrawn"
        return {"kind": kind, "scope": scope, "text": text, "condition": str(item.get("condition") or "").strip(),
                "validity": validity, "params": dict(self.presets[preset]) if preset else {}, "preset": preset,
                "source": source, "evidence": evidence}

    def _apply(self, p: dict, op: str, target: str | None) -> str | None:
        if op in ("ADD", "UPDATE"):   # an UPDATE that holds for a shorter time suspends the old entry (ProfileStore.add)
            supersedes = target if op == "UPDATE" else None
            entry = self.store.add(p["kind"], p["scope"], p["text"], p["condition"], p["params"], p["source"], p["evidence"],
                                   supersedes=supersedes, validity=p.get("validity", "until_withdrawn"))
            return entry.id
        if op == "DELETE":
            self.store.retire(target)
        elif op == "NOOP" and target:
            self.store.upvote(target)   # a repetition confirms the rule, and keeps the longer of the two validities
            if p["kind"] != WITHDRAWAL:
                self.store.extend(target, p.get("validity", "until_withdrawn"))
        return None

    def _max_tokens(self, system: str, user: str, max_tokens: int) -> int:
        """The answer room: max_tokens plus the reasoning allowance, never more than the context leaves (prompts measured
        at 3.4 or more characters per token)."""
        mt = max_tokens + self.reasoning_tokens
        if self.context_tokens:
            mt = max(max_tokens, min(mt, self.context_tokens - (len(system) + len(user)) // 3 - 128))
        return mt

    def _ask(self, name: str, system: str, user: str, schema: dict, max_tokens: int, ctx: dict | None = None) -> dict | None:
        rec: dict[str, Any] = {"name": name}
        if self.keep_prompts:
            rec["prompt"] = [system, user]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            r = self.vlm.chat(messages, schema=schema, name=name, max_tokens=self._max_tokens(system, user, max_tokens))
            rec.update(answer=r.get("json"), ms=r.get("ms"), tokens=[r.get("prompt_tokens"), r.get("completion_tokens")])
            if r.get("reasoning"):
                rec["reasoning"] = r["reasoning"]
            if r.get("json") is None and r.get("text"):   # unparsable answer: keep its start for diagnosis
                rec["raw"] = r["text"][:600]
        except Exception as e:  # server down, timeout, unparsable answer: nothing is written from this call
            rec["error"] = repr(e)
        self.calls.append(rec)
        return rec.get("answer")
