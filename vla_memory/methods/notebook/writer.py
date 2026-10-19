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
from .writer_log import render_log
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
        proposals = self.extract(text, gold)
        return {"proposals": proposals, "ops": [self.merge(proposal) for proposal in proposals], "calls": self.calls}

    def from_failure(self, records, outcome: str = "", complaint: str = "", gold=None) -> dict:
        """A failed episode's planner records (+ the outcome, + the user's remark) -> at most one correction, merged."""
        self.calls = []
        self.store.reload()
        proposals = self.distil(records, outcome, complaint, gold)
        return {"proposals": proposals, "ops": [self.merge(proposal) for proposal in proposals], "calls": self.calls}

    def feedback(self, applied_ids, success: bool, kinds=("correction",), cap: int = 5) -> dict:
        """+1 for every applied entry of these kinds after a success, up to `cap` (so that a rule which later harms is
        retired within a few failures), -1 after a failure (retired at 0)."""
        self.store.reload()
        votes = {"up": [], "down": [], "retired": []}
        for entry_id in dict.fromkeys(applied_ids):
            entry = self.store.get(entry_id)
            if entry is None or not entry.active or entry.kind not in kinds:
                continue
            if success and entry.importance < cap:
                self.store.upvote(entry_id)
                votes["up"].append(entry_id)
            elif not success:
                self.store.downvote(entry_id)
                votes["down"].append(entry_id)
                if not self.store.get(entry_id).active:
                    votes["retired"].append(entry_id)
        return votes

    # ------------------------------------------------------------------ the three questions
    def extract(self, text: str, gold=None) -> list[dict]:
        """A user message -> the proposed entries (at most max_entries), checked against the vocabulary."""
        system = f"{ROLE}\n{self._vocab()}\n" + self.EXTRACT_TASK.format(n=self.max_entries)
        schema = entry_schema(self.task, self.max_entries, self.presets)
        answer = self._ask("extract", system, f'Message: "{text}"', schema, 400, {"gold": gold})
        items = answer_entries(answer)[: self.max_entries]
        proposals = (self._proposal(item, source="user", evidence=f'"{text}"') for item in items)
        return [proposal for proposal in proposals if proposal]

    def distil(self, records, outcome: str = "", complaint: str = "", gold=None) -> list[dict]:
        """A failed episode's records -> the proposed corrections (at most DISTIL_MAX), with the episode and step as evidence."""
        system = f"{ROLE}\n{self._vocab()}\n" + self.DISTIL_TASK
        user = self.render_log(records, outcome, complaint)
        schema = entry_schema(self.task, 1, self.presets, distil=True)
        answer = self._ask("distil", system, user, schema, 300, {"gold": gold})
        episode_id = next((record.get("episode") for record in records if record.get("episode") is not None), "?")
        proposals = []
        for item in answer_entries(answer)[: self.DISTIL_MAX]:
            evidence = f"episode {episode_id} step {item.get('step')}" + (f'; user: "{complaint}"' if complaint else "")
            proposal = self._proposal(item, kind="correction", source="distilled", evidence=evidence)
            if proposal:
                proposals.append(proposal)
        return proposals

    def merge(self, proposal: dict) -> dict:
        """Compare a proposal with the active entries of its scope, choose the operation and apply it to the store."""
        candidates = self.store.active(proposal["scope"])[: self.max_candidates]
        ids = [entry.id for entry in candidates]
        withdrawal = proposal["kind"] == WITHDRAWAL
        operations = ("DELETE", "NOOP") if withdrawal else OPS
        operation = "NOOP" if withdrawal else "ADD"
        target = None
        record = {"scope": proposal["scope"], "text": proposal["text"], "candidates": ids}
        if candidates:
            operation, target = self._merge_answer(proposal, candidates, operations, operation, record)
        if target is None and operation != "ADD":   # an update of nothing is a new rule; deleting or repeating nothing does nothing
            operation = "ADD" if operation == "UPDATE" else "NOOP"
        record.update(op=operation, target=target, entry=self._apply(proposal, operation, target))
        return record

    def _merge_answer(self, proposal: dict, candidates: list, operations: tuple, operation: str, record: dict) -> tuple:
        """Ask the merge question; returns the chosen operation and its target id (None for none)."""
        ids = [entry.id for entry in candidates]
        withdrawal = proposal["kind"] == WITHDRAWAL
        system = f"{ROLE}\n" + (self.WITHDRAW_TASK if withdrawal else self.MERGE_TASK).format(scope=proposal["scope"])
        rules = "\n".join(f"- [{entry.id}] {self._show(entry.text, entry.condition, entry.params, entry.validity)}"
                          for entry in candidates)
        if withdrawal:
            user = "Notebook rules:\n" + rules + f"\nTaken back: {proposal['text']}"
        else:
            proposed = self._show(proposal["text"], proposal["condition"], proposal["params"], proposal["validity"])
            user = "Notebook rules:\n" + rules + f"\nProposed rule: {proposed}"
        answer = self._ask("merge", system, user, merge_schema(ids, operations), 64,
                           {"proposal": proposal, "candidates": candidates})
        if answer and answer.get("op") in operations:
            return answer["op"], (answer.get("id") if answer.get("id") in ids else None)
        record["fallback"] = True   # failed call: keep a statement rather than lose it; a withdrawal then changes nothing
        return operation, None

    # ------------------------------------------------------------------ helpers
    def render_log(self, records, outcome: str = "", complaint: str = "") -> str:
        """The failed episode as the distiller reads it (writer_log.py)."""
        return render_log(self.task, records, outcome, complaint, self.log_lines)

    def _vocab(self) -> str:
        """The skills, the scopes and the presets in words (writer_schema.vocabulary)."""
        return vocabulary(self.task, self.presets)

    def _show(self, text: str, condition: str, params: dict, validity: str = "until_withdrawn") -> str:
        """One rule as the merge shows it (writer_schema.show)."""
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

    def _apply(self, proposal: dict, operation: str, target: str | None) -> str | None:
        """Carry the operation out on the store; returns the id of a new entry (ADD, UPDATE), else None."""
        if operation in ("ADD", "UPDATE"):   # an UPDATE that holds for a shorter time suspends the old entry (ProfileStore.add)
            supersedes = target if operation == "UPDATE" else None
            entry = self.store.add(proposal["kind"], proposal["scope"], proposal["text"], proposal["condition"], proposal["params"],
                                   proposal["source"], proposal["evidence"], supersedes=supersedes,
                                   validity=proposal.get("validity", "until_withdrawn"))
            return entry.id
        if operation == "DELETE":
            self.store.retire(target)
        elif operation == "NOOP" and target:
            self.store.upvote(target)   # a repetition confirms the rule, and keeps the longer of the two validities
            if proposal["kind"] != WITHDRAWAL:
                self.store.extend(target, proposal.get("validity", "until_withdrawn"))
        return None

    def _max_tokens(self, system: str, user: str, max_tokens: int) -> int:
        """The answer room: max_tokens plus the reasoning allowance, never more than the context leaves (prompts measured
        at 3.4 or more characters per token)."""
        allowed = max_tokens + self.reasoning_tokens
        if self.context_tokens:
            allowed = max(max_tokens, min(allowed, self.context_tokens - (len(system) + len(user)) // 3 - 128))
        return allowed

    def _ask(self, name: str, system: str, user: str, schema: dict, max_tokens: int, context: dict | None = None) -> dict | None:
        """One question to the model; its answer (None when the call failed), and a call record with the answer, the
        latency, the token counts, the reasoning and the start of an unparsable answer."""
        record: dict[str, Any] = {"name": name}
        if self.keep_prompts:
            record["prompt"] = [system, user]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            reply = self.vlm.chat(messages, schema=schema, name=name, max_tokens=self._max_tokens(system, user, max_tokens))
            record.update(answer=reply.get("json"), ms=reply.get("ms"), tokens=[reply.get("prompt_tokens"), reply.get("completion_tokens")])
            if reply.get("reasoning"):
                record["reasoning"] = reply["reasoning"]
            if reply.get("json") is None and reply.get("text"):   # unparsable answer: keep its start for diagnosis
                record["raw"] = reply["text"][:600]
        except Exception as error:  # server down, timeout, unparsable answer: nothing is written from this call
            record["error"] = repr(error)
        self.calls.append(record)
        return record.get("answer")
