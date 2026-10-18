"""The oracle writer: the writer's prompts, schemas and store operations, with the answers taken from annotations.

An annotation (gold item) is {label, kind, scope, text, validity, condition, params: <preset name>} plus at most one of
replaces | revokes | repeats: <label of an earlier item>, which makes the merge an UPDATE, DELETE or NOOP of that item's
entry; without them the merge is an ADD. An item that revokes is a withdrawal. A failed episode gets every annotated
correction (e.g. the same completion rule for each object's skill). No model is called; the dry runs use it.
"""
from __future__ import annotations

from .store import ProfileStore
from .writer import PreferenceWriter
from .writer_schema import WITHDRAWAL

MERGE_OPS = {"replaces": "UPDATE", "revokes": "DELETE", "repeats": "NOOP"}
DEFAULTS = {"kind": "preference", "validity": "until_withdrawn", "condition": "", "params": "none", "step": 0}


class GoldWriter(PreferenceWriter):
    DISTIL_MAX = None

    def __init__(self, task, store: ProfileStore, **kw):
        super().__init__(None, task, store, **kw)
        self.ids: dict[str, str] = {}      # label -> entry id
        self._gold: list[dict] = []        # the annotated items of the current question

    def _ask(self, name, system, user, schema, max_tokens, ctx=None):
        ctx = ctx or {}
        if name == "merge":
            ans = self._merge_answer_from(ctx["proposal"], ctx["candidates"])
        else:
            ans = self._entries_answer_from(ctx.get("gold"), schema)
        rec = {"name": name, "answer": ans, "ms": 0.0, "gold": True}
        if self.keep_prompts:
            rec["prompt"] = [system, user]
        self.calls.append(rec)
        return ans

    def _merge_answer_from(self, proposal: dict, candidates: list) -> dict:
        """The merge operation the annotation implies; the same annotated rule again (e.g. a correction) is a NOOP."""
        g = self._match(proposal)
        ids = [e.id for e in candidates]
        key = next((k for k in ("replaces", "revokes", "repeats") if g.get(k)), None)
        i = self.ids.get(g[key]) if key else None
        same = next((e.id for e in candidates if e.text == proposal["text"]), None)
        if i in ids:
            return {"op": MERGE_OPS[key], "id": i}
        if same and not key:
            return {"op": "NOOP", "id": same}
        return {"op": "NOOP" if key == "revokes" else "ADD", "id": "none"}

    def _entries_answer_from(self, gold, schema: dict) -> dict:
        """The annotated items as the answer of an extraction or a distillation; an item that revokes is a withdrawal."""
        self._gold = [dict(g) for g in gold or []]
        fields = schema["properties"]["entries"]["items"]["properties"]
        items = []
        for g in self._gold:
            given = {k: v for k, v in g.items() if v is not None}
            items.append({**DEFAULTS, **given, **({"kind": WITHDRAWAL} if g.get("revokes") else {})})
        return {"entries": [{k: it.get(k) for k in fields} for it in items]}

    def merge(self, p: dict) -> dict:
        rec = super().merge(p)
        g = self._match(p)
        if rec["entry"] and g.get("label"):
            self.ids[g["label"]] = rec["entry"]
        return rec

    def _match(self, p: dict) -> dict:
        same = (g for g in self._gold if g.get("scope") == p["scope"] and str(g.get("text", "")).strip() == p["text"])
        return next(same, {})
