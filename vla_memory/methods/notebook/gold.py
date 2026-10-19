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
    """The PreferenceWriter with every model question answered from the annotations of the current step."""

    DISTIL_MAX = None

    def __init__(self, task, store: ProfileStore, **options):
        super().__init__(None, task, store, **options)
        self.ids: dict[str, str] = {}      # label -> entry id
        self._gold: list[dict] = []        # the annotated items of the current question

    def _ask(self, name, system, user, schema, max_tokens, context=None):
        """The annotation's answer to the question `name`; the call record notes it as gold."""
        context = context or {}
        if name == "merge":
            answer = self._merge_answer_from(context["proposal"], context["candidates"])
        else:
            answer = self._entries_answer_from(context.get("gold"), schema)
        record = {"name": name, "answer": answer, "ms": 0.0, "gold": True}
        if self.keep_prompts:
            record["prompt"] = [system, user]
        self.calls.append(record)
        return answer

    def _merge_answer_from(self, proposal: dict, candidates: list) -> dict:
        """The merge operation the annotation implies; the same annotated rule again (e.g. a correction) is a NOOP."""
        annotation = self._match(proposal)
        ids = [entry.id for entry in candidates]
        relation = next((name for name in ("replaces", "revokes", "repeats") if annotation.get(name)), None)
        target = self.ids.get(annotation[relation]) if relation else None
        same = next((entry.id for entry in candidates if entry.text == proposal["text"]), None)
        if target in ids:
            return {"op": MERGE_OPS[relation], "id": target}
        if same and not relation:
            return {"op": "NOOP", "id": same}
        return {"op": "NOOP" if relation == "revokes" else "ADD", "id": "none"}

    def _entries_answer_from(self, gold, schema: dict) -> dict:
        """The annotated items as the answer of an extraction or a distillation; an item that revokes is a withdrawal."""
        self._gold = [dict(annotation) for annotation in gold or []]
        fields = schema["properties"]["entries"]["items"]["properties"]
        items = []
        for annotation in self._gold:
            given = {name: value for name, value in annotation.items() if value is not None}
            items.append({**DEFAULTS, **given, **({"kind": WITHDRAWAL} if annotation.get("revokes") else {})})
        return {"entries": [{name: item.get(name) for name in fields} for item in items]}

    def merge(self, proposal: dict) -> dict:
        """The merge of the PreferenceWriter; a stored entry is remembered under its annotation's label."""
        record = super().merge(proposal)
        annotation = self._match(proposal)
        if record["entry"] and annotation.get("label"):
            self.ids[annotation["label"]] = record["entry"]
        return record

    def _match(self, proposal: dict) -> dict:
        """The annotated item with the proposal's scope and text, or {}."""
        same = (annotation for annotation in self._gold
                if annotation.get("scope") == proposal["scope"] and str(annotation.get("text", "")).strip() == proposal["text"])
        return next(same, {})
