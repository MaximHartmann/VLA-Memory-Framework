"""What a session's records say: the entries the planner applied in an episode, whether it followed what the session
expects, and the summary of a run with the writer's scores against the annotations.

    applied_entries(records)   the plan rules, the skill rules (per skill, with the number of queries that carried them),
                               the planner-scope entries and their parameters of one episode, from the planner's records
    check(expect, plan, params)   whether the plan and the parameters are what the session expects
    summarise(records, store)  per phase: episodes, successes, followed, applied, held back; the writer's operations, calls
                               and scores; the notebook at the end
    score_writer(records)      extraction precision and recall, false memories, merge accuracy and validity on the
                               annotated messages
"""
from __future__ import annotations

import collections

from .writer_schema import WITHDRAWAL

MERGE_OPS = {"replaces": "UPDATE", "revokes": "DELETE", "repeats": "NOOP"}
BRIEF = ("id", "kind", "scope", "text", "condition", "params", "importance", "source", "validity", "expires")


def applied_entries(records, planner_ids=()) -> dict:
    """What the planner applied in one episode, from its records. Planner-scope ids come from the plan record when the
    planner logs them, otherwise from `planner_ids` (the store's planner rules at the episode's start)."""
    applied = {"plan": [], "skill": {}, "planner": [], "params": {}}
    for record in records:
        if record.get("event") == "plan":
            applied["plan"] = list(record.get("profile_plan_rules") or [])
            applied["params"] = dict(record.get("profile_params") or {})
            applied["planner"] = list(record.get("profile_planner_rules", planner_ids if applied["params"] else []))
        elif record.get("event") == "monitor":
            for entry_id in record.get("profile_rules") or []:
                counts = applied["skill"].setdefault(record.get("skill"), {})
                counts[entry_id] = counts.get(entry_id, 0) + 1
    return applied


def applied_ids(applied: dict) -> list[str]:
    """Every entry id an episode applied, once each: plan, planner, then the skill rules."""
    skill_ids = [entry_id for counts in applied["skill"].values() for entry_id in counts]
    return list(dict.fromkeys(applied["plan"] + applied["planner"] + skill_ids))


def check(expect: dict, plan: list, params: dict) -> dict | None:
    """Following: is the plan expect.plan, does it start with expect.plan_prefix, are the overrides expect.params in
    force?"""
    if not expect:
        return None
    following = {}
    if expect.get("plan") is not None:
        following["plan_exact"] = list(plan) == list(expect["plan"])
    if expect.get("plan_prefix") is not None:
        following["plan"] = list(plan[: len(expect["plan_prefix"])]) == list(expect["plan_prefix"])
    if expect.get("params"):
        following["params"] = all(params.get(name) == value for name, value in expect["params"].items())
    following["followed"] = all(following.values())
    return following


def brief(entry) -> dict:
    """An entry as the records show it."""
    return {name: getattr(entry, name) for name in BRIEF}


def phase_counts(records: list[dict]) -> dict:
    """Per phase: episodes, successes, checked and followed, with an applied entry, with a held-back rule."""
    phases: dict[str, dict] = {}
    for record in (record for record in records if record["event"] == "episode"):
        empty = dict(episodes=0, success=0, checked=0, followed=0, applied_any=0, held_any=0)
        counts = phases.setdefault(record.get("phase") or "", empty)
        counts["episodes"] += 1
        counts["success"] += bool(record["success"])
        counts["applied_any"] += bool(applied_ids(record["applied"]))
        counts["held_any"] += bool(record.get("held"))
        if record.get("check"):
            counts["checked"] += 1
            counts["followed"] += bool(record["check"]["followed"])
    return phases


def summarise(records: list[dict], store=None) -> dict:
    """The summary record of a run: the phases, the merge operations, the writer's calls and scores, the notebook's end."""
    calls = [call for record in records if record["event"] in ("message", "distil") for call in record.get("calls", [])]
    writer_calls = {"model": sum(not call.get("gold") for call in calls), "gold": sum(bool(call.get("gold")) for call in calls),
                    "errors": sum("error" in call for call in calls), "ms": round(sum(call.get("ms") or 0 for call in calls), 1)}
    return {"event": "summary", "phases": phase_counts(records),
            "ops": dict(collections.Counter(operation["op"] for record in records for operation in record.get("ops", []))),
            "writer_calls": writer_calls,
            "writer": score_writer(records),
            "active_end": [brief(entry) for entry in store.reload().active()] if store is not None else [],
            "entries_end": len(store.entries) if store is not None else 0}


def matches(proposal: dict, annotation: dict) -> bool:
    """Does a proposed entry match an annotation: scope, kind, preset and keywords?"""
    kind = WITHDRAWAL if annotation.get("revokes") else annotation.get("kind")   # an annotation that revokes is a withdrawal
    if proposal["scope"] != annotation.get("scope") or kind not in (None, proposal["kind"]):
        return False
    if proposal["scope"] == "planner" and proposal.get("preset") != annotation.get("params"):
        return False
    return all(str(keyword).lower() in proposal["text"].lower() for keyword in annotation.get("keywords") or [])


def score_message(record: dict, counter: collections.Counter, ids: dict, validity: collections.Counter) -> None:
    """Score the proposals and operations of one annotated message into the counters."""
    gold = list(record["expect"])
    used = set()
    for proposal, operation in zip(record["proposals"], record["ops"]):
        index = next((index for index, annotation in enumerate(gold) if index not in used and matches(proposal, annotation)), None)
        if index is None:
            counter["fp"] += 1
            counter["false_memories"] += not gold
            continue
        used.add(index)
        annotation = gold[index]
        counter["tp"] += 1
        if not annotation.get("revokes"):
            validity[f"{annotation.get('validity') or 'until_withdrawn'}>{proposal.get('validity') or 'until_withdrawn'}"] += 1
        relation = next((name for name in ("replaces", "revokes", "repeats") if annotation.get(name)), None)
        wanted = (MERGE_OPS[relation], ids.get(annotation[relation])) if relation else ("ADD", None)
        counter["merge_ok"] += (operation["op"], operation.get("target")) == wanted
        if annotation.get("label") and operation.get("entry"):
            ids[annotation["label"]] = operation["entry"]
    counter["fn"] += len(gold) - len(used)


def score_writer(records: list[dict]) -> dict:
    """The writer's measures on the annotated messages: extraction precision and recall (scope, kind, preset and keywords
    must match), false memories (rules from messages that state none), merge accuracy (ADD, or UPDATE / DELETE / NOOP of
    the entry that the annotation's replaces / revokes / repeats label points to) and, for every matched rule that is not
    a withdrawal, the validity (annotation `validity`, default until_withdrawn) against the written one: `confusion` counts
    "annotated>written"."""
    counter, ids, validity = collections.Counter(), {}, collections.Counter()
    for record in records:
        if record.get("event") == "message" and "proposals" in record and record.get("expect") is not None:
            score_message(record, counter, ids, validity)
    true_positives, false_positives, false_negatives = counter["tp"], counter["fp"], counter["fn"]
    right = sum(count for pair, count in validity.items() if len(set(pair.split(">"))) == 1)
    return {"proposed": true_positives + false_positives, "annotated": true_positives + false_negatives, "matched": true_positives,
            "precision": round(true_positives / (true_positives + false_positives), 3) if true_positives + false_positives else None,
            "recall": round(true_positives / (true_positives + false_negatives), 3) if true_positives + false_negatives else None,
            "false_memories": counter["false_memories"],
            "merge_correct": counter["merge_ok"],
            "validity": {"right": right, "of": sum(validity.values()), "confusion": dict(sorted(validity.items()))}}
