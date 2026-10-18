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
    out = {"plan": [], "skill": {}, "planner": [], "params": {}}
    for r in records:
        if r.get("event") == "plan":
            out["plan"] = list(r.get("profile_plan_rules") or [])
            out["params"] = dict(r.get("profile_params") or {})
            out["planner"] = list(r.get("profile_planner_rules", planner_ids if out["params"] else []))
        elif r.get("event") == "monitor":
            for i in r.get("profile_rules") or []:
                counts = out["skill"].setdefault(r.get("skill"), {})
                counts[i] = counts.get(i, 0) + 1
    return out


def applied_ids(applied: dict) -> list[str]:
    skill_ids = [i for counts in applied["skill"].values() for i in counts]
    return list(dict.fromkeys(applied["plan"] + applied["planner"] + skill_ids))


def check(expect: dict, plan: list, params: dict) -> dict | None:
    """Following: is the plan expect.plan, does it start with expect.plan_prefix, are the overrides expect.params in
    force?"""
    if not expect:
        return None
    out = {}
    if expect.get("plan") is not None:
        out["plan_exact"] = list(plan) == list(expect["plan"])
    if expect.get("plan_prefix") is not None:
        out["plan"] = list(plan[: len(expect["plan_prefix"])]) == list(expect["plan_prefix"])
    if expect.get("params"):
        out["params"] = all(params.get(k) == v for k, v in expect["params"].items())
    out["followed"] = all(out.values())
    return out


def brief(e) -> dict:
    """An entry as the records show it."""
    return {k: getattr(e, k) for k in BRIEF}


def phase_counts(recs: list[dict]) -> dict:
    """Per phase: episodes, successes, checked and followed, with an applied entry, with a held-back rule."""
    phases: dict[str, dict] = {}
    for r in (r for r in recs if r["event"] == "episode"):
        empty = dict(episodes=0, success=0, checked=0, followed=0, applied_any=0, held_any=0)
        p = phases.setdefault(r.get("phase") or "", empty)
        p["episodes"] += 1
        p["success"] += bool(r["success"])
        p["applied_any"] += bool(applied_ids(r["applied"]))
        p["held_any"] += bool(r.get("held"))
        if r.get("check"):
            p["checked"] += 1
            p["followed"] += bool(r["check"]["followed"])
    return phases


def summarise(recs: list[dict], store=None) -> dict:
    calls = [c for r in recs if r["event"] in ("message", "distil") for c in r.get("calls", [])]
    writer_calls = {"model": sum(not c.get("gold") for c in calls), "gold": sum(bool(c.get("gold")) for c in calls),
                    "errors": sum("error" in c for c in calls), "ms": round(sum(c.get("ms") or 0 for c in calls), 1)}
    return {"event": "summary", "phases": phase_counts(recs),
            "ops": dict(collections.Counter(o["op"] for r in recs for o in r.get("ops", []))),
            "writer_calls": writer_calls,
            "writer": score_writer(recs),
            "active_end": [brief(e) for e in store.reload().active()] if store is not None else [],
            "entries_end": len(store.entries) if store is not None else 0}


def matches(p: dict, g: dict) -> bool:
    """Does a proposed entry match an annotation: scope, kind, preset and keywords?"""
    kind = WITHDRAWAL if g.get("revokes") else g.get("kind")   # an annotation that revokes an earlier one is a withdrawal
    if p["scope"] != g.get("scope") or kind not in (None, p["kind"]):
        return False
    if p["scope"] == "planner" and p.get("preset") != g.get("params"):
        return False
    return all(str(k).lower() in p["text"].lower() for k in g.get("keywords") or [])


def score_message(r: dict, s: collections.Counter, ids: dict, validity: collections.Counter) -> None:
    """Score the proposals and operations of one annotated message into the counters."""
    gold = list(r["expect"])
    used = set()
    for p, op in zip(r["proposals"], r["ops"]):
        j = next((j for j, g in enumerate(gold) if j not in used and matches(p, g)), None)
        if j is None:
            s["fp"] += 1
            s["false_memories"] += not gold
            continue
        used.add(j)
        g = gold[j]
        s["tp"] += 1
        if not g.get("revokes"):
            validity[f"{g.get('validity') or 'until_withdrawn'}>{p.get('validity') or 'until_withdrawn'}"] += 1
        key = next((k for k in ("replaces", "revokes", "repeats") if g.get(k)), None)
        want = (MERGE_OPS[key], ids.get(g[key])) if key else ("ADD", None)
        s["merge_ok"] += (op["op"], op.get("target")) == want
        if g.get("label") and op.get("entry"):
            ids[g["label"]] = op["entry"]
    s["fn"] += len(gold) - len(used)


def score_writer(recs: list[dict]) -> dict:
    """The writer's measures on the annotated messages: extraction precision and recall (scope, kind, preset and keywords
    must match), false memories (rules from messages that state none), merge accuracy (ADD, or UPDATE / DELETE / NOOP of
    the entry that the annotation's replaces / revokes / repeats label points to) and, for every matched rule that is not
    a withdrawal, the validity (annotation `validity`, default until_withdrawn) against the written one: `confusion` counts
    "annotated>written"."""
    s, ids, validity = collections.Counter(), {}, collections.Counter()
    for r in recs:
        if r.get("event") == "message" and "proposals" in r and r.get("expect") is not None:
            score_message(r, s, ids, validity)
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    right = sum(n for k, n in validity.items() if len(set(k.split(">"))) == 1)
    return {"proposed": tp + fp, "annotated": tp + fn, "matched": tp,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None, "false_memories": s["false_memories"],
            "merge_correct": s["merge_ok"],
            "validity": {"right": right, "of": sum(validity.values()), "confusion": dict(sorted(validity.items()))}}
