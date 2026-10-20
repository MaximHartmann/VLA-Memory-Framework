"""The memory-vote gate: the exemplar store's own verdict on the current skill, which may settle a monitor call without
asking the model.

Each retrieved moment votes "done" when its stored skill is later in the task's skill order than the current skill, the
same rule that labels the examples shown to the model. A full vote with fewer than `gate_min_done` "done" votes settles
the call as "not done"; any "done" vote asks the model. The last skill of the plan, and the task's last skill, are always
asked: no stored moment is labelled later than the task's last skill, so its vote is never "done". A task whose
plan order varies from episode to episode has no order of skills across objects: the vote is right between the skills of
one object and between plan stages, and arbitrary between two objects of one stage (--no-gate asks at every call).
"""
from __future__ import annotations

import time


def vote(memory, hits, label: int) -> dict | None:
    """The retrieved moments' verdict on the skill with index `label`: how many say "done", of how many, and the lowest
    similarity. None when the store does not label its entries with a skill index (the gate then cannot decide)."""
    labels = [memory.describe_hit(index).get("subgoal") for index, _ in hits]
    if not hits or any(stored is None for stored in labels):
        return None
    done = sum(int(stored) > label for stored in labels)
    lowest_similarity = min(similarity for _, similarity in hits)
    return {"done": done, "of": len(labels), "min_sim": round(lowest_similarity, 3)}


def gate_skips(verdict: dict | None, k_vote: int, gate_min_done: int, last_in_plan: bool, last_in_task: bool) -> bool:
    """Whether the gate settles the call as "not done" instead of asking the model."""
    if verdict is None or verdict["of"] < k_vote or verdict["done"] >= gate_min_done:
        return False
    return not last_in_plan and not last_in_task


def retrieve_with_gate(planner, pictures, record: dict) -> list:
    """The stored moments most similar to the pictures, as the planner shows them to the model (its k nearest); with the
    gate, the k_vote nearest vote first and may settle the call (record["gated"] True, no hits). The record gets the
    retrieval time (with log_memory), the vote and the examples shown."""
    memory = planner.memory
    if memory is None or not len(memory):
        return []
    label = planner.current_label()
    use_gate = planner.gate and planner.k_vote > 0
    count = max(planner.k, planner.k_vote) if use_gate else planner.k
    started = time.perf_counter()
    hits = memory.retrieve(pictures, k=count, exclude_episode=planner.exclude_episode)
    if planner.log_memory:
        record["retrieval_ms"] = round(1000 * (time.perf_counter() - started), 3)
    if use_gate:
        verdict = vote(memory, hits[:planner.k_vote], label)
        record["vote"] = verdict
        last_in_plan = planner.skill_index >= len(planner.plan) - 1
        last_in_task = label >= len(planner.task.skills) - 1
        if gate_skips(verdict, planner.k_vote, planner.gate_min_done, last_in_plan, last_in_task):
            record["gated"] = True
            return []
        hits = hits[:planner.k]                 # the model sees the k nearest of them
    record["exemplars"] = [{"i": index, "sim": round(similarity, 3), **memory.describe_hit(index)} for index, similarity in hits]
    return hits
