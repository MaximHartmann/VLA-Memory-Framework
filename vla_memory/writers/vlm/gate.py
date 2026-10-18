"""The memory-vote gate: the exemplar store's own verdict on the current skill, which may settle a monitor call without
asking the model.

Each retrieved moment votes "done" when its stored skill is later in the task's skill order than the current skill, the
same rule that labels the examples shown to the model. A full vote with fewer than `gate_min_done` "done" votes settles
the call as "not done"; any "done" vote asks the model. The last skill of the plan, and the task's last skill, are always
asked: no stored moment is labelled later than the task's last skill, so its vote is never "done".
"""
from __future__ import annotations


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
