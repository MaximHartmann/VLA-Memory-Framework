"""A failed episode as the distiller reads it: the planner's own decision log, shortened to the switches and the end.

    render_log(task, records, outcome, complaint, log_lines)
        the instruction, the plan, the task's completion criteria, the outcome and the user's remark, then one line per
        monitor query; a long log keeps every switch with the query before it and the end of the episode, up to about
        log_lines lines, and merges runs of identical queries into one line with the ranges of the sensor readings
"""
from __future__ import annotations


def query_key(r: dict) -> tuple:
    """What makes two queries the same for the log: the skill, the verdict and what the monitor saw."""
    v = r.get("vlm") or {}
    verdict = "no answer" if r.get("error") else ("complete" if v.get("current_skill_done") else "not complete")
    if r.get("advanced"):
        verdict += " -> plan finished" if r.get("plan_complete") else f" -> next: {r.get('skill_after')}"
    if r.get("recovery_inserted"):
        verdict += f" -> inserted: {r['recovery_inserted']}"
    seen = ", ".join(f"{k} {x}" for k, x in v.items() if k not in ("current_skill_done", "reason"))
    return r.get("skill"), verdict, seen


def sensor_ranges(rs: list[dict]) -> str:
    """The sensor readings of merged queries: one value, or the range when they differ in two decimals."""
    values: dict[str, list] = {}
    for r in rs:
        for k, x in (r.get("proprio") or {}).items():
            if isinstance(x, (int, float)) and x == x:
                values.setdefault(k, []).append(x)
    parts = []
    for k, x in values.items():
        low, high = f"{min(x):.2f}", f"{max(x):.2f}"
        parts.append(f"{k} {low}" if low == high else f"{k} {low}-{high}")
    return ", ".join(parts)


def query_line(rs: list[dict]) -> str:
    """One line for a run of identical queries (or one query)."""
    skill, verdict, seen = query_key(rs[0])
    sensors = sensor_ranges(rs)
    if len(rs) == 1:
        steps = f"step {rs[0].get('step')}"
    else:
        steps = f"steps {rs[0].get('step')}-{rs[-1].get('step')} ({len(rs)} queries)"
    line = f"{steps}: {skill} | {verdict}"
    if sensors:
        line += f" | {sensors}"
    if seen:
        line += f" | {seen}"
    return line


def kept_queries(q: list[dict], log_lines: int) -> set:
    """The queries the log shows: every switch with the query before it, and the end of the episode."""
    keep = {j for i, r in enumerate(q) if r.get("advanced") or r.get("recovery_inserted") for j in (i - 1, i) if j >= 0}
    keep |= set(range(max(0, len(q) - max(3, log_lines - len(keep))), len(q)))
    return keep


def query_runs(q: list[dict], keep: set) -> list[list]:
    """Runs of consecutive kept queries with the same skill, verdict and answer, none of them a switch:
    [first, last, key, queries] each."""
    runs: list[list] = []
    for i in sorted(keep):
        key = None if q[i].get("advanced") or q[i].get("recovery_inserted") else query_key(q[i])
        if runs and key is not None and runs[-1][2] == key and runs[-1][1] == i - 1:
            runs[-1][1] = i
            runs[-1][3].append(q[i])
        else:
            runs.append([i, i, key, [q[i]]])
    return runs


def header(task, records: list, queries: int, outcome: str, complaint: str) -> list[str]:
    """The lines before the queries: instruction, plan, criteria, outcome, remark."""
    plan = next((r for r in records if r.get("event") == "plan"), {})
    criteria = "\n".join(f'  {v}: {c.format(object="X")}' for v, c in task.criteria.items())
    lines = [f'Instruction: "{plan.get("instruction", "")}"',
             "Plan: " + "; ".join(f"{i + 1}. {s}" for i, s in enumerate(plan.get("plan") or [])),
             f"Completion criteria in force (X = the skill's object):\n{criteria}",
             f"Outcome: {outcome or 'failed'}"]
    if complaint:
        lines.append(f'User\'s remark: "{complaint}"')
    lines.append(f"Monitor queries ({queries}): step, current skill, verdict, sensor readings (ranges over merged queries), "
                 "what the monitor saw")
    return lines


def render_log(task, records: list, outcome: str = "", complaint: str = "", log_lines: int = 40) -> str:
    """The failed episode as the distiller reads it (module docstring)."""
    q = [r for r in records if r.get("event") == "monitor" and r.get("queried")]
    lines = header(task, records, len(q), outcome, complaint)
    last = -1
    for first, end, _, rs in query_runs(q, kept_queries(q, log_lines)):
        if first > last + 1:
            lines.append(f"  ... {first - last - 1} queries omitted")
        lines.append("  " + query_line(rs))
        last = end
    return "\n".join(lines)
