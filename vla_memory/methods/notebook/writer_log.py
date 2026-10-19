"""A failed episode as the distiller reads it: the planner's own decision log, shortened to the switches and the end.

    render_log(task, records, outcome, complaint, log_lines)
        the instruction, the plan, the task's completion criteria, the outcome and the user's remark, then one line per
        monitor query; a long log keeps every switch with the query before it and the end of the episode, up to about
        log_lines lines, and merges runs of identical queries into one line with the ranges of the sensor readings
"""
from __future__ import annotations


def query_key(record: dict) -> tuple:
    """What makes two queries the same for the log: the skill, the verdict and what the monitor saw."""
    answer = record.get("vlm") or {}
    verdict = "no answer" if record.get("error") else ("complete" if answer.get("current_skill_done") else "not complete")
    if record.get("advanced"):
        verdict += " -> plan finished" if record.get("plan_complete") else f" -> next: {record.get('skill_after')}"
    if record.get("recovery_inserted"):
        verdict += f" -> inserted: {record['recovery_inserted']}"
    seen = ", ".join(f"{name} {value}" for name, value in answer.items() if name not in ("current_skill_done", "reason"))
    return record.get("skill"), verdict, seen


def sensor_ranges(records: list[dict]) -> str:
    """The sensor readings of merged queries: one value, or the range when they differ in two decimals."""
    values: dict[str, list] = {}
    for record in records:
        for name, value in (record.get("proprio") or {}).items():
            if isinstance(value, (int, float)) and value == value:
                values.setdefault(name, []).append(value)
    parts = []
    for name, readings in values.items():
        low, high = f"{min(readings):.2f}", f"{max(readings):.2f}"
        parts.append(f"{name} {low}" if low == high else f"{name} {low}-{high}")
    return ", ".join(parts)


def query_line(records: list[dict]) -> str:
    """One line for a run of identical queries (or one query)."""
    skill, verdict, seen = query_key(records[0])
    sensors = sensor_ranges(records)
    if len(records) == 1:
        steps = f"step {records[0].get('step')}"
    else:
        steps = f"steps {records[0].get('step')}-{records[-1].get('step')} ({len(records)} queries)"
    line = f"{steps}: {skill} | {verdict}"
    if sensors:
        line += f" | {sensors}"
    if seen:
        line += f" | {seen}"
    return line


def kept_queries(queries: list[dict], log_lines: int) -> set:
    """The queries the log shows: every switch with the query before it, and the end of the episode."""
    keep = {kept for index, record in enumerate(queries) if record.get("advanced") or record.get("recovery_inserted")
            for kept in (index - 1, index) if kept >= 0}
    keep |= set(range(max(0, len(queries) - max(3, log_lines - len(keep))), len(queries)))
    return keep


def query_runs(queries: list[dict], keep: set) -> list[list]:
    """Runs of consecutive kept queries with the same skill, verdict and answer, none of them a switch:
    [first, last, key, queries] each."""
    runs: list[list] = []
    for index in sorted(keep):
        query = queries[index]
        key = None if query.get("advanced") or query.get("recovery_inserted") else query_key(query)
        if runs and key is not None and runs[-1][2] == key and runs[-1][1] == index - 1:
            runs[-1][1] = index
            runs[-1][3].append(query)
        else:
            runs.append([index, index, key, [query]])
    return runs


def header(task, records: list, queries: int, outcome: str, complaint: str) -> list[str]:
    """The lines before the queries: instruction, plan, criteria, outcome, remark."""
    plan = next((record for record in records if record.get("event") == "plan"), {})
    criteria = "\n".join(f'  {verb}: {rule.format(object="X")}' for verb, rule in task.criteria.items())
    lines = [f'Instruction: "{plan.get("instruction", "")}"',
             "Plan: " + "; ".join(f"{number}. {skill}" for number, skill in enumerate(plan.get("plan") or [], 1)),
             f"Completion criteria in force (X = the skill's object):\n{criteria}",
             f"Outcome: {outcome or 'failed'}"]
    if complaint:
        lines.append(f'User\'s remark: "{complaint}"')
    lines.append(f"Monitor queries ({queries}): step, current skill, verdict, sensor readings (ranges over merged queries), "
                 "what the monitor saw")
    return lines


def render_log(task, records: list, outcome: str = "", complaint: str = "", log_lines: int = 40) -> str:
    """The failed episode as the distiller reads it (module docstring)."""
    queries = [record for record in records if record.get("event") == "monitor" and record.get("queried")]
    lines = header(task, records, len(queries), outcome, complaint)
    last = -1
    for first, end, _, run in query_runs(queries, kept_queries(queries, log_lines)):
        if first > last + 1:
            lines.append(f"  ... {first - last - 1} queries omitted")
        lines.append("  " + query_line(run))
        last = end
    return "\n".join(lines)
