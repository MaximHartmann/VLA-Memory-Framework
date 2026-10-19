"""Writing authored rules into a copy of the task file: history.rules as flow-style YAML lines, and the copy with only the
rules block (and, optionally, the comment above it) replaced. Every other line is copied unchanged, and the copy is loaded
back and checked before it replaces the destination.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from typing import Sequence

from ...task import TaskSpec
from .author_rules_spec import check_rules, with_rules
from .rules import parse_rules

_PLAIN_KEY = re.compile(r"[A-Za-z][A-Za-z0-9 _-]*")   # a YAML key that needs no quotes


def _yaml_key(text: str) -> str:
    """A verb as a YAML key: plain if it can be, else quoted."""
    if _PLAIN_KEY.fullmatch(text) and not text.endswith(" "):
        return text
    return json.dumps(text)


def _number_text(value: float) -> str:
    """A threshold as the shortest text with up to six decimals."""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _indent_of(line: str) -> str:
    """The leading whitespace of a line."""
    return re.match(r"\s*", line).group(0)


def rules_yaml_lines(rules: dict, indent: str = "    ") -> list[str]:
    """history.rules as flow-style YAML lines, one per verb (as in the reference task file)."""
    width = max(len(_yaml_key(verb)) for verb in rules) + 1
    lines = []
    for verb, spec in rules.items():
        items = [f"trigger: {spec['trigger']}"]
        for name, value in spec.items():
            if name != "trigger" and value is not None:
                items.append(f"{name}: {_number_text(value)}")
        lines.append(f"{indent}{(_yaml_key(verb) + ':').ljust(width)} {{{', '.join(items)}}}")
    return lines


def write_task_yaml(source, destination, rules: dict, comment: Sequence[str] | None = None) -> pathlib.Path:
    """A copy of the task file `source` with history.rules replaced by `rules`; every other line is copied unchanged. With
    `comment`, the comment lines directly above `rules:` are replaced by these lines (without '#'). The copy is loaded back
    and checked: same task apart from history.rules, and the rules writer accepts it."""
    source = pathlib.Path(source)
    destination = pathlib.Path(destination)
    check_rules(TaskSpec.load(source), rules)
    lines = source.read_text().splitlines()
    history, history_end = _find_history_section(lines, source)
    rules_line = _find_or_insert_rules_line(lines, history, history_end)
    _replace_block(lines, history, rules_line, rules, comment)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_checked_copy(source, destination, lines, rules)
    return destination


def _find_history_section(lines: list[str], source: pathlib.Path) -> tuple[int, int]:
    """The line of the top-level `history:` key, and the line of the next top-level key (or the end of the file)."""
    start = None
    for index, line in enumerate(lines):
        if re.match(r"history:\s*(#.*)?$", line):
            start = index
            break
    if start is None:
        raise ValueError(f"{source}: no top-level history: section")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].strip() and not lines[index][0].isspace() and not lines[index].lstrip().startswith("#"):
            end = index
            break
    return start, end


def _find_or_insert_rules_line(lines: list[str], history: int, history_end: int) -> int:
    """The line of `rules:` inside the history section. Without one, `rules:` is appended at the end of the section, at
    the indentation of its keys."""
    for index in range(history + 1, history_end):
        if re.match(r"\s+rules:\s*(#.*)?$", lines[index]):
            return index
    body = [index for index in range(history + 1, history_end) if lines[index].strip() and not lines[index].lstrip().startswith("#")]
    indent = _indent_of(lines[body[0]]) if body else "  "
    last = body[-1] if body else history
    lines.insert(last + 1, f"{indent}rules:")
    return last + 1


def _rules_block_end(lines: list[str], rules_line: int) -> int:
    """The first line after the rules block (every line indented deeper than `rules:`); blank lines after the block stay."""
    indent = _indent_of(lines[rules_line])
    stop = rules_line + 1
    while stop < len(lines) and (not lines[stop].strip() or len(_indent_of(lines[stop])) > len(indent)):
        stop += 1
    while stop > rules_line + 1 and not lines[stop - 1].strip():
        stop -= 1
    return stop


def _replace_block(lines: list[str], history: int, rules_line: int, rules: dict, comment: Sequence[str] | None) -> None:
    """Replace the rules block in place; with a comment, also the comment lines directly above `rules:`."""
    indent = _indent_of(lines[rules_line])
    stop = _rules_block_end(lines, rules_line)
    replacement = [lines[rules_line]] + rules_yaml_lines(rules, indent + "  ")
    top = rules_line
    if comment is not None:
        while top > history + 1 and re.match(rf"{indent}#", lines[top - 1]):
            top -= 1
        replacement = [f"{indent}# {text}".rstrip() for text in comment] + replacement
    lines[top:stop] = replacement


def _write_checked_copy(source: pathlib.Path, destination: pathlib.Path, lines: list[str], rules: dict) -> None:
    """Write the lines to a temporary file beside the destination and let it replace the destination only if it loads back
    as the task with the new rules."""
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")   # the same suffix: TaskSpec.load reads by suffix
    temporary.write_text("\n".join(lines) + "\n")
    try:
        if not _check_written_copy(source, temporary, rules):
            raise RuntimeError(f"{destination}: the written copy does not reproduce the task with the new rules")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _check_written_copy(source: pathlib.Path, written: pathlib.Path, rules: dict) -> bool:
    """Whether the written file is the same task as the source apart from history.rules, and its rules parse to the given
    ones."""
    original = dataclasses.asdict(TaskSpec.load(source))
    copied = dataclasses.asdict(TaskSpec.load(written))
    written_rules = copied["history"].pop("rules", None)
    original["history"].pop("rules", None)
    if original != copied:
        return False
    source_task = TaskSpec.load(source)
    return parse_rules(with_rules(source_task, written_rules or {})) == parse_rules(with_rules(source_task, rules))
