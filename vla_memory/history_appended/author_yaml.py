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

from ..task import TaskSpec
from .author_rules_spec import check_rules, with_rules
from .rules import parse_rules

_PLAIN = re.compile(r"[A-Za-z][A-Za-z0-9 _-]*")   # a YAML key that needs no quotes


def _key(text: str) -> str:
    """A verb as a YAML key: plain if it can be, else quoted."""
    if _PLAIN.fullmatch(text) and not text.endswith(" "):
        return text
    return json.dumps(text)


def _num(value: float) -> str:
    """A threshold as the shortest text with up to six decimals."""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _indent_of(line: str) -> str:
    return re.match(r"\s*", line).group(0)


def rules_yaml_lines(rules: dict, indent: str = "    ") -> list[str]:
    """history.rules as flow-style YAML lines, one per verb (as in the reference task file)."""
    width = max(len(_key(verb)) for verb in rules) + 1
    out = []
    for verb, spec in rules.items():
        items = [f"trigger: {spec['trigger']}"]
        for name, value in spec.items():
            if name != "trigger" and value is not None:
                items.append(f"{name}: {_num(value)}")
        out.append(f"{indent}{(_key(verb) + ':').ljust(width)} {{{', '.join(items)}}}")
    return out


def write_task_yaml(src, dst, rules: dict, comment: Sequence[str] | None = None) -> pathlib.Path:
    """A copy of the task file `src` with history.rules replaced by `rules`; every other line is copied unchanged. With
    `comment`, the comment lines directly above `rules:` are replaced by these lines (without '#'). The copy is loaded back
    and checked: same task apart from history.rules, and the rules writer accepts it."""
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    check_rules(TaskSpec.load(src), rules)
    lines = src.read_text().splitlines()
    history, history_end = _find_history_section(lines, src)
    rules_line = _find_or_insert_rules_line(lines, history, history_end)
    _replace_block(lines, history, rules_line, rules, comment)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_checked_copy(src, dst, lines, rules)
    return dst


def _find_history_section(lines: list[str], src: pathlib.Path) -> tuple[int, int]:
    """The line of the top-level `history:` key, and the line of the next top-level key (or the end of the file)."""
    start = None
    for i, line in enumerate(lines):
        if re.match(r"history:\s*(#.*)?$", line):
            start = i
            break
    if start is None:
        raise ValueError(f"{src}: no top-level history: section")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and not lines[i][0].isspace() and not lines[i].lstrip().startswith("#"):
            end = i
            break
    return start, end


def _find_or_insert_rules_line(lines: list[str], history: int, history_end: int) -> int:
    """The line of `rules:` inside the history section. Without one, `rules:` is appended at the end of the section, at
    the indentation of its keys."""
    for i in range(history + 1, history_end):
        if re.match(r"\s+rules:\s*(#.*)?$", lines[i]):
            return i
    body = [i for i in range(history + 1, history_end) if lines[i].strip() and not lines[i].lstrip().startswith("#")]
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
    new = [lines[rules_line]] + rules_yaml_lines(rules, indent + "  ")
    top = rules_line
    if comment is not None:
        while top > history + 1 and re.match(rf"{indent}#", lines[top - 1]):
            top -= 1
        new = [f"{indent}# {text}".rstrip() for text in comment] + new
    lines[top:stop] = new


def _write_checked_copy(src: pathlib.Path, dst: pathlib.Path, lines: list[str], rules: dict) -> None:
    """Write the lines to a temporary file beside dst and let it replace dst only if it loads back as the task with the new
    rules."""
    tmp = dst.with_name(f".{dst.stem}.tmp{dst.suffix}")   # the same suffix: TaskSpec.load reads by suffix
    tmp.write_text("\n".join(lines) + "\n")
    try:
        if not _check_written_copy(src, tmp, rules):
            raise RuntimeError(f"{dst}: the written copy does not reproduce the task with the new rules")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def _check_written_copy(src: pathlib.Path, written: pathlib.Path, rules: dict) -> bool:
    """Whether the written file is the same task as src apart from history.rules, and its rules parse to the given ones."""
    original = dataclasses.asdict(TaskSpec.load(src))
    copied = dataclasses.asdict(TaskSpec.load(written))
    got = copied["history"].pop("rules", None)
    original["history"].pop("rules", None)
    if original != copied:
        return False
    source = TaskSpec.load(src)
    return parse_rules(with_rules(source, got or {})) == parse_rules(with_rules(source, rules))
