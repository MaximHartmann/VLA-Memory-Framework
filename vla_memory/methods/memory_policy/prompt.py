"""The policy's prompt with a memory: the skill prompt, the marker ". memory:" and the words of the retrieved sentence.

    with_memory("pick up the red block", "memory: release at step 1, up 2 cm")
        -> "pick up the red block. memory: release at step 1, up 2 cm"
    strip_memory("pick up the red block. memory: release at step 1, up 2 cm")
        -> "pick up the red block"

The training prompts (training/relabel_memory.py) have exactly this format, and strip_memory cuts a prompt back to the skill
at the same marker, which is how a policy trained on the same frames without the memory (the control) reads them. A skill
prompt never contains the marker.
"""
from __future__ import annotations

from .sentence import PREFIX

MEMORY_MARKER = ". memory:"          # separates the skill prompt from the memory's words
SENTENCE_START = PREFIX.strip()      # "memory:", how every memory sentence begins


def words_of(sentence: str) -> str:
    """The words of a memory sentence: 'memory: up 2 cm' -> 'up 2 cm'."""
    return sentence.strip().removeprefix(SENTENCE_START).strip()


def with_memory(prompt: str, sentence: str) -> str:
    """The prompt with the memory: '<prompt>. memory: <words>'."""
    return prompt + MEMORY_MARKER + " " + words_of(sentence)


def strip_memory(prompt: str) -> str:
    """The prompt without its memory: cut at the first marker (a prompt without one comes back unchanged)."""
    return prompt.split(MEMORY_MARKER, 1)[0]
