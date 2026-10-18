"""Preference and correction memory: a notebook about the user, read by the planner before every episode.

Entries are text rules with a scope: `plan` (task order, what to skip), `skill:<skill string>` (an extra completion rule for
that skill) or `planner` (verification strictness as parameter overrides). They come from the user's own words or are
distilled from failed episodes, are never similarity-matched (scope decides applicability), carry a vote count that removes
rules that keep harming, and are superseded rather than deleted when the user changes their mind. The planner reads the
profile at the start of each episode; nothing in the policy or the planning model changes. Reading is a file lookup and
costs no latency. See docs/methods/notebook/README.md.

  extension.NotebookExtension   the planner loop's extension: --with notebook --profile FILE [--profile-off]
  store.ProfileStore            the notebook: load/save a JSON file, add, supersede, up/down-vote, query by scope, render
                                prompt text; its own clock and session start (entries that hold for the day or the session
                                end there), and plan rules held back when the instruction names the task's objects itself
  entry.Entry                   one rule with its scope, condition, source, evidence, importance, validity and expiry
  writer.PreferenceWriter       writes the notebook with the planning model: a user message -> entries, a failed episode's
                                planner log -> a correction, each merged Mem0-style (ADD / UPDATE / DELETE / NOOP); votes
                                after episodes. Its parts: writer_schema (the answer schemas and texts), writer_log (the
                                failed episode as the distiller reads it)
  gold.GoldWriter               the same, answered from annotations (oracle writer, dry runs)
  session                       scripted sessions of user messages and episodes: run_session, schedule; session_score: the
                                records' applied entries, the following checks, the summary and the writer's scores
"""
from .entry import STRICTNESS_PRESETS, Entry
from .extension import NotebookExtension
from .gold import GoldWriter
from .store import ProfileStore
from .writer import PreferenceWriter

__all__ = ["Entry", "ProfileStore", "STRICTNESS_PRESETS", "PreferenceWriter", "GoldWriter", "NotebookExtension"]
