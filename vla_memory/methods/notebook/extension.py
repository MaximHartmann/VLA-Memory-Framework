"""The notebook as an extension of the planner loop: --with notebook --profile FILE.

The planner re-reads the notebook at the start of every episode. Its plan-scope rules go into the plan request (except
those held back because the instruction names the task's objects itself: the instruction has precedence, logged as
`profile_plan_held`), its skill-scope rules after that skill's completion criterion, and its planner-scope parameters
override votes_to_advance, call_every and min_calls_per_skill. --profile-off runs the same command line without the
notebook (the baseline).
"""
from __future__ import annotations

import logging

from ...proxy.extensions import Extension
from .store import ProfileStore

log = logging.getLogger("vla_memory.methods.notebook")


class NotebookExtension(Extension):
    """The notebook in the planner loop: its rules in the plan request and the questions, its parameters as overrides.
    store None: the notebook is switched off (--profile-off)."""

    name = "notebook"

    @classmethod
    def add_arguments(cls, parser):
        """--profile and --profile-off."""
        group = parser.add_argument_group("the preference notebook about the user (--with notebook)")
        group.add_argument("--profile", default=None, help="user preference memory (notebook), a JSON file")
        group.add_argument("--profile-off", action="store_true",
                           help="ignore --profile (baseline without the notebook, same command line)")

    @classmethod
    def build(cls, args, task, built):
        """The extension with the notebook of --profile; it needs the VLM writer, which reads the notebook."""
        if args.writer != "vlm":
            raise ValueError("the notebook is read by the VLM planner: it needs --writer vlm")
        if not args.profile:
            raise ValueError("the notebook needs --profile FILE")
        if args.profile_off:
            return cls(None, task, path=None)
        store = ProfileStore(args.profile)
        log.info("user profile (notebook): %d active entries from %s", len(store), args.profile)
        return cls(store, task, path=args.profile)

    def __init__(self, store: ProfileStore | None, task, path=None):
        self.store = store                 # None: the notebook is switched off (--profile-off)
        self.task = task
        self.path = path
        self._overrides: dict = {}

    def parameters(self):
        """Re-read the notebook and return its planner-scope overrides (called before plan_request)."""
        if self.store is None:
            return {}
        self.store.reload()
        self._overrides = self.store.planner_overrides()
        return dict(self._overrides)

    def plan_request(self, instruction):
        """The plan rules for this instruction as text, and the record fields: the rules applied, held back and the
        parameters in force."""
        if self.store is None:
            return "", {}
        plan_rules, held = self.store.plan_rules_for(instruction, self.task.objects)
        planner_rules = self.store.planner_rules() if self._overrides else []
        fields = {"profile_plan_rules": [entry.id for entry in plan_rules]}
        if held:
            fields["profile_plan_held"] = held
        fields["profile_planner_rules"] = [entry.id for entry in planner_rules]
        fields["profile_params"] = self._overrides
        self.store.note_applied(plan_rules + planner_rules)
        return self.store.plan_text(plan_rules), fields

    def criterion_text(self, skill):
        """The skill's rules after its completion criterion, and their ids for the record."""
        if self.store is None:
            return "", {}
        rules = self.store.rules_for_skill(skill)
        if not rules:
            return "", {}
        self.store.note_applied(rules)
        return "\n" + self.store.rules_text(rules), {"profile_rules": [entry.id for entry in rules]}

    def metadata(self):
        """The notebook file under "profile"."""
        return {"profile": self.path}
