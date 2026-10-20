"""The session store as the VLM writer's long-term memory, loaded by the proxy with

    --with session_store --store-dir DIR [--store-key colour|dinov2|siglip] [--store-filter none] [--store-cap N] ...

It provides the planner's memory (in place of --memory), sets the query context at every call (instruction, plan, session,
episode, step), records every monitored call, and writes the finished episode when the next one starts or the proxy stops
(the store_update record goes to the planner's log). The database needs lancedb and pyarrow, the keys dinov2 and siglip also
torch and transformers; a missing package is a usage error naming the extra to install. See
docs/methods/session_store/README.md.
"""
from __future__ import annotations

import argparse
import logging

from ...proxy.extensions import Extension
from .keys import require_encoders
from .online import LABELLERS, EpisodeRecorder
from .options import DEFAULT_EVICT, DEFAULT_FILTER, DEFAULT_KEY, EVICT, FILTERS, KEY_NAMES

log = logging.getLogger("vla_memory.methods.session_store")


def import_database():
    """The module of the database store, which needs lancedb and pyarrow: a missing package is a usage error that names the
    extra to install."""
    try:
        from . import store
    except ImportError as error:
        raise ValueError(f"the session store needs lancedb and pyarrow ({error}): install them with the extra "
                         f"(pip install -e '.[session_store]' in the framework checkout)") from error
    return store


class SessionStoreExtension(Extension):
    """The session store in the VLM writer's planner and the proxy: the database (store.SessionStore) and the recorder of the
    robot's own episodes (online.EpisodeRecorder; None with --no-store-write, which only reads the database)."""

    name = "session_store"

    # ------------------------------------------------------------------ command line and construction
    @classmethod
    def add_arguments(cls, parser):
        """The --store-* options."""
        group = parser.add_argument_group("the session store (--with session_store): the VLM writer's memory on a database")
        group.add_argument("--store-dir", required=True, metavar="DIR",
                           help="the database directory (created when missing, else resumed)")
        group.add_argument("--store-import", default=None, metavar="FILE",
                           help="an exemplar store .npz imported as demonstrations into a new database")
        group.add_argument("--store-import-instruction", default=None, metavar="TEXT",
                           help="the instruction recorded with the imported demonstrations (default: the task file's first "
                                "instruction example)")
        group.add_argument("--store-key", choices=KEY_NAMES, default=DEFAULT_KEY,
                           help=f"the key that retrieves (default {DEFAULT_KEY}; dinov2 and siglip need torch and transformers)")
        group.add_argument("--store-keys", nargs="+", choices=KEY_NAMES, default=None, metavar="KEY",
                           help="further keys a new database stores with every moment, besides the colour key (which decides "
                                "the near-duplicates of an episode) and the retrieval key; an existing database keeps its own")
        group.add_argument("--store-filter", choices=FILTERS, default=DEFAULT_FILTER,
                           help=f"payload filter of the retrieval (default {DEFAULT_FILTER})")
        group.add_argument("--store-session", default=None, metavar="NAME",
                           help="the session written with new moments (default: the start time)")
        group.add_argument("--store-write", action=argparse.BooleanOptionalAction, default=True,
                           help="add the robot's own episodes (default on)")
        group.add_argument("--store-labeller", choices=sorted(LABELLERS), default="rules",
                           help="who labels the robot's calls (default rules)")
        group.add_argument("--store-cap", type=int, default=0, metavar="N", help="most moments kept (0 = unbounded)")
        group.add_argument("--store-evict", choices=EVICT, default=DEFAULT_EVICT,
                           help=f"eviction rule above the cap (default {DEFAULT_EVICT})")
        group.add_argument("--store-device", default="cpu", metavar="DEVICE",
                           help="where the encoders of the learned keys run (cpu or cuda; default cpu)")
        group.add_argument("--store-threads", type=int, default=8, metavar="N", help="CPU threads of the encoders (default 8)")

    @classmethod
    def build(cls, args, task, built):
        """The extension from the parsed options; an unusable combination or a missing package is a usage error (ValueError)."""
        if getattr(args, "writer", "vlm") != "vlm":
            raise ValueError("the session store is the VLM writer's memory: it needs --writer vlm")
        instruction = args.store_import_instruction or next(iter(task.instruction_examples), None)
        if args.store_import and instruction is None:
            raise ValueError("--store-import needs --store-import-instruction: the task file has no instruction examples")
        database = import_database()
        require_encoders(database.keys_of_new_database(args.store_key, args.store_keys))      # before a database is created
        store = database.SessionStore(args.store_dir, task, key=args.store_key, filter=args.store_filter, cap=args.store_cap,
                                      evict=args.store_evict, session=args.store_session, keys=args.store_keys,
                                      device=args.store_device, threads=args.store_threads)
        if args.store_write:
            try:
                require_encoders(store.keys)       # an existing database may hold learned keys that new moments need as well
            except ValueError:
                store.close()
                raise
        if args.store_import and not len(store):
            added = store.import_exemplar_store(args.store_import, instruction)
            log.info("session store: imported %d demonstrations from %s", added, args.store_import)
        log.info("session store %s: %d moments, keys %s, retrieval key %s, filter %s, cap %d (%s), session %s, write %s",
                 args.store_dir, len(store), ", ".join(store.keys), args.store_key, args.store_filter, args.store_cap,
                 args.store_evict, store.session, args.store_write)
        return cls(store, EpisodeRecorder(task, labeller=args.store_labeller) if args.store_write else None)

    def __init__(self, store, recorder: EpisodeRecorder | None):
        self.store = store
        self.recorder = recorder
        self.episode = None      # the episode the planner is in, set at every call

    def config(self) -> dict:
        """The store's settings, for the plan record."""
        return {"dir": str(self.store.path), "key": self.store.key, "keys": list(self.store.keys), "filter": self.store.filter,
                "cap": self.store.cap, "evict": self.store.evict_rule, "session": self.store.session,
                "write": self.recorder is not None}

    def metadata(self) -> dict:
        """The database directory under "session_store"."""
        return {"session_store": str(self.store.path)}

    def log_fields(self) -> dict:
        """The number of stored moments in every proxy log record."""
        return {"memory_entries": len(self.store)}

    # ------------------------------------------------------------------ the planner's memory and the episode
    def memory(self):
        """The store, as the planner's long-term memory."""
        return self.store

    def plan_request(self, instruction: str):
        """The new episode's instruction becomes the retrieval's context; the plan request itself is not changed."""
        self.store.set_context(instruction=instruction)
        return "", {}

    def before_decision(self, control_keys, images, planner):
        """The episode, the plan and the step of this call become the retrieval's context."""
        self.episode = planner.episode
        self.store.set_context(plan=list(planner.plan), episode=planner.episode, step=control_keys.get("step"))

    def after_decision(self, control_keys, images, record, planner):
        """Keep the call for the episode's labelling."""
        if self.recorder is not None:
            self.recorder.record(control_keys, images, record)

    def end_episode(self, writer):
        """Label the episode's calls and write them into the store; the store_update record goes to the planner's log."""
        if self.recorder is None or self.episode is None:
            self.store.flush_usage()
            return None
        record = self.recorder.finish(self.store, self.episode, dict(self.store.context))
        if record is not None:
            record["bytes"] = self.store.size_bytes()
            log.info("session store after %s: %d of %d calls added, %d removed, %d moments, %.0f ms", record["episode"],
                     record["added"], record["calls"], record["removed"], record["size"], record["ms"]["total"])
        return record

    def close(self):
        """The proxy stops: close the store (the last episode was taken by end_episode)."""
        self.store.close()
