"""The memory policy's proxy extension: at every policy call, the most similar stored moment of an earlier episode goes to
the policy as a third picture and as a sentence in the prompt.

    --with memory_policy --memory-store DIR [--memory-mode normal|off|wrong] [--memory-seed N] [--memory-cache N]

before_policy runs after the proxy decided the prompt and before the observation goes to the policy. It computes the key of
the raw wrist frame (key.py), retrieves the nearest admissible entry (retrieve.py: same skill, same gripper state, release
entries only for a gripper closed on nothing), and sets

    observation/memory_image   the entry's wrist picture (240 x 320 x 3 uint8 in the reference store)
    prompt                     "<prompt>. memory: <words>" (prompt.py)

The policy server must run a policy trained on such inputs (the reference: openpi config pi05_red_blue_v2_memory, whose
input transform puts observation/memory_image into its third image slot). The modes: normal; off sends no memory at all
(the bare prompt, no picture key); wrong sends a random entry of ANOTHER skill, picture and sentence (seeded; a control).

Every call's retrieval is logged under "memory" in the proxy's log record (LOG_KEYS: entry, episode, step, source, skill,
width, similarity, rule, sentence, ms; rule is one of retrieve.RULES, "wrong", or why nothing was retrieved: no_skill,
no_frame, no_width); every record carries memory_mode. Mode off logs memory None.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ...proxy.extensions import Extension
from .key import FRAME_SHAPE, KEY_NAME, WristKey
from .pictures import Pictures
from .prompt import with_memory
from .query import finger_width, skill_of, wrist_frame
from .retrieve import neighbour
from .store import MomentStore

log = logging.getLogger("vla_memory.methods.memory_policy")
MODES = ("normal", "off", "wrong")
LOG_KEYS = ("entry", "episode", "step", "source", "skill", "width", "similarity", "rule", "sentence", "ms")
MEMORY_IMAGE = "observation/memory_image"
DEFAULT_WRIST_IMAGE_KEY = "observation/wrist_image"     # the proxy's default second --image-keys entry


class MemoryPolicyExtension(Extension):
    name = "memory"

    # ------------------------------------------------------------------ command line and construction
    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group("the memory policy (--with memory_policy): the nearest stored moment as a third "
                                          "picture and a sentence in the prompt")
        group.add_argument("--memory-store", required=True, metavar="DIR",
                           help="the store directory (entries.jsonl, keys_fingerprint.npy, meta.json, wrist/*.png)")
        group.add_argument("--memory-mode", choices=list(MODES), default="normal",
                           help="normal: the retrieved entry (default); off: no memory at all (bare prompt, no picture); "
                                "wrong: a random entry of another skill")
        group.add_argument("--memory-seed", type=int, default=0, metavar="N",
                           help="seed of the random choice in mode wrong (default 0)")
        group.add_argument("--memory-cache", type=int, default=1024, metavar="N",
                           help="decoded wrist pictures kept in memory, 230 KB each (default 1024; 0 decodes at every call)")

    @classmethod
    def build(cls, args, task, built):
        """The extension from the parsed options; a store that cannot be read is a usage error (ValueError)."""
        image_keys = getattr(args, "image_keys", None) or ("observation/image", DEFAULT_WRIST_IMAGE_KEY)
        wrist_image_key = image_keys[1] if len(image_keys) > 1 else None
        try:
            return cls(args.memory_store, task, mode=args.memory_mode, seed=args.memory_seed, cache=args.memory_cache,
                       wrist_image_key=wrist_image_key)
        except (OSError, KeyError) as error:
            raise ValueError(f"memory store {args.memory_store!r}: {error}") from error

    def __init__(self, path, task, mode: str = "normal", seed: int = 0, cache: int = 1024,
                 wrist_image_key: str | None = DEFAULT_WRIST_IMAGE_KEY):
        """path: the store directory; task: the task the store was built for (its skills, its fingerprint); mode: one of
        MODES; seed: of mode wrong; cache: decoded pictures kept; wrist_image_key: the observation key of the policy's
        wrist image, the frame used when the environment does not send memory/wrist_raw."""
        if mode not in MODES:
            raise ValueError(f"memory mode {mode!r}: one of {MODES}")
        started = time.perf_counter()
        self.store = MomentStore.load(path)
        self.load_s = time.perf_counter() - started
        self.wrist_key = WristKey(task)
        self._check_key_size()
        self.skills = tuple(task.skills)
        self.mode = mode
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.pictures = Pictures(self.store.pictures, cache)
        self.wrist_image_key = wrist_image_key
        self.calls = 0
        self.retrieved = 0
        self._warned: set = set()
        log.info("memory policy: %d entries from %s loaded in %.2f s; mode %s", len(self.store), path, self.load_s, mode)

    def _check_key_size(self) -> None:
        """The task's key must have as many values as the store's keys (the store was built with the same task file)."""
        size = len(self.wrist_key.key_of(np.zeros(FRAME_SHAPE, np.uint8)))
        stored = self.store.keys.shape[1]
        if len(self.store) and size != stored:
            raise ValueError(f"the task's key has {size} values, the store's keys {stored}: the store was built with "
                             f"another task file")

    # ------------------------------------------------------------------ what the proxy is told
    def config(self) -> dict:
        return {"store": str(self.store.path), "entries": len(self.store), "mode": self.mode, "key": KEY_NAME,
                "seed": self.seed, "cache": self.pictures.capacity}

    def metadata(self) -> dict:
        return {"memory_policy": self.config()}

    def log_fields(self) -> dict:
        return {"memory_mode": self.mode}

    # ------------------------------------------------------------------ every policy call
    def before_policy(self, obs: dict, ctrl: dict, prompt: str, planner) -> dict:
        """Retrieve for this call and put the memory into the observation; the log record goes under "memory"."""
        started = time.perf_counter()
        self.calls += 1
        if self.mode == "off":
            return {"memory": None}
        record = self.retrieve(obs, ctrl, prompt, planner)
        if record["entry"] is not None:
            obs[MEMORY_IMAGE] = self.pictures[record["entry"]]
            obs["prompt"] = with_memory(prompt, record["sentence"])
            self.retrieved += 1
        record["ms"] = 1000 * (time.perf_counter() - started)
        return {"memory": record}

    def retrieve(self, obs: dict, ctrl: dict, prompt: str, planner) -> dict:
        """The retrieval of one call as a log record (LOG_KEYS): the entry and its provenance, or why there is none."""
        record = dict.fromkeys(LOG_KEYS)
        skill = skill_of(prompt, planner, self.skills)
        frame, fallback = wrist_frame(obs, ctrl, self.wrist_image_key)
        width = finger_width(ctrl)
        record.update(skill=skill, width=width)
        reason = self._missing(skill, frame, fallback, width)
        if reason is not None:
            record["rule"] = reason
            return record
        key = self.wrist_key.key_of(frame)
        if self.mode == "wrong":
            index, similarity, rule = self._wrong_entry(key, skill)
        else:
            index, similarity, rule = neighbour(self.store, key, skill, width)
        record["rule"] = rule
        if index >= 0:
            record.update(self._entry_fields(index, similarity))
        return record

    def _entry_fields(self, index: int, similarity: float) -> dict:
        """The log fields of one store entry."""
        return {"entry": int(index), "episode": str(self.store.episode[index]), "step": int(self.store.step[index]),
                "source": str(self.store.source[index]), "similarity": float(similarity),
                "sentence": str(self.store.sentence[index])}

    def _wrong_entry(self, key: np.ndarray, skill: str) -> tuple:
        """Mode wrong: a random entry of another skill, with its similarity to the call's key."""
        others = np.flatnonzero(self.store.skill != skill)
        if not len(others):
            return -1, float("nan"), "wrong"
        index = int(self.rng.choice(others))
        return index, float(self.store.keys[index] @ np.asarray(key, np.float32)), "wrong"

    def _missing(self, skill, frame, fallback: bool, width) -> str | None:
        """Why nothing can be retrieved at this call (None: everything is there). Each problem is logged once."""
        if skill is None:
            self._warn_once("skill", "the prompt is not a skill and the monitor has no current skill: no memory")
            return "no_skill"
        if frame is None:
            self._warn_once("frame", "neither memory/wrist_raw nor the policy's wrist image is sent: no memory")
            return "no_frame"
        if fallback:
            self._warn_once("fallback", f"memory/wrist_raw is not sent: the key is computed on {self.wrist_image_key} "
                                        f"(the policy's input, not the raw frame the store was built on)")
        if tuple(frame.shape) != FRAME_SHAPE:
            self._warn_once("shape", f"wrist frame {tuple(frame.shape)}: the reference store's keys were computed on "
                                     f"{FRAME_SHAPE} frames")
        if width is None:
            self._warn_once("width", "no finger width (memory/proprio_steps or memory/finger_width): no memory")
            return "no_width"
        return None

    def _warn_once(self, topic: str, message: str) -> None:
        if topic not in self._warned:
            log.warning("memory policy: %s", message)
            self._warned.add(topic)

    # ------------------------------------------------------------------ the end of the run
    def close(self) -> None:
        log.info("memory policy: %d calls, %d with a memory; %d pictures decoded, %d cached (%.0f MB)", self.calls,
                 self.retrieved, self.pictures.decoded, self.pictures.cached, self.pictures.cache_bytes() / 1e6)
