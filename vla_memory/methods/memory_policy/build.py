"""The memory store and the training labels of the memory policy, built from recorded episodes given as arrays.

The store holds one entry per call moment of the training episodes: every HORIZON-th frame from the first one that still
has HORIZON actions after it, the moments at which the policy is called. An entry keeps

    provenance     the episode id, the step, the skill and the source label of the episode's dataset ("demo", "expert")
    gripper        the measured finger width (m, both fingers together) and the gripper state (gripper.py)
    sentence       the movement of the moment's own next HORIZON actions in words (sentence.py), and whether it begins
                   with a release (a gripper closed on nothing is shown only such entries)
    key, picture   the retrieval key of the raw wrist frame (key.py) and the frame itself as PNG

A frame of the training data is labelled with its nearest entry of ANOTHER episode (retrieve.neighbour with its own
episode excluded): the entry's wrist picture becomes the frame's memory picture and the entry's sentence is appended to
the frame's skill prompt (prompt.py). A fixed one frame in ten gets no memory, an all-zero picture and the bare skill
prompt, picked by a checksum of episode id and step: a call without a memory is then part of the training distribution,
and every run picks the same frames.

Only arrays come in and go out; reading and writing datasets is left to the caller (training/relabel_memory.py).
"""
from __future__ import annotations

import dataclasses
import datetime
import io
import zlib
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from .gripper import EMPTY_WIDTH, OPEN_WIDTH, STATES, gripper_state
from .key import KEY_NAME, WristKey
from .prompt import with_memory
from .retrieve import neighbour
from .sentence import CLOSED_COMMAND, HORIZON, is_release, sentence
from .store import MomentStore

NO_MEMORY_EVERY = 10         # one frame in ten gets no memory
NO_SEED = -1                 # the seed field of an entry whose episode has no simulator seed (a recorded dataset)


@dataclasses.dataclass
class Episode:
    """One recorded episode as arrays, frame by frame."""

    episode_id: str          # unique among the episodes of one build: the leave-one-episode-out rule compares it
    source: str              # the source label of the episode's dataset ("demo", "expert", ...), kept as provenance
    joints: np.ndarray       # (n, 7) the measured arm joints (rad)
    actions: np.ndarray      # (n, 8) the frame's action: 7 joint targets and the gripper command
    widths: np.ndarray       # (n,) the measured finger width (m, both fingers together)
    skills: Sequence[str]    # (n,) the skill the policy was given at the frame

    def __len__(self) -> int:
        return len(self.skills)


@dataclasses.dataclass
class StoreParts:
    """What a store is saved from (store.save_store), one item per entry in every list."""

    entries: list = dataclasses.field(default_factory=list)    # the fields of store.GIVEN_FIELDS
    keys: list = dataclasses.field(default_factory=list)       # the key of the wrist frame
    pngs: list = dataclasses.field(default_factory=list)       # the wrist frame as PNG bytes
    q_now: list = dataclasses.field(default_factory=list)      # the joints the sentence starts from (chunks.npz)
    chunks: list = dataclasses.field(default_factory=list)     # the HORIZON actions the sentence describes (chunks.npz)

    def extend(self, other: "StoreParts") -> None:
        """Append the entries of other (one more episode) to these."""
        for field in dataclasses.fields(self):
            getattr(self, field.name).extend(getattr(other, field.name))


@dataclasses.dataclass
class FrameLabel:
    """The memory of one training frame."""

    neighbour: int           # the store entry whose picture and sentence the frame gets; -1: no memory
    similarity: float        # the cosine similarity of the keys (NaN when no entry was admissible)
    rule: str                # the retrieval rule that chose the candidates (retrieve.RULES)
    prompt: str              # the skill prompt, with the entry's sentence when the frame has a memory


# ------------------------------------------------------------------------------------------------------- the store
def call_moments(frame_count: int) -> list[int]:
    """The policy-call moments of an episode: every HORIZON-th frame from 0 that has HORIZON actions after it."""
    return [step for step in range(0, frame_count, HORIZON) if step + HORIZON <= frame_count]


def store_entry(episode: Episode, step: int, skills: Sequence[str]) -> dict:
    """The fields of the store entry of one call moment; skills: the task's skill list (for skill_index)."""
    width = float(episode.widths[step])
    skill = episode.skills[step]
    text = sentence(episode.joints[step], episode.actions[step:step + HORIZON], width)
    return {"episode": episode.episode_id, "seed": NO_SEED, "step": int(step), "skill": skill,
            "skill_index": skills.index(skill) if skill in skills else -1, "source": episode.source,
            "width": width, "gripper": gripper_state(width), "release": is_release(text), "sentence": text}


def encode_png(frame: np.ndarray) -> bytes:
    """A uint8 RGB frame as PNG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame)).save(buffer, format="PNG")
    return buffer.getvalue()


def episode_entries(episode: Episode, wrist_frames: Mapping[int, np.ndarray], wrist_key: WristKey,
                    skills: Sequence[str]) -> StoreParts:
    """The store entries of one training episode. wrist_frames: the raw wrist frame of every call moment (by step);
    wrist_key: the task's key; skills: the task's skill list."""
    steps = call_moments(len(episode))
    frames = [wrist_frames[step] for step in steps]
    return StoreParts(entries=[store_entry(episode, step, skills) for step in steps],
                      keys=[wrist_key.key_of(frame) for frame in frames],
                      pngs=[encode_png(frame) for frame in frames],
                      q_now=[episode.joints[step] for step in steps],
                      chunks=[episode.actions[step:step + HORIZON] for step in steps])


def count_by(entries: list[dict], field: str, values: Sequence = ()) -> dict:
    """How many entries have each value of one field (the given values first, then the others as they appear)."""
    found = [entry[field] for entry in entries]
    return {value: found.count(value) for value in dict.fromkeys([*values, *found])}


def store_meta(entries: list[dict], keys: Sequence[np.ndarray]) -> dict:
    """The store's meta.json: what it holds, counted by source, skill and gripper state, and how it was built."""
    releases = [entry for entry in entries if entry["release"]]
    return {"key": KEY_NAME, "key_dim": int(len(keys[0])) if len(keys) else 0, "entries": len(entries),
            "episodes": len(count_by(entries, "episode")), "by_source": count_by(entries, "source"),
            "by_skill": count_by(entries, "skill"), "by_gripper": count_by(entries, "gripper", STATES),
            "release_by_skill": count_by(releases, "skill"),
            "retrieval": "retrieve.neighbour: cosine similarity of the keys, same skill and gripper state, release "
                         "entries only for a gripper closed on nothing, k = 1",
            "frame": "raw wrist frame, uint8 RGB", "horizon": HORIZON, "closed_command": CLOSED_COMMAND,
            "open_width": OPEN_WIDTH, "empty_width": EMPTY_WIDTH,
            "built": datetime.datetime.now().isoformat(timespec="seconds")}


# ---------------------------------------------------------------------------------------------------- the labelling
def no_memory(episode_id: str, step: int) -> bool:
    """The fixed one frame in ten without a memory: the checksum of "<episode id>:<step>"."""
    return zlib.crc32(f"{episode_id}:{step}".encode()) % NO_MEMORY_EVERY == 0


def label_frame(store: MomentStore, episode_id: str, step: int, skill: str, width: float,
                key: np.ndarray) -> FrameLabel:
    """The memory of one frame: its nearest admissible entry of another episode, unless it is a frame without memory."""
    index, similarity, rule = neighbour(store, key, skill, width, exclude_episode=episode_id)
    if no_memory(episode_id, step) or index < 0:
        return FrameLabel(neighbour=-1, similarity=similarity, rule=rule, prompt=skill)
    prompt = with_memory(skill, store.sentence[index])
    return FrameLabel(neighbour=index, similarity=similarity, rule=rule, prompt=prompt)


def label_episode(store: MomentStore, episode: Episode, wrist_frames: Sequence[np.ndarray],
                  wrist_key: WristKey) -> list[FrameLabel]:
    """The memory of every frame of one episode. wrist_frames: the raw wrist frame of every frame, in order;
    wrist_key: the task's key, the one the store was built with."""
    keys = wrist_key.keys_of(wrist_frames)
    labels = []
    for step, key in enumerate(keys):
        width = float(episode.widths[step])
        labels.append(label_frame(store, episode.episode_id, step, episode.skills[step], width, key))
    return labels
