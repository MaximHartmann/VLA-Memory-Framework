"""The memory store on disk: the stored moments of earlier episodes that the policy's memory is retrieved from.

A store is a directory with these files:

    entries.jsonl          one JSON object per stored moment, in index order (the fields below)
    keys_fingerprint.npy   (n, d) float32: the retrieval key of every entry's wrist picture (key.py); row i is entry i
    wrist/NNNNN.png        the entry's raw wrist picture (240 x 320 RGB in the reference store); NNNNN is its index
    meta.json              what the store holds: key (the key's name, "fingerprint"), key_dim, entries, episodes, the counts
                           by source, skill and gripper state, the release entries per skill, horizon, the thresholds, built
    chunks.npz             optional: q_now (n, 7) and chunk (n, HORIZON, 8), the joint angles and the actions every sentence
                           was computed from (sentence.py), kept for checks

The fields of an entry, in file order:

    id            the entry's index, 0 to n - 1
    episode       the episode it comes from, e.g. "demo_train_000056" or "nudge_2662"; retrieval can leave one episode out
    seed          the simulator seed of that episode (-1 for a demonstration)
    step          the control step of the moment in its episode
    skill         the skill the robot worked on (one of the task's skill prompts)
    skill_index   the skill's index in the task's skill list
    source        who acted: "demo" (a demonstration), "policy" (the policy's own episode) or "expert" (an expert correction)
    width         the measured finger width at the moment (m, both fingers together, rounded to 0.1 mm)
    gripper       the gripper state from the unrounded width: open, closed_object or closed_nothing (gripper.py)
    release       whether the sentence begins with a release (sentence.is_release)
    sentence      the memory sentence of the moment's next HORIZON actions (sentence.py)
    wrist_png     the path of the wrist picture, relative to the store directory

MomentStore.load(path) reads a store for retrieval; save_store(...) writes one (the store builder).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

from .key import KEY_NAME, unit_rows

ENTRIES_FILE = "entries.jsonl"
KEYS_FILE = f"keys_{KEY_NAME}.npy"
META_FILE = "meta.json"
CHUNKS_FILE = "chunks.npz"
PICTURES_DIR = "wrist"
ENTRY_FIELDS = ("id", "episode", "seed", "step", "skill", "skill_index", "source", "width", "gripper", "release", "sentence",
                "wrist_png")
GIVEN_FIELDS = ENTRY_FIELDS[1:-1]     # what the builder provides; save_store sets id and wrist_png from the entry's index


@dataclasses.dataclass
class MomentStore:
    """The stored moments, read for retrieval: one row per entry in every array, in index order. (Named so to keep it apart
    from the planner's long-term memory interface vla_memory.memory.MemoryStore.)"""
    path: pathlib.Path
    meta: dict
    keys: np.ndarray          # (n, d) float32, rows of length 1
    skill: np.ndarray         # (n,) str
    closed: np.ndarray        # (n,) bool: the gripper was closed at the moment (closed_object or closed_nothing)
    release: np.ndarray       # (n,) bool: the sentence begins with a release
    episode: np.ndarray       # (n,) str
    step: np.ndarray          # (n,) int
    source: np.ndarray        # (n,) str
    sentence: list            # (n,) str: the memory sentences
    pictures: list            # (n,) str: the paths of the wrist pictures

    def __len__(self) -> int:
        return len(self.sentence)

    @classmethod
    def load(cls, path) -> "MomentStore":
        """Read a store directory (the files above; chunks.npz is not needed)."""
        path = pathlib.Path(path)
        meta = json.loads((path / META_FILE).read_text())
        check_key_name(meta, path)
        rows = read_entries(path)
        keys = unit_rows(np.load(path / KEYS_FILE))
        if len(keys) != len(rows):
            raise ValueError(f"{path}: {len(keys)} keys for {len(rows)} entries")
        return cls(path=path, meta=meta, keys=keys,
                   skill=np.array([row["skill"] for row in rows], dtype=str),
                   closed=np.array([row["gripper"] != "open" for row in rows], dtype=bool),
                   release=np.array([bool(row["release"]) for row in rows], dtype=bool),
                   episode=np.array([row["episode"] for row in rows], dtype=str),
                   step=np.array([int(row["step"]) for row in rows], dtype=int),
                   source=np.array([row["source"] for row in rows], dtype=str),
                   sentence=[row["sentence"] for row in rows],
                   pictures=[str(path / row["wrist_png"]) for row in rows])


def read_entries(path) -> list[dict]:
    """The entries of a store directory, in index order (entries.jsonl)."""
    lines = (pathlib.Path(path) / ENTRIES_FILE).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def check_key_name(meta: dict, path) -> None:
    """A store whose keys are not this package's key cannot be queried with it."""
    name = meta.get("key", KEY_NAME)
    if name != KEY_NAME:
        raise ValueError(f"{path}: the store's keys are {name!r} keys; the memory policy computes {KEY_NAME!r} keys")


# ---------------------------------------------------------------------- writing a store
def save_store(path, entries, keys, wrist_pngs, meta: dict, q_now=None, chunk=None) -> None:
    """Write a store directory in the format above; files of a store already there are overwritten.

    entries: one dict per entry with the fields GIVEN_FIELDS (gripper from the unrounded width, gripper.gripper_state;
    release from the sentence, sentence.is_release); id and wrist_png are set from the order, the width is rounded to 0.1 mm.
    keys: (n, d), saved as float32. wrist_pngs: the PNG bytes of every entry's wrist picture. meta: written as meta.json.
    q_now (n, 7) and chunk (n, HORIZON, 8): written as chunks.npz when given."""
    path = pathlib.Path(path)
    keys = np.asarray(keys, np.float32)
    if not len(entries) == len(keys) == len(wrist_pngs):
        raise ValueError(f"{len(entries)} entries, {len(keys)} keys and {len(wrist_pngs)} pictures: one of each per entry")
    (path / PICTURES_DIR).mkdir(parents=True, exist_ok=True)
    np.save(path / KEYS_FILE, keys)
    write_entries(path, entries, wrist_pngs)
    if q_now is not None or chunk is not None:
        save_chunks(path, q_now, chunk)
    (path / META_FILE).write_text(json.dumps(meta, indent=1))


def write_entries(path: pathlib.Path, entries, wrist_pngs) -> None:
    """entries.jsonl and the wrist pictures, in entry order."""
    with (path / ENTRIES_FILE).open("w") as file:
        for index, (entry, png) in enumerate(zip(entries, wrist_pngs)):
            record = entry_record(index, entry)
            (path / record["wrist_png"]).write_bytes(png)
            file.write(json.dumps(record) + "\n")


def entry_record(index: int, entry: dict) -> dict:
    """One line of entries.jsonl: the entry's fields in file order, with its index and its picture's path."""
    missing = [name for name in GIVEN_FIELDS if name not in entry]
    if missing:
        raise ValueError(f"store entry {index}: no field(s) {missing}")
    return {"id": index, "episode": str(entry["episode"]), "seed": int(entry["seed"]), "step": int(entry["step"]),
            "skill": str(entry["skill"]), "skill_index": int(entry["skill_index"]), "source": str(entry["source"]),
            "width": round(float(entry["width"]), 4), "gripper": str(entry["gripper"]), "release": bool(entry["release"]),
            "sentence": str(entry["sentence"]), "wrist_png": f"{PICTURES_DIR}/{index:05d}.png"}


def save_chunks(path: pathlib.Path, q_now, chunk) -> None:
    """chunks.npz: the joint angles and the actions of every entry."""
    if q_now is None or chunk is None:
        raise ValueError("chunks.npz needs both q_now and chunk")
    np.savez(path / CHUNKS_FILE, q_now=np.asarray(q_now), chunk=np.asarray(chunk))
