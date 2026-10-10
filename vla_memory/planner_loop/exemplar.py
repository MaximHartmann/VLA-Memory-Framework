"""Exemplar store: labelled camera frames of past episodes, retrieved by fingerprint similarity and shown to the
planner model as few-shot examples (retrieval-augmented in-context learning).

Each entry holds the camera images of one past moment, its fingerprint, and labels that were known when the entry
was written (which skill was active, the state of every object, the gripper). For a query while skill k is active,
an entry's answer is current_skill_done = (entry's skill index > k): the exact switch timing of the labelled data.

Build once from recorded episodes (scripts/build_exemplar_store.py) and load at run time; during an episode the store
is only read. Between episodes a memory method may let it grow (append, keep, a store directory): each entry may carry
its provenance (meta "provenance": source demo | self, episode, labeller, why it was admitted, wall time, update number);
entries without one are demonstrations built from recordings (provenance()).

File formats, both loaded by load(path): one .npz file (the image stacks, the key matrix and the metadata as JSON, written
by save; stores built by the research code load unchanged), or a store directory for a store that grows between episodes
(exemplar_dir.py: save_dir, load).
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Sequence

import numpy as np

from ..memory import MemoryStore
from .exemplar_dir import load_directory, save_directory
from .fingerprint import Fingerprint


class ExemplarStore(MemoryStore):
    def __init__(self, fingerprint: Fingerprint, task, path=None, camera_names=("exterior", "wrist")):
        self.fp = fingerprint
        self.task = task
        self.camera_names = tuple(camera_names)
        self.images: list[np.ndarray] = []      # one uint8 stack per camera, each (N, H, W, 3)
        self.keys: np.ndarray | None = None
        self.meta: list[dict] = []
        self.loc: np.ndarray | None = None      # per entry (segment, row) of its frames in the directory loc_dir; -1: not there
        self.loc_dir: pathlib.Path | None = None
        self.info: dict = {}                    # free-form JSON state kept in a store directory's index
        if path:
            self.load(path)

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(cls, fingerprint: Fingerprint, task, episodes: Sequence[dict], labeller: Callable, stride: int = 8,
              out=None, camera_names=("exterior", "wrist")) -> "ExemplarStore":
        """episodes: iterable of dicts with the camera stacks (keys = camera_names, each (T, H, W, 3)), an 'episode'
        id and whatever the labeller needs (e.g. 'phase', 'state'); labeller(episode_dict, frame) -> label dict
        with the task's object fields, 'gripper', 'gripper_high_above_table' and 'subgoal' (skill index)."""
        store = cls(fingerprint, task, camera_names=camera_names)
        stacks = [[] for _ in camera_names]
        keys = []
        meta = []
        for episode in episodes:
            cameras = [np.asarray(episode[c]) for c in camera_names]   # decompress each array once, not once per frame
            for i in range(0, len(cameras[0]), stride):
                frames = [c[i] for c in cameras]
                for stack, frame in zip(stacks, frames):
                    stack.append(frame)
                keys.append(fingerprint(frames))
                meta.append(dict(episode=episode["episode"], frame=int(i), **labeller(episode, i)))
        store.images = [np.stack(s) for s in stacks]
        store.keys = np.stack(keys).astype(np.float32)
        store.meta = meta
        if out:
            store.save(out)
        return store

    def save(self, path):
        """One .npz with the image stacks, the keys and the metadata as JSON."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"img{i}": s for i, s in enumerate(self.images)}
        # keep the names of the research-code format for two cameras so both directions stay loadable
        if len(self.images) == 2:
            arrays = {"ext": self.images[0], "wri": self.images[1]}
        np.savez_compressed(path, keys=self.keys, meta=json.dumps(self.meta), cameras=json.dumps(self.camera_names), **arrays)

    def load(self, path):
        """Load the store from one .npz or from a store directory (module docstring)."""
        if pathlib.Path(path).is_dir():
            return self._load_dir(pathlib.Path(path))
        z = np.load(path)
        if "ext" in z.files:
            self.images = [z["ext"], z["wri"]]
        else:
            self.images = [z[f"img{i}"] for i in range(len(json.loads(str(z["cameras"]))))]
        self.keys = z["keys"]
        self.meta = json.loads(str(z["meta"]))
        self.loc = None
        self.loc_dir = None
        return self

    def rekey(self):
        """Recompute all keys with the current fingerprint (after changing it or the camera subset)."""
        keys = []
        for i in range(len(self.meta)):
            keys.append(self.fp([stack[i] for stack in self.images]))
        self.keys = np.stack(keys).astype(np.float32)
        return self

    def __len__(self):
        return len(self.meta)

    # ------------------------------------------------------------------ growing, pruning, provenance
    def append(self, images: Sequence[np.ndarray], keys: np.ndarray, meta: Sequence[dict]):
        """Add n entries: images = one (n, H, W, 3) uint8 stack per camera (camera_names order), keys (n, D), meta n dicts.
        The frames must have the size of the frames already stored, and the keys their length."""
        n = len(meta)
        images = [np.asarray(s) for s in images]
        if len(images) != len(self.camera_names) or any(len(s) != n for s in images):
            raise ValueError(f"append: expected one stack of {n} frames per camera {self.camera_names}, got "
                             f"{[np.shape(s) for s in images]}")
        if n == 0:
            return self
        keys = np.asarray(keys, np.float32).reshape(n, -1)
        if len(self):
            self._check_fits(images, keys)
        self._concatenate(images, keys, meta)
        return self

    def _check_fits(self, images, keys):
        """New frames must have the size of the stored frames, and new keys the length of the stored keys."""
        for name, old, new in zip(self.camera_names, self.images, images):
            if old.shape[1:] != new.shape[1:]:
                raise ValueError(f"append: {name} frames {new.shape[1:]} differ from the stored {old.shape[1:]}")
        if keys.shape[1] != self.keys.shape[1]:
            raise ValueError(f"append: keys of length {keys.shape[1]}, the store's are {self.keys.shape[1]}")

    def _concatenate(self, images, keys, meta):
        """Put the new entries after the stored ones; their frames are in no segment file yet (location -1)."""
        if len(self):
            self.images = [np.concatenate([old, new]) for old, new in zip(self.images, images)]
            self.keys = np.concatenate([self.keys, keys])
        else:
            self.images = [np.ascontiguousarray(s) for s in images]
            self.keys = keys
        self.loc = np.concatenate([self.locations(), np.full((len(meta), 2), -1, np.int64)])
        self.meta = list(self.meta) + [dict(m) for m in meta]

    def keep(self, indices):
        """Keep the entries at `indices`, in that order, and remove all others."""
        idx = np.asarray(indices, np.int64).reshape(-1)
        loc = self.locations()
        self.images = [s[idx] for s in self.images]
        if self.keys is not None:
            self.keys = self.keys[idx]
        self.meta = [self.meta[i] for i in idx]
        self.loc = loc[idx]
        return self

    def provenance(self, index) -> dict:
        """Where entry `index` came from: source (demo | self), episode, labeller, admitted (why), time, update. Stores
        written before provenance existed hold demonstrations labelled when the store was built."""
        m = self.meta[index]
        return m.get("provenance") or {"source": "demo", "episode": m.get("episode"), "labeller": None, "admitted": "built",
                                       "time": None, "update": 0}

    def locations(self) -> np.ndarray:
        """Per entry (segment, row) of its frames in the store directory; -1 when they are not in one."""
        """Per entry (segment, row) of its frames in loc_dir; -1 everywhere when unknown."""
        if self.loc is None or len(self.loc) != len(self.meta):
            return np.full((len(self.meta), 2), -1, np.int64)
        return self.loc

    # ------------------------------------------------------------------ the directory format (exemplar_dir.py)
    def save_dir(self, path) -> dict:
        """Save as a store directory: only the frames not yet there are written. Returns the new segment, how many
        entries it holds, and the directory's size in bytes."""
        return save_directory(self, path)

    def _load_dir(self, directory: pathlib.Path):
        return load_directory(self, directory)

    # ------------------------------------------------------------------ reading
    def retrieve(self, images, k=4, exclude_episode=None, per_episode=2):
        """k most similar entries (cosine), at most `per_episode` from the same past episode, never from
        `exclude_episode` (leave-one-out evaluation on the episodes the store was built from)."""
        query = self.fp(list(images))
        similarities = self.keys @ query
        order = np.argsort(-similarities, kind="stable")
        hits = []
        taken: dict = {}   # entries taken per past episode
        for j in order:
            episode = self.meta[j]["episode"]
            if exclude_episode is not None and episode == exclude_episode:
                continue
            if taken.get(episode, 0) >= per_episode:
                continue
            taken[episode] = taken.get(episode, 0) + 1
            hits.append((int(j), float(similarities[j])))
            if len(hits) >= k:
                break
        return hits

    def describe_hit(self, index):
        m = self.meta[index]
        return {"episode": m.get("episode"), "frame": m.get("frame"), "subgoal": m.get("subgoal")}

    def fewshot_content(self, hits, skill_index, skill, encode_png):
        """The retrieved entries as chat content: a header text, then per example one text and its camera pictures."""
        items = [{"type": "text", "text": self._fewshot_header(len(hits))}]
        for n, (j, _similarity) in enumerate(hits):
            items.append({"type": "text", "text": self._example_text(n, j, skill_index, skill)})
            for stack in self.images:
                items.append({"type": "image_url", "image_url": {"url": encode_png(stack[j])}})
        return items

    def _fewshot_header(self, count: int) -> str:
        """The text before the examples: how many there are and how the pictures of each are ordered."""
        cameras = " then ".join(self.camera_names)
        return (f"Here are {count} labelled observations from PREVIOUS episodes that look "
                f"similar to the current one (each is a group of {len(self.images)} pictures: {cameras} camera). "
                f"Use them to calibrate how the states look from these cameras.")

    def _example_text(self, n: int, j: int, skill_index: int, skill: str) -> str:
        """Example n (entry j): which pictures are its cameras, its labelled states, and its answer for the current skill
        (done when the entry's skill index is past the current one)."""
        task = self.task
        labels = self.meta[j]
        ncam = len(self.images)
        done = labels["subgoal"] > skill_index
        pictures = ", ".join(f"Picture {ncam * n + c + 1}: {name}" for c, name in enumerate(self.camera_names))
        objects = ", ".join(f"{o} {task.object_noun} = {labels.get(task.field(o), 'unsure')}" for o in task.objects)
        # the task's state fields (the reference task: "gripper = open, gripper high above table = false")
        state = "".join(f", {k.replace('_', ' ')} = {_label_text(labels[k])}" for k in task.state_fields() if k in labels)
        return (f"Example {n + 1} ({pictures}): {objects}{state}; "
                f"for the skill \"{skill}\": current_skill_done = {str(done).lower()}.")


def _label_text(v) -> str:
    """A label value in an example's text: booleans in JSON spelling (true / false), anything else as it is."""
    if isinstance(v, (bool, np.bool_)):
        return str(v).lower()
    return str(v)

