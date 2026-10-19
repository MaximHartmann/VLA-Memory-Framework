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
by save; the two-camera stacks are named ext and wri), or a store directory for a store that grows between episodes
(exemplar_dir.py: save_dir, load).
"""
from __future__ import annotations

import json
import pathlib
from typing import Callable, Sequence

import numpy as np

from ...memory import MemoryStore
from .exemplar_dir import load_directory, save_directory
from ...fingerprint import Fingerprint


class ExemplarStore(MemoryStore):
    """The store: one uint8 image stack per camera, the key matrix (one unit vector per entry) and the entries' labels
    (meta). path: a .npz file or a store directory to load."""

    def __init__(self, fingerprint: Fingerprint, task, path=None, camera_names=("exterior", "wrist")):
        self.fingerprint = fingerprint
        self.task = task
        self.camera_names = tuple(camera_names)
        self.images: list[np.ndarray] = []      # one uint8 stack per camera, each (N, H, W, 3)
        self.keys: np.ndarray | None = None
        self.meta: list[dict] = []
        self.frame_locations: np.ndarray | None = None   # per entry (segment, row) of its frames in location_directory; -1: not there
        self.location_directory: pathlib.Path | None = None
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
            cameras = [np.asarray(episode[name]) for name in camera_names]   # decompress each array once, not once per frame
            for frame_index in range(0, len(cameras[0]), stride):
                frames = [camera[frame_index] for camera in cameras]
                for stack, frame in zip(stacks, frames):
                    stack.append(frame)
                keys.append(fingerprint(frames))
                meta.append(dict(episode=episode["episode"], frame=int(frame_index), **labeller(episode, frame_index)))
        store.images = [np.stack(stack) for stack in stacks]
        store.keys = np.stack(keys).astype(np.float32)
        store.meta = meta
        if out:
            store.save(out)
        return store

    def save(self, path):
        """One .npz with the image stacks, the keys and the metadata as JSON."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"img{camera_index}": stack for camera_index, stack in enumerate(self.images)}
        if len(self.images) == 2:   # the two-camera names every store so far uses: ext and wri
            arrays = {"ext": self.images[0], "wri": self.images[1]}
        np.savez_compressed(path, keys=self.keys, meta=json.dumps(self.meta), cameras=json.dumps(self.camera_names), **arrays)

    def load(self, path):
        """Load the store from one .npz or from a store directory (module docstring)."""
        if pathlib.Path(path).is_dir():
            return load_directory(self, pathlib.Path(path))
        archive = np.load(path)
        if "ext" in archive.files:
            self.images = [archive["ext"], archive["wri"]]
        else:
            self.images = [archive[f"img{camera_index}"] for camera_index in range(len(json.loads(str(archive["cameras"]))))]
        self.keys = archive["keys"]
        self.meta = json.loads(str(archive["meta"]))
        self.frame_locations = None
        self.location_directory = None
        return self

    def rekey(self):
        """Recompute all keys with the current fingerprint (after changing it or the camera subset)."""
        keys = []
        for entry in range(len(self.meta)):
            keys.append(self.fingerprint([stack[entry] for stack in self.images]))
        self.keys = np.stack(keys).astype(np.float32)
        return self

    def __len__(self):
        return len(self.meta)

    # ------------------------------------------------------------------ growing, pruning, provenance
    def append(self, images: Sequence[np.ndarray], keys: np.ndarray, meta: Sequence[dict]):
        """Add n entries: images = one (n, H, W, 3) uint8 stack per camera (camera_names order), keys (n, D), meta n dicts.
        The frames must have the size of the frames already stored, and the keys their length."""
        count = len(meta)
        images = [np.asarray(stack) for stack in images]
        if len(images) != len(self.camera_names) or any(len(stack) != count for stack in images):
            raise ValueError(f"append: expected one stack of {count} frames per camera {self.camera_names}, got "
                             f"{[np.shape(stack) for stack in images]}")
        if count == 0:
            return self
        keys = np.asarray(keys, np.float32).reshape(count, -1)
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
            self.images = [np.ascontiguousarray(stack) for stack in images]
            self.keys = keys
        self.frame_locations = np.concatenate([self.locations(), np.full((len(meta), 2), -1, np.int64)])
        self.meta = list(self.meta) + [dict(entry) for entry in meta]

    def keep(self, indices):
        """Keep the entries at `indices`, in that order, and remove all others."""
        kept = np.asarray(indices, np.int64).reshape(-1)
        locations = self.locations()
        self.images = [stack[kept] for stack in self.images]
        if self.keys is not None:
            self.keys = self.keys[kept]
        self.meta = [self.meta[index] for index in kept]
        self.frame_locations = locations[kept]
        return self

    def provenance(self, index) -> dict:
        """Where entry `index` came from: source (demo | self), episode, labeller, admitted (why), time, update. Stores
        written before provenance existed hold demonstrations labelled when the store was built."""
        entry = self.meta[index]
        return entry.get("provenance") or {"source": "demo", "episode": entry.get("episode"), "labeller": None,
                                           "admitted": "built", "time": None, "update": 0}

    def locations(self) -> np.ndarray:
        """Per entry (segment, row) of its frames in the store directory; -1 everywhere when unknown."""
        if self.frame_locations is None or len(self.frame_locations) != len(self.meta):
            return np.full((len(self.meta), 2), -1, np.int64)
        return self.frame_locations

    def save_dir(self, path) -> dict:
        """Save as a store directory (exemplar_dir.py): only the frames not yet there are written. Returns the new
        segment, how many entries it holds, and the directory's size in bytes."""
        return save_directory(self, path)

    # ------------------------------------------------------------------ reading
    def retrieve(self, images, k=4, exclude_episode=None, per_episode=2):
        """k most similar entries (cosine), at most `per_episode` from the same past episode, never from
        `exclude_episode` (leave-one-out evaluation on the episodes the store was built from)."""
        query = self.fingerprint(list(images))
        similarities = self.keys @ query
        order = np.argsort(-similarities, kind="stable")
        hits = []
        taken: dict = {}   # entries taken per past episode
        for index in order:
            episode = self.meta[index]["episode"]
            if exclude_episode is not None and episode == exclude_episode:
                continue
            if taken.get(episode, 0) >= per_episode:
                continue
            taken[episode] = taken.get(episode, 0) + 1
            hits.append((int(index), float(similarities[index])))
            if len(hits) >= k:
                break
        return hits

    def describe_hit(self, index):
        """The episode, the frame and the skill index of one entry, for the decision log and the gate."""
        entry = self.meta[index]
        return {"episode": entry.get("episode"), "frame": entry.get("frame"), "subgoal": entry.get("subgoal")}

    def fewshot_content(self, hits, skill_index, skill, encode_png):
        """The retrieved entries as chat content: a header text, then per example one text and its camera pictures."""
        items = [{"type": "text", "text": self._fewshot_header(len(hits))}]
        for number, (entry, _similarity) in enumerate(hits):
            items.append({"type": "text", "text": self._example_text(number, entry, skill_index, skill)})
            for stack in self.images:
                items.append({"type": "image_url", "image_url": {"url": encode_png(stack[entry])}})
        return items

    def _fewshot_header(self, count: int) -> str:
        """The text before the examples: how many there are and how the pictures of each are ordered."""
        cameras = " then ".join(self.camera_names)
        return (f"Here are {count} labelled observations from PREVIOUS episodes that look "
                f"similar to the current one (each is a group of {len(self.images)} pictures: {cameras} camera). "
                f"Use them to calibrate how the states look from these cameras.")

    def _example_text(self, number: int, entry: int, skill_index: int, skill: str) -> str:
        """Example `number` (the store's entry `entry`): which pictures are its cameras, its labelled states, and its answer
        for the current skill (done when the entry's skill index is past the current one)."""
        task = self.task
        labels = self.meta[entry]
        camera_count = len(self.images)
        done = labels["subgoal"] > skill_index
        pictures = ", ".join(f"Picture {camera_count * number + camera_index + 1}: {name}"
                             for camera_index, name in enumerate(self.camera_names))
        objects = ", ".join(f"{object_name} {task.object_noun} = {labels.get(task.field(object_name), 'unsure')}"
                            for object_name in task.objects)
        # the task's state fields (the reference task: "gripper = open, gripper high above table = false")
        state = "".join(f", {field.replace('_', ' ')} = {_label_text(labels[field])}" for field in task.state_fields() if field in labels)
        return (f"Example {number + 1} ({pictures}): {objects}{state}; "
                f"for the skill \"{skill}\": current_skill_done = {str(done).lower()}.")


def _label_text(value) -> str:
    """A label value in an example's text: booleans in JSON spelling (true / false), anything else as it is."""
    if isinstance(value, (bool, np.bool_)):
        return str(value).lower()
    return str(value)
