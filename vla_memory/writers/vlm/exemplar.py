"""Exemplar store: labelled camera frames of past episodes, retrieved by fingerprint similarity and shown to the
planner model as few-shot examples (retrieval-augmented in-context learning).

Each entry holds the camera images of one past moment, its fingerprint, and labels that were known when the entry
was written (which skill was active, the state of every object, the gripper). For a query while skill k is active,
an entry's answer is current_skill_done = (entry's skill index > k): the exact switch timing of the labelled data.

Build once from recorded episodes (scripts/build_exemplar_store.py) and load at run time; the store is only read.

File format: one .npz file (the image stacks, the key matrix and the metadata as JSON), written by save and read by load;
the two-camera stacks are named ext and wri. The store directory of earlier versions is no longer read: the store that
grows between episodes is the session store (vla_memory/methods/session_store).
"""
from __future__ import annotations

import json
import pathlib
from typing import Callable, Sequence

import numpy as np

from ...memory import MemoryStore
from ...fingerprint import Fingerprint


class ExemplarStore(MemoryStore):
    """The store: one uint8 image stack per camera, the key matrix (one unit vector per entry) and the entries' labels
    (meta). path: a .npz file to load."""

    def __init__(self, fingerprint: Fingerprint, task, path=None, camera_names=("exterior", "wrist")):
        self.fingerprint = fingerprint
        self.task = task
        self.camera_names = tuple(camera_names)
        self.images: list[np.ndarray] = []      # one uint8 stack per camera, each (N, H, W, 3)
        self.keys: np.ndarray | None = None
        self.meta: list[dict] = []
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
        """Load the store from one .npz file (module docstring)."""
        if pathlib.Path(path).is_dir():
            raise ValueError(f"exemplar store {path} is a directory: the store's directory format is gone. A store is one "
                             f".npz file (scripts/build_exemplar_store.py); the store that grows between episodes is the "
                             f"session store (--with session_store)")
        archive = np.load(path)
        if "ext" in archive.files:
            self.images = [archive["ext"], archive["wri"]]
        else:
            self.images = [archive[f"img{camera_index}"] for camera_index in range(len(json.loads(str(archive["cameras"]))))]
        self.keys = archive["keys"]
        self.meta = json.loads(str(archive["meta"]))
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
                            for object_name in labels.get("objects", task.objects))   # an entry may name the objects of its episode
        # the task's state fields (the reference task: "gripper = open, gripper high above table = false")
        state = "".join(f", {field.replace('_', ' ')} = {_label_text(labels[field])}" for field in task.state_fields() if field in labels)
        return (f"Example {number + 1} ({pictures}): {objects}{state}; "
                f"for the skill \"{skill}\": current_skill_done = {str(done).lower()}.")


def _label_text(value) -> str:
    """A label value in an example's text: booleans in JSON spelling (true / false), anything else as it is."""
    if isinstance(value, (bool, np.bool_)):
        return str(value).lower()
    return str(value)
