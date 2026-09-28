"""Exemplar store: labelled camera frames of past episodes, retrieved by fingerprint similarity and shown to the
planner model as few-shot examples (retrieval-augmented in-context learning).

Each entry holds the camera images of one past moment, its fingerprint, and labels that were known when the entry
was written (which skill was active, the state of every object, the gripper). For a query while skill k is active,
an entry's answer is current_skill_done = (entry's skill index > k): the exact switch timing of the labelled data.

Build once from recorded episodes (scripts/build_exemplar_store.py), load at run time; the store is read-only during
a run. File format: one .npz with the image stacks, the key matrix and the metadata as JSON, so stores built by the
research code are loaded unchanged.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Sequence

import numpy as np

from ..memory import MemoryStore
from .fingerprint import Fingerprint


class ExemplarStore(MemoryStore):
    def __init__(self, fingerprint: Fingerprint, task, path=None, camera_names=("exterior", "wrist")):
        self.fp = fingerprint; self.task = task; self.camera_names = tuple(camera_names)
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
        st = cls(fingerprint, task, camera_names=camera_names)
        stacks = [[] for _ in camera_names]; keys = []; meta = []
        for ep in episodes:
            cams = [np.asarray(ep[c]) for c in camera_names]   # decompress each array once, not once per frame
            n = len(cams[0])
            for i in range(0, n, stride):
                frames = [c[i] for c in cams]
                for s, f in zip(stacks, frames):
                    s.append(f)
                keys.append(fingerprint(frames))
                meta.append(dict(episode=ep["episode"], frame=int(i), **labeller(ep, i)))
        st.images = [np.stack(s) for s in stacks]; st.keys = np.stack(keys).astype(np.float32); st.meta = meta
        if out:
            st.save(out)
        return st

    def save(self, path):
        path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"img{i}": s for i, s in enumerate(self.images)}
        # keep the names of the research-code format for two cameras so both directions stay loadable
        if len(self.images) == 2:
            arrays = {"ext": self.images[0], "wri": self.images[1]}
        np.savez_compressed(path, keys=self.keys, meta=json.dumps(self.meta), cameras=json.dumps(self.camera_names), **arrays)

    def load(self, path):
        z = np.load(path)
        if "ext" in z.files:
            self.images = [z["ext"], z["wri"]]
        else:
            self.images = [z[f"img{i}"] for i in range(len(json.loads(str(z["cameras"]))))]
        self.keys = z["keys"]; self.meta = json.loads(str(z["meta"]))
        return self

    def rekey(self):
        """Recompute all keys with the current fingerprint (after changing it or the camera subset)."""
        self.keys = np.stack([self.fp([s[i] for s in self.images]) for i in range(len(self.meta))]).astype(np.float32)
        return self

    def __len__(self):
        return len(self.meta)

    # ------------------------------------------------------------------ reading
    def retrieve(self, images, k=4, exclude_episode=None, per_episode=2):
        """k most similar entries (cosine), at most `per_episode` from the same past episode, never from
        `exclude_episode` (leave-one-out evaluation on the episodes the store was built from)."""
        q = self.fp(list(images)); sims = self.keys @ q
        order = np.argsort(-sims, kind="stable"); out = []; per: dict = {}
        for j in order:
            m = self.meta[j]
            if exclude_episode is not None and m["episode"] == exclude_episode:
                continue
            if per.get(m["episode"], 0) >= per_episode:
                continue
            per[m["episode"]] = per.get(m["episode"], 0) + 1
            out.append((int(j), float(sims[j])))
            if len(out) >= k:
                break
        return out

    def describe_hit(self, index):
        m = self.meta[index]
        return {"episode": m.get("episode"), "frame": m.get("frame"), "subgoal": m.get("subgoal")}

    def fewshot_content(self, hits, skill_index, skill, encode_png):
        t = self.task; ncam = len(self.images)
        cams = " then ".join(self.camera_names)
        items = [{"type": "text", "text": f"Here are {len(hits)} labelled observations from PREVIOUS episodes that look "
                                          f"similar to the current one (each is a group of {ncam} pictures: {cams} camera). "
                                          f"Use them to calibrate how the states look from these cameras."}]
        for n, (j, sim) in enumerate(hits):
            m = self.meta[j]; done = m["subgoal"] > skill_index
            pics = ", ".join(f"Picture {ncam * n + c + 1}: {name}" for c, name in enumerate(self.camera_names))
            objs = ", ".join(f"{o} {t.object_noun} = {m.get(t.field(o), 'unsure')}" for o in t.objects)
            items.append({"type": "text", "text": f"Example {n + 1} ({pics}): {objs}, gripper = {m['gripper']}, "
                                                  f"gripper high above table = {str(m['gripper_high_above_table']).lower()}; "
                                                  f"for the skill \"{skill}\": current_skill_done = {str(done).lower()}."})
            for s in self.images:
                items.append({"type": "image_url", "image_url": {"url": encode_png(s[j])}})
        return items
