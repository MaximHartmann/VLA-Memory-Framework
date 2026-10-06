"""Exemplar store: labelled camera frames of past episodes, retrieved by fingerprint similarity and shown to the
planner model as few-shot examples (retrieval-augmented in-context learning).

Each entry holds the camera images of one past moment, its fingerprint, and labels that were known when the entry
was written (which skill was active, the state of every object, the gripper). For a query while skill k is active,
an entry's answer is current_skill_done = (entry's skill index > k): the exact switch timing of the labelled data.

Build once from recorded episodes (scripts/build_exemplar_store.py) and load at run time; during an episode the store
is only read. Between episodes it may grow (append, keep): the online store (online.py) adds the agent's own labelled
episodes, removes near-duplicates and evicts above a capacity. Each entry may carry its provenance (meta "provenance":
source demo | self, episode, labeller, why it was admitted, wall time, update number); entries without one are
demonstrations built from recordings (provenance()).

File formats, both loaded by load(path):
  one .npz       the image stacks, the key matrix and the metadata as JSON (save), so stores built by the research code
                 are loaded unchanged;
  a directory    index.npz (keys, metadata, camera names, where each entry's frames are) and segments/seg_NNNNNN.npz
                 (frames, compressed, written once). save_dir writes only the frames not yet in the directory and then
                 replaces the index atomically (tmp file + rename), so saving after every episode costs the new frames,
                 not the whole store, and a crash leaves the previous index valid.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Callable, Sequence

import numpy as np

from ..memory import MemoryStore
from .fingerprint import Fingerprint

DIR_FORMAT = "vla_memory exemplar store directory v1"


class ExemplarStore(MemoryStore):
    def __init__(self, fingerprint: Fingerprint, task, path=None, camera_names=("exterior", "wrist")):
        self.fp = fingerprint; self.task = task; self.camera_names = tuple(camera_names)
        self.images: list[np.ndarray] = []      # one uint8 stack per camera, each (N, H, W, 3)
        self.keys: np.ndarray | None = None
        self.meta: list[dict] = []
        self.loc: np.ndarray | None = None      # per entry (segment, row) of its frames in the directory loc_dir; -1: not there
        self.loc_dir: pathlib.Path | None = None
        self.info: dict = {}                    # free-form JSON state kept in a store directory's index (online.py: updates)
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
        if pathlib.Path(path).is_dir():
            return self._load_dir(pathlib.Path(path))
        z = np.load(path)
        if "ext" in z.files:
            self.images = [z["ext"], z["wri"]]
        else:
            self.images = [z[f"img{i}"] for i in range(len(json.loads(str(z["cameras"]))))]
        self.keys = z["keys"]; self.meta = json.loads(str(z["meta"])); self.loc = self.loc_dir = None
        return self

    def rekey(self):
        """Recompute all keys with the current fingerprint (after changing it or the camera subset)."""
        self.keys = np.stack([self.fp([s[i] for s in self.images]) for i in range(len(self.meta))]).astype(np.float32)
        return self

    def __len__(self):
        return len(self.meta)

    # ------------------------------------------------------------------ growing, pruning, provenance
    def append(self, images: Sequence[np.ndarray], keys: np.ndarray, meta: Sequence[dict]):
        """Add n entries: images = one (n, H, W, 3) uint8 stack per camera (camera_names order), keys (n, D), meta n dicts.
        The frames must have the size of the frames already stored, and the keys their length."""
        n = len(meta); images = [np.asarray(s) for s in images]
        if len(images) != len(self.camera_names) or any(len(s) != n for s in images):
            raise ValueError(f"append: expected one stack of {n} frames per camera {self.camera_names}, got "
                             f"{[np.shape(s) for s in images]}")
        if n == 0:
            return self
        keys = np.asarray(keys, np.float32).reshape(n, -1)
        if len(self):
            for name, old, new in zip(self.camera_names, self.images, images):
                if old.shape[1:] != new.shape[1:]:
                    raise ValueError(f"append: {name} frames {new.shape[1:]} differ from the stored {old.shape[1:]}")
            if keys.shape[1] != self.keys.shape[1]:
                raise ValueError(f"append: keys of length {keys.shape[1]}, the store's are {self.keys.shape[1]}")
            self.images = [np.concatenate([old, new]) for old, new in zip(self.images, images)]
            self.keys = np.concatenate([self.keys, keys])
        else:
            self.images = [np.ascontiguousarray(s) for s in images]; self.keys = keys
        self.loc = np.concatenate([self._locations(), np.full((n, 2), -1, np.int64)])
        self.meta = list(self.meta) + [dict(m) for m in meta]
        return self

    def keep(self, indices):
        """Keep the entries at `indices`, in that order, and remove all others."""
        idx = np.asarray(indices, np.int64).reshape(-1)
        loc = self._locations()
        self.images = [s[idx] for s in self.images]
        self.keys = self.keys[idx] if self.keys is not None else None
        self.meta = [self.meta[i] for i in idx]; self.loc = loc[idx]
        return self

    def provenance(self, index) -> dict:
        """Where entry `index` came from: source (demo | self), episode, labeller, admitted (why), time, update. Stores
        written before provenance existed hold demonstrations labelled when the store was built."""
        m = self.meta[index]
        return m.get("provenance") or {"source": "demo", "episode": m.get("episode"), "labeller": None, "admitted": "built",
                                       "time": None, "update": 0}

    def _locations(self) -> np.ndarray:
        if self.loc is None or len(self.loc) != len(self.meta):
            return np.full((len(self.meta), 2), -1, np.int64)
        return self.loc

    # ------------------------------------------------------------------ the directory format
    def save_dir(self, path) -> dict:
        """Save as a store directory (see the module docstring). Returns the new segment (or None), the number of entries
        whose frames it holds, and the directory's size in bytes."""
        d = pathlib.Path(path); seg_dir = d / "segments"; seg_dir.mkdir(parents=True, exist_ok=True)
        same = self.loc_dir is not None and self.loc_dir == d.resolve()
        loc = self._locations().copy() if same else np.full((len(self), 2), -1, np.int64)
        new = np.flatnonzero(loc[:, 0] < 0); seg = None
        if len(new):
            used = [int(p.stem[4:]) for p in seg_dir.glob("seg_*.npz") if p.stem[4:].isdigit()]
            num = max(used + [int(x) for x in loc[:, 0]] + [-1]) + 1; seg = f"seg_{num:06d}"
            _atomic_savez(seg_dir / f"{seg}.npz", True, **{f"img{c}": s[new] for c, s in enumerate(self.images)})
            loc[new, 0] = num; loc[new, 1] = np.arange(len(new))
        keys = self.keys if self.keys is not None else np.zeros((0, 0), np.float32)
        _atomic_savez(d / "index.npz", False, keys=keys, meta=json.dumps(self.meta), cameras=json.dumps(self.camera_names),
                      loc=loc, info=json.dumps(self.info), format=DIR_FORMAT)
        self.loc, self.loc_dir = loc, d.resolve()
        keep = {f"seg_{int(x):06d}.npz" for x in set(loc[:, 0].tolist())}
        for p in seg_dir.glob("seg_*.npz"):   # segments no entry refers to any more (all their entries were removed)
            if p.name not in keep:
                p.unlink()
        return {"segment": seg, "new": int(len(new)), "bytes": dir_bytes(d)}

    def _load_dir(self, d: pathlib.Path):
        with np.load(d / "index.npz") as z:
            self.meta = json.loads(str(z["meta"])); self.camera_names = tuple(json.loads(str(z["cameras"])))
            loc = z["loc"].astype(np.int64).reshape(-1, 2); n = len(self.meta)
            self.keys = z["keys"].astype(np.float32) if n else None
            self.info = json.loads(str(z["info"])) if "info" in z.files else {}
        stacks: list = [None] * len(self.camera_names)
        for num in sorted(set(loc[:, 0].tolist())):
            rows = np.flatnonzero(loc[:, 0] == num)
            with np.load(d / "segments" / f"seg_{int(num):06d}.npz") as seg:
                for c in range(len(self.camera_names)):
                    frames = seg[f"img{c}"]
                    if stacks[c] is None:
                        stacks[c] = np.empty((n, *frames.shape[1:]), frames.dtype)
                    stacks[c][rows] = frames[loc[rows, 1]]
        self.images = [s for s in stacks if s is not None]
        self.loc, self.loc_dir = loc, d.resolve()
        return self

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
            # the task's state fields (the reference task: "gripper = open, gripper high above table = false")
            state = "".join(f", {k.replace('_', ' ')} = {_label_text(m[k])}" for k in t.state_fields() if k in m)
            items.append({"type": "text", "text": f"Example {n + 1} ({pics}): {objs}{state}; "
                                                  f"for the skill \"{skill}\": current_skill_done = {str(done).lower()}."})
            for s in self.images:
                items.append({"type": "image_url", "image_url": {"url": encode_png(s[j])}})
        return items


def _label_text(v) -> str:
    """A label value in an example's text: booleans in JSON spelling (true / false), anything else as it is."""
    return str(v).lower() if isinstance(v, (bool, np.bool_)) else str(v)


def _atomic_savez(path: pathlib.Path, compressed: bool, **arrays):
    """np.savez(_compressed) to a temporary file next to `path`, synced to disk, then renamed over `path`: readers see the
    old file or the new one, never a partial one."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        (np.savez_compressed if compressed else np.savez)(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def dir_bytes(path) -> int:
    """Bytes of all files below `path` (a store directory, or one file)."""
    p = pathlib.Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
