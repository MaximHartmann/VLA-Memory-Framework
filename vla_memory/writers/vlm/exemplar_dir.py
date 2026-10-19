"""The directory format of the exemplar store, for a store that grows between episodes.

    index.npz                 the keys, the metadata (JSON), the camera names, where each entry's frames are (segment, row),
                              free-form `info` (JSON) and the format name
    segments/seg_NNNNNN.npz   the frames of the entries written at one save, compressed, written once

save_directory writes only the frames that are not yet in the directory as a new segment and then replaces the index
atomically (temporary file, then rename), so saving after every episode costs the new frames, not the whole store, and a
crash leaves the previous index valid. load_directory reads the index and gathers every entry's frames from the segments.
Both work on an ExemplarStore (its images, keys, meta, info, camera_names, frame_locations and location_directory).
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np

DIR_FORMAT = "vla_memory exemplar store directory v1"


# ---------------------------------------------------------------------- saving
def save_directory(store, path) -> dict:
    """Save `store` as a directory. Returns the new segment's name (or None), how many entries it holds, and the
    directory's size in bytes."""
    directory = pathlib.Path(path)
    segment_dir = directory / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    locations = _locations_in(store, directory)
    new_entries = np.flatnonzero(locations[:, 0] < 0)
    segment = None
    if len(new_entries):
        segment = _write_new_segment(store, segment_dir, locations, new_entries)
    _write_index(store, directory, locations)
    store.frame_locations = locations
    store.location_directory = directory.resolve()
    _remove_unused_segments(segment_dir, locations)
    return {"segment": segment, "new": int(len(new_entries)), "bytes": dir_bytes(directory)}


def _locations_in(store, directory: pathlib.Path) -> np.ndarray:
    """Where each entry's frames are in `directory`: a copy of the known locations when it is the store's own directory,
    else -1 everywhere (every entry's frames have to be written)."""
    if store.location_directory is not None and store.location_directory == directory.resolve():
        return store.locations().copy()
    return np.full((len(store), 2), -1, np.int64)


def _write_new_segment(store, segment_dir: pathlib.Path, locations: np.ndarray, new_entries: np.ndarray) -> str:
    """Write the frames of the new entries as the next segment file (numbered after every segment on disk or in the
    locations) and record their locations. Returns the segment's name."""
    used = [int(file.stem[4:]) for file in segment_dir.glob("seg_*.npz") if file.stem[4:].isdigit()]
    number = max(used + [int(segment) for segment in locations[:, 0]] + [-1]) + 1
    segment = f"seg_{number:06d}"
    frames = {f"img{camera_index}": stack[new_entries] for camera_index, stack in enumerate(store.images)}
    atomic_savez(segment_dir / f"{segment}.npz", True, **frames)
    locations[new_entries, 0] = number
    locations[new_entries, 1] = np.arange(len(new_entries))
    return segment


def _write_index(store, directory: pathlib.Path, locations: np.ndarray):
    """Replace index.npz atomically: the keys, the metadata, the camera names, the entries' locations and `info`."""
    keys = store.keys
    if keys is None:
        keys = np.zeros((0, 0), np.float32)
    atomic_savez(directory / "index.npz", False, keys=keys, meta=json.dumps(store.meta), cameras=json.dumps(store.camera_names),
                 loc=locations, info=json.dumps(store.info), format=DIR_FORMAT)


def _remove_unused_segments(segment_dir: pathlib.Path, locations: np.ndarray):
    """Delete the segment files no entry refers to any more (all their entries were removed)."""
    keep = {f"seg_{int(segment):06d}.npz" for segment in set(locations[:, 0].tolist())}
    for file in segment_dir.glob("seg_*.npz"):
        if file.name not in keep:
            file.unlink()


# ---------------------------------------------------------------------- loading
def load_directory(store, path):
    """Load a store directory into `store`: the index, then every entry's frames from the segment files."""
    directory = pathlib.Path(path)
    locations = _read_index(store, directory)
    store.images = _read_segments(store, directory, locations)
    store.frame_locations = locations
    store.location_directory = directory.resolve()
    return store


def _read_index(store, directory: pathlib.Path) -> np.ndarray:
    """Read index.npz into meta, camera_names, keys and info. Returns the entries' (segment, row) locations."""
    with np.load(directory / "index.npz") as archive:
        store.meta = json.loads(str(archive["meta"]))
        store.camera_names = tuple(json.loads(str(archive["cameras"])))
        locations = archive["loc"].astype(np.int64).reshape(-1, 2)
        store.keys = None
        if len(store.meta):
            store.keys = archive["keys"].astype(np.float32)
        store.info = {}
        if "info" in archive.files:
            store.info = json.loads(str(archive["info"]))
    return locations


def _read_segments(store, directory: pathlib.Path, locations: np.ndarray) -> list[np.ndarray]:
    """Gather every entry's frames from the segment files into one stack per camera, in entry order."""
    count = len(store.meta)
    stacks: list = [None] * len(store.camera_names)
    for number in sorted(set(locations[:, 0].tolist())):
        rows = np.flatnonzero(locations[:, 0] == number)
        with np.load(directory / "segments" / f"seg_{int(number):06d}.npz") as segment:
            for camera_index in range(len(store.camera_names)):
                frames = segment[f"img{camera_index}"]
                if stacks[camera_index] is None:
                    stacks[camera_index] = np.empty((count, *frames.shape[1:]), frames.dtype)
                stacks[camera_index][rows] = frames[locations[rows, 1]]
    return [stack for stack in stacks if stack is not None]


# ---------------------------------------------------------------------- files
def atomic_savez(path: pathlib.Path, compressed: bool, **arrays):
    """np.savez(_compressed) to a temporary file next to `path`, synced to disk, then renamed over `path`: readers see the
    old file or the new one, never a partial one."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as file:
        if compressed:
            np.savez_compressed(file, **arrays)
        else:
            np.savez(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def dir_bytes(path) -> int:
    """Bytes of all files below `path` (a store directory, or one file)."""
    target = pathlib.Path(path)
    if target.is_file():
        return target.stat().st_size
    return sum(file.stat().st_size for file in target.rglob("*") if file.is_file())
