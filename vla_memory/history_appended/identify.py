"""Which object is in the gripper: the "what" of an event, asked only at the moments the rules writer detects a pick.

The rules writer knows WHEN an object was picked up from the robot's own signals; it does not know WHICH object. Without an
identifier it assumes the plan order. An identifier looks at the wrist camera at the pick and names the held object, so the
history states what really happened and tasks without a fixed order become possible. Because it runs only at events (two to
four times per episode), even a slow identifier is affordable.

    ColourIdentifier   counts each object's colour (the task file's fingerprint colour rules) in the region of the wrist
                       image where a held object appears (history.held_region, fractions of height and width); the colour that
                       fills at least history.held_min_share of the region names the object. For distinctly coloured objects.
Further identifiers implement ObjectIdentifier.identify (a VLM question restricted to the task's objects, an image embedding
compared with reference crops). The task file picks one with history.object_from: plan_order (default) | colour.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..planner_loop.fingerprint import colour_mask


class ObjectIdentifier:
    def identify(self, image) -> str | None:
        """The task object held in the gripper, or None if it cannot tell."""
        raise NotImplementedError


class ColourIdentifier(ObjectIdentifier):
    def __init__(self, task, region=None, min_share=None, colours=None):
        h = task.history or {}
        self.colours = {k: [tuple(r) for r in v] for k, v in (colours or (task.fingerprint or {}).get("colours", {})).items()}
        missing = [o for o in task.objects if o not in self.colours]
        if missing:
            raise ValueError(f"ColourIdentifier: no colour rules for object(s) {missing} (task file: fingerprint.colours)")
        self.region = tuple(region or h.get("held_region", (0.30, 0.60, 0.33, 0.67)))   # row0, row1, col0, col1
        self.min_share = float(min_share if min_share is not None else h.get("held_min_share", 0.25))
        self.objects = list(task.objects)

    def shares(self, image) -> dict[str, float]:
        """Share of the held-object region filled by each object's colour."""
        img = np.asarray(image); H, W = img.shape[:2]; r0, r1, c0, c1 = self.region
        reg = img[int(r0 * H):int(r1 * H), int(c0 * W):int(c1 * W)]
        return {o: float(colour_mask(reg, self.colours[o]).mean()) for o in self.objects}

    def identify(self, image) -> str | None:
        s = self.shares(image); best = max(s, key=s.get)
        return best if s[best] >= self.min_share else None


def make_identifier(task) -> ObjectIdentifier | None:
    """The identifier the task file asks for (history.object_from), or None for the plan order."""
    kind = (task.history or {}).get("object_from", "plan_order")
    if kind == "plan_order":
        return None
    if kind == "colour":
        return ColourIdentifier(task)
    raise ValueError(f"history.object_from: unknown identifier {kind!r} (plan_order | colour)")


def validate_identifier(identifier: ObjectIdentifier, samples: Iterable[tuple]) -> dict:
    """samples = (image, true object or None); returns counts of correct, wrong and unknown answers per true object."""
    out: dict = {}
    for image, truth in samples:
        got = identifier.identify(image); d = out.setdefault(str(truth), {"correct": 0, "wrong": 0, "unknown": 0})
        d["unknown" if got is None else ("correct" if got == truth else "wrong")] += 1
    return out
