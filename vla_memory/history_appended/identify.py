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
        history = task.history or {}
        colour_rules = colours or (task.fingerprint or {}).get("colours", {})
        self.colours = {}
        for name, ranges in colour_rules.items():
            self.colours[name] = [tuple(colour_range) for colour_range in ranges]
        missing = [name for name in task.objects if name not in self.colours]
        if missing:
            raise ValueError(f"ColourIdentifier: no colour rules for object(s) {missing} (task file: fingerprint.colours)")
        self.region = tuple(region or history.get("held_region", (0.30, 0.60, 0.33, 0.67)))   # row0, row1, col0, col1
        if min_share is None:
            min_share = history.get("held_min_share", 0.25)
        self.min_share = float(min_share)
        self.objects = list(task.objects)

    def shares(self, image) -> dict[str, float]:
        """Share of the held-object region filled by each object's colour."""
        picture = np.asarray(image)
        height, width = picture.shape[:2]
        row0, row1, col0, col1 = self.region
        region = picture[int(row0 * height):int(row1 * height), int(col0 * width):int(col1 * width)]
        shares = {}
        for name in self.objects:
            shares[name] = float(colour_mask(region, self.colours[name]).mean())
        return shares

    def identify(self, image) -> str | None:
        shares = self.shares(image)
        best = max(shares, key=shares.get)
        if shares[best] >= self.min_share:
            return best
        return None


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
        got = identifier.identify(image)
        counts = out.setdefault(str(truth), {"correct": 0, "wrong": 0, "unknown": 0})
        if got is None:
            counts["unknown"] += 1
        elif got == truth:
            counts["correct"] += 1
        else:
            counts["wrong"] += 1
    return out
