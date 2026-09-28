"""Proprioception summary: the robot's own sensor readings over the last calls, rendered as one sentence.

The planner model receives it with every monitor query. It carries what cameras cannot show reliably (whether the
fingers stopped on an object, how high the hand is, and the trend), and it is available on any real robot: finger
opening from the gripper encoder, hand height from forward kinematics.
"""
from __future__ import annotations


class ProprioSummary:
    def __init__(self, closed_threshold: float = 0.3, history: int = 12, note: str = ""):
        self.closed_threshold = closed_threshold; self.history = history; self.note = note
        self.hist: list[dict] = []

    def reset(self):
        self.hist = []

    def push(self, values: dict | None):
        if values:
            self.hist.append(dict(values)); self.hist = self.hist[-self.history:]

    @property
    def latest(self) -> dict | None:
        return self.hist[-1] if self.hist else None

    def text(self) -> str:
        h = self.hist
        if not h:
            return ""
        c = h[-1].get("gripper_closedness"); z = h[-1].get("hand_height")
        closed = [x.get("gripper_closedness", 0) >= self.closed_threshold for x in h]
        n_closed = 0
        for v in reversed(closed):
            if not v: break
            n_closed += 1
        n_open = 0
        for v in reversed(closed):
            if v: break
            n_open += 1
        ever_closed = any(closed)
        parts = []
        if c is not None:
            state = "closed (fingers stopped on an object)" if c >= self.closed_threshold else "open"
            parts.append(f"gripper {state}, closedness {c:.2f} (0 = fully open); " +
                         (f"closed for the last {n_closed} calls" if n_closed else
                          f"open for the last {n_open} calls" + (", after having been closed earlier" if ever_closed else ", never closed so far")))
        if z is not None and z == z:
            zs = [x.get("hand_height") for x in h[-3:] if x.get("hand_height") == x.get("hand_height")]
            trend = "" if len(zs) < 2 else (f", rising ({100 * (zs[-1] - zs[0]):+.1f} cm over the last {len(zs) - 1} calls)" if zs[-1] - zs[0] > 0.01
                                            else f", descending ({100 * (zs[-1] - zs[0]):+.1f} cm)" if zs[-1] - zs[0] < -0.01 else ", holding height")
            parts.append(f"hand (gripper) height above the table {100 * z:.1f} cm{trend}" + (f"; {self.note}" if self.note else ""))
        return "Robot proprioception (its own joint sensors, reliable): " + "; ".join(parts) + "."
