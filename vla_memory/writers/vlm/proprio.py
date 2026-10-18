"""Proprioception summary: the robot's own sensor readings over the last calls, rendered as one sentence.

The planner model receives it with every monitor query. It carries what cameras cannot show reliably (whether the
fingers stopped on an object, how high the hand is, and the trend), and it is available on any real robot: finger
opening from the gripper encoder, hand height from forward kinematics.
"""
from __future__ import annotations


def _trailing_count(flags: list, value: bool) -> int:
    """How many of the last flags equal `value`, counted from the end up to the first other one."""
    count = 0
    for flag in reversed(flags):
        if flag != value:
            break
        count += 1
    return count


class ProprioSummary:
    """The last few readings of gripper closedness and hand height; text() is the sentence for the planner model."""

    def __init__(self, closed_threshold: float = 0.3, history: int = 12, note: str = ""):
        self.closed_threshold = closed_threshold
        self.history = history
        self.note = note
        self.hist: list[dict] = []

    def reset(self):
        self.hist = []

    def push(self, values: dict | None):
        """Add one call's readings (gripper_closedness, hand_height); None adds nothing. Keeps the last `history` calls."""
        if values:
            self.hist.append(dict(values))
            self.hist = self.hist[-self.history:]

    @property
    def latest(self) -> dict | None:
        return self.hist[-1] if self.hist else None

    def text(self) -> str:
        """The sentence: the gripper part and the height part, each only with a reading; "" before any call."""
        if not self.hist:
            return ""
        parts = []
        gripper = self._gripper_sentence()
        if gripper is not None:
            parts.append(gripper)
        height = self._height_sentence()
        if height is not None:
            parts.append(height)
        return "Robot proprioception (its own joint sensors, reliable): " + "; ".join(parts) + "."

    # ------------------------------------------------------------------ the gripper
    def _gripper_sentence(self) -> str | None:
        """The gripper now (closed or open, its closedness) and for how many calls it has been so; None without a reading."""
        closedness = self.hist[-1].get("gripper_closedness")
        if closedness is None:
            return None
        if closedness >= self.closed_threshold:
            state = "closed (fingers stopped on an object)"
        else:
            state = "open"
        return f"gripper {state}, closedness {closedness:.2f} (0 = fully open); " + self._since_text()

    def _since_text(self) -> str:
        """For how many calls in a row the gripper has been closed, or open (and whether it was ever closed before)."""
        closed = self._closed_flags()
        closed_calls = _trailing_count(closed, True)
        if closed_calls:
            return f"closed for the last {closed_calls} calls"
        open_calls = _trailing_count(closed, False)
        if any(closed):
            return f"open for the last {open_calls} calls, after having been closed earlier"
        return f"open for the last {open_calls} calls, never closed so far"

    def _closed_flags(self) -> list:
        """Per call of the history, whether the gripper counted as closed (closedness at or above the threshold)."""
        return [x.get("gripper_closedness", 0) >= self.closed_threshold for x in self.hist]

    # ------------------------------------------------------------------ the hand height
    def _height_sentence(self) -> str | None:
        """The hand's height above the table, its trend over the last calls and the task's note; None without a reading
        (a NaN height is no reading)."""
        height = self.hist[-1].get("hand_height")
        if height is None or height != height:
            return None
        sentence = f"hand (gripper) height above the table {100 * height:.1f} cm{self._trend_text()}"
        if self.note:
            sentence += f"; {self.note}"
        return sentence

    def _trend_text(self) -> str:
        """Rising, descending or holding height over the last three calls (more than 1 cm counts); "" with one reading."""
        heights = self._recent_heights()
        if len(heights) < 2:
            return ""
        change = heights[-1] - heights[0]
        if change > 0.01:
            return f", rising ({100 * change:+.1f} cm over the last {len(heights) - 1} calls)"
        if change < -0.01:
            return f", descending ({100 * change:+.1f} cm)"
        return ", holding height"

    def _recent_heights(self) -> list:
        """The hand heights of the last three calls, NaN readings left out."""
        return [x.get("hand_height") for x in self.hist[-3:] if x.get("hand_height") == x.get("hand_height")]
