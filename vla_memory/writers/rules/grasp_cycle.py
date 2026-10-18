"""The grasp cycle: a state machine on the gripper signals that fires "lift" and "set_down", for pick and place.

It is followed at every control step (width: the finger distance in m, effort: the squeeze, fingertip: x, y, z in m):
    free      -> holding   the fingers stopped between empty_width and open_width and squeeze above squeeze_force
    holding   -> lifted    the fingertips rose `rise` above the point where the grasp began, still squeezing   => "lift"
                           (with `at_top`: only once they also rise less than at_top per step, i.e. at the top of the lift)
    lifted    -> released  the fingers opened (width >= open_width) less than `release_below` above the grasp point
                           (set down, or let go just above the table); opening higher up is a drop and ends the cycle
    released  -> free      the fingertips moved `up` above or `away` from the release point                     => "set_down"
A touch after the release (squeeze, then open again without a lift) keeps the set-down pending.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class GraspCycle:
    """The per-step state machine of one grasp cycle; step() returns "lift", "set_down" or None."""

    def __init__(self, open_width: float, empty_width: float, squeeze_force: float, rise: float,
                 release_below: float, up: float, away: float | None, at_top: float | None = None):
        self.open_width = open_width
        self.empty_width = empty_width
        self.squeeze_force = squeeze_force
        self.rise = rise
        self.release_below = release_below
        self.up = up
        self.away = away
        self.at_top = at_top
        self.reset()

    def reset(self):
        self.state = "free"
        self.z_grasp = None       # fingertip height where the grasp began
        self.z_rel = None         # fingertip height where the fingers opened after a lift
        self.p_rel = None         # fingertip position at that release
        self.pending = False      # a set-down waits for the retreat from the release point
        self.z_prev = None        # fingertip height at the previous step (for at_top)

    def step(self, width: float, effort: float, fingertip: Sequence[float]) -> str | None:
        """One control step. Returns "lift", "set_down" or None."""
        fingertip = np.asarray(fingertip, float)
        height = float(fingertip[2])
        trigger = None
        if self.state == "free":
            trigger = self._step_free(width, effort, height)
        elif self.state == "holding":
            trigger = self._step_holding(width, effort, height, fingertip)
        elif self.state == "lifted":
            trigger = self._step_lifted(width, height, fingertip)
        elif self.state == "released":
            trigger = self._step_released(width, effort, height, fingertip)
        self.z_prev = height
        return trigger

    def _step_free(self, width, effort, height):
        """Waiting for a grasp: the fingers stop on something and squeeze."""
        if self._squeezing(width, effort):
            self.z_grasp = height
            self.state = "holding"
        return None

    def _step_holding(self, width, effort, height, fingertip):
        """Squeezing at the grasp point: the lift fires once the fingertips rose `rise` above it; letting go ends the cycle."""
        if width >= self.open_width or width <= self.empty_width:
            if self.pending:
                # touched the block that was just set down, and let go again: the set-down stays pending
                self._release_at(height, fingertip)
            else:
                self.state = "free"
            return None
        if self._lift_complete(effort, height):
            self.state = "lifted"
            self.pending = False
            return "lift"
        return None

    def _step_lifted(self, width, height, fingertip):
        """Carrying: opening less than `release_below` above the grasp point is a set-down, opening higher up a drop."""
        if width >= self.open_width:
            if height - self.z_grasp < self.release_below:
                self._release_at(height, fingertip)
                self.pending = True
            else:
                self.state = "free"   # dropped from height: no set-down
        elif width <= self.empty_width:
            self.state = "free"   # closed fully: the object slipped out
        return None

    def _step_released(self, width, effort, height, fingertip):
        """Open after a set-down: the retreat fires set_down; squeezing again (a touch) keeps it pending."""
        if height - self.z_rel >= self.up or self._moved_away(fingertip):
            self.state = "free"
            self.pending = False
            return "set_down"
        if self._squeezing(width, effort):
            self.z_grasp = height
            self.state = "holding"
        return None

    def _squeezing(self, width, effort) -> bool:
        """The fingers stopped on something (between empty and open) and press on it."""
        return self.empty_width < width < self.open_width and effort > self.squeeze_force

    def _lift_complete(self, effort, height) -> bool:
        """Still squeezing and `rise` above the grasp point; with at_top, also rising less than at_top per step."""
        risen = height - self.z_grasp >= self.rise and effort > self.squeeze_force
        if not risen:
            return False
        if self.at_top is None:
            return True
        return self.z_prev is not None and height - self.z_prev < self.at_top

    def _moved_away(self, fingertip) -> bool:
        """At least `away` from the release point in any direction (only with finite positions)."""
        if self.away is None or self.p_rel is None:
            return False
        if not (np.all(np.isfinite(fingertip)) and np.all(np.isfinite(self.p_rel))):
            return False
        return float(np.linalg.norm(fingertip - self.p_rel)) >= self.away

    def _release_at(self, height, fingertip):
        """The fingers opened: remember the release point and wait for the retreat."""
        self.z_rel = height
        self.p_rel = fingertip
        self.state = "released"
