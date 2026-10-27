"""The keyframe policy as a method: the policy keeps the plain instruction as its prompt, and the raw wrist frame of the call
at which the writer completed a skill goes to the policy as its third picture from then on.

    --method keyframe [--keyframe-mode normal|off] [--keyframe-max-events N]
                      [--keyframe-placed-later [--keyframe-placed-height M] [--keyframe-placed-calls N]]

The events are the writer's completed skills (picked up red, placed red, picked up blue in the reference task). At every
call, after the writer's before_call: when the count has grown since the keyframe was taken, the call's raw wrist frame
(memory/wrist_raw, 240 x 320 x 3 uint8 in the reference bench, the kind of picture the training keyframes are; without it
the policy's wrist image, reported once) becomes the keyframe, and from then on every observation carries it as
observation/memory_image, which the policy reads in its third image slot (the dataset: training/relabel_keyframe.py). Before
the first event nothing is sent and the slot stays masked out, as on the training frames before the first event. A new
episode clears the keyframe; events after the N-th (default: the task's history.max_lines) do not replace it, and neither
does the last skill of the episode's plan, which has no event in the training data. Mode off is the no-keyframe control: the
keyframe is taken and logged, never sent. With --keyframe-placed-later the keyframe of a placing event (a skill of one of the
task's placing verbs) waits for the hand to rise (placed.py).

Every proxy log record carries keyframe_mode and, under "keyframe": source (the writer), events (the count, capped as above),
stored (a keyframe was taken at this call), event (the event of the current keyframe, None before the first), step
(memory/step of the call it was taken at), sent (observation/memory_image went to the policy), and frame ("raw" or
"fallback") at a call that stores one. With --keyframe-placed-later also placed_later: height (m above a resting cube, None
without a joint state) and, while the placing event waits or is taken, pending_calls, signal_step and reason (waiting,
height or fallback); dropped_pending when a later event replaced a waiting one.
"""
from __future__ import annotations

import logging

import numpy as np

from ..base import Method, most_shown
from ..memory_policy.query import wrist_frame
from . import placed
from .placed import PlacedLater

log = logging.getLogger("vla_memory.methods.keyframe")
MODES = ("normal", "off")
MEMORY_IMAGE = "observation/memory_image"
DEFAULT_WRIST_IMAGE_KEY = "observation/wrist_image"     # the proxy's default second --image-keys entry


class KeyframeMethod(Method):
    """The plain instruction as the prompt, and the wrist frame of the last event as observation/memory_image (module
    docstring). wrist_image_key: the policy's wrist image, used when the environment sends no raw wrist frame."""

    name = "keyframe"

    # ------------------------------------------------------------------ command line and construction
    @classmethod
    def add_arguments(cls, parser):
        """The keyframe options: the mode, the event cap and the later placed keyframe."""
        group = parser.add_argument_group("the keyframe policy (--method keyframe): the wrist picture of the last key "
                                          "moment as the third picture")
        group.add_argument("--keyframe-mode", choices=list(MODES), default="normal",
                           help="normal: send the keyframe (default); off: take and log it, send nothing (the control)")
        group.add_argument("--keyframe-max-events", type=int, default=None, metavar="N",
                           help="events that take a new keyframe (default: the task's history.max_lines)")
        group.add_argument("--keyframe-placed-later", action="store_true",
                           help="take the placing event's keyframe at the first call from its signal on with the fingertips "
                                "at least --keyframe-placed-height above the placed cube, at the latest at the "
                                "--keyframe-placed-calls-th call after the signal (default: at the signal)")
        group.add_argument("--keyframe-placed-height", type=float, default=placed.HEIGHT, metavar="M",
                           help=f"with --keyframe-placed-later: the fingertips' height above the cube in m (default "
                                f"{placed.HEIGHT})")
        group.add_argument("--keyframe-placed-calls", type=int, default=placed.CALLS, metavar="N",
                           help=f"with --keyframe-placed-later: the latest call after the signal (default {placed.CALLS})")

    @classmethod
    def build(cls, args, task):
        """The method with the options; the event cap defaults to the task's history.max_lines."""
        image_keys = getattr(args, "image_keys", None) or ("observation/image", DEFAULT_WRIST_IMAGE_KEY)
        max_events = args.keyframe_max_events
        if max_events is None:
            max_events = (task.history or {}).get("max_lines", len(task.skills) - 1)
        placed_later = None
        if args.keyframe_placed_later:
            placed_later = PlacedLater.for_task(task, args.keyframe_placed_height, args.keyframe_placed_calls)
        wrist_image_key = image_keys[1] if len(image_keys) > 1 else None
        return cls(task, args.keyframe_mode, max_events, wrist_image_key, placed_later)

    def __init__(self, task=None, mode: str = "normal", max_events: int = 3,
                 wrist_image_key: str | None = DEFAULT_WRIST_IMAGE_KEY, placed_later: PlacedLater | None = None):
        if mode not in MODES:
            raise ValueError(f"keyframe mode {mode!r}: one of {MODES}")
        super().__init__(task)
        self.mode = mode
        self.max_events = int(max_events)
        self.wrist_image_key = wrist_image_key
        self.placed_later = placed_later
        self.taken = 0             # keyframes taken in the whole run
        self._warned: set = set()
        self.start_episode(None, {})

    def start_episode(self, episode, control_keys) -> None:
        """A new episode: no keyframe yet."""
        self.keyframe = None       # the raw wrist frame of the last event
        self.event = 0             # the event it belongs to (0: none yet)
        self.step = None           # the control step of the call it was taken at
        if self.placed_later is not None:
            self.placed_later.reset()

    def config(self) -> dict:
        """The mode, the event cap and the placed-later rule, for the metadata."""
        config = {"mode": self.mode, "max_events": self.max_events}
        if self.placed_later is not None:
            config["placed_later"] = self.placed_later.config()
        return config

    def metadata(self) -> dict:
        """The configuration under "keyframe"."""
        return {"keyframe": self.config()}

    # ------------------------------------------------------------------ every policy call
    def prompt(self, instruction: str, writer) -> str:
        """The plain instruction: the keyframe policy reads its memory from the picture, not the prompt."""
        return instruction

    def before_policy(self, observation: dict, control_keys: dict, writer) -> dict:
        """Take a keyframe when the event count has grown, and send the current one; the log record goes under "keyframe"."""
        events = min(len(writer.completed()), most_shown(self.max_events, writer))
        record = {"source": writer.name, "events": events, "stored": False}
        if self.placed_later is None:
            if events > self.event:
                record.update(self._take(observation, control_keys, events))
        else:
            record.update(self._take_later(observation, control_keys, events, writer))
        sent = self.keyframe is not None and self.mode == "normal"
        if sent:
            observation[MEMORY_IMAGE] = self.keyframe
        record.update(event=self.event or None, step=self.step, sent=sent)
        return {"keyframe_mode": self.mode, "keyframe": record}

    def _take(self, observation: dict, control_keys: dict, events: int) -> dict:
        """The call's wrist frame becomes the keyframe of the newest event (without a frame: tried at the next call)."""
        frame, fallback = wrist_frame(observation, control_keys, self.wrist_image_key)
        if frame is None:
            self._warn_once("frame", "neither memory/wrist_raw nor the policy's wrist image is sent: no keyframe")
            return {}
        if fallback:
            self._warn_once("fallback", f"memory/wrist_raw is not sent: the keyframe is the policy's "
                                        f"{self.wrist_image_key} (not the raw frame the training keyframes are)")
        self.keyframe = np.array(frame, copy=True)
        self.event = events
        self.step = None if control_keys.get("step") is None else int(control_keys["step"])
        self.taken += 1
        return {"stored": True, "frame": "fallback" if fallback else "raw"}

    def _take_later(self, observation: dict, control_keys: dict, events: int, writer) -> dict:
        """As _take, except for a placing event, whose keyframe waits for the hand; a later event drops a waiting one."""
        rule = self.placed_later
        height = rule.fingertip_height(observation)
        if height is None:
            self._warn_once("state", "no joint state (7 arm joints in observation/state): the placing event's keyframe is "
                                     "taken by the fallback only")
        info = {"height": height}
        fields = {}
        if events > self.event and rule.is_placing(events, writer, self.task):
            fields.update(self._wait_for_hand(observation, control_keys, events, height, info))
        elif events > self.event:
            if rule.pending is not None:
                info["dropped_pending"] = dict(rule.pending)
                rule.reset()
            fields.update(self._take(observation, control_keys, events))
        fields["placed_later"] = info
        return fields

    def _wait_for_hand(self, observation: dict, control_keys: dict, events: int, height, info: dict) -> dict:
        """One call of the waiting placing event: take its keyframe when the rule says so, else keep the previous one."""
        rule = self.placed_later
        step = None if control_keys.get("step") is None else int(control_keys["step"])
        reason = rule.wait(step, height)
        info.update(pending_calls=rule.pending["calls"], signal_step=rule.pending["signal_step"], reason=reason)
        if reason == "waiting":
            return {}
        taken = self._take(observation, control_keys, events)
        if taken:
            rule.reset()           # without a frame it stays pending and is tried again at the next call
        return taken

    def _warn_once(self, topic: str, message: str) -> None:
        """A warning once per topic and run."""
        if topic not in self._warned:
            log.warning("keyframe: %s", message)
            self._warned.add(topic)

    def close(self) -> None:
        """The proxy stops: log how many keyframes were taken."""
        log.info("keyframe: %d keyframes taken (mode %s)", self.taken, self.mode)
