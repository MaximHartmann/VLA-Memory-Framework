"""The keyframe policy: a VLA policy whose third image slot shows the wrist picture of the last key moment of its episode.

The policy keeps the plain instruction as its prompt. The writer (writers/) signals the key moments, the events of the task
(picked up red, placed red, picked up blue in the reference task): the simulator's ground truth or the rules writer in the
evaluation, any writer in principle. At the call at which an event is signalled, the method keeps that call's raw wrist
frame and sends it from then on as observation/memory_image, which the policy's input transform puts into its third image
slot; before the first event the slot stays empty. The policy was fine-tuned on demonstrations relabelled the same way,
each frame with the wrist picture of a frame drawn from the 16 frames after its last event, never later than the frame
itself. See docs/methods/keyframe/README.md.

  method.KeyframeMethod   the method: --method keyframe [--keyframe-mode off] [--keyframe-placed-later]
  placed.PlacedLater      the option that takes the placing event's keyframe once the hand has risen
  labels                  the training labels: history_counts, boundaries, source_frame, label_episode (used by
                          training/relabel_keyframe.py, which reads and writes the datasets)
"""
from .labels import WINDOW, boundaries, history_counts, label_episode, source_frame
from .method import MODES, KeyframeMethod
from .placed import PlacedLater

__all__ = ["KeyframeMethod", "MODES", "PlacedLater", "WINDOW", "boundaries", "history_counts", "label_episode",
           "source_frame"]
