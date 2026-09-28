"""The planner loop: a VLM planner drives a VLA policy that knows only short skill prompts.

The planner decomposes the user's instruction once into the task's skills, monitors progress at every policy call
from the cameras and the proprioception, and advances a forward-only pointer; the proxy relays observations between
the environment and the policy server and rewrites the prompt to the current skill. Memory: the working memory of
the episode (plan, pointer, recent proprioception) inside Planner, and the exemplar store (labelled frames of past
episodes, retrieved by fingerprint similarity) as long-term memory. See docs/methods/planner_loop/README.md.

  planner.Planner            plan once, monitor every call, hysteresis and forward-only pointer
  proprio.ProprioSummary     the last few gripper / hand-height readings as one sentence
  exemplar.ExemplarStore     the long-term memory: build, save, load, retrieve, few-shot content
  fingerprint.Fingerprint    the retrieval key of an observation (colour-blob key, plain thumbnail, or your own)
  labels.PhaseLabeller       per-frame labels of recorded demonstrations for building a store
  proxy                      the relay and command-line entry point: python -m vla_memory.planner_loop.proxy
"""
from .planner import Planner
from .proprio import ProprioSummary
from .exemplar import ExemplarStore
from .fingerprint import Fingerprint, ColourBlobFingerprint, ThumbnailFingerprint, fingerprint_from_task
from .labels import PhaseLabeller, load_npz_episodes, parse_ranges

__all__ = ["Planner", "ProprioSummary", "ExemplarStore", "Fingerprint", "ColourBlobFingerprint", "ThumbnailFingerprint",
           "fingerprint_from_task", "PhaseLabeller", "load_npz_episodes", "parse_ranges"]
