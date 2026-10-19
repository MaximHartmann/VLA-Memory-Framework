"""The VLM planner as a writer: a vision-language model decomposes the instruction into the task's skills once, monitors
progress at every policy call from the cameras and the proprioception, and advances a forward-only pointer. Its memory: the
working memory of the episode (plan, pointer, recent proprioception) inside Planner, and the exemplar store (labelled frames
of past episodes, retrieved by fingerprint similarity) as long-term memory. See docs/writers/vlm/README.md.

  writer.VlmWriter           the writer: --writer vlm and the options of options.py
  planner.Planner            plan once, monitor every call, hysteresis and forward-only pointer
  plan                       the plan request, the fallback to the task's skill order, the recovery step
  gate                       the memory-vote gate: the store's own verdict, which may settle a call without the model
  question                   the texts of the plan request and the monitor question
  proprio.ProprioSummary     the last few gripper / hand-height readings as one sentence
  exemplar.ExemplarStore     the long-term memory: build, save, load, retrieve, few-shot content
  exemplar_dir               the store's directory format, for a store that grows between episodes
  labels.PhaseLabeller       per-frame labels of recorded demonstrations for building a store (parse_episode_ids: "0-19,25")
  options                    the command line of the planning model and of the planner's decision rule
The retrieval keys (fingerprints) are shared with other components: vla_memory/fingerprint.py.
"""
from .planner import Planner
from .proprio import ProprioSummary
from .exemplar import ExemplarStore
from .labels import PhaseLabeller, load_npz_episodes, parse_episode_ids, parse_ranges

__all__ = ["Planner", "ProprioSummary", "ExemplarStore", "PhaseLabeller", "load_npz_episodes", "parse_episode_ids",
           "parse_ranges"]
