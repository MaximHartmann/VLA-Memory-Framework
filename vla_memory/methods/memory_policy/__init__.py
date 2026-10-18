"""The memory policy: a VLA policy that reads a memory retrieved for it at every call.

At every policy call the planner loop's proxy retrieves the most similar stored moment of an earlier episode (the store:
moments of demonstrations, of the policy's own episodes and of expert corrections) and hands it to the policy next to the
two camera images: the stored moment's WRIST picture as a third image (observation/memory_image) and, appended to the
prompt, one sentence that says how the robot moved during the moment's next actions ("pick up the red block. memory: release
at step 1, up 2 cm"). The policy was fine-tuned on frames labelled the same way, so it can use the hint and stays in control.
The retrieval key is the colour fingerprint of the raw wrist frame; hard filters keep the same skill and gripper state, and
a gripper closed on nothing is only shown moments that release. See docs/methods/memory_policy/README.md.

  extension.MemoryPolicyExtension   the proxy extension: --with memory_policy --memory-store DIR [--memory-mode ...]
  store.MomentStore, save_store     the store directory: its files and entry fields, reading and writing it
  retrieve.neighbour                the retrieval rule: hard filters, then the most similar entry (k = 1)
  key.WristKey                      the retrieval key of a raw wrist frame (the task's colour fingerprint, length 1)
  pictures.Pictures                 the store's wrist pictures, decoded when first needed, a bounded cache
  query                             what a call provides: the raw wrist frame, the finger width, the skill
  sentence.sentence                 the memory sentence of a moment from its joints, its next actions and its finger width
  prompt.with_memory, strip_memory  the prompt with the memory's words, and back
  gripper.gripper_state             open / closed on an object / closed on nothing from the measured finger width
  kinematics.FrankaKinematics       the TCP position of the Franka arm (forward kinematics), for the sentence's movement
  build.Episode, episode_entries, label_episode, no_memory
                                    the store builder and the training labels, from recorded episodes given as arrays
                                    (used by training/relabel_memory.py, which reads and writes the datasets)
"""
from .extension import MODES, MemoryPolicyExtension
from .gripper import STATES, gripper_state
from .key import WristKey
from .kinematics import FrankaKinematics
from .prompt import MEMORY_MARKER, strip_memory, with_memory
from .retrieve import RULES, neighbour
from .sentence import HORIZON, is_release, sentence
from .store import MomentStore, save_store

__all__ = ["MemoryPolicyExtension", "MODES", "MomentStore", "save_store", "neighbour", "RULES", "WristKey", "sentence",
           "is_release", "HORIZON", "with_memory", "strip_memory", "MEMORY_MARKER", "gripper_state", "STATES",
           "FrankaKinematics"]
