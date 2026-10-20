"""The session store: the VLM writer's long-term memory on an embedded vector database (LanceDB): the labelled moments of
past episodes, retrieved by a key, with payload filters, provenance and a size cap, growing from the robot's own episodes.
See docs/methods/session_store/README.md.

  extension.SessionStoreExtension   the proxy extension: --with session_store --store-dir DIR [--store-key colour|dinov2|siglip]
  options                           what the options choose from: the keys, the payload filters, the eviction rules
  keys.KeyEncoder                   the keys of a moment: colour (the colour-blob fingerprint), dinov2 and siglip (learned
                                    embeddings; torch and transformers)
  store.SessionStore                the database: the moments with their keys, pictures and payload, insert, nearest-k with
                                    payload filters, eviction, import of an exemplar store file; the framework's MemoryStore
                                    interface (lancedb and pyarrow)
  online                            the write path: the calls of an episode, labelled in hindsight by the rules writer (or the
                                    simulator's prompts), admitted into the store when the episode ends

Importing the package needs none of the heavy packages: the extension imports store.py when it builds the store, and keys.py
imports torch and transformers when a learned key encodes a picture.
"""
from .extension import SessionStoreExtension
from .keys import KeyEncoder
from .options import KEY_NAMES

__all__ = ["SessionStoreExtension", "KeyEncoder", "KEY_NAMES"]
