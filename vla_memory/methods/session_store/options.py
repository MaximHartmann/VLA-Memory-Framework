"""What the session store's command-line options choose from. This module imports nothing, so that the command line can offer
the choices without lancedb, pyarrow, torch or transformers installed (store.py is imported when the store is built)."""

KEY_NAMES = ("colour", "dinov2", "siglip")                      # --store-key, --store-keys: what retrieves a moment (keys.py)
FILTERS = ("none", "task", "instruction", "plan", "session")    # --store-filter: what a moment must share with the current call
EVICT = ("fifo", "use_recency")                                 # --store-evict: which moments go above the cap

DEFAULT_KEY = "colour"
DEFAULT_FILTER = "none"
DEFAULT_EVICT = "use_recency"
