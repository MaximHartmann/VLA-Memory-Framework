"""The policy's state in a policy server's reply: what a server with a state tap sends with the actions of every call, for the
head writer and the proxy's recording (vla_memory/writers/head).

    reply["policy_state"] = {"names": [name, ...], "values": (n, width) float16}

n summaries of the policy's state at this call, each named by the block and the tokens it averages: "block12:prompt" is
block 12 averaged over the prompt's words. A head is fitted on one of them and finds it by its name.
"""
from __future__ import annotations

import numpy as np

KEY = "policy_state"


def state_entry(names, values) -> dict:
    """The reply's entry for the summaries `values` (n, width), named `names`."""
    values = np.asarray(values, np.float32).reshape(len(names), -1)
    return {"names": [str(name) for name in names], "values": values.astype(np.float16)}


def read_state(reply: dict) -> tuple[list[str], np.ndarray]:
    """The names and the (n, width) float32 values of a reply's policy state; ValueError when the reply has none."""
    entry = reply.get(KEY)
    if not isinstance(entry, dict) or "names" not in entry or "values" not in entry:
        raise ValueError(f"the policy's reply has no {KEY!r}: serve the policy with a state tap (docs/taps/README.md)")
    names = [name.decode() if isinstance(name, bytes) else str(name) for name in entry["names"]]
    return names, np.asarray(entry["values"], np.float32).reshape(len(names), -1)
