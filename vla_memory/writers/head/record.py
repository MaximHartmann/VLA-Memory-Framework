"""Recording for the event head: while another writer runs the robot (the rules writer, the simulator's ground truth), the
proxy keeps what the head is fitted on (--record-head DIR [--record-head-calls 3]).

At every policy call it keeps the policy's state (reply["policy_state"], vla_memory/taps/reply.py) together with the number
of skills the writer counted as complete when the call's input was decided: a row with label 0, "the next skill is not done
yet", as far as this input goes. At the first `calls` calls after the writer completed a skill, it asks the policy once more,
with the same pictures but the input of the call before the switch (the old prompt, the old memory picture): a row with label
1, "done, but not shown yet", the moment the head must recognise. The answer to that extra question goes nowhere; the robot
acts on the real call only. A policy that keeps its own memory would count the question as a call: do not record with one.

One file per episode, <episode>.npz:
    values   (rows, n, width) float16   the policy's state summaries of every row
    rows     (rows, 3) int64            the step, the completed skills the row's input shows, the label
    names    (n,) str                   the summaries' names, as the server sent them
    meta     str (JSON)                 the episode, its instruction and plan, the writer and the method
training/fit_event_head.py --recordings fits a head on these files.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np

from ...taps.reply import read_state


class StateRecorder:
    """The recording of one proxy: start_episode, on_call at every policy call, end_episode (also at the next start)."""

    def __init__(self, directory, calls: int = 3):
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = int(calls)
        self.episode = None

    def start_episode(self, episode: str, meta: dict) -> None:
        """A new episode: the last one is written, the rows start empty."""
        self.end_episode()
        self.episode, self.meta = str(episode), dict(meta)
        self.values, self.rows, self.names = [], [], None
        self.count = 0                # the completed skills the last call's input showed
        self.inputs = None            # the last call's input: the keys the method set
        self.old_inputs, self.old_count, self.questions_left = None, 0, 0

    def on_call(self, observation: dict, inputs: dict, reply: dict, count: int, step, ask) -> dict:
        """One policy call: the observation as the policy got it, the keys of it the method set (`inputs`), the policy's
        reply, the completed skills the input shows, the environment's step and ask(observation) -> reply, the policy.
        Returns fields for the proxy's log."""
        if self.episode is None:
            return {}
        self._add(reply, step, count, 0)
        if count > self.count and self.inputs is not None:
            self.old_inputs, self.old_count, self.questions_left = self.inputs, self.count, self.calls
        asked = self.questions_left > 0
        if asked:
            question = {key: value for key, value in observation.items() if key not in inputs}
            question.update(self.old_inputs)
            self._add(ask(question), step, self.old_count, 1)
            self.questions_left -= 1
        self.count, self.inputs = count, dict(inputs)
        return {"rows": len(self.rows), "asked": asked}

    def _add(self, reply: dict, step, count: int, label: int) -> None:
        """One row: the reply's state summaries, the step, the completed skills shown and the label."""
        names, values = read_state(reply)
        if self.names is None:
            self.names = names
        elif names != self.names:
            raise ValueError(f"the policy server changed its state summaries from {self.names} to {names}")
        self.values.append(values.astype(np.float16))
        self.rows.append((-1 if step is None else int(step), int(count), int(label)))

    def end_episode(self) -> None:
        """Write the episode's file (an episode without a row writes none)."""
        if self.episode is not None and self.rows:
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", self.episode)
            np.savez(self.directory / f"{name}.npz", values=np.stack(self.values), rows=np.asarray(self.rows, np.int64),
                     names=np.array(self.names), meta=json.dumps(self.meta))
        self.episode = None
