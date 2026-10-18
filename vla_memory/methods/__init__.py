"""Methods: what the policy receives, and the memories that act on it. base.Method is the interface of the methods that act
on a writer's signal, base.METHODS their registry; one of them is chosen with --method NAME:
    subgoal.SubgoalMethod          the current skill as the prompt
    history.method.HistoryMethod   the instruction with a line per completed skill (the history-appended prompt)
    keyframe.method.KeyframeMethod the instruction, and the wrist picture of the last key moment as a third picture

Other memories plug in with --with NAME (proxy/extensions.py):
    memory_policy                  a store of moments of recorded episodes, the nearest one handed to the policy
    notebook                       the preference notebook about the user, read by the VLM planner
"""
from .base import METHODS, Method, register_method, resolve_method

__all__ = ["Method", "METHODS", "resolve_method", "register_method"]
