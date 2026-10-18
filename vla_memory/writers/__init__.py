"""Writers: the components that signal that a step of the task is done, so that a method can act on it.

    base.Writer, base.Call      the interface and what a writer sees of a policy call; base.WRITERS, resolve_writer
    oracle.OracleWriter         the simulator's ground truth (memory/oracle_prompt): the reference, simulation only
    oracle.OffWriter            no writer: the policy gets the instruction
    rules.writer.RulesWriter    rules on the robot's gripper and hand signals, declared in the task file
    vlm.writer.VlmWriter        the VLM planner: it plans and monitors every call, with the exemplar store as its memory
    head.writer.HeadWriter      the policy's own event head: its verdict comes with the policy's reply

Every writer keeps the plan and a forward-only pointer; chosen with --writer NAME. Oracle, rules and vlm decide before the
policy call, from the call's inputs, so the prompt of the same call shows the decision.
The event head decides after the call, from the policy's reply, so its decision shows from the next call on.
"""
from .base import WRITERS, Call, Writer, register_writer, resolve_writer

__all__ = ["Writer", "Call", "WRITERS", "resolve_writer", "register_writer"]
