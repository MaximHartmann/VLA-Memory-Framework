"""Planner proxy: the relay between an environment client and the VLA policy server.

    environment  --ws-->  PlannerProxy (:LISTEN)  --ws-->  policy server (:UPSTREAM)
                                |  HTTP (OpenAI-compatible chat API)
                                v
                          planner model (vLLM, ...)

The environment sends its ordinary observation with the FULL instruction as `prompt`, plus control keys under the
`memory/` prefix (see PlannerProxy below); the proxy strips them, asks the planner for the current skill, writes it into
`prompt`, forwards the observation and returns the policy's reply with an extra `planner` info dict. Neither the
environment nor the policy server needs to know about the planner.

Modes:  vlm      the planner decides the prompt (plan + monitor)
        oracle   forwards memory/oracle_prompt (a ground-truth reference planner, for evaluation)
        off      passthrough
A method's proxy may add modes (MODES, build_monitor, decide): the history-appended proxy adds `rules`.

The planner's defaults are the settled configuration of the reference evaluation: the memory-vote gate (with --memory; the
6 nearest stored moments vote, one "done" vote asks the model, the last skill is always asked), a verdict-only answer, 4
examples per question, votes 1, min-calls 2. --no-gate, --no-verdict-only (with --brief: no reason field), --k, --k-vote,
--gate-min-done, --votes and --min-calls change them.

Online memory (vlm mode): --online-store DIR makes the exemplar memory grow from the agent's own episodes (online.py). The
proxy hands every call (frames, gripper signals, proprioception, oracle prompt, the planner's decision) to the store; when
the next episode starts, and when the proxy is stopped (SIGTERM, as the launchers do), the store labels the finished
episode (--online-labeller rules | planner | oracle), admits it (--online-admit all | complete | none), leaves out the calls
after the task is complete (--online-terminal) and near-duplicates (--online-dedup), evicts above --online-max-entries
(--online-evict), and saves the directory; a later run with the same DIR resumes it, and --memory seeds a new DIR. The
planner log gets one store_update record per episode and every monitor record the store size (memory_entries) and the
retrieval time; the proxy log gets memory_entries. Without --online-store nothing changes.

Keyframe memory (vlm mode): --keyframes N shows the monitor labelled frames of earlier moments of the running episode, by
default the start and every switch of the plan pointer (keyframes.py; --keyframe-policy, --keyframe-cameras,
--keyframe-render, --max-images-per-prompt); every monitor record lists the keyframes shown. Without --keyframes nothing
changes.

Reflection memory (vlm mode, with --profile): --reflect lets the robot write lessons into the notebook itself
(vla_memory.reflection). The proxy collects the gripper signals of every call; when an episode has ended (the next one
starts, or the proxy is stopped) the judge decides from the robot's own records whether it failed and which decision of the
planner caused it (--reflect-judge signals: a hindsight check with the task's rules writer, default; plan: the decision log
only), the lessons applied in the episode are voted, and after a failure the planner caused the planning model writes one
lesson for that skill (source "self"), which the planner reads from the next episode on. The planner log gets one
reflection record per episode. Without --reflect nothing changes.

World memory (vlm mode): --world-state, --world-verify and --world-store keep a structured symbolic memory of the episode
(vla_memory.world): object states, positions, events and completed skills as facts with validity intervals, written by the
proxy at every call from the gripper signals (memory/proprio_steps), the wrist camera's colour identifier and the external
camera's colour blobs (mapped to the table with --world-calibration), without a model. --world-state adds its state block to
every monitor question; --world-verify lets a "done" move the plan pointer only if the world state agrees; --world-store
writes every episode's facts to an SQLite file for queries across episodes (python -m vla_memory.world.query). The planner log
gets one world_update record per episode. Without these options nothing changes.

Action memory (any mode): --action-memory STORE makes the proxy pass the policy's reply through the action-memory hook
(vla_memory.action_memory) before returning it: --action-mode recover (default) replaces the action chunk with the memory's
stall recovery (release, clear, re-targeted chunks of earlier successful episodes) from a stall trigger until the skill changes;
--action-mode blend mixes every chunk with the memory's (--action-blend). Object positions come from the external camera's
colour blobs through --action-calibration (default --world-calibration). --action-write complete|all adds the policy's own
chunks of finished episodes to the store and saves it. Every proxy log record gets an action_memory field. Without
--action-memory nothing changes.

The frontend speaks the openpi websocket protocol (msgpack) because that is what openpi environment clients use;
another frontend replaces ProxyServer and keeps PlannerProxy.

Components (vla_memory/plugins.py): --vlm-backend NAME picks the planning model's client (default openai: OpenAICompatibleVLM on
--vlm-url and --vlm-model; or a registered name, an entry point of the group vla_memory.vlm_backends, or module:Class), with
--vlm-arg KEY=VALUE for its own options and --vlm-api-key (default: the environment variable VLM_API_KEY). --transport NAME picks
the connection to the policy (default openpi: OpenPIWebsocketTransport on --upstream-host and --upstream-port; or a registered
name, an entry point of vla_memory.transports, or module:Class), with --transport-arg KEY=VALUE. The defaults build exactly the
client and transport of earlier versions.
"""
from __future__ import annotations

import argparse
import asyncio
import http
import json
import logging
import signal
import time
import traceback

import numpy as np

from .planner import Planner
from ..plugins import construct, key_value
from ..task import TaskSpec
from ..transport import OpenPIWebsocketTransport, PolicyTransport, resolve_transport
from ..vlm import OpenAICompatibleVLM, VLMBackend, resolve_api_key, resolve_vlm_backend

log = logging.getLogger("vla_memory.planner_loop.proxy")
CTRL = "memory/"


class PlannerProxy:
    MODES = ("vlm", "oracle", "off")

    def __init__(self, transport: PolicyTransport, mode: str, planner: Planner | None, log_path=None,
                 image_keys=("observation/image", "observation/wrist_image"), task: TaskSpec | None = None, reflection=None):
        # planner = the monitor that keeps the plan and its pointer: the VLM planner, or another monitor a method supplies
        self.transport = transport; self.mode = mode; self.planner = planner; self.image_keys = tuple(image_keys)
        self.reflection = reflection   # reflection memory (vla_memory.reflection.Reflection, --reflect): lessons after failures
        self.task = task if task is not None else getattr(planner, "task", None)
        log.info("upstream metadata: %s", transport.metadata())
        self._log = open(log_path, "a") if log_path else None
        self.episode = None; self.n = 0
        mem = getattr(planner, "memory", None)   # a memory that learns from the run (online.OnlineStore) is fed by the proxy
        self.online = mem if callable(getattr(mem, "record", None)) and callable(getattr(mem, "end_episode", None)) else None
        self.world = getattr(planner, "world", None)   # world memory (vla_memory.world.WorldMemory): written here at every call
        self.action_memory = None   # action-memory hook (vla_memory.action_memory, --action-memory): None = off, the reply passes as is

    @classmethod
    def add_arguments(cls, ap: argparse.ArgumentParser):
        """Command-line options of a method's proxy (none here)."""

    def configure(self, args):
        """Apply those options after construction (nothing here)."""

    @classmethod
    def build_monitor(cls, args, task: TaskSpec):
        """The monitor for the chosen mode (None: the prompt needs no monitor)."""
        return build_planner(args, task) if args.mode == "vlm" else None

    def prompt_for(self, instruction: str, rec: dict) -> str:
        """The prompt handed to the policy after a planner decision: here the current skill. Other methods override."""
        return rec.get("skill_after", self.planner.current_skill)

    def oracle_prompt_for(self, instruction: str, oracle_prompt: str) -> str:
        """The prompt in oracle mode: the environment's ground-truth current skill, forwarded as it is."""
        return oracle_prompt

    def decide(self, instruction: str, ctrl: dict, obs: dict) -> tuple[str, dict]:
        """The prompt of this call and the fields it adds to the log record."""
        if self.mode == "vlm":
            ext = ctrl.get("exterior_raw", obs.get(self.image_keys[0]))
            wri = ctrl.get("wrist_raw", obs.get(self.image_keys[1]) if len(self.image_keys) > 1 else None)
            images = [np.asarray(ext)] + ([np.asarray(wri)] if wri is not None else [])
            proprio = {k: float(ctrl[k]) for k in ("gripper_closedness", "hand_height") if k in ctrl}
            if self.world is not None:   # the world state of this call, before the planner decides
                self.world.observe(ctrl, images, self.episode, skill=self.planner.current_skill)
            rec = self.planner.observe(images, int(ctrl.get("step", -1)), proprio=proprio or None)
            if self.online is not None:
                self.online.record(ctrl, images, rec, self.planner)
            return self.prompt_for(instruction, rec), dict(
                queried=rec["queried"], advanced=rec["advanced"], vlm_ms=rec.get("ms", 0.0), skill_index=self.planner.idx,
                vlm=rec.get("vlm"), recovery=rec.get("recovery_inserted"))
        if self.mode == "oracle":
            if "oracle_prompt" not in ctrl and not getattr(self, "_warned_oracle", False):
                log.warning("oracle mode without memory/oracle_prompt: forwarding the instruction"); self._warned_oracle = True
            op = ctrl.get("oracle_prompt", instruction); op = op.decode() if isinstance(op, bytes) else op
            return self.oracle_prompt_for(instruction, op), {}
        return instruction, {}

    def infer(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        ctrl = {k[len(CTRL):]: v for k, v in obs.items() if isinstance(k, str) and k.startswith(CTRL)}
        obs = {k: v for k, v in obs.items() if not (isinstance(k, str) and k.startswith(CTRL))}
        task = obs.get("prompt", ""); task = task.decode() if isinstance(task, bytes) else task
        info = {"mode": self.mode, "queried": False, "advanced": False, "vlm_ms": 0.0}
        eid = ctrl.get("episode"); eid = eid.decode() if isinstance(eid, bytes) else eid
        if ctrl.get("new_episode") or self.episode is None or (eid is not None and str(eid) != self.episode):
            if self.online is not None:
                self.store_update()   # the episode that just ended goes into the memory before the next one starts
            if self.reflection is not None:
                self.reflect_episode()   # judged by the robot itself; a lesson after a failure, read from the next episode on
            if self.world is not None:
                self.world_update()   # the finished episode's facts go to the world store
            self.episode = str(eid if eid is not None else f"ep{self.n}"); self.n += 1
            if self.world is not None:
                self.world.start_episode(self.episode, int(ctrl.get("step", 0)))
            if self.planner is not None:
                rec = self.planner.start_episode(task, self.episode)
                info.update(plan=rec["plan"], plan_source=rec["source"], plan_ms=rec.get("ms"))
                log.info("episode %s plan (%s): %s", self.episode, rec["source"], rec["plan"])
        prompt, extra = self.decide(task, ctrl, obs)
        info.update(extra)
        if self.reflection is not None:
            self.reflection.record(ctrl)
        if self.online is not None:
            info["memory_entries"] = len(self.online)
        obs["prompt"] = prompt
        info["prompt"] = prompt; info["oracle_prompt"] = ctrl.get("oracle_prompt")
        t1 = time.perf_counter(); result = self.transport.infer(obs); t_up = time.perf_counter() - t1
        if self.action_memory is not None:   # may replace or blend the action chunk (--action-memory)
            info["action_memory"] = self.action_memory.on_reply(obs, ctrl, result, self.memory_skill(prompt), self.episode)
        info["upstream_ms"] = 1000 * t_up; info["total_ms"] = 1000 * (time.perf_counter() - t0)
        result["planner"] = info
        if self._log:
            self._log.write(json.dumps({"episode": self.episode, "step": ctrl.get("step"), **info}, default=str) + "\n"); self._log.flush()
        return result

    def memory_skill(self, prompt: str):
        """The skill in force for the action memory: the prompt if it is a skill (oracle and planner modes), else the monitor's
        current skill (the history proxy's writers), else None."""
        skills = list(self.task.skills) if self.task is not None else []
        if prompt in skills:
            return prompt
        p = self.planner
        if p is not None and getattr(p, "plan", None):
            return p.plan[min(getattr(p, "idx", 0), len(p.plan) - 1)]
        return None

    def store_update(self) -> dict | None:
        """The online store labels, admits and saves the recorded episode; its store_update record goes to the planner's log."""
        rec = self.online.end_episode()
        if rec is not None:
            log_event = getattr(self.planner, "log_event", None)
            if log_event is not None:
                log_event(rec)
            log.info("memory update after %s: %s of %d calls added, %d removed, %d entries, %.0f ms", rec["episode"],
                     rec["added"], rec["calls"], rec["removed"], rec["size"], rec["ms"]["total"])
        return rec

    def reflect_episode(self) -> dict | None:
        """The reflection memory judges the episode that just ended (and may write a lesson); its record goes to the planner's log."""
        rec = self.reflection.end_episode(list(getattr(self.planner, "history", []) or []))
        if rec is not None:
            log_event = getattr(self.planner, "log_event", None)
            if log_event is not None:
                log_event(rec)
            log.info("reflection after %s: %s (%s%s), %d lesson(s), %.0f ms", rec["episode"], "failed" if rec["failed"] else "ok",
                     rec["kind"], f", {rec['skill']}" if rec["skill"] else "", len(rec["ops"]), rec["ms"])
        return rec

    def world_update(self) -> dict | None:
        """The world memory writes the finished episode's facts to its store; its world_update record goes to the planner's log."""
        rec = self.world.end_episode()
        if rec is not None:
            log_event = getattr(self.planner, "log_event", None)
            if log_event is not None:
                log_event(rec)
            log.info("world memory after %s: %d facts, state %s, done %s", rec["episode"], rec["facts"], rec["state"], rec["done"])
        return rec

    def close(self):
        """End of the run: the online store takes the last episode and releases its directory; the reflection memory judges
        the last episode; the world memory stores it; the proxy log is closed."""
        if self.online is not None:
            self.store_update(); self.online.close()
        if self.reflection is not None:
            self.reflect_episode()
        if self.world is not None:
            self.world_update(); self.world.close()
        if self.action_memory is not None:
            self.action_memory.close()
        if self._log:
            self._log.close(); self._log = None


class ProxyServer:
    """openpi-protocol frontend (msgpack over websocket); one connection at a time is enough for an evaluation loop."""

    def __init__(self, proxy: PlannerProxy, host: str, port: int, metadata: dict):
        self.p, self.host, self.port, self.meta = proxy, host, port, metadata

    async def _run(self):
        import websockets.asyncio.server as _server
        async with _server.serve(self._handler, self.host, self.port, compression=None, max_size=None,
                                 process_request=_health) as s:
            log.info("planner proxy listening on %s:%s (mode %s)", self.host, self.port, self.p.mode)
            await s.serve_forever()

    async def _handler(self, ws):
        import websockets.frames
        from openpi_client import msgpack_numpy
        packer = msgpack_numpy.Packer(); await ws.send(packer.pack(self.meta))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
                res = await asyncio.to_thread(self.p.infer, obs)
                await ws.send(packer.pack(res))
            except websockets.exceptions.ConnectionClosed:
                log.info("client disconnected"); break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="proxy error"); raise


def _health(conn, req):
    return conn.respond(http.HTTPStatus.OK, "OK\n") if req.path == "/healthz" else None


def build_vlm(args) -> VLMBackend:
    """The planning model's client of --vlm-backend (default openai: OpenAICompatibleVLM, taken from this module's namespace)
    with base_url, model, chat_template_kwargs and api_key (if set) as far as its constructor takes them, and --vlm-arg.
    --no-chat-template-kwargs sends none (an empty dict; None would mean the client's default, thinking off)."""
    name = getattr(args, "vlm_backend", None) or "openai"
    factory = OpenAICompatibleVLM if name == "openai" else resolve_vlm_backend(name)
    std = {"base_url": args.vlm_url, "model": args.vlm_model,
           "chat_template_kwargs": {} if args.no_chat_template_kwargs else {"enable_thinking": bool(args.thinking)}}
    key = resolve_api_key(getattr(args, "vlm_api_key", None))
    if key:
        std["api_key"] = key
    return construct(factory, std, dict(getattr(args, "vlm_arg", None) or []), f"VLM backend {name!r}")


def build_transport(args) -> PolicyTransport:
    """The connection to the policy of --transport (default openpi: OpenPIWebsocketTransport, taken from this module's
    namespace) with host and port (if set) as far as its constructor takes them, and --transport-arg."""
    name = getattr(args, "transport", None) or "openpi"
    factory = OpenPIWebsocketTransport if name == "openpi" else resolve_transport(name)
    std = {k: v for k, v in (("host", args.upstream_host), ("port", args.upstream_port)) if v is not None}
    return construct(factory, std, dict(getattr(args, "transport_arg", None) or []), f"transport {name!r}")


def check_components(ap: argparse.ArgumentParser, args):
    """Usage errors of --transport and --vlm-backend before anything is built: the openpi transport needs --upstream-port, and
    a named component must resolve."""
    transport = args.transport or "openpi"
    if transport == "openpi" and args.upstream_port is None:
        ap.error("the following arguments are required: --upstream-port")
    try:
        if transport != "openpi":
            resolve_transport(transport)
        if args.mode == "vlm" and (args.vlm_backend or "openai") != "openai":
            resolve_vlm_backend(args.vlm_backend)
    except ValueError as e:
        ap.error(str(e))


def component_meta(args) -> dict:
    """The proxy metadata's entries for components other than the defaults (none for the defaults: the metadata stays as it was)."""
    out = {}
    if args.mode == "vlm" and (args.vlm_backend or "openai") != "openai":
        out["vlm_backend"] = args.vlm_backend
    if (args.transport or "openpi") != "openpi":
        out["transport"] = args.transport
    return out


def build_planner(args, task: TaskSpec) -> Planner:
    vlm = build_vlm(args)
    store = None
    if getattr(args, "online_store", None):   # the memory grows from the run's own episodes (resumed, or from --memory)
        from .online import OnlineStore
        store = OnlineStore.from_args(args, task)
        log.info("exemplar memory: %s", store.describe())
    elif args.memory:
        from .exemplar import ExemplarStore
        from .fingerprint import fingerprint_from_task
        store = ExemplarStore(fingerprint_from_task(task), task, args.memory)
        log.info("exemplar memory: %d entries from %s", len(store), args.memory)
    profile = None
    if args.profile and not args.profile_off:
        from ..preference import ProfileStore
        if getattr(args, "reflect", False):   # the robot writes lessons too: they are shown under their own heading
            from ..reflection import SelfNotebook as ProfileStore
        profile = ProfileStore(args.profile); log.info("user profile (notebook): %d active entries from %s", len(profile), args.profile)
    keyframes = None
    if getattr(args, "keyframes", 0):   # frames of earlier moments of the running episode in every question
        from .keyframes import KeyframeMemory
        keyframes = KeyframeMemory.from_args(args, task); log.info("keyframe memory: %s", keyframes.config())
    world = None
    if getattr(args, "world_state", False) or getattr(args, "world_verify", False) or getattr(args, "world_store", None):
        from ..world import WorldMemory   # facts of the episode: state block, switch verifier, store across episodes
        world = WorldMemory.from_args(args, task); log.info("world memory: %s", world.config())
    planner = Planner(vlm, task, memory=store, k=args.k, votes_to_advance=args.votes, call_every=args.call_every, profile=profile,
                      min_calls_per_skill=args.min_calls, recovery=args.recovery,
                      log=(args.log + ".planner.jsonl") if args.log else None, use_wrist=not args.no_wrist, upscale=args.upscale,
                      brief=args.brief, gate=args.gate, k_vote=args.k_vote, gate_min_done=args.gate_min_done,
                      verdict_only=args.verdict_only, log_memory=bool(getattr(args, "online_store", None)), keyframes=keyframes,
                      world=world)
    gate = (f"gate on (the {args.k_vote} nearest stored moments vote; asked from {args.gate_min_done} 'done' vote(s); last skill "
            f"always asked)" if args.gate and store is not None else "gate off" + (" (no memory store)" if args.gate else ""))
    answer = "verdict only" if args.verdict_only else ("states without reason" if args.brief else "states + reason + verdict")
    log.info("planner: %s; answer: %s; %d example(s) per question, votes %d, min-calls %d, call-every %d",
             gate, answer, args.k if store is not None else 0, args.votes, args.min_calls, args.call_every)
    return planner


def build_reflection(args, task: TaskSpec, planner):
    """The reflection memory of --reflect (None without it): judge, lesson writer and votes on the planner's notebook."""
    if not getattr(args, "reflect", False) or getattr(planner, "profile", None) is None:
        return None
    from ..reflection import Reflection, ReflectionWriter, make_judge
    judge = make_judge(args.reflect_judge, task, tolerance=args.reflect_tolerance, stall_calls=args.reflect_stall)
    writer = ReflectionWriter(planner.vlm, task, planner.profile, judge=judge, evidence=args.reflect_evidence, keep_prompts=True)
    log.info("reflection memory: %s judge, evidence %s, lessons into %s", judge.name, "on" if args.reflect_evidence else "off",
             args.profile)
    return Reflection(writer)


def build_parser(proxy_cls=PlannerProxy) -> argparse.ArgumentParser:
    """The command line of the proxy (and of a method's proxy class, which adds its own options)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--upstream-port", type=int, default=None, help="the policy server's port (required by the openpi transport)")
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--transport", default="openpi", metavar="NAME",
                    help="the connection to the policy: openpi (default, websocket + msgpack), a registered name, an entry point "
                         "of vla_memory.transports, or module:Class (a PolicyTransport; gets host and port if it takes them)")
    ap.add_argument("--transport-arg", action="append", type=key_value, default=[], metavar="KEY=VALUE",
                    help="a keyword argument of the transport's constructor (repeatable; the value is read as JSON if it parses)")
    ap.add_argument("--task", default=None, help="task YAML/JSON (default: the packaged red_blue_blocks task)")
    ap.add_argument("--mode", choices=list(proxy_cls.MODES), default="vlm")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--vlm-model", default="Qwen3.8-27B-INT4")
    ap.add_argument("--vlm-backend", default="openai", metavar="NAME",
                    help="the planning model's client: openai (default, any OpenAI-compatible chat API: vLLM, SGLang, Ollama, a "
                         "hosted API), a registered name, an entry point of vla_memory.vlm_backends, or module:Class (a VLMBackend; "
                         "gets base_url, model, api_key and chat_template_kwargs if it takes them)")
    ap.add_argument("--vlm-arg", action="append", type=key_value, default=[], metavar="KEY=VALUE",
                    help="a keyword argument of the backend's constructor, e.g. timeout=60 (repeatable; read as JSON if it parses)")
    ap.add_argument("--vlm-api-key", default=None, metavar="KEY",
                    help="API key of the VLM server (sent as a bearer token); default: the environment variable VLM_API_KEY, "
                         "which keeps the key out of the process list")
    ap.add_argument("--thinking", action="store_true", help="enable the model's reasoning mode (Qwen3 chat template)")
    ap.add_argument("--no-chat-template-kwargs", action="store_true", help="for servers that reject chat_template_kwargs")
    ap.add_argument("--call-every", type=int, default=1, help="query the planner model every N policy calls")
    ap.add_argument("--votes", type=int, default=1, help="consecutive 'done' answers needed to advance")
    ap.add_argument("--min-calls", type=int, default=2, help="a skill cannot end before its N-th monitor call")
    ap.add_argument("--recovery", action="store_true", help="put down a wrongly held object first (needs --no-verdict-only)")
    ap.add_argument("--memory", default=None, help="exemplar store .npz (scripts/build_exemplar_store.py)")
    ap.add_argument("--profile", default=None, help="user preference memory (notebook), a JSON file; absent = notebook off")
    ap.add_argument("--profile-off", action="store_true", help="ignore --profile (baseline without the notebook, same command line)")
    ap.add_argument("--k", type=int, default=4, help="stored examples shown to the planner model per question")
    ap.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True,
                    help="memory-vote gate (with --memory): skip the question when fewer than --gate-min-done of the --k-vote "
                         "nearest stored moments are labelled with a later skill; the last skill of the plan is always asked "
                         "(default: on; --no-gate asks at every monitor call)")
    ap.add_argument("--k-vote", type=int, default=6, help="stored moments consulted by the gate (the model sees --k of them)")
    ap.add_argument("--gate-min-done", type=int, default=1, help="ask the model from this many 'done' votes on (1 = any)")
    ap.add_argument("--verdict-only", action=argparse.BooleanOptionalAction, default=True,
                    help="the monitor answers only current_skill_done (default: on; --no-verdict-only: object states, "
                         "gripper, reason and verdict)")
    ap.add_argument("--brief", action="store_true", help="with --no-verdict-only: the state answer without the free-text reason")
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--image-keys", nargs="+", default=["observation/image", "observation/wrist_image"],
                    help="observation keys of the camera images used when memory/*_raw frames are not sent")
    ap.add_argument("--log", default=None)
    on = ap.add_argument_group("online memory (vlm mode): the exemplar store grows from the agent's own episodes (online.py)")
    on.add_argument("--online-store", default=None, metavar="DIR",
                    help="store directory: resumed if it exists, else started from --memory (or empty); updated and saved "
                         "after every episode")
    on.add_argument("--online-labeller", choices=["rules", "planner", "oracle"], default="rules",
                    help="who labels the agent's calls: the rules writer on the robot's own signals (default), the "
                         "planner's own pointer (self-training baseline), or memory/oracle_prompt (simulation only)")
    on.add_argument("--online-admit", choices=["all", "complete", "none"], default="all",
                    help="store every episode, only those the labeller saw complete, or none (a frozen store with the same "
                         "logging, for baselines) (default: all)")
    on.add_argument("--online-dedup", type=float, default=0.9999, metavar="COS",
                    help="near-duplicates: an episode keeps at most 2 entries (retrieval's per-episode cap) with the same "
                         "labels and a key cosine >= COS to each other (default 0.9999; 0 = off)")
    on.add_argument("--online-max-entries", type=int, default=0, metavar="N",
                    help="capacity; above it entries are evicted by --online-evict (default 0 = no limit)")
    on.add_argument("--online-evict", choices=["oldest", "redundant"], default="oldest",
                    help="eviction: own entries of the oldest episode first, or the entry with the most similar same-label "
                         "neighbour first; demonstrations last (default: oldest)")
    on.add_argument("--online-terminal", choices=["drop", "last", "done"], default="drop",
                    help="calls after the last skill is complete: not stored (default), labelled as the last skill like "
                         "the demos, or as done (the index after the last skill); the gate asks the last skill either way")
    kf = ap.add_argument_group("keyframe memory (vlm mode): labelled frames of earlier moments of the running episode in every "
                               "question (keyframes.py)")
    kf.add_argument("--keyframes", type=int, default=0, metavar="N",
                    help="keep at most N keyframes per episode and show them to the monitor (default 0 = off)")
    kf.add_argument("--keyframe-policy", choices=["events", "start", "switches", "uniform", "model"], default="events",
                    help="which moments: the start and every switch of the plan pointer (default), the start only, the "
                         "switches only, every --keyframe-every-th call, or the moments the monitor asks to keep (keep_frame "
                         "in its answer; clusters kept as their median call, as in MemER)")
    kf.add_argument("--keyframe-cameras", choices=["all", "exterior"], default="all",
                    help="the pictures of a keyframe: every camera (default) or the first (external) camera only")
    kf.add_argument("--keyframe-render", choices=["images", "text"], default="images",
                    help="show the keyframes as pictures with a label (default) or the same moments as text only")
    kf.add_argument("--keyframe-every", type=int, default=8, metavar="K", help="the uniform policy: every K-th call (default 8)")
    kf.add_argument("--max-images-per-prompt", type=int, default=12, metavar="M",
                    help="the planner model server's limit of images per prompt (vLLM --limit-mm-per-prompt; 12 on the "
                         "reference server): keyframes that do not fit next to the examples and the current pictures are left "
                         "out, the oldest after the start frame first (default 12)")
    rf = ap.add_argument_group("reflection memory (vlm mode, with --profile): lessons the robot writes after its own failed "
                               "episodes (vla_memory.reflection)")
    rf.add_argument("--reflect", action="store_true",
                    help="judge every finished episode from the robot's own records and, after a failure the planner caused, "
                         "write a lesson into the --profile notebook (default: off)")
    rf.add_argument("--reflect-judge", choices=["signals", "plan"], default="signals",
                    help="signals: a hindsight check of the planner's switches with the robot's gripper signals and the task's "
                         "rules writer (needs memory/proprio_steps; default); plan: the decision log only (plan incomplete at "
                         "the end, or a stall)")
    rf.add_argument("--reflect-tolerance", type=int, default=1, metavar="CALLS",
                    help="signals judge: a switch confirmed at most this many calls later is not a wrong decision (default 1)")
    rf.add_argument("--reflect-stall", type=int, default=20, metavar="CALLS",
                    help="plan judge: a skill current for this many calls is a stall (default 20)")
    rf.add_argument("--reflect-evidence", action=argparse.BooleanOptionalAction, default=True,
                    help="give the lesson writer the judge's evidence (readings at the wrong switch and at the confirmation); "
                         "--no-reflect-evidence: the decision log and the outcome only (default: on)")
    wd = ap.add_argument_group("world memory (vlm mode): object states, positions, events and completed skills as facts with "
                               "time, written at every call from the gripper signals and the cameras (vla_memory.world)")
    wd.add_argument("--world-state", action="store_true",
                    help="add the world memory's state block to every monitor question (default: off)")
    wd.add_argument("--world-verify", action="store_true",
                    help="a 'done' moves the plan pointer only if the world state agrees that the skill is complete (default: off)")
    wd.add_argument("--world-store", default=None, metavar="FILE",
                    help="SQLite file that keeps every episode's facts across runs, for queries (default: none)")
    wd.add_argument("--world-verify-patience", type=int, default=0, metavar="N",
                    help="with --world-verify: let the monitor's 'done' through after N consecutive vetoes of the same skill "
                         "(default 0 = never: a switch always needs the world state's agreement)")
    wd.add_argument("--world-calibration", default=None, metavar="JSON",
                    help="the external camera's pixel-to-table mapping and region of interest (world.calibration); without it "
                         "positions are kept as picture coordinates")
    am = ap.add_argument_group("action memory (any mode): re-targeted action chunks of earlier successful episodes replace (stall "
                               "recovery) or blend the policy's chunk (vla_memory.action_memory)")
    am.add_argument("--action-memory", default=None, metavar="STORE",
                    help="action store .npz (vla_memory.action_memory.ActionStore); absent = off, the reply passes unchanged")
    am.add_argument("--action-mode", choices=["recover", "blend"], default="recover",
                    help="recover: a stall trigger starts the memory's recovery until the skill changes (default); blend: every "
                         "chunk becomes (1 - b) policy + b memory")
    am.add_argument("--action-representation", choices=["absolute", "joint_delta", "ee_delta", "ee_anchor"], default="ee_anchor",
                    help="how a stored chunk is re-targeted (default ee_anchor: TCP motion in the object's frame, inverse kinematics)")
    am.add_argument("--action-k", type=int, default=5, help="nearest stored chunks averaged after re-targeting (default 5)")
    am.add_argument("--action-no-follow", action="store_true", help="re-query from scratch at every call instead of following "
                                                                     "the stored sub-trajectory")
    am.add_argument("--action-max-distance", type=float, default=0.0, metavar="D",
                    help="do not act when the nearest stored chunk is farther than D key units (default 0 = no gate)")
    am.add_argument("--action-blend", type=float, default=0.5, help="blend mode: the memory's weight b (default 0.5)")
    am.add_argument("--action-horizon", type=int, default=8,
                    help="actions the environment executes per call (the reference bench: 8 of 10; default 8)")
    am.add_argument("--action-calibration", default=None, metavar="JSON",
                    help="the external camera's table calibration for the objects' positions (default: --world-calibration)")
    am.add_argument("--action-write", choices=["none", "complete", "all"], default="none",
                    help="add the policy's own executed chunks of a finished episode to the store and save it: of episodes "
                         "that reached the last skill (complete), of every episode (all), or never (default)")
    proxy_cls.add_arguments(ap)
    return ap


def main(argv=None, proxy_cls=PlannerProxy):
    ap = build_parser(proxy_cls)
    args = ap.parse_args(argv)
    if args.mode == "vlm" and args.recovery and args.verdict_only:
        ap.error("--recovery reads the object states from the monitor's answer: add --no-verdict-only")
    if args.online_store and args.mode != "vlm":
        ap.error("--online-store grows the planner's exemplar memory: it needs --mode vlm")
    if args.keyframes and args.mode != "vlm":
        ap.error("--keyframes shows frames to the planner's monitor: it needs --mode vlm")
    if args.reflect and (args.mode != "vlm" or not args.profile or args.profile_off):
        ap.error("--reflect writes lessons into the notebook the planner reads: it needs --mode vlm and --profile (not --profile-off)")
    if (args.world_state or args.world_verify or args.world_store) and args.mode != "vlm":
        ap.error("--world-state / --world-verify / --world-store keep the planner's world memory: they need --mode vlm")
    check_components(ap, args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    task = TaskSpec.load(args.task) if args.task else TaskSpec.default()
    planner = proxy_cls.build_monitor(args, task)
    transport = build_transport(args)
    proxy = proxy_cls(transport, args.mode, planner, args.log, image_keys=args.image_keys, task=task,
                      reflection=build_reflection(args, task, planner))
    proxy.configure(args)
    if args.action_memory:   # the action-memory hook between the policy's reply and the environment
        from ..action_memory import build_hook
        proxy.action_memory = build_hook(args, task)
        log.info("action memory: %s", proxy.action_memory.config())
    meta = {"proxy": proxy_cls.__module__, "mode": args.mode, "task": task.name, "profile": (args.profile if not args.profile_off else None),
            "vlm": args.vlm_model if args.mode == "vlm" else None, "memory": args.memory, "k": args.k}
    meta.update(component_meta(args))
    if proxy.action_memory is not None:
        meta["action_memory"] = args.action_memory
    if proxy.online is None and proxy.reflection is None and proxy.world is None and proxy.action_memory is None:
        asyncio.run(ProxyServer(proxy, args.listen_host, args.listen_port, meta)._run())
        return
    # the online store must take the last episode (and reflection judge it, the world memory store it): the launchers stop
    # the proxy with SIGTERM once the bench is done
    if proxy.online is not None:
        meta["online_store"] = args.online_store
    if proxy.reflection is not None:
        meta["reflect"] = args.reflect_judge
    if proxy.world is not None:
        meta["world"] = proxy.world.config()
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        asyncio.run(ProxyServer(proxy, args.listen_host, args.listen_port, meta)._run())
    except (KeyboardInterrupt, SystemExit):
        log.info("stopping: the online store takes the last episode, reflection judges it")
    finally:
        proxy.close()


def _exit_on_sigterm(signum, frame):
    raise SystemExit(0)


if __name__ == "__main__":
    main()
