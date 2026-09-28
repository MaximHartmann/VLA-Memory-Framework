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

The frontend speaks the openpi websocket protocol (msgpack) because that is what openpi environment clients use;
another frontend replaces ProxyServer and keeps PlannerProxy.
"""
from __future__ import annotations

import argparse
import asyncio
import http
import json
import logging
import time
import traceback

import numpy as np

from .planner import Planner
from ..task import TaskSpec
from ..transport import OpenPIWebsocketTransport, PolicyTransport
from ..vlm import OpenAICompatibleVLM

log = logging.getLogger("vla_memory.planner_loop.proxy")
CTRL = "memory/"


class PlannerProxy:
    def __init__(self, transport: PolicyTransport, mode: str, planner: Planner | None, log_path=None,
                 image_keys=("observation/image", "observation/wrist_image")):
        self.transport = transport; self.mode = mode; self.planner = planner; self.image_keys = tuple(image_keys)
        log.info("upstream metadata: %s", transport.metadata())
        self._log = open(log_path, "a") if log_path else None
        self.episode = None; self.n = 0

    def infer(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        ctrl = {k[len(CTRL):]: v for k, v in obs.items() if isinstance(k, str) and k.startswith(CTRL)}
        obs = {k: v for k, v in obs.items() if not (isinstance(k, str) and k.startswith(CTRL))}
        task = obs.get("prompt", ""); task = task.decode() if isinstance(task, bytes) else task
        info = {"mode": self.mode, "queried": False, "advanced": False, "vlm_ms": 0.0}
        if ctrl.get("new_episode") or self.episode is None:
            self.episode = str(ctrl.get("episode", f"ep{self.n}")); self.n += 1
            if self.mode == "vlm":
                rec = self.planner.start_episode(task, self.episode)
                info.update(plan=rec["plan"], plan_source=rec["source"], plan_ms=rec.get("ms"))
                log.info("episode %s plan (%s): %s", self.episode, rec["source"], rec["plan"])
        if self.mode == "vlm":
            ext = ctrl.get("exterior_raw", obs.get(self.image_keys[0]))
            wri = ctrl.get("wrist_raw", obs.get(self.image_keys[1]) if len(self.image_keys) > 1 else None)
            images = [np.asarray(ext)] + ([np.asarray(wri)] if wri is not None else [])
            proprio = {k: float(ctrl[k]) for k in ("gripper_closedness", "hand_height") if k in ctrl}
            rec = self.planner.observe(images, int(ctrl.get("step", -1)), proprio=proprio or None)
            prompt = rec.get("skill_after", self.planner.current_skill)
            info.update(queried=rec["queried"], advanced=rec["advanced"], vlm_ms=rec.get("ms", 0.0),
                        skill_index=self.planner.idx, vlm=rec.get("vlm"), recovery=rec.get("recovery_inserted"))
        elif self.mode == "oracle":
            prompt = ctrl.get("oracle_prompt", task); prompt = prompt.decode() if isinstance(prompt, bytes) else prompt
        else:
            prompt = task
        obs["prompt"] = prompt
        info["prompt"] = prompt; info["oracle_prompt"] = ctrl.get("oracle_prompt")
        t1 = time.perf_counter(); result = self.transport.infer(obs); t_up = time.perf_counter() - t1
        info["upstream_ms"] = 1000 * t_up; info["total_ms"] = 1000 * (time.perf_counter() - t0)
        result["planner"] = info
        if self._log:
            self._log.write(json.dumps({"episode": self.episode, "step": ctrl.get("step"), **info}, default=str) + "\n"); self._log.flush()
        return result


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


def build_planner(args, task: TaskSpec) -> Planner:
    vlm = OpenAICompatibleVLM(args.vlm_url, args.vlm_model,
                              chat_template_kwargs=None if args.no_chat_template_kwargs else {"enable_thinking": bool(args.thinking)})
    store = None
    if args.memory:
        from .exemplar import ExemplarStore
        from .fingerprint import fingerprint_from_task
        store = ExemplarStore(fingerprint_from_task(task), task, args.memory)
        log.info("exemplar memory: %d entries from %s", len(store), args.memory)
    return Planner(vlm, task, memory=store, k=args.k, votes_to_advance=args.votes, call_every=args.call_every,
                   min_calls_per_skill=args.min_calls, recovery=args.recovery,
                   log=(args.log + ".planner.jsonl") if args.log else None, use_wrist=not args.no_wrist, upscale=args.upscale)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--upstream-port", type=int, required=True)
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--task", default=None, help="task YAML/JSON (default: the packaged red_blue_blocks task)")
    ap.add_argument("--mode", choices=["vlm", "oracle", "off"], default="vlm")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--vlm-model", default="Qwen3.8-27B-INT4")
    ap.add_argument("--thinking", action="store_true", help="enable the model's reasoning mode (Qwen3 chat template)")
    ap.add_argument("--no-chat-template-kwargs", action="store_true", help="for servers that reject chat_template_kwargs")
    ap.add_argument("--call-every", type=int, default=1, help="query the planner model every N policy calls")
    ap.add_argument("--votes", type=int, default=1, help="consecutive 'done' answers needed to advance")
    ap.add_argument("--min-calls", type=int, default=2, help="a skill cannot end before its N-th monitor call")
    ap.add_argument("--recovery", action="store_true")
    ap.add_argument("--memory", default=None, help="exemplar store .npz (scripts/build_exemplar_store.py)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--image-keys", nargs="+", default=["observation/image", "observation/wrist_image"],
                    help="observation keys of the camera images used when memory/*_raw frames are not sent")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    task = TaskSpec.load(args.task) if args.task else TaskSpec.default()
    planner = build_planner(args, task) if args.mode == "vlm" else None
    transport = OpenPIWebsocketTransport(args.upstream_host, args.upstream_port)
    proxy = PlannerProxy(transport, args.mode, planner, args.log, image_keys=args.image_keys)
    meta = {"proxy": "vla_memory.planner_loop.proxy", "mode": args.mode, "task": task.name,
            "vlm": args.vlm_model if args.mode == "vlm" else None, "memory": args.memory, "k": args.k}
    asyncio.run(ProxyServer(proxy, args.listen_host, args.listen_port, meta)._run())


if __name__ == "__main__":
    main()
