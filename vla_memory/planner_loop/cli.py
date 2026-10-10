"""The command line of the planner loop's proxy (and, through proxy_cls, of a method's proxy such as the history proxy).

    python -m vla_memory.planner_loop.proxy --listen-port <port> --upstream-port <policy port> [options]

Option groups: the connection (--listen-port, --upstream-port, --upstream-host, --transport, --transport-arg), the task and
the mode (--task, --mode), the planning model (--vlm-url, --vlm-model, --vlm-backend, --vlm-arg, --vlm-api-key, --thinking,
--no-chat-template-kwargs), the planner's decision rule (--memory, --k, --gate, --k-vote, --gate-min-done, --verdict-only,
--brief, --votes, --min-calls, --call-every, --recovery, --no-wrist, --upscale, --image-keys, --log), and extensions
(--with NAME, repeatable; each extension then adds its own options, see extensions.py).

The planner's defaults are the settled configuration of the reference evaluation: the memory-vote gate (with --memory; the
6 nearest stored moments vote, one "done" vote asks the model, the last skill is always asked), a verdict-only answer, 4
examples per question, votes 1, min-calls 2. --no-gate, --no-verdict-only (with --brief: no reason field), --k, --k-vote,
--gate-min-done, --votes and --min-calls change them.

Components (vla_memory/plugins.py): --vlm-backend NAME picks the planning model's client (default openai: OpenAICompatibleVLM
on --vlm-url and --vlm-model; or a registered name, an entry point of vla_memory.vlm_backends, or module:Class), with
--vlm-arg KEY=VALUE for its own options and --vlm-api-key (default: the environment variable VLM_API_KEY). --transport NAME
picks the connection to the policy (default openpi: OpenPIWebsocketTransport on --upstream-host and --upstream-port; or a
registered name, an entry point of vla_memory.transports, or module:Class), with --transport-arg KEY=VALUE.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from typing import Sequence

from .extensions import Extension, build_extensions, memory_from, resolve_extension
from .planner import Planner
from .proxy import PlannerProxy
from .server import ProxyServer
from ..plugins import construct, key_value
from ..task import TaskSpec
from ..transport import OpenPIWebsocketTransport, PolicyTransport, resolve_transport
from ..vlm import OpenAICompatibleVLM, VLMBackend, resolve_api_key, resolve_vlm_backend

log = logging.getLogger("vla_memory.planner_loop.proxy")


# ---------------------------------------------------------------------- the options
def add_connection_options(parser: argparse.ArgumentParser):
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=None,
                        help="the policy server's port (required by the openpi transport)")
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--transport", default="openpi", metavar="NAME",
                        help="the connection to the policy: openpi (default, websocket + msgpack), a registered name, an "
                             "entry point of vla_memory.transports, or module:Class (a PolicyTransport; gets host and port "
                             "if it takes them)")
    parser.add_argument("--transport-arg", action="append", type=key_value, default=[], metavar="KEY=VALUE",
                        help="a keyword argument of the transport's constructor (repeatable; read as JSON if it parses)")
    parser.add_argument("--log", default=None, help="the proxy's log file, one JSON record per call")


def add_task_options(parser: argparse.ArgumentParser, modes: Sequence[str]):
    parser.add_argument("--task", default=None, help="task YAML/JSON (default: the packaged red_blue_blocks task)")
    parser.add_argument("--mode", choices=list(modes), default="vlm")
    parser.add_argument("--image-keys", nargs="+", default=["observation/image", "observation/wrist_image"],
                        help="observation keys of the camera images used when memory/*_raw frames are not sent")


def add_model_options(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("the planning model (vlm mode)")
    group.add_argument("--vlm-url", default="http://127.0.0.1:8100/v1")
    group.add_argument("--vlm-model", default="Qwen3.8-27B-INT4")
    group.add_argument("--vlm-backend", default="openai", metavar="NAME",
                       help="the planning model's client: openai (default, any OpenAI-compatible chat API), a registered "
                            "name, an entry point of vla_memory.vlm_backends, or module:Class (a VLMBackend; gets base_url, "
                            "model, api_key and chat_template_kwargs if it takes them)")
    group.add_argument("--vlm-arg", action="append", type=key_value, default=[], metavar="KEY=VALUE",
                       help="a keyword argument of the backend's constructor, e.g. timeout=60 (repeatable; read as JSON if "
                            "it parses)")
    group.add_argument("--vlm-api-key", default=None, metavar="KEY",
                       help="API key of the VLM server (sent as a bearer token); default: the environment variable "
                            "VLM_API_KEY, which keeps the key out of the process list")
    group.add_argument("--thinking", action="store_true", help="enable the model's reasoning mode (Qwen3 chat template)")
    group.add_argument("--no-chat-template-kwargs", action="store_true", help="for servers that reject chat_template_kwargs")


def add_planner_options(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("the planner's decision rule (vlm mode)")
    group.add_argument("--memory", default=None, help="exemplar store .npz (scripts/build_exemplar_store.py)")
    group.add_argument("--k", type=int, default=4, help="stored examples shown to the planning model per question")
    group.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True,
                       help="memory-vote gate (with --memory): skip the question when fewer than --gate-min-done of the "
                            "--k-vote nearest stored moments are labelled with a later skill; the last skill of the plan is "
                            "always asked (default: on; --no-gate asks at every monitor call)")
    group.add_argument("--k-vote", type=int, default=6, help="stored moments consulted by the gate (the model sees --k of them)")
    group.add_argument("--gate-min-done", type=int, default=1, help="ask the model from this many 'done' votes on (1 = any)")
    group.add_argument("--verdict-only", action=argparse.BooleanOptionalAction, default=True,
                       help="the monitor answers only current_skill_done (default: on; --no-verdict-only: object states, "
                            "gripper, reason and verdict)")
    group.add_argument("--brief", action="store_true", help="with --no-verdict-only: the state answer without the free-text reason")
    group.add_argument("--votes", type=int, default=1, help="consecutive 'done' answers needed to advance")
    group.add_argument("--min-calls", type=int, default=2, help="a skill cannot end before its N-th monitor call")
    group.add_argument("--call-every", type=int, default=1, help="query the planning model every N policy calls")
    group.add_argument("--recovery", action="store_true", help="put down a wrongly held object first (needs --no-verdict-only)")
    group.add_argument("--no-wrist", action="store_true", help="show the planning model the external camera only")
    group.add_argument("--upscale", type=int, default=1, help="enlarge the pictures by this factor before sending them")


def add_extension_option(parser: argparse.ArgumentParser):
    parser.add_argument("--with", dest="extensions", action="append", default=[], metavar="NAME",
                        help="an extension: an optional memory that plugs into the proxy and the planner (a registered "
                             "name, an entry point of vla_memory.extensions, or module:Class; repeatable). Its own options "
                             "follow; --help with --with NAME lists them")


def build_parser(proxy_cls=PlannerProxy, extension_classes: Sequence[type] = ()) -> argparse.ArgumentParser:
    """The command line of the proxy, of its extensions, and of a method's proxy class (which adds its own options)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_connection_options(parser)
    add_task_options(parser, proxy_cls.MODES)
    add_model_options(parser)
    add_planner_options(parser)
    add_extension_option(parser)
    for extension_cls in extension_classes:
        extension_cls.add_arguments(parser)
    proxy_cls.add_arguments(parser)
    return parser


def extension_names(argv) -> list[str]:
    """The --with values, read before the full parse so the extensions can add their options."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    add_extension_option(pre_parser)
    known, _ = pre_parser.parse_known_args(argv)
    return known.extensions


def parse_command_line(argv=None, proxy_cls=PlannerProxy):
    """Parse in two passes: the extensions named with --with first, then everything. Returns (parser, args, classes)."""
    try:
        classes = [resolve_extension(name) for name in extension_names(argv)]
    except ValueError as error:
        build_parser(proxy_cls).error(str(error))
    parser = build_parser(proxy_cls, classes)
    return parser, parser.parse_args(argv), classes


def check_arguments(parser: argparse.ArgumentParser, args):
    """Usage errors before anything is built."""
    if args.mode == "vlm" and args.recovery and args.verdict_only:
        parser.error("--recovery reads the object states from the monitor's answer: add --no-verdict-only")
    transport = args.transport or "openpi"
    if transport == "openpi" and args.upstream_port is None:
        parser.error("the following arguments are required: --upstream-port")
    try:
        if transport != "openpi":
            resolve_transport(transport)
        if args.mode == "vlm" and (args.vlm_backend or "openai") != "openai":
            resolve_vlm_backend(args.vlm_backend)
    except ValueError as error:
        parser.error(str(error))


# ---------------------------------------------------------------------- the builders
def load_task(path) -> TaskSpec:
    return TaskSpec.load(path) if path else TaskSpec.default()


def build_vlm(args) -> VLMBackend:
    """The planning model's client of --vlm-backend with base_url, model, chat_template_kwargs and api_key (if set) as far
    as its constructor takes them, and --vlm-arg. --no-chat-template-kwargs sends none."""
    name = getattr(args, "vlm_backend", None) or "openai"
    factory = OpenAICompatibleVLM if name == "openai" else resolve_vlm_backend(name)
    if args.no_chat_template_kwargs:
        template_kwargs = {}
    else:
        template_kwargs = {"enable_thinking": bool(args.thinking)}
    standard = {"base_url": args.vlm_url, "model": args.vlm_model, "chat_template_kwargs": template_kwargs}
    key = resolve_api_key(getattr(args, "vlm_api_key", None))
    if key:
        standard["api_key"] = key
    return construct(factory, standard, dict(getattr(args, "vlm_arg", None) or []), f"VLM backend {name!r}")


def build_transport(args) -> PolicyTransport:
    """The connection to the policy of --transport with host and port (if set) as far as its constructor takes them, and
    --transport-arg."""
    name = getattr(args, "transport", None) or "openpi"
    factory = OpenPIWebsocketTransport if name == "openpi" else resolve_transport(name)
    standard = {}
    if args.upstream_host is not None:
        standard["host"] = args.upstream_host
    if args.upstream_port is not None:
        standard["port"] = args.upstream_port
    return construct(factory, standard, dict(getattr(args, "transport_arg", None) or []), f"transport {name!r}")


def build_memory(args, task: TaskSpec, extensions: Sequence[Extension] = ()):
    """The planner's long-term memory: an extension's store, else the exemplar store of --memory, else None."""
    store = memory_from(extensions)
    if store is not None:
        return store
    if not args.memory:
        return None
    from .exemplar import ExemplarStore
    from .fingerprint import fingerprint_from_task
    store = ExemplarStore(fingerprint_from_task(task), task, args.memory)
    log.info("exemplar memory: %d entries from %s", len(store), args.memory)
    return store


def build_planner(args, task: TaskSpec, extensions: Sequence[Extension] = ()) -> Planner:
    store = build_memory(args, task, extensions)
    planner = Planner(build_vlm(args), task, memory=store, k=args.k, votes_to_advance=args.votes, call_every=args.call_every,
                      min_calls_per_skill=args.min_calls, recovery=args.recovery, use_wrist=not args.no_wrist,
                      upscale=args.upscale, brief=args.brief, gate=args.gate, k_vote=args.k_vote,
                      gate_min_done=args.gate_min_done, verdict_only=args.verdict_only,
                      log=(args.log + ".planner.jsonl") if args.log else None,
                      log_memory=memory_from(extensions) is not None, extensions=extensions)
    log_planner_configuration(args, store)
    return planner


def log_planner_configuration(args, store):
    if args.gate and store is not None:
        gate = (f"gate on (the {args.k_vote} nearest stored moments vote; asked from {args.gate_min_done} 'done' vote(s); "
                f"last skill always asked)")
    elif args.gate:
        gate = "gate off (no memory store)"
    else:
        gate = "gate off"
    if args.verdict_only:
        answer = "verdict only"
    elif args.brief:
        answer = "states without reason"
    else:
        answer = "states + reason + verdict"
    log.info("planner: %s; answer: %s; %d example(s) per question, votes %d, min-calls %d, call-every %d",
             gate, answer, args.k if store is not None else 0, args.votes, args.min_calls, args.call_every)


def component_meta(args) -> dict:
    """The metadata's entries for components other than the defaults (none for the defaults)."""
    meta = {}
    if args.mode == "vlm" and (args.vlm_backend or "openai") != "openai":
        meta["vlm_backend"] = args.vlm_backend
    if (args.transport or "openpi") != "openpi":
        meta["transport"] = args.transport
    return meta


def proxy_metadata(args, task: TaskSpec, proxy_cls, extensions: Sequence[Extension]) -> dict:
    """What the environment client is told about the proxy when it connects."""
    meta = {"proxy": proxy_cls.__module__, "mode": args.mode, "task": task.name,
            "vlm": args.vlm_model if args.mode == "vlm" else None, "memory": args.memory, "k": args.k}
    meta.update(component_meta(args))
    for extension in extensions:
        meta.update(extension.metadata())
    return meta


# ---------------------------------------------------------------------- running
def main(argv=None, proxy_cls=PlannerProxy):
    parser, args, extension_classes = parse_command_line(argv, proxy_cls)
    check_arguments(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    task = load_task(args.task)
    try:
        extensions = build_extensions(extension_classes, args, task)
    except ValueError as error:
        parser.error(str(error))
    planner = proxy_cls.build_monitor(args, task, extensions)
    for extension in extensions:
        extension.attach(planner)
    proxy = proxy_cls(build_transport(args), args.mode, planner, args.log, image_keys=args.image_keys, task=task,
                      extensions=extensions)
    proxy.configure(args)
    serve(proxy, args.listen_host, args.listen_port, proxy_metadata(args, task, proxy_cls, extensions))


def serve(proxy: PlannerProxy, host: str, port: int, metadata: dict):
    """Run the frontend until the process is stopped (Ctrl-C, or SIGTERM as the launchers send it), then close the proxy
    so that the extensions take the last episode."""
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        asyncio.run(ProxyServer(proxy, host, port, metadata).run())
    except (KeyboardInterrupt, SystemExit):
        log.info("stopping")
    finally:
        proxy.close()


def _exit_on_sigterm(signum, frame):
    raise SystemExit(0)


if __name__ == "__main__":
    main()
