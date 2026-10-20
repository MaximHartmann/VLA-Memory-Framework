"""The command line of the proxy: one writer, one method, and the extensions.

    python -m vla_memory.proxy --writer NAME --method NAME --listen-port <port> --upstream-port <policy port> [options]

Writers (--writer):
    oracle     the simulator's ground truth (memory/oracle_prompt)
    rules      rules on the robot's gripper and hand signals (the task file's history.rules)
    vlm        the VLM planner, with the exemplar store (default)
    head       the policy's own event head (a policy server that returns its verdict)
    off        none: the policy gets the instruction
Methods (--method):
    subgoal    the current skill as the prompt (default)
    history    the instruction with a line per completed skill
    keyframe   the instruction, and the wrist picture of the last key moment as a third picture

Option groups: the connection (--listen-port, --upstream-port, --upstream-host, --transport, --transport-arg, --log), the
task and the choices (--task, --writer, --method, --image-keys), the policy's own memory (--memory-reset) and the extensions
(--with NAME, repeatable). The chosen writer, method and extensions add their own options; --help with the choices lists
them. The defaults are --writer vlm --method subgoal: the planner loop.

The episode markers memory/new_episode and memory/episode always reach the policy as the environment sent them, so that a
policy that keeps its own memory resets it when an episode starts. --memory-reset every-call marks every call as a new
episode instead: the policy clears its memory before every call, the test of whether it uses its memory at all.

--writer, --method and --with take a registered name (writers/base.py WRITERS, methods/base.py METHODS, extensions.py
EXTENSIONS), an entry point of vla_memory.writers / vla_memory.methods / vla_memory.extensions, or module:Class.
--transport NAME picks the connection to the policy (default openpi: OpenPIWebsocketTransport on --upstream-host and
--upstream-port; or a registered name, an entry point of vla_memory.transports, or module:Class), with --transport-arg
KEY=VALUE.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from typing import Sequence

from .extensions import Extension, build_extensions, resolve_extension
from .relay import MEMORY_RESETS, Relay
from .server import ProxyServer
from ..methods.base import resolve_method
from ..plugins import construct, key_value
from ..task import TaskSpec
from ..transport import OpenPIWebsocketTransport, PolicyTransport, resolve_transport
from ..writers.base import resolve_writer

log = logging.getLogger("vla_memory.proxy")
DEFAULT_WRITER = "vlm"
DEFAULT_METHOD = "subgoal"


# ---------------------------------------------------------------------- the options
def add_connection_options(parser: argparse.ArgumentParser):
    """The ports, the transport to the policy and the log file."""
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


def add_choice_options(parser: argparse.ArgumentParser):
    """The writer, the method and the extensions; parsed first, so that their own options can be added."""
    parser.add_argument("--writer", default=DEFAULT_WRITER, metavar="NAME",
                        help="who signals that a step is done (default vlm): a registered name (the list above), an entry "
                             "point of vla_memory.writers, or module:Class")
    parser.add_argument("--method", default=DEFAULT_METHOD, metavar="NAME",
                        help="what the policy receives (default subgoal): a registered name (the list above), an entry "
                             "point of vla_memory.methods, or module:Class")
    parser.add_argument("--with", dest="extensions", action="append", default=[], metavar="NAME",
                        help="an extension: an optional memory that plugs into the proxy and the VLM writer's planner (a "
                             "registered name, an entry point of vla_memory.extensions, or module:Class; repeatable)")


def add_task_options(parser: argparse.ArgumentParser):
    """The task file and the observation keys of the policy's cameras."""
    parser.add_argument("--task", default=None, help="task YAML/JSON (default: the packaged red_blue_blocks task)")
    parser.add_argument("--image-keys", nargs="+", default=["observation/image", "observation/wrist_image"],
                        help="observation keys of the camera images used when memory/*_raw frames are not sent")


def add_policy_memory_options(parser: argparse.ArgumentParser):
    """When a policy that keeps its own memory is told that an episode starts."""
    parser.add_argument("--memory-reset", choices=MEMORY_RESETS, default="episode",
                        help="episode: forward the environment's episode markers (default); every-call: mark every call as "
                             "a new episode, so that a policy with its own memory clears it before every call (the reset "
                             "test)")


def build_parser(writer_class, method_class, extension_classes: Sequence[type] = ()) -> argparse.ArgumentParser:
    """The command line with the options of the chosen writer, method and extensions."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_connection_options(parser)
    add_task_options(parser)
    add_policy_memory_options(parser)
    add_choice_options(parser)
    writer_class.add_arguments(parser)
    method_class.add_arguments(parser)
    for extension_class in extension_classes:
        extension_class.add_arguments(parser)
    return parser


def parse_choices(argv) -> argparse.Namespace:
    """--writer, --method and --with, read before the full parse."""
    choice_parser = argparse.ArgumentParser(add_help=False)
    add_choice_options(choice_parser)
    choices, _ = choice_parser.parse_known_args(argv)
    return choices


def parse_command_line(argv=None):
    """Parse in two passes: the choices first, then everything. Returns (parser, args, extension classes); the writer's
    and the method's classes are resolve_writer(args.writer) and resolve_method(args.method)."""
    choices = parse_choices(argv)
    try:
        writer_class = resolve_writer(choices.writer)
        method_class = resolve_method(choices.method)
        extension_classes = [resolve_extension(name) for name in choices.extensions]
    except ValueError as error:
        parser = argparse.ArgumentParser()
        add_connection_options(parser)
        parser.error(str(error))
    parser = build_parser(writer_class, method_class, extension_classes)
    return parser, parser.parse_args(argv), extension_classes


def check_arguments(parser: argparse.ArgumentParser, args):
    """Usage errors before anything is built: the openpi transport needs --upstream-port, another must be resolvable."""
    transport = args.transport or "openpi"
    if transport == "openpi" and args.upstream_port is None:
        parser.error("the following arguments are required: --upstream-port")
    try:
        if transport != "openpi":
            resolve_transport(transport)
    except ValueError as error:
        parser.error(str(error))


# ---------------------------------------------------------------------- the builders
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


def build_components(args, task: TaskSpec, extension_classes) -> tuple:
    """The extensions, the writer and the method (ValueError for an unusable combination of options)."""
    extensions = build_extensions(extension_classes, args, task)
    writer = resolve_writer(args.writer).build(args, task, extensions)
    method = resolve_method(args.method).build(args, task)
    for extension in extensions:
        extension.attach(writer.monitor)
    return extensions, writer, method


def component_metadata(args) -> dict:
    """The metadata's entries for components other than the defaults (none for the defaults)."""
    metadata = {}
    if (getattr(args, "vlm_backend", None) or "openai") != "openai":
        metadata["vlm_backend"] = args.vlm_backend
    if (args.transport or "openpi") != "openpi":
        metadata["transport"] = args.transport
    if getattr(args, "memory_reset", "episode") != "episode":
        metadata["memory_reset"] = args.memory_reset
    return metadata


def proxy_metadata(args, task: TaskSpec, writer, method, extensions: Sequence[Extension]) -> dict:
    """What the environment client is told about the proxy when it connects."""
    metadata = {"proxy": Relay.__module__, "mode": writer.name, "method": method.name, "task": task.name,
                "vlm": getattr(args, "vlm_model", None) if writer.name == "vlm" else None,
                "memory": getattr(args, "memory", None), "k": getattr(args, "k", None)}
    metadata.update(component_metadata(args))
    metadata.update(method.metadata())
    for extension in extensions:
        metadata.update(extension.metadata())
    return metadata


# ---------------------------------------------------------------------- running
def main(argv=None):
    """Parse the command line, build the components and serve until the process is stopped."""
    parser, args, extension_classes = parse_command_line(argv)
    check_arguments(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    task = TaskSpec.load_or_default(args.task)
    try:
        extensions, writer, method = build_components(args, task, extension_classes)
    except ValueError as error:
        parser.error(str(error))
    relay = Relay(build_transport(args), writer, method, args.log, image_keys=args.image_keys, task=task,
                  extensions=extensions, memory_reset=args.memory_reset)
    serve(relay, args.listen_host, args.listen_port, proxy_metadata(args, task, writer, method, extensions))


def serve(relay: Relay, host: str, port: int, metadata: dict):
    """Run the frontend until the process is stopped (Ctrl-C, or SIGTERM as the launchers send it), then close the relay
    so that the extensions take the last episode."""
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        asyncio.run(ProxyServer(relay, host, port, metadata).run())
    except (KeyboardInterrupt, SystemExit):
        log.info("stopping")
    finally:
        relay.close()


def _exit_on_sigterm(signal_number, frame):
    """Turn SIGTERM into a SystemExit, so that serve() closes the relay like after Ctrl-C."""
    raise SystemExit(0)


if __name__ == "__main__":
    main()
