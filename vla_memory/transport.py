"""Connection to the VLA policy server, shared by every method: an observation goes in, the policy's actions come
out, and nothing in the framework talks to the policy any other way. A method that sits between the environment
and the policy (rewriting the prompt, adding history, attaching memory) sends its modified observation through a
PolicyTransport and returns the reply unchanged.

OpenPIWebsocketTransport speaks the openpi serving protocol (msgpack over a websocket), used by pi0 / pi0.5 servers.
For another VLA, implement infer(observation) -> {"actions": ...} against its own server; metadata() may return
what the server reports about itself.

Choosing the transport (the proxies' --transport, plugins.py): a registered name (TRANSPORTS: openpi =
OpenPIWebsocketTransport; register_transport adds one), an entry point of the group vla_memory.transports, or an import path
module:Class. make_transport() builds it with host and port (--upstream-host, --upstream-port) if its constructor takes them,
and the --transport-arg KEY=VALUE options.
"""
from __future__ import annotations

from typing import Any

from .plugins import construct, resolve

TRANSPORTS: dict[str, Any] = {"openpi": "vla_memory.transport:OpenPIWebsocketTransport"}   # name -> class or import path
TRANSPORT_ENTRY_POINTS = "vla_memory.transports"


class PolicyTransport:
    def infer(self, observation: dict) -> dict[str, Any]:
        raise NotImplementedError

    def metadata(self) -> dict:
        return {}


class OpenPIWebsocketTransport(PolicyTransport):
    def __init__(self, host="127.0.0.1", port=8000):
        from openpi_client import websocket_client_policy   # optional dependency
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    def infer(self, observation):
        return self.client.infer(observation)

    def metadata(self):
        return self.client.get_server_metadata()


def register_transport(name: str, factory) -> None:
    """Make `factory` (a PolicyTransport class, or a callable returning one) selectable as --transport NAME."""
    TRANSPORTS[name] = factory


def resolve_transport(spec):
    """The class --transport names: a registered name, an entry point of vla_memory.transports, or module:Class."""
    return resolve(spec, TRANSPORTS, TRANSPORT_ENTRY_POINTS, "transport")


def make_transport(spec="openpi", options: dict | None = None, **standard) -> PolicyTransport:
    """The transport `spec` names, built with the `standard` keyword arguments it declares (host, port) and every one of
    `options` (--transport-arg KEY=VALUE)."""
    return construct(resolve_transport(spec), standard, options, f"transport {spec!r}")
