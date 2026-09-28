"""Connection to the VLA policy server, shared by every method: an observation goes in, the policy's actions come
out, and nothing in the framework talks to the policy any other way. A method that sits between the environment
and the policy (rewriting the prompt, adding history, attaching memory) sends its modified observation through a
PolicyTransport and returns the reply unchanged.

OpenPIWebsocketTransport speaks the openpi serving protocol (msgpack over a websocket), used by pi0 / pi0.5 servers.
For another VLA, implement infer(observation) -> {"actions": ...} against its own server; metadata() may return
what the server reports about itself.
"""
from __future__ import annotations

from typing import Any


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
