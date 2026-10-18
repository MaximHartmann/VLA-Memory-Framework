"""The proxy: the relay between an environment client and the VLA policy server, which runs one writer and one method.

  relay.Relay              one policy call: the writer decides, the method sets what the policy receives, the reply goes back
  server.ProxyServer       the websocket frontend (the openpi serving protocol)
  cli                      the command line and main(): python -m vla_memory.proxy --writer NAME --method NAME
  extensions.Extension     how an optional memory plugs into the proxy and the VLM writer's planner (--with NAME)
"""
from .extensions import EXTENSIONS, Extension, register_extension, resolve_extension
from .relay import Relay

__all__ = ["Relay", "Extension", "EXTENSIONS", "register_extension", "resolve_extension"]
