"""The proxy's network frontend: the openpi serving protocol (msgpack over a websocket), because that is what openpi
environment clients speak. One connection at a time is enough for an evaluation loop. Another frontend replaces this
class and keeps the relay. When it stops it closes the open connections itself, so that it stops even while an environment
client is still connected.
"""
from __future__ import annotations

import asyncio
import http
import logging
import traceback

from .relay import Relay

log = logging.getLogger("vla_memory.proxy")


class ProxyServer:
    """The websocket server in front of the relay: every connected client first receives the metadata, then one reply per
    observation."""

    def __init__(self, proxy: Relay, host: str, port: int, metadata: dict):
        self.proxy = proxy
        self.host = host
        self.port = port
        self.metadata = metadata
        self.stop = asyncio.Event()         # set: stop serving (cli.py sets it at SIGTERM and Ctrl-C)

    async def run(self):
        """Serve until `stop` is set or the task is cancelled. Then accept no new connection, close the open ones (close code
        1001, going away) and wait until their handlers end, so that nothing uses the relay any more."""
        import websockets.asyncio.server as server
        async with server.serve(self._handle_connection, self.host, self.port, compression=None, max_size=None,
                                process_request=health_check):
            log.info("proxy listening on %s:%s (writer %s, method %s)", self.host, self.port, self.proxy.writer.name,
                     self.proxy.method.name)
            await self.stop.wait()
            log.info("proxy stopping: closing the open connections")

    async def _handle_connection(self, websocket):
        """One client: send the metadata, then answer every observation with the proxy's reply."""
        import websockets.frames
        from openpi_client import msgpack_numpy
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        while True:
            try:
                observation = msgpack_numpy.unpackb(await websocket.recv())
                reply = await asyncio.to_thread(self.proxy.infer, observation)
                await websocket.send(packer.pack(reply))
            except websockets.exceptions.ConnectionClosed:
                log.info("client disconnected")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="proxy error")
                raise


def health_check(connection, request):
    """GET /healthz answers OK (for launchers that wait until the proxy is up)."""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
