"""The proxy's network frontend: the openpi serving protocol (msgpack over a websocket), because that is what openpi
environment clients speak. One connection at a time is enough for an evaluation loop. Another frontend replaces this
class and keeps PlannerProxy.
"""
from __future__ import annotations

import asyncio
import http
import logging
import traceback

from .proxy import PlannerProxy

log = logging.getLogger("vla_memory.planner_loop.proxy")


class ProxyServer:
    def __init__(self, proxy: PlannerProxy, host: str, port: int, metadata: dict):
        self.proxy = proxy
        self.host = host
        self.port = port
        self.metadata = metadata

    async def run(self):
        """Serve until the process is stopped."""
        import websockets.asyncio.server as server
        async with server.serve(self._handle_connection, self.host, self.port, compression=None, max_size=None,
                                process_request=health_check) as websocket_server:
            log.info("planner proxy listening on %s:%s (mode %s)", self.host, self.port, self.proxy.mode)
            await websocket_server.serve_forever()

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
