"""The policy server of the built-in memory: openpi's websocket policy server with the stateful policy.

    python -m builtin_memory.serve --checkpoint <checkpoint dir> [--port 8000] [--center-crop 0.95] [--reset-every-call]

Runs in openpi's environment, with this folder (integrations/openpi) and the framework on PYTHONPATH. The proxy connects to
the port as to any openpi server (--upstream-port) and forwards the environment's episode markers, at which the policy
resets its state; every reply carries reply["memory"] (policy.py). --reset-every-call clears the state before every call,
as the proxy's --memory-reset every-call does. The checkpoint is one of the reference recipe (config.py); its phase head and
its norm stats are read from it. --center-crop is the centre crop of the exterior picture that the reference policies run
at inference (0.95).
"""
from __future__ import annotations

import argparse
import logging

from openpi.serving import websocket_policy_server

from .config import serving_config
from .policy import create_policy

log = logging.getLogger("builtin_memory.serve")


def build_parser() -> argparse.ArgumentParser:
    """The command line: the checkpoint, the server's address and the policy's options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="the checkpoint directory (with params/ and assets/)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--center-crop", type=float, default=0.0,
                        help="centre crop of the exterior picture at inference (0: off; the reference policies run 0.95)")
    parser.add_argument("--reset-every-call", action="store_true",
                        help="clear the memory before every call (the reset test)")
    parser.add_argument("--default-prompt", default=None, help="the prompt of a call that sends none")
    return parser


def main(argv=None) -> None:
    """Load the checkpoint and serve the stateful policy until the process is stopped."""
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_parser().parse_args(argv)
    train_config = serving_config(args.checkpoint)
    policy = create_policy(train_config, args.checkpoint, default_prompt=args.default_prompt, center_crop=args.center_crop,
                           reset_every_call=args.reset_every_call)
    log.info("memory policy %s: phases %d, reset every call %s, centre crop %s", args.checkpoint,
             train_config.model.memory_num_phases, args.reset_every_call, args.center_crop)
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host=args.host, port=args.port,
                                                           metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
