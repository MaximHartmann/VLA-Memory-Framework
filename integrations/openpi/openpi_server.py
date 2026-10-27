"""openpi's policy server with the policy's state in every reply, for the head writer (docs/writers/head): the policy runs as
openpi's serve_policy.py runs it, and a state tap (openpi_tap.py) reads one block of its backbone during the policy's own
pass. Every reply gets reply["policy_state"] (vla_memory/taps/reply.py): the block's output averaged over the prompt's tokens
(or a camera's picture tokens), named "block<block>:prompt" (":exterior", ":wrist"), as the automatic tap names it.

    python -m openpi_server --port <port> --config <openpi config> --checkpoint <checkpoint dir> [--center-crop 0.95] \
        [--layer 12] [--pool text] [--default-prompt TEXT]

A JAX or a PyTorch checkpoint (openpi tells them apart by model.safetensors). Runs in openpi's environment with the framework
and this folder (integrations/openpi) on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import logging

import openpi_tap
from vla_memory.taps.reply import KEY, state_entry
from vla_memory.writers.head.file import OLD_POOLS
from vla_memory.writers.head.crop import CenterCrop

log = logging.getLogger("openpi_server")


def create_policy(config: str, checkpoint: str, center_crop: float = 0.0, default_prompt: str | None = None):
    """The trained policy as openpi's serve_policy.py creates it, with the optional centre crop of the exterior picture
    (crop.py) behind its input transforms."""
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config
    policy = policy_config.create_trained_policy(openpi_config.get_config(config), checkpoint, default_prompt=default_prompt)
    if center_crop:
        input_transform, crop = policy._input_transform, CenterCrop(center_crop)
        policy._input_transform = lambda data: crop(input_transform(data))
    return policy


class StatePolicy:
    """The policy, with the tapped state of every call added to its reply."""

    def __init__(self, policy, tap, name: str):
        self._policy = policy
        self._tap = tap
        self._name = name

    def infer(self, observation: dict) -> dict:
        """The policy's reply to one observation, with reply["policy_state"]."""
        outputs = self._policy.infer(observation)
        outputs[KEY] = state_entry([self._name], self._tap.read()[:1])
        return outputs

    @property
    def metadata(self) -> dict:
        """The policy's metadata plus the name of the state it hands out."""
        return {**self._policy.metadata, KEY: {"names": [self._name]}}


def main(argv=None) -> None:
    """Load the policy, tap it and serve it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--config", required=True, help="the policy's openpi training config")
    parser.add_argument("--checkpoint", required=True, help="the policy's checkpoint directory")
    parser.add_argument("--center-crop", type=float, default=0.0,
                        help="centre crop of the exterior picture at inference (0: off; our policies run 0.95)")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--layer", type=int, default=12, help="the block whose output is read, counted from 0")
    parser.add_argument("--pool", choices=openpi_tap.POOLS, default="text",
                        help="averaged over the prompt's tokens (text) or a camera's picture tokens")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    from openpi.serving import websocket_policy_server
    policy = create_policy(args.config, args.checkpoint, args.center_crop, args.default_prompt)
    name = f"block{args.layer}:{OLD_POOLS[args.pool]}"
    served = StatePolicy(policy, openpi_tap.attach(policy, layer=args.layer, pool=args.pool), name)
    log.info("serving %s with its state %s", args.checkpoint, served.metadata[KEY]["names"])
    websocket_policy_server.WebsocketPolicyServer(policy=served, host=args.host, port=args.port,
                                                  metadata=served.metadata).serve_forever()


if __name__ == "__main__":
    main()
