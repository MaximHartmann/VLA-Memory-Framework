"""The policy server of the event-head writer: openpi's websocket policy server, with the head's verdict in every reply.

Runs in the openpi environment (openpi with our changes, training/openpi/) with the framework on PYTHONPATH:

    python -m vla_memory.writers.head.server --port <port> --config pi05_red_blue_v2_history \
        --checkpoint <checkpoint dir> --head-dir <head dir> [--center-crop 0.95] [--head-batch 8] [--jax-cache DIR]

Every call runs the policy's infer() unchanged. An identity wrapped around its input transform keeps the transformed
observation, and after the call the head scores it (features.LayerFeature, file.EventHead). The reply gets one more
entry, which the event-head writer reads (writer.py, --writer head):

    reply["event_head"] = {"score": float, "done": bool, "threshold": float, "ms": the time of the head pass}

The head pass is compiled in the first call, after the policy's own first call. --head-batch 1 is about four times faster
but does not reproduce the training numerics (features.py); --jax-cache names JAX's persistent compilation cache, so
that the kernels chosen once are used again.
"""
from __future__ import annotations

import argparse
import logging
import time

from .file import load_head

log = logging.getLogger("vla_memory.writers.head.server")


class HeadPolicy:
    """The policy, and after every call the head's verdict on the transformed observation that call used."""

    def __init__(self, policy, feature, head):
        self._policy = policy
        self._feature = feature
        self._head = head
        self._inputs = None
        transform = policy._input_transform

        def keep(data):
            """The policy's input transform, with its result kept for the head: the policy gets exactly what it returns."""
            self._inputs = transform(data)
            return self._inputs

        policy._input_transform = keep

    def infer(self, observation: dict) -> dict:
        """The policy's reply to one observation, with the head's verdict on that call added as event_head."""
        self._inputs = None
        outputs = self._policy.infer(observation)
        started = time.perf_counter()
        verdict = self._head.verdict(self._feature.feature(self._inputs))
        verdict["ms"] = 1000 * (time.perf_counter() - started)
        outputs["event_head"] = verdict
        return outputs

    @property
    def metadata(self) -> dict:
        """The policy's metadata plus the head's threshold, N, layer, pool, batch shape and digests."""
        metadata = dict(self._policy.metadata)
        metadata["event_head"] = {"threshold": self._head.threshold, "N": self._head.consecutive, "layer": self._head.layer,
                                  "pool": self._head.pool, "batch": self._feature.batch, "sha256": self._head.meta.get("sha256")}
        return metadata


def build_parser() -> argparse.ArgumentParser:
    """The command line: the server's port, the policy's config and checkpoint, the head's directory and the head pass."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--config", required=True, help="the openpi training config of the policy")
    parser.add_argument("--checkpoint", required=True, help="the checkpoint directory of the policy")
    parser.add_argument("--center-crop", type=float, default=0.0,
                        help="centre crop of the exterior image at inference (0: off; the reference policy runs 0.95)")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--head-dir", required=True, help="the head's directory (head.npz, head.json, head.sha256)")
    parser.add_argument("--head-batch", type=int, default=8,
                        help="batch shape of the head pass; 8 = the batch shape of the training rows (default)")
    parser.add_argument("--jax-cache", default=None, help="JAX's persistent compilation cache directory")
    return parser


def use_compilation_cache(directory: str) -> None:
    """Keep every compiled program in JAX's persistent cache, and load it from there when it exists."""
    import jax
    jax.config.update("jax_compilation_cache_dir", directory)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def create_policy(args):
    """The trained policy as openpi's serve_policy.py creates it, with the optional centre crop."""
    from openpi import transforms
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config
    extra_transforms = [transforms.CenterCropResizeImages(args.center_crop)] if args.center_crop else []
    train_config = openpi_config.get_config(args.config)
    policy = policy_config.create_trained_policy(train_config, args.checkpoint, default_prompt=args.default_prompt,
                                                 extra_input_transforms=extra_transforms)
    return policy, train_config


def main(argv=None) -> None:
    """Load the head and the policy, then serve the policy with the head's verdict in every reply."""
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_parser().parse_args(argv)
    if args.jax_cache:
        use_compilation_cache(args.jax_cache)
    from openpi.serving import websocket_policy_server
    from .features import LayerFeature
    head = load_head(args.head_dir)
    policy, train_config = create_policy(args)
    feature = LayerFeature(policy, train_config, layer=head.layer, pool_name=head.pool, batch=args.head_batch)
    served = HeadPolicy(policy, feature, head)
    log.info("event head %s: layer %d %s, threshold %.6g, N %d, batch %d", args.head_dir, head.layer, head.pool,
             head.threshold, head.consecutive, feature.batch)
    server = websocket_policy_server.WebsocketPolicyServer(policy=served, host=args.host, port=args.port,
                                                           metadata=served.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
