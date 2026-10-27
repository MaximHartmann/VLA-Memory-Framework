"""A policy server for any LeRobot policy, in openpi's serving protocol (msgpack over a websocket, what the framework's proxy
speaks), with the policy's state in every reply for the head writer (docs/writers/head).

    python -m lerobot_server --policy <hub id or folder> --port 8000 [--device cuda] \
        [--camera observation/image=observation.images.camera1 ...] [--state observation/state=observation.state]

The policy and its pre- and post-processors are loaded with LeRobot's own functions, for any policy type; a policy whose
config would load its backbone's weights separately (load_vlm_weights) takes them from its own weights file. At every call
the observation's keys are renamed to the policy's inputs (--camera, --state; by default openpi's pictures observation/image
and observation/wrist_image become the policy's cameras in order, and observation/state its state), a uint8 picture becomes
the float tensor LeRobot expects, and the prompt becomes the task. The reply holds the action chunk, "actions" (steps,
action size), and "policy_state" (vla_memory/taps/reply.py): the automatic tap's summary of the policy's language backbone
(vla_memory/taps/auto.py), two thirds of the way up and averaged over the prompt's words, as every server of the framework
hands it out ("block11:prompt" for SmolVLA). memory/new_episode True resets the policy. LeRobotPolicy is the policy part,
serve() the network part. Runs in a LeRobot environment with the framework, this folder, openpi_client and websockets on
the path.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import traceback

import numpy as np
import torch

from vla_memory.taps.auto import AutoTap
from vla_memory.taps.reply import KEY, state_entry

log = logging.getLogger("lerobot_server")
OPENPI_CAMERAS = ("observation/image", "observation/wrist_image")      # the pictures openpi's clients send, in this order
OPENPI_STATE = "observation/state"


def default_keys(features: dict) -> tuple[dict, dict]:
    """The default renaming: openpi's pictures in order to the policy's cameras in order, openpi's state to the policy's.
    features: {name: (kind, shape)} with kind "VISUAL" or "STATE", in the order of the policy's config."""
    cameras = [name for name, (kind, _) in features.items() if kind == "VISUAL"]
    states = [name for name, (kind, _) in features.items() if kind == "STATE"]
    return dict(zip(OPENPI_CAMERAS, cameras)), dict(zip([OPENPI_STATE], states))


def picture(array, size) -> torch.Tensor:
    """(1, 3, H, W) float in [0, 1] at the policy's size from a uint8 picture, H x W x 3 or 3 x H x W (resized bilinearly,
    as LeRobot's own policy server does)."""
    image = torch.as_tensor(np.asarray(array))
    image = image.permute(2, 0, 1) if image.shape[-1] == 3 else image
    image = image[None].float() / (255 if image.dtype == torch.uint8 else 1)
    if tuple(image.shape[-2:]) != tuple(size):
        image = torch.nn.functional.interpolate(image, size=tuple(size), mode="bilinear", align_corners=False)
    return image


def state_vector(values, size: int) -> torch.Tensor:
    """(1, size) float32: the state, cut or zero-padded to the policy's size."""
    state = np.zeros(size, np.float32)
    values = np.asarray(values, np.float32).reshape(-1)[:size]
    state[:len(values)] = values
    return torch.as_tensor(state)[None]


def lerobot_batch(observation: dict, cameras: dict, state: dict, features: dict, default_prompt: str = "") -> dict:
    """The policy's input from an openpi-style observation: the renamed pictures and state, and the prompt as the task.
    Keys that are not renamed (the memory/ control keys among them) stay out."""
    batch = {"task": str(observation.get("prompt", default_prompt))}
    for key, name in cameras.items():
        if key in observation:
            batch[name] = picture(observation[key], features[name][1][1:])
    for key, name in state.items():
        if key in observation:
            batch[name] = state_vector(observation[key], features[name][1][0])
    return batch


class LeRobotPolicy:
    """A LeRobot policy with openpi's interface: infer(observation) -> {"actions", "policy_state"}, and metadata."""

    def __init__(self, policy, preprocessor, postprocessor, cameras: dict | None = None, state: dict | None = None,
                 name: str = "", default_prompt: str = ""):
        self.policy, self.preprocessor, self.postprocessor = policy, preprocessor, postprocessor
        self.features = {key: (feature.type.name, tuple(feature.shape))
                         for key, feature in policy.config.input_features.items()}
        default_cameras, default_state = default_keys(self.features)
        self.cameras, self.state = cameras or default_cameras, state or default_state
        self.name, self.default_prompt = name, default_prompt
        try:
            self.tap = AutoTap(policy)
        except ValueError as error:          # no stack of transformer blocks (a diffusion policy's U-Net): actions only
            log.warning("the policy's state cannot be handed out: %s", error)
            self.tap = None

    def infer(self, observation: dict) -> dict:
        """The policy's action chunk for one observation, and its state; memory/new_episode resets the policy first."""
        if observation.get("memory/new_episode"):
            self.policy.reset()
        batch = self.preprocessor(lerobot_batch(observation, self.cameras, self.state, self.features, self.default_prompt))
        if self.tap is not None:
            self.tap.start_call()
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch)
        chunk = chunk[:, None] if chunk.dim() == 2 else chunk
        actions = torch.stack([self.postprocessor(chunk[:, step]) for step in range(chunk.shape[1])], 1)
        reply = {"actions": actions[0].float().cpu().numpy()}
        if self.tap is not None:
            values = self.tap.read()                                 # its first read() also sets the names
            reply[KEY] = state_entry(self.tap.names, values)
        return reply

    def blank_observation(self) -> dict:
        """Black pictures, a zero state at the policy's sizes and a short prompt: the warm-up call's observation."""
        observation = {key: np.zeros((*self.features[name][1][1:], 3), np.uint8) for key, name in self.cameras.items()}
        observation["prompt"] = self.default_prompt or "warm up"
        observation.update({key: np.zeros(self.features[name][1][0], np.float32) for key, name in self.state.items()})
        return observation

    @property
    def metadata(self) -> dict:
        """The policy, the renaming of the observation's keys, and the state's names (after a first call) and stack."""
        tap = self.tap
        return {"policy": self.name, "policy_type": self.policy.config.type, "cameras": self.cameras, "state": self.state,
                KEY: {"names": list(tap.names) if tap else [], "stack": tap.stack if tap else None}}


def load_policy(path: str, device: str | None = None):
    """(policy, preprocessor, postprocessor) of the LeRobot policy at `path` (a hub id or a folder), for any policy type."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    config = PreTrainedConfig.from_pretrained(path)
    if getattr(config, "load_vlm_weights", False):          # the policy's own weights file holds its backbone too
        config.load_vlm_weights = False
    config.device = device or config.device
    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": config.device}})
    return policy.eval(), preprocessor, postprocessor


async def serve(served, host: str, port: int) -> None:
    """Serve until SIGTERM or Ctrl-C: every client first gets the metadata, then one reply per observation; calls run one
    after another, so that every reply holds the state of its own call. At the stop the open connections are closed."""
    import websockets.asyncio.server
    import websockets.exceptions
    import websockets.frames
    from openpi_client import msgpack_numpy
    from vla_memory.proxy.server import health_check
    one_call = asyncio.Lock()
    stop = asyncio.Event()
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(signal_number, stop.set)

    async def handle(websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(served.metadata))
        while True:
            try:
                observation = msgpack_numpy.unpackb(await websocket.recv())
                async with one_call:
                    reply = await asyncio.to_thread(served.infer, observation)
                await websocket.send(packer.pack(reply))
            except websockets.exceptions.ConnectionClosed:
                log.info("client disconnected")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="policy error")
                raise

    async with websockets.asyncio.server.serve(handle, host, port, compression=None, max_size=None,
                                               process_request=health_check):
        log.info("listening on %s:%s", host, port)
        await stop.wait()
        log.info("stopping: closing the open connections")


def pair(text: str) -> tuple[str, str]:
    """KEY=FEATURE from the command line."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"{text!r}: expected KEY=FEATURE")
    key, feature = text.split("=", 1)
    return key, feature


def main(argv=None) -> None:
    """Load the policy, make one warm-up call (the tap picks its blocks, which the metadata names), and serve it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="a hub id or a folder with the policy's config, weights, processors")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default=None, help="the policy's device (default: its config's)")
    parser.add_argument("--camera", type=pair, action="append", default=[], metavar="KEY=FEATURE",
                        help="an observation's picture and the policy's camera it is (repeatable; default: openpi's "
                             "pictures observation/image, observation/wrist_image to the policy's cameras in order)")
    parser.add_argument("--state", type=pair, action="append", default=[], metavar="KEY=FEATURE",
                        help="the observation's state and the policy's (default: observation/state to the policy's state)")
    parser.add_argument("--default-prompt", default="", help="the task when an observation has no prompt")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    served = LeRobotPolicy(*load_policy(args.policy, args.device), cameras=dict(args.camera), state=dict(args.state),
                           name=args.policy, default_prompt=args.default_prompt)
    served.infer(served.blank_observation())
    served.policy.reset()
    log.info("serving %s: cameras %s, state %s, policy state %s", args.policy, served.cameras, served.state,
             served.metadata[KEY]["names"])
    asyncio.run(serve(served, args.host, args.port))


if __name__ == "__main__":
    main()
