"""Policy transforms for the red/blue sequential pick-and-place task.

Modelled on ``libero_policy``, which our dataset matches in shape (8-D state,
two camera views) but NOT in semantics. Two differences are load-bearing:

  * STATE/ACTION ARE JOINT SPACE, NOT EEF. ``state`` is the 7 *measured* Franka
    arm joint positions plus one normalised gripper-closedness scalar.
    ``actions`` is 7 joint VELOCITIES -- generate_demos.py:301 records
    ``(q_cmd - q_meas) / dt`` -- plus one absolute gripper command. LIBERO's
    8-D state is end-effector pose instead, but nothing here depends on which:
    the vector is passed through untouched.

    Because the actions are already relative to the measured state, the data
    config applies NO delta transform. Adding one subtracts the state twice and
    silently corrupts the action distribution.

  * ACTION DIMENSION IS 8, NOT 7. ``LiberoOutputs`` hardcodes
    ``actions[..., :7]`` because LIBERO has 6 EEF deltas + 1 gripper. Reusing it
    for this dataset would slice off the gripper command, and a pick-and-place
    policy whose gripper never actuates fails in a way that looks like a
    grasping bug rather than a plumbing bug. Hence a separate Outputs class.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 7 arm joints + 1 gripper. Kept as a named constant because it appears both
# here and in the delta-action mask in training/config.py; the two must agree.
ACTION_DIM = 8


def make_red_blue_example() -> dict:
    """Creates a random input example for the red/blue policy."""
    return {
        "observation/state": np.random.rand(ACTION_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the red block with your arm, place it back, and then grasp the blue block",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RedBlueInputs(transforms.DataTransformFn):
    """Dataset/inference inputs -> model inputs. Used for both training and inference."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # LeRobot hands back float32 (C,H,W); the sim hands back uint8 (H,W,C).
        # _parse_image normalises both so training and inference agree.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Single-arm setup: there is no right wrist camera, so pad with
                # zeros and mask it out below.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only present during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RedBlueOutputs(transforms.DataTransformFn):
    """Model outputs -> dataset format. Inference only."""

    def __call__(self, data: dict) -> dict:
        # Actions were padded up to the model action dimension on the way in;
        # take back exactly our 8 (7 joints + gripper). See module docstring for
        # why this is 8 and not LIBERO's 7.
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
