"""Forward kinematics of the Franka arm (FER / Panda): where the tool centre point (TCP) is for 7 joint angles.

The memory sentence (sentence.py) says how far the hand moves during a stored moment's next actions: the TCP position of
the chunk's last joint target minus the TCP position of the current joints. Only the position is needed, so this is the
plain chain of homogeneous transforms of the robot's URDF: the base pose in the world; for every joint its fixed origin
(xyz, roll-pitch-yaw) followed by its rotation about its own z axis; then the fixed flange, hand and TCP frames.

The numbers are those of the reference bench: the joint origins of the FER URDF (fer_isaac.urdf of franka_description) and
the bench's base pose (at (-0.4, -0.65, 0.75) m, turned +90 degrees about z). Positions are in the bench's world frame,
whose axes the sentence names x forward, y left, z up.

Another arm needs its own chain: its joint origins, its tool frames and its base pose (the arguments of FrankaKinematics),
or any object with a method tcp_position(joints) -> (x, y, z).
"""
from __future__ import annotations

import math

import numpy as np

# fer_isaac.urdf: the origin (xyz, rpy) of fer_joint1..7; every joint turns about the z axis of its own frame
FER_JOINTS = (((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
              ((0.0, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0)),
              ((0.0, -0.316, 0.0), (math.pi / 2, 0.0, 0.0)),
              ((0.0825, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)),
              ((-0.0825, 0.384, 0.0), (-math.pi / 2, 0.0, 0.0)),
              ((0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)),
              ((0.088, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)))
# the fixed frames after the last joint: the flange (fer_joint8), the hand (fer_hand_joint), the TCP between the fingertips
FER_TOOL = (((0.0, 0.0, 0.107), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, -math.pi / 4)),
            ((0.0, 0.0, 0.1034), (0.0, 0.0, 0.0)))
BENCH_BASE_POS = (-0.4, -0.65, 0.75)              # m: the arm's base in the bench's world
BENCH_BASE_QUAT = (0.70711, 0.0, 0.0, 0.70711)    # (w, x, y, z): turned +90 degrees about z


def rpy_matrix(rpy) -> np.ndarray:
    """The rotation of URDF roll, pitch and yaw (fixed axes x, y, z): Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = (float(angle) for angle in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quaternion_matrix(quat) -> np.ndarray:
    """The rotation of a quaternion (w, x, y, z), normalised first."""
    q = np.asarray(quat, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def transform(xyz, rotation) -> np.ndarray:
    """The 4 x 4 homogeneous transform of a translation (m) and a rotation matrix."""
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = xyz
    return out


def z_turn(cos: float, sin: float) -> np.ndarray:
    """The 4 x 4 rotation about z by the angle with this cosine and sine (a revolute joint)."""
    out = np.eye(4)
    out[0, 0] = cos
    out[0, 1] = -sin
    out[1, 0] = sin
    out[1, 1] = cos
    return out


class FrankaKinematics:
    """The TCP position (m, world) for the joint angles (rad) of a serial arm, by default the Franka arm of the bench.

    joints: the origin (xyz, rpy) of every joint; tool: the fixed frames after the last joint; base_pos, base_quat: the base
    in the world (position, quaternion w, x, y, z)."""

    def __init__(self, joints=FER_JOINTS, tool=FER_TOOL, base_pos=BENCH_BASE_POS, base_quat=BENCH_BASE_QUAT):
        self.origins = [transform(xyz, rpy_matrix(rpy)) for xyz, rpy in joints]
        self.tool = np.eye(4)
        for xyz, rpy in tool:
            self.tool = self.tool @ transform(xyz, rpy_matrix(rpy))
        self.base = transform(base_pos, quaternion_matrix(base_quat))

    def tcp_position(self, joints) -> np.ndarray:
        """(3,): the TCP position of one configuration (one angle per joint, rad)."""
        q = np.asarray(joints, float)
        if q.shape != (len(self.origins),):
            raise ValueError(f"{len(self.origins)} joint angles expected, got shape {q.shape}")
        pose = self.base
        for origin, cos, sin in zip(self.origins, np.cos(q), np.sin(q)):
            pose = pose @ origin @ z_turn(cos, sin)
        return (pose @ self.tool)[:3, 3]
